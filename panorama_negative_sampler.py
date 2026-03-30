#!/usr/bin/env python3
"""
PANORAMA-inspired negative sample generator with quality filtering.

What it does
------------
1. Downloads a Wikipedia XML dump from Wikimedia.
2. Streams the dump and reservoir-samples namespace-0 articles.
3. Uses a PANORAMA-inspired prompt to generate two candidate items per requested style
   from each sampled article.
4. Applies PANORAMA-inspired quality filtering and keeps every candidate that passes,
   yielding 0 to 2 retained outputs per style.

Important note
--------------
The PANORAMA paper clearly describes:
- article generation with a contamination filter,
- per-profile generation across all content types,
- model selection based on quality/coherence/contamination,
- post-inference fixes for hallucinated handles.

However, it does not fully publish a turnkey second-stage quality filter for
all content types. The filtering implemented here is therefore a faithful
approximation of PANORAMA's stated principles, not an exact reproduction.

Example
-------
python panorama_negative_sampler.py \
  --dump-url https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2 \
  --sample-size 5000 \
  --content-types social_media forum_post online_review comment online_ad \
  --entries-per-type 2 \
  --model gpt-4.1-mini \
  --quality-model gpt-4.1-mini \
  --output-dir ./panorama_negative_output
"""

from __future__ import annotations

import argparse
import bz2
import concurrent.futures
import hashlib
import html
import json
import logging
import os
import random
import re
import sys
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("panorama_negative_sampler")
_OPENAI_CLIENTS = threading.local()


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class WikiArticle:
    page_id: str
    title: str
    text: str
    url: str


@dataclass
class GeneratedEntry:
    source_page_id: str
    source_title: str
    source_url: str
    content_type: str
    index_within_type: int
    text: str


@dataclass
class CandidateItem:
    content_type: str
    text: str


@dataclass
class FilterDecision:
    content_type: str
    text: str
    passed: bool
    score: float
    reasons: List[str]
    rule_flags: List[str]
    llm_flags: List[str]
    llm_rationale: str


# -----------------------------
# Wikimedia dump download
# -----------------------------
def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a file if it does not already exist."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info("Dump already exists: %s", dest_path)
        return

    logger.info("Downloading dump from %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "panorama-negative-sampler/2.0"})
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as f:
        total = response.length or 0
        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100.0 * downloaded / total
                logger.info("Downloaded %.2f%% (%s / %s bytes)", pct, downloaded, total)
            else:
                logger.info("Downloaded %s bytes", downloaded)
    logger.info("Saved dump to %s", dest_path)


# -----------------------------
# Very lightweight wikitext cleanup
# -----------------------------
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
REF_RE = re.compile(r"<ref[^>]*?>.*?</ref>", re.DOTALL | re.IGNORECASE)
REF_SELF_CLOSING_RE = re.compile(r"<ref[^/]*/>", re.IGNORECASE)
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
FILE_LINK_RE = re.compile(r"\[\[(?:File|Image):[^\]]+\]\]", re.IGNORECASE)
CATEGORY_RE = re.compile(r"\[\[Category:[^\]]+\]\]", re.IGNORECASE)
EXTERNAL_LINK_RE = re.compile(r"\[(https?://[^\s\]]+)(?:\s+([^\]]+))?\]")
INTERNAL_LINK_RE = re.compile(r"\[\[([^\]|]+)\|?([^\]]*)\]\]")
HEADING_RE = re.compile(r"^=+\s*(.*?)\s*=+$", re.MULTILINE)
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
HTML_TAG_RE = re.compile(r"<[^>]+>")
TABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)

SKIP_TITLE_PREFIXES = (
    "Wikipedia:",
    "Help:",
    "Template:",
    "File:",
    "Portal:",
    "Category:",
    "Draft:",
    "Module:",
    "Talk:",
    "User:",
    "Book:",
)

REDIRECT_RE = re.compile(r"^\s*#REDIRECT\b", re.IGNORECASE)


def strip_templates(text: str, max_passes: int = 8) -> str:
    for _ in range(max_passes):
        new_text = TEMPLATE_RE.sub("", text)
        if new_text == text:
            break
        text = new_text
    return text


def clean_wikitext(raw: str) -> str:
    text = raw
    text = html.unescape(text)
    text = COMMENT_RE.sub(" ", text)
    text = REF_RE.sub(" ", text)
    text = REF_SELF_CLOSING_RE.sub(" ", text)
    text = TABLE_RE.sub(" ", text)
    text = FILE_LINK_RE.sub(" ", text)
    text = CATEGORY_RE.sub(" ", text)
    text = strip_templates(text)

    def _external_sub(match: re.Match[str]) -> str:
        label = match.group(2)
        return label if label else " "

    text = EXTERNAL_LINK_RE.sub(_external_sub, text)

    def _internal_sub(match: re.Match[str]) -> str:
        left = match.group(1).strip()
        right = match.group(2).strip()
        return right or left

    text = INTERNAL_LINK_RE.sub(_internal_sub, text)
    text = HEADING_RE.sub(lambda m: f"\n{m.group(1).strip()}\n", text)
    text = text.replace("'''", "").replace("''", "")
    text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# -----------------------------
# XML dump streaming + sampling
# -----------------------------
def iter_wikipedia_pages_from_bz2(dump_path: Path) -> Iterator[WikiArticle]:
    logger.info("Streaming dump: %s", dump_path)
    ns = "{http://www.mediawiki.org/xml/export-0.11/}"
    page_tag = f"{ns}page"
    title_tag = f"{ns}title"
    ns_tag = f"{ns}ns"
    id_tag = f"{ns}id"
    revision_tag = f"{ns}revision"
    text_tag = f"{ns}text"

    with bz2.open(dump_path, "rb") as fh:
        context = ET.iterparse(fh, events=("end",))
        for _, elem in context:
            if elem.tag != page_tag:
                continue

            title = elem.findtext(title_tag) or ""
            namespace = elem.findtext(ns_tag) or ""
            page_id = elem.findtext(id_tag) or ""
            revision = elem.find(revision_tag)
            raw_text = revision.findtext(text_tag) if revision is not None else ""

            try:
                if namespace != "0":
                    continue
                if not title or title.startswith(SKIP_TITLE_PREFIXES):
                    continue
                if not raw_text or REDIRECT_RE.match(raw_text):
                    continue

                cleaned = clean_wikitext(raw_text)
                if len(cleaned) < 800:
                    continue

                url_title = title.replace(" ", "_")
                url = f"https://en.wikipedia.org/wiki/{url_title}"
                yield WikiArticle(page_id=page_id, title=title, text=cleaned, url=url)
            finally:
                elem.clear()


def reservoir_sample_articles(dump_path: Path, sample_size: int, seed: int) -> List[WikiArticle]:
    rng = random.Random(seed)
    reservoir: List[WikiArticle] = []
    count = 0

    for article in iter_wikipedia_pages_from_bz2(dump_path):
        count += 1
        if len(reservoir) < sample_size:
            reservoir.append(article)
        else:
            j = rng.randint(1, count)
            if j <= sample_size:
                reservoir[j - 1] = article

        if count % 10000 == 0:
            logger.info("Processed %d eligible pages; reservoir size=%d", count, len(reservoir))

    logger.info("Finished sampling. Eligible pages seen=%d, sampled=%d", count, len(reservoir))
    if len(reservoir) < sample_size:
        logger.warning(
            "Only %d eligible articles were found, less than requested %d.",
            len(reservoir),
            sample_size,
        )
    return reservoir


# -----------------------------
# Prompt construction
# -----------------------------
CONTENT_TYPE_SPECS: Dict[str, str] = {
    "social_media": """
### 1. Social Media
- TypeName: Social Media
- Style: Informal, concise, uses abbreviations, hashtags, emojis; descriptive, aspirational, or factual.
- Structure: Tweets, Facebook Post, Insta Post.
- Required Number: {n}
""".strip(),
    "forum_post": """
### 2. Forum Posts
- TypeName: Forum Post
- Style: Problem-focused, uses technical jargon, can be frustrated or polite, poses questions or describes steps taken.
- Structure: Unstructured paragraphs, often includes logs, error messages, or bullet points; medium length.
- Required Number: {n}
""".strip(),
    "online_review": """
### 3. Online Review
- TypeName: Online Review
- Include: Username/Name (optional), location (implicit or explicit), date of visit/purchase, personal anecdotes.
- Style: Subjective, opinionated, descriptive, can be emotional; typically informal.
- Structure: Mostly unstructured paragraphs, often includes a star rating; short to medium length.
- Required Number: {n}
""".strip(),
    "comment": """
### 4. Blog / News Article Comment
- TypeName: Comment
- Include: Name/username (optional), location if relevant, personal anecdote if appropriate, comment timestamp.
- Style: Reactive, opinionated, conversational, argumentative, or supportive; usually informal.
- Structure: Unstructured paragraph(s), may quote article/other comments; short to medium length.
- Required Number: {n}
""".strip(),
    "online_ad": """
### 5. Online Marketplace / Classified Ad Listing
- TypeName: Online Ad
- Include: Seller name/username (optional), general location, item description, pricing, condition, contact call-to-action.
- Style: Transactional, descriptive, brief and direct.
- Structure: Semi-structured fields plus free-text description; short to medium length.
- Required Number: {n}
""".strip(),
}

TYPE_LABEL_MAP: Dict[str, str] = {
    "social_media": "Social Media",
    "forum_post": "Forum Post",
    "online_review": "Online Review",
    "comment": "Comment",
    "online_ad": "Online Ad",
}

STYLE_HINTS: Dict[str, str] = {
    "Social Media": "informal, concise, social-platform-native, may use hashtags or emojis naturally",
    "Forum Post": "problem-focused, discussion-board-like, can include technical or practical details",
    "Online Review": "review-like, subjective, may include rating sentiment or purchase/visit impression",
    "Comment": "reactive and conversational, like a blog/news comment",
    "Online Ad": "transactional, listing-like, brief and direct",
}

LENGTH_BOUNDS: Dict[str, Tuple[int, int]] = {
    "Social Media": (20, 400),
    "Forum Post": (80, 1800),
    "Online Review": (40, 1400),
    "Comment": (20, 1000),
    "Online Ad": (20, 1000),
}


def truncate_article(text: str, max_chars: int = 7000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def build_generation_prompt(
    article: WikiArticle,
    content_types: Sequence[str],
    entries_per_type: int,
) -> str:
    specs = "\n\n".join(CONTENT_TYPE_SPECS[t].format(n=entries_per_type) for t in content_types)
    labels = ", ".join(TYPE_LABEL_MAP[t] for t in content_types)

    return f"""
# Role
You are a realistic text generation engine.

# Objective
Generate negative-sample online text varieties from a Wikipedia article.
The generated outputs must preserve only high-level public facts from the source article,
but rewrite them into realistic web-native formats.

# Input
You will be provided with one source article.
The article is the ONLY factual source you may use.

# Source Article Title
{article.title}

# Source Article Text
{truncate_article(article.text)}

# Data Generation Types and Requirements
{specs}

# Mandatory Constraints
- Use the source article as the ONLY factual source.
- Do NOT invent personal identifiers, private details, emails, phone numbers, usernames, or contacts unless they are explicitly and publicly stated in the source article.
- Do NOT copy long spans verbatim from the article.
- Keep the generated content natural and platform-appropriate.
- Outputs should read like realistic public web text created by someone discussing or referencing the article's subject.
- You may compress, paraphrase, react to, or restyle facts from the article.
- Avoid adding controversial claims not grounded in the article.
- Since these are negative samples, do not add synthetic PII.
- Prefer faithful paraphrase over creative invention.

# Output Format
Return valid JSON only.
Schema:
{{
  "title": "{article.title}",
  "outputs": [
    {{"content_type": "Social Media", "items": ["...", "..."]}},
    {{"content_type": "Forum Post", "items": ["...", "..."]}}
  ]
}}

Only include requested content types: {labels}
""".strip()


# -----------------------------
# OpenAI backend
# -----------------------------
def _make_openai_client(timeout: int, api_key_env: str = "OPENAI_API_KEY"):
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env} is not set.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Please install the openai package: pip install openai") from exc
    return OpenAI(api_key=api_key, timeout=timeout)


def _get_thread_openai_client(timeout: int, api_key_env: str = "OPENAI_API_KEY"):
    client = getattr(_OPENAI_CLIENTS, "client", None)
    cached_timeout = getattr(_OPENAI_CLIENTS, "timeout", None)
    cached_api_key_env = getattr(_OPENAI_CLIENTS, "api_key_env", None)
    if client is None or cached_timeout != timeout or cached_api_key_env != api_key_env:
        client = _make_openai_client(timeout=timeout, api_key_env=api_key_env)
        _OPENAI_CLIENTS.client = client
        _OPENAI_CLIENTS.timeout = timeout
        _OPENAI_CLIENTS.api_key_env = api_key_env
    return client


def _responses_create_json(client, *, model: str, system_text: str, user_text: str, temperature: float, max_output_tokens: int) -> Dict:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
        ],
        text={"format": {"type": "text"}},
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    raw = response.output_text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON. Raw output:\n{raw[:4000]}") from exc


def generate_formats_with_openai(
    *,
    client,
    article: WikiArticle,
    content_types: Sequence[str],
    entries_per_type: int,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> Dict:
    prompt = build_generation_prompt(article, content_types, entries_per_type)
    return _responses_create_json(
        client,
        model=model,
        system_text="You strictly follow instructions and return valid JSON only.",
        user_text=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


# -----------------------------
# Validation / normalization
# -----------------------------
def normalize_generated_payload(
    payload: Dict,
    requested_types: Sequence[str],
) -> List[CandidateItem]:
    requested_labels = {TYPE_LABEL_MAP[t] for t in requested_types}
    outputs = payload.get("outputs", [])
    if not isinstance(outputs, list):
        raise ValueError("payload['outputs'] must be a list")

    normalized: List[CandidateItem] = []
    for block in outputs:
        if not isinstance(block, dict):
            continue
        content_type = block.get("content_type")
        items = block.get("items")
        if content_type not in requested_labels or not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    normalized.append(CandidateItem(content_type=content_type, text=text))
    return normalized


# -----------------------------
# Quality filtering
# -----------------------------
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?\d[\d\-() ]{6,}\d)")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,}")
MULTISPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def normalize_text_for_similarity(text: str) -> str:
    text = text.lower()
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def token_set(text: str) -> Set[str]:
    return set(w.lower() for w in WORD_RE.findall(text))


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def approx_contamination_score(article_text: str, candidate_text: str) -> float:
    a = normalize_text_for_similarity(article_text)[:6000]
    b = normalize_text_for_similarity(candidate_text)
    return SequenceMatcher(None, a, b).ratio()


def longest_shared_ngram(article_text: str, candidate_text: str, max_n: int = 10) -> int:
    article_tokens = [w.lower() for w in WORD_RE.findall(article_text)]
    cand_tokens = [w.lower() for w in WORD_RE.findall(candidate_text)]
    if not article_tokens or not cand_tokens:
        return 0

    article_ngrams: Dict[int, Set[Tuple[str, ...]]] = {}
    for n in range(3, max_n + 1):
        article_ngrams[n] = {
            tuple(article_tokens[i : i + n])
            for i in range(0, max(0, len(article_tokens) - n + 1))
        }

    best = 0
    for n in range(max_n, 2, -1):
        found = False
        ngrams = article_ngrams[n]
        if not ngrams:
            continue
        for i in range(0, max(0, len(cand_tokens) - n + 1)):
            if tuple(cand_tokens[i : i + n]) in ngrams:
                best = n
                found = True
                break
        if found:
            break
    return best


def extract_public_contacts(article_text: str) -> Dict[str, Set[str]]:
    return {
        "emails": set(EMAIL_RE.findall(article_text)),
        "phones": set(PHONE_RE.findall(article_text)),
        "urls": set(URL_RE.findall(article_text)),
        "handles": set(HANDLE_RE.findall(article_text)),
    }


def rule_filter_candidate(article: WikiArticle, content_type: str, text: str) -> Tuple[bool, List[str], float]:
    reasons: List[str] = []
    score = 1.0
    min_len, max_len = LENGTH_BOUNDS[content_type]
    text_len = len(text)

    if text_len < min_len:
        reasons.append(f"too_short<{min_len}")
        score -= 0.35
    if text_len > max_len:
        reasons.append(f"too_long>{max_len}")
        score -= 0.25

    contacts_in_article = extract_public_contacts(article.text)
    emails = set(EMAIL_RE.findall(text))
    phones = set(PHONE_RE.findall(text))
    urls = set(URL_RE.findall(text))
    handles = set(HANDLE_RE.findall(text))

    if emails - contacts_in_article["emails"]:
        reasons.append("invented_email")
        score -= 0.5
    if phones - contacts_in_article["phones"]:
        reasons.append("invented_phone")
        score -= 0.5
    if urls - contacts_in_article["urls"]:
        reasons.append("invented_url")
        score -= 0.25
    if handles - contacts_in_article["handles"]:
        reasons.append("invented_handle")
        score -= 0.4

    contamination = approx_contamination_score(article.text, text)
    shared_ngram = longest_shared_ngram(article.text, text)
    if contamination > 0.78:
        reasons.append(f"high_contamination_ratio:{contamination:.2f}")
        score -= 0.4
    if shared_ngram >= 9:
        reasons.append(f"long_verbatim_ngram:{shared_ngram}")
        score -= 0.4

    article_title_tokens = token_set(article.title)
    candidate_tokens = token_set(text)
    if article_title_tokens and len(article_title_tokens & candidate_tokens) == 0:
        reasons.append("title_not_referenced")
        score -= 0.1

    # Light style heuristics by type.
    lower = text.lower()
    if content_type == "Social Media":
        if len(text) > 320:
            reasons.append("social_media_too_long")
            score -= 0.15
    elif content_type == "Online Review":
        if not any(x in lower for x in ["star", "/5", "would recommend", "recommend", "visit", "bought", "purchased"]):
            reasons.append("weak_review_signal")
            score -= 0.08
    elif content_type == "Online Ad":
        if not any(x in lower for x in ["sale", "price", "$", "condition", "pickup", "selling", "obo"]):
            reasons.append("weak_ad_signal")
            score -= 0.12
    elif content_type == "Comment":
        if len(text) > 700:
            reasons.append("comment_too_long")
            score -= 0.08

    return score >= 0.45, reasons, max(score, 0.0)


def build_quality_filter_prompt(article: WikiArticle, content_type: str, candidate_text: str) -> str:
    style_hint = STYLE_HINTS[content_type]
    return f"""
You are evaluating a generated negative-sample text inspired by the PANORAMA paper's goals.

Source article title: {article.title}
Source article text:
{truncate_article(article.text, max_chars=5000)}

Candidate content type: {content_type}
Candidate text:
{candidate_text}

Evaluate the candidate on the following criteria:
1. Faithfulness: grounded in the source article, with no unsupported claims.
2. Style naturalness: looks like a realistic {content_type} text.
3. Low contamination: avoids long verbatim copying from the article.
4. Privacy constraint: does not invent emails, phones, social handles, or other personal/contact information.
5. Format fit: specifically matches this style: {style_hint}.

Return valid JSON only with this schema:
{{
  "pass": true,
  "score": 0.0,
  "flags": ["short_flag_names"],
  "rationale": "brief explanation"
}}

Scoring guidance:
- 0.85 to 1.00: strong candidate
- 0.70 to 0.84: acceptable candidate
- 0.50 to 0.69: weak candidate, usually fail unless issues are minor
- below 0.50: fail

Use conservative judgment. If the candidate invents facts or contact details, fail it.
""".strip()


def llm_quality_filter_candidate(
    *,
    client,
    article: WikiArticle,
    content_type: str,
    text: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> Tuple[bool, float, List[str], str]:
    prompt = build_quality_filter_prompt(article, content_type, text)
    payload = _responses_create_json(
        client,
        model=model,
        system_text="You are a strict data quality judge. Return valid JSON only.",
        user_text=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    passed = bool(payload.get("pass", False))
    score = float(payload.get("score", 0.0))
    flags = payload.get("flags", [])
    if not isinstance(flags, list):
        flags = []
    flags = [str(x) for x in flags]
    rationale = str(payload.get("rationale", ""))
    return passed, max(0.0, min(1.0, score)), flags, rationale



def quality_filter_candidates(
    *,
    client,
    article: WikiArticle,
    candidates: List[CandidateItem],
    quality_model: str,
    quality_temperature: float,
    quality_max_output_tokens: int,
    keep_per_type: int,
    min_combined_score: float,
) -> Tuple[List[GeneratedEntry], List[FilterDecision]]:
    decisions: List[FilterDecision] = []

    for cand in candidates:
        rule_pass, rule_flags, rule_score = rule_filter_candidate(article, cand.content_type, cand.text)
        llm_pass, llm_score, llm_flags, llm_rationale = llm_quality_filter_candidate(
            client=client,
            article=article,
            content_type=cand.content_type,
            text=cand.text,
            model=quality_model,
            temperature=quality_temperature,
            max_output_tokens=quality_max_output_tokens,
        )
        combined = 0.45 * rule_score + 0.55 * llm_score
        passed = rule_pass and llm_pass and combined >= min_combined_score
        reasons: List[str] = []
        if not rule_pass:
            reasons.append("rule_fail")
        if not llm_pass:
            reasons.append("llm_fail")
        if combined < min_combined_score:
            reasons.append(f"combined_score_below_threshold:{combined:.2f}")
        decisions.append(
            FilterDecision(
                content_type=cand.content_type,
                text=cand.text,
                passed=passed,
                score=combined,
                reasons=reasons,
                rule_flags=rule_flags,
                llm_flags=llm_flags,
                llm_rationale=llm_rationale,
            )
        )

    accepted_rows: List[GeneratedEntry] = []
    by_type: Dict[str, List[FilterDecision]] = {}
    for d in decisions:
        if d.passed:
            by_type.setdefault(d.content_type, []).append(d)

    for content_type, items in by_type.items():
        retained = items[:keep_per_type]
        for idx, item in enumerate(retained, start=1):
            accepted_rows.append(
                GeneratedEntry(
                    source_page_id=article.page_id,
                    source_title=article.title,
                    source_url=article.url,
                    content_type=content_type,
                    index_within_type=idx,
                    text=item.text,
                )
            )

    return accepted_rows, decisions


# -----------------------------
# I/O helpers
# -----------------------------
def save_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_sampled_articles(path: Path, articles: Sequence[WikiArticle]) -> None:
    save_jsonl(path, [asdict(a) for a in articles])


def stable_article_output_name(article: WikiArticle) -> str:
    digest = hashlib.md5(f"{article.page_id}:{article.title}".encode("utf-8")).hexdigest()[:10]
    safe_title = re.sub(r"[^A-Za-z0-9._-]+", "_", article.title)[:80].strip("_")
    return f"{article.page_id}_{safe_title}_{digest}.json"


def process_article(
    *,
    article_index: int,
    total_articles: int,
    article: WikiArticle,
    args: argparse.Namespace,
    quality_model: str,
    requested_generation_count: int,
    per_article_dir: Path,
) -> Tuple[int, List[Dict], List[Dict]]:
    client = _get_thread_openai_client(timeout=args.timeout)
    per_article_path = per_article_dir / stable_article_output_name(article)

    logger.info("[%d/%d] Generating for '%s'", article_index, total_articles, article.title)
    raw_payload = generate_formats_with_openai(
        client=client,
        article=article,
        content_types=args.content_types,
        entries_per_type=requested_generation_count,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    candidates = normalize_generated_payload(raw_payload, args.content_types)
    accepted_rows, decisions = quality_filter_candidates(
        client=client,
        article=article,
        candidates=candidates,
        quality_model=quality_model,
        quality_temperature=args.quality_temperature,
        quality_max_output_tokens=args.quality_max_output_tokens,
        keep_per_type=args.entries_per_type,
        min_combined_score=args.min_combined_score,
    )

    serializable_payload = {
        "title": article.title,
        "source_page_id": article.page_id,
        "source_url": article.url,
        "requested_content_types": [TYPE_LABEL_MAP[t] for t in args.content_types],
        "requested_entries_per_type": args.entries_per_type,
        "generated_entries_per_type_before_filtering": requested_generation_count,
        "raw_generation": raw_payload,
        "accepted_rows": [asdict(r) for r in accepted_rows],
        "filter_decisions": [asdict(d) for d in decisions],
    }
    with open(per_article_path, "w", encoding="utf-8") as f:
        json.dump(serializable_payload, f, ensure_ascii=False, indent=2)

    if args.sleep_seconds > 0:
        time.sleep(args.sleep_seconds)

    return article_index, serializable_payload["accepted_rows"], serializable_payload["filter_decisions"]


# -----------------------------
# Main pipeline
# -----------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PANORAMA-inspired negative sample generator")
    parser.add_argument(
        "--dump-url",
        type=str,
        default="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2",
        help="Wikimedia dump URL",
    )
    parser.add_argument(
        "--dump-path",
        type=Path,
        default=Path("./data/enwiki-latest-pages-articles-multistream.xml.bz2"),
        help="Local path for the downloaded dump",
    )
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--content-types",
        nargs="+",
        default=["social_media", "forum_post", "online_review", "comment", "online_ad"],
        choices=list(CONTENT_TYPE_SPECS.keys()),
        help="One or more content types to generate",
    )
    parser.add_argument("--entries-per-type", type=int, default=2, help="Number of candidates to generate per style before filtering. Use 2 to get 0-2 retained outputs per style.")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--quality-model", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--quality-temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=7000)
    parser.add_argument("--quality-max-output-tokens", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-workers", type=int, default=32, help="Number of articles to process concurrently for OpenAI API calls")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true", help="Skip articles whose output file already exists")
    parser.add_argument("--min-combined-score", type=float, default=0.68)
    parser.add_argument("--output-dir", type=Path, default=Path("./panorama_negative_output"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    quality_model = args.quality_model or args.model

    output_dir: Path = args.output_dir
    sampled_articles_path = output_dir / "sampled_articles.jsonl"
    generated_jsonl_path = output_dir / "generated_negative_samples.jsonl"
    filter_log_path = output_dir / "quality_filter_decisions.jsonl"
    per_article_dir = output_dir / "per_article"
    manifest_path = output_dir / "manifest.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    per_article_dir.mkdir(parents=True, exist_ok=True)
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")

    _make_openai_client(timeout=args.timeout)

    # Step 1: download dump
    download_file(args.dump_url, args.dump_path)

    # Step 2: sample articles
    if sampled_articles_path.exists():
        logger.info("Loading cached sampled articles from %s", sampled_articles_path)
        sampled_articles: List[WikiArticle] = []
        with open(sampled_articles_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                sampled_articles.append(WikiArticle(**obj))
    else:
        sampled_articles = reservoir_sample_articles(args.dump_path, args.sample_size, args.seed)
        save_sampled_articles(sampled_articles_path, sampled_articles)

    logger.info("Sampled %d articles", len(sampled_articles))

    all_rows: List[Dict] = []
    all_decisions: List[Dict] = []
    requested_generation_count = args.entries_per_type

    logger.info("Each style will generate %d candidate(s); every candidate that passes the PANORAMA-inspired filter will be retained.", requested_generation_count)

    pending_futures: Dict[concurrent.futures.Future[Tuple[int, List[Dict], List[Dict]]], Tuple[int, WikiArticle, Path]] = {}
    completed_results: Dict[int, Tuple[List[Dict], List[Dict]]] = {}

    for i, article in enumerate(sampled_articles, start=1):
        per_article_path = per_article_dir / stable_article_output_name(article)
        if args.resume and per_article_path.exists():
            logger.info("[%d/%d] Skipping existing %s", i, len(sampled_articles), article.title)
            try:
                with open(per_article_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                accepted = payload.get("accepted_rows", [])
                decisions = payload.get("filter_decisions", [])
                all_rows.extend(accepted)
                all_decisions.extend(decisions)
            except Exception as exc:
                logger.warning("Could not read cached article payload for %s: %s", article.title, exc)
            continue

    articles_to_process: List[Tuple[int, WikiArticle, Path]] = []
    for i, article in enumerate(sampled_articles, start=1):
        per_article_path = per_article_dir / stable_article_output_name(article)
        if not (args.resume and per_article_path.exists()):
            articles_to_process.append((i, article, per_article_path))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for i, article, per_article_path in articles_to_process:
            future = executor.submit(
                process_article,
                article_index=i,
                total_articles=len(sampled_articles),
                article=article,
                args=args,
                quality_model=quality_model,
                requested_generation_count=requested_generation_count,
                per_article_dir=per_article_dir,
            )
            pending_futures[future] = (i, article, per_article_path)

        for future in concurrent.futures.as_completed(pending_futures):
            i, article, per_article_path = pending_futures[future]
            try:
                article_index, accepted, decisions = future.result()
                completed_results[article_index] = (accepted, decisions)
            except Exception as exc:
                logger.exception("Generation failed for '%s': %s", article.title, exc)
                error_path = per_article_dir / (per_article_path.stem + ".error.txt")
                error_path.write_text(str(exc), encoding="utf-8")

    for i in range(1, len(sampled_articles) + 1):
        result = completed_results.get(i)
        if result is None:
            continue
        accepted, decisions = result
        all_rows.extend(accepted)
        all_decisions.extend(decisions)

    save_jsonl(generated_jsonl_path, all_rows)
    save_jsonl(filter_log_path, all_decisions)

    manifest = {
        "dump_url": args.dump_url,
        "dump_path": str(args.dump_path),
        "sample_size_requested": args.sample_size,
        "sample_size_actual": len(sampled_articles),
        "content_types": args.content_types,
        "entries_per_type_after_filter": args.entries_per_type,
        "generated_entries_per_type_before_filter": requested_generation_count,
        "generation_model": args.model,
        "quality_model": quality_model,
        "temperature": args.temperature,
        "quality_temperature": args.quality_temperature,
        "max_output_tokens": args.max_output_tokens,
        "quality_max_output_tokens": args.quality_max_output_tokens,
        "max_workers": args.max_workers,
        "min_combined_score": args.min_combined_score,
        "generated_rows_after_filter": len(all_rows),
        "filter_decision_rows": len(all_decisions),
        "sampled_articles_file": str(sampled_articles_path),
        "aggregate_output_file": str(generated_jsonl_path),
        "filter_log_file": str(filter_log_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Done. Aggregate output: %s", generated_jsonl_path)
    logger.info("Filter log: %s", filter_log_path)
    logger.info("Generated rows after filtering: %d", len(all_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
