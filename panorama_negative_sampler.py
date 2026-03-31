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
  --sample-size 1000 \
  --content-types social_media forum_post online_review comment online_ad \
  --entries-per-type 2 \
  --model gpt-5-mini \
  --quality-model gpt-5-mini \
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
from collections import Counter
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


@dataclass
class BinaryClassificationRow:
    label: str
    content_type: str
    text: str
    source: str
    source_id: str


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
    few_shot_examples: Optional[Dict[str, List[str]]] = None,
) -> str:
    specs = "\n\n".join(CONTENT_TYPE_SPECS[t].format(n=entries_per_type) for t in content_types)
    labels = ", ".join(TYPE_LABEL_MAP[t] for t in content_types)
    few_shot_section = ""
    if few_shot_examples:
        blocks: List[str] = []
        for content_type in content_types:
            label = TYPE_LABEL_MAP[content_type]
            examples = few_shot_examples.get(label, [])
            if not examples:
                continue
            example_lines = "\n".join(
                f"- Example {idx}: {example_text}" for idx, example_text in enumerate(examples, start=1)
            )
            blocks.append(f"## {label} style examples\n{example_lines}")
        if blocks:
            few_shot_section = f"""

# Style Reference Examples
Use these PANORAMA examples as style references only.
They are not factual sources for this article.
Do not copy names, usernames, handles, locations, dates, contact details, or any unique identifiers from them.
Treat bracketed placeholders as non-factual style cues only.

{"\n\n".join(blocks)}
""".rstrip()

    return f"""
# Role
You are a realistic syntetic data generation engine.

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
{few_shot_section}

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
- Match the stylistic register and structure of the style reference examples, while using only facts from the source article.
- If a style example contains a username, location, date, handle, URL, phone number, or email, do not reuse it.

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


def _generation_json_schema(article_title: str, content_types: Sequence[str], entries_per_type: int) -> Dict[str, object]:
    allowed_labels = [TYPE_LABEL_MAP[t] for t in content_types]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string", "const": article_title},
            "outputs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "content_type": {"type": "string", "enum": allowed_labels},
                        "items": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": max(1, entries_per_type),
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["content_type", "items"],
                },
            },
        },
        "required": ["title", "outputs"],
    }


def _quality_filter_json_schema() -> Dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pass": {"type": "boolean"},
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "flags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "rationale": {"type": "string"},
        },
        "required": ["pass", "score", "flags", "rationale"],
    }


def _responses_create_json(
    client,
    *,
    model: str,
    system_text: str,
    user_text: str,
    max_output_tokens: int,
    schema_name: str,
    schema: Dict[str, object],
    max_attempts: int = 3,
) -> Dict:
    last_error: Optional[Exception] = None
    current_max_output_tokens = max_output_tokens

    for attempt in range(1, max_attempts + 1):
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
                "verbosity": "low",
            },
            max_output_tokens=current_max_output_tokens,
        )

        if response.status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            last_error = ValueError(
                f"Model response incomplete for schema '{schema_name}'"
                + (f": {details}" if details else "")
            )
            if getattr(details, "reason", None) == "max_output_tokens":
                current_max_output_tokens = max(current_max_output_tokens + 400, current_max_output_tokens * 2)
        elif response.status == "failed":
            last_error = ValueError(
                f"Model response failed for schema '{schema_name}': {response.error}"
            )
        else:
            raw = response.output_text.strip()
            if not raw:
                last_error = ValueError(
                    f"Model returned empty output for schema '{schema_name}' with status={response.status}"
                )
            else:
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as exc:
                    last_error = ValueError(
                        f"Model did not return valid JSON for schema '{schema_name}'. Raw output:\n{raw[:4000]}"
                    )
                    last_error.__cause__ = exc

        if attempt < max_attempts:
            logger.warning(
                "Retrying structured response for schema '%s' after attempt %d/%d: %s",
                schema_name,
                attempt,
                max_attempts,
                last_error,
            )
            time.sleep(0.5 * attempt)

    assert last_error is not None
    raise last_error


def generate_formats_with_openai(
    *,
    client,
    article: WikiArticle,
    content_types: Sequence[str],
    entries_per_type: int,
    model: str,
    max_output_tokens: int,
    few_shot_examples: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    prompt = build_generation_prompt(article, content_types, entries_per_type, few_shot_examples)
    return _responses_create_json(
        client,
        model=model,
        system_text="You strictly follow instructions and return valid JSON only.",
        user_text=prompt,
        max_output_tokens=max_output_tokens,
        schema_name="panorama_generation",
        schema=_generation_json_schema(article.title, content_types, entries_per_type),
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
LABEL_FIELD_RE = re.compile(
    r"\b(username|user name|name|location|date|timestamp|time)\s*:\s*([^,\n]+)",
    re.IGNORECASE,
)


def sanitize_few_shot_text(text: str) -> str:
    text = LABEL_FIELD_RE.sub(lambda m: f"{m.group(1)}: [{m.group(1).lower().replace(' ', '_')}_placeholder]", text)
    text = EMAIL_RE.sub("[email_placeholder]", text)
    text = PHONE_RE.sub("[phone_placeholder]", text)
    text = URL_RE.sub("[url_placeholder]", text)
    text = HANDLE_RE.sub("[handle_placeholder]", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "[date_placeholder]", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(?:\s?[AP]M)?\b", "[time_placeholder]", text, flags=re.IGNORECASE)
    return MULTISPACE_RE.sub(" ", text).strip()


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
    max_output_tokens: int,
) -> Tuple[bool, float, List[str], str]:
    prompt = build_quality_filter_prompt(article, content_type, text)
    payload = _responses_create_json(
        client,
        model=model,
        system_text="You are a strict data quality judge. Return valid JSON only.",
        user_text=prompt,
        max_output_tokens=max_output_tokens,
        schema_name="panorama_quality_filter",
        schema=_quality_filter_json_schema(),
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


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_cached_results_from_per_article(per_article_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    accepted_rows: List[Dict] = []
    decisions: List[Dict] = []
    if not per_article_dir.exists():
        logger.info("No per-article cache directory found at %s", per_article_dir)
        return accepted_rows, decisions

    json_paths = sorted(per_article_dir.glob("*.json"))
    for path in json_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read per-article payload %s: %s", path, exc)
            continue
        accepted = payload.get("accepted_rows", [])
        filter_decisions = payload.get("filter_decisions", [])
        if isinstance(accepted, list):
            accepted_rows.extend(row for row in accepted if isinstance(row, dict))
        if isinstance(filter_decisions, list):
            decisions.extend(row for row in filter_decisions if isinstance(row, dict))

    logger.info(
        "Loaded %d accepted rows and %d filter decisions from %d per-article payload(s)",
        len(accepted_rows),
        len(decisions),
        len(json_paths),
    )
    return accepted_rows, decisions


def sample_existing_negative_rows(
    per_article_dir: Path,
    *,
    content_types: Sequence[str],
    sample_size: int,
    seed: int,
) -> List[Dict]:
    rows, _ = load_cached_results_from_per_article(per_article_dir)
    by_type: Dict[str, List[Dict]] = {}
    for row in rows:
        content_type = str(row.get("content_type", "")).strip()
        if content_type in content_types:
            by_type.setdefault(content_type, []).append(row)

    rng = random.Random(seed)
    sampled: List[Dict] = []
    unique_counts: Dict[str, int] = {}
    for content_type in content_types:
        items = list(by_type[content_type])
        rng.shuffle(items)
        seen_source_titles: Set[str] = set()
        selected_for_type: List[Dict] = []
        for item in items:
            source_title = str(item.get("source_title", "")).strip()
            if source_title in seen_source_titles:
                continue
            seen_source_titles.add(source_title)
            selected_for_type.append(item)
            if len(selected_for_type) >= sample_size:
                break
        unique_counts[content_type] = len(selected_for_type)
        sampled.extend(selected_for_type)
    logger.info("Cached negative counts by content type after unique source_title sampling: %s", unique_counts)
    return sampled


def has_enough_cached_negative_rows(
    per_article_dir: Path,
    *,
    content_types: Sequence[str],
    sample_size: int,
    seed: int,
) -> Tuple[bool, List[Dict], Dict[str, int]]:
    sampled_rows = sample_existing_negative_rows(
        per_article_dir,
        content_types=content_types,
        sample_size=sample_size,
        seed=seed,
    )
    counts = {content_type: 0 for content_type in content_types}
    for row in sampled_rows:
        counts[str(row["content_type"])] += 1
    enough = all(counts[content_type] >= sample_size for content_type in content_types)
    logger.info(
        "Cached negative sufficiency check: enough=%s, required_per_type=%d, counts=%s",
        enough,
        sample_size,
        counts,
    )
    return enough, sampled_rows, counts


def _get_hf_token() -> str:
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.getenv(env_name, "").strip()
        if token:
            return token
    raise RuntimeError(
        "PANORAMA is gated on Hugging Face. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN after accepting the dataset terms."
    )


def normalize_content_type_label(value: str) -> Optional[str]:
    normalized = value.strip().lower()
    alias_map = {
        "social media": "Social Media",
        "social_media": "Social Media",
        "forum post": "Forum Post",
        "forum_post": "Forum Post",
        "online review": "Online Review",
        "online_review": "Online Review",
        "comment": "Comment",
        "blog/news article comment": "Comment",
        "blog_news_article_comment": "Comment",
        "blog / news article comment": "Comment",
        "online ad": "Online Ad",
        "online_ad": "Online Ad",
    }
    if normalized in alias_map:
        return alias_map[normalized]
    fallback = normalized.replace("-", "_").replace(" ", "_")
    return TYPE_LABEL_MAP.get(fallback)


def normalize_hf_file_url(url: str) -> str:
    return url.replace("/blob/", "/resolve/")


def download_hf_file(url: str, dest_path: Path, token: str) -> Path:
    resolved_url = normalize_hf_file_url(url)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info("PANORAMA parquet already exists: %s", dest_path)
        return dest_path

    logger.info("Downloading PANORAMA parquet from %s", resolved_url)
    req = urllib.request.Request(
        resolved_url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "panorama-negative-sampler/2.0",
        },
    )
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    logger.info("Saved PANORAMA parquet to %s", dest_path)
    return dest_path


def iter_panorama_rows_from_parquet(parquet_path: Path) -> Iterator[Dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Please install pyarrow to read PANORAMA parquet files.") from exc

    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches():
        for row in batch.to_pylist():
            if isinstance(row, dict):
                yield row


def sample_panorama_positive_rows(
    *,
    parquet_url: str,
    parquet_path: Path,
    content_types: Sequence[str],
    sample_size: int,
    seed: int,
) -> List[Dict]:
    hf_token = _get_hf_token()
    requested_labels = [TYPE_LABEL_MAP[content_type] for content_type in content_types]
    reservoirs: Dict[str, List[Dict]] = {label: [] for label in requested_labels}
    seen_counts: Counter[str] = Counter()
    rng = random.Random(seed)

    local_parquet_path = download_hf_file(parquet_url, parquet_path, hf_token)
    logger.info("Reading PANORAMA rows from %s", local_parquet_path)
    for row in iter_panorama_rows_from_parquet(local_parquet_path):
        text = str(row.get("text", "")).strip()
        if not text:
            continue

        raw_content_type = row.get("content-type")
        if raw_content_type is None:
            raw_content_type = row.get("content_type")
        if raw_content_type is None:
            continue

        content_type = normalize_content_type_label(str(raw_content_type))
        if content_type not in reservoirs:
            continue

        seen_counts[content_type] += 1
        candidate = {
            "label": "positive",
            "content_type": content_type,
            "text": text,
            "source": "PANORAMA",
            "source_id": str(row.get("id", f"{content_type}-{seen_counts[content_type]}")),
        }
        bucket = reservoirs[content_type]
        if len(bucket) < sample_size:
            bucket.append(candidate)
        else:
            replace_index = rng.randint(0, seen_counts[content_type] - 1)
            if replace_index < sample_size:
                bucket[replace_index] = candidate

    missing = [content_type for content_type in requested_labels if len(reservoirs[content_type]) < sample_size]
    if missing:
        counts = {content_type: len(reservoirs[content_type]) for content_type in requested_labels}
        raise RuntimeError(
            f"PANORAMA does not contain enough rows for requested content types. counts={counts}, required={sample_size}"
        )

    sampled: List[Dict] = []
    for content_type in requested_labels:
        sampled.extend(reservoirs[content_type])
    logger.info("Sampled PANORAMA positives by content type: %s", {k: len(v) for k, v in reservoirs.items()})
    return sampled


def build_few_shot_examples(
    panorama_rows: Sequence[Dict],
    *,
    content_types: Sequence[str],
    max_examples_per_type: int,
) -> Dict[str, List[str]]:
    examples: Dict[str, List[str]] = {}
    requested_labels = [TYPE_LABEL_MAP[content_type] for content_type in content_types]
    for label in requested_labels:
        examples[label] = []

    for row in panorama_rows:
        content_type = str(row.get("content_type", "")).strip()
        if content_type not in examples or len(examples[content_type]) >= max_examples_per_type:
            continue
        sanitized = sanitize_few_shot_text(str(row.get("text", "")).strip())
        if sanitized:
            examples[content_type].append(sanitized)
    return examples


def build_binary_classification_rows(positive_rows: Sequence[Dict], negative_rows: Sequence[Dict], seed: int) -> List[Dict]:
    combined: List[Dict] = []
    for row in positive_rows:
        combined.append(asdict(BinaryClassificationRow(**row)))
    for row in negative_rows:
        combined.append(
            asdict(
                BinaryClassificationRow(
                    label="negative",
                    content_type=str(row["content_type"]),
                    text=str(row["text"]),
                    source="synthetic_generation",
                    source_id=str(row.get("source_page_id", "")),
                )
            )
        )

    rng = random.Random(seed)
    rng.shuffle(combined)
    return combined


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
    few_shot_examples: Optional[Dict[str, List[str]]] = None,
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
        max_output_tokens=args.max_output_tokens,
        few_shot_examples=few_shot_examples,
    )
    candidates = normalize_generated_payload(raw_payload, args.content_types)
    accepted_rows, decisions = quality_filter_candidates(
        client=client,
        article=article,
        candidates=candidates,
        quality_model=quality_model,
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
def log_progress(prefix: str, completed: int, total: int) -> None:
    if total <= 0:
        logger.info("%s: 0/0 (0.0%%)", prefix)
        return
    pct = 100.0 * completed / total
    logger.info("%s: %d/%d (%.1f%%)", prefix, completed, total, pct)


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
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--content-types",
        nargs="+",
        default=["social_media", "forum_post", "online_review", "comment", "online_ad"],
        choices=list(CONTENT_TYPE_SPECS.keys()),
        help="One or more content types to generate",
    )
    parser.add_argument("--entries-per-type", type=int, default=5, help="Number of candidates to generate per style before filtering. Use 5 to get 0-5 retained outputs per style.")
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--quality-model", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.8, help="Deprecated and ignored for Responses API models that do not support temperature.")
    parser.add_argument("--quality-temperature", type=float, default=0.2, help="Deprecated and ignored for Responses API models that do not support temperature.")
    parser.add_argument("--max-output-tokens", type=int, default=7000)
    parser.add_argument("--quality-max-output-tokens", type=int, default=2000)
    parser.add_argument("--few-shot-per-type", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-workers", type=int, default=32, help="Number of articles to process concurrently for OpenAI API calls")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse cached per-article outputs if present. Sampled articles are reused automatically from output-dir/sampled_articles.jsonl when that file exists.")
    parser.add_argument("--min-combined-score", type=float, default=0.68)
    parser.add_argument(
        "--panorama-parquet-url",
        type=str,
        default="https://huggingface.co/datasets/srirxml/PANORAMA/resolve/main/data/train-00000-of-00001.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./panorama_negative_output"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    quality_model = args.quality_model or args.model
    requested_labels = [TYPE_LABEL_MAP[content_type] for content_type in args.content_types]

    output_dir: Path = args.output_dir
    sampled_articles_path = output_dir / "sampled_articles.jsonl"
    generated_jsonl_path = output_dir / "generated_negative_samples.jsonl"
    filter_log_path = output_dir / "quality_filter_decisions.jsonl"
    binary_dataset_path = output_dir / "panorama_positive_negative_dataset.jsonl"
    panorama_parquet_path = output_dir / "cache" / "PANORAMA_train.parquet"
    per_article_dir = output_dir / "per_article"
    manifest_path = output_dir / "manifest.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    per_article_dir.mkdir(parents=True, exist_ok=True)
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")

    all_rows: List[Dict] = []
    all_decisions: List[Dict] = []
    requested_generation_count = args.entries_per_type
    positive_rows: List[Dict] = []
    few_shot_examples: Dict[str, List[str]] = {}

    has_sufficient_cached_negatives, sampled_negative_rows, negative_counts = has_enough_cached_negative_rows(
        per_article_dir,
        content_types=requested_labels,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    if args.force_regenerate:
        logger.info("Forcing synthetic regeneration; cached negatives will not be used for skip decisions.")
        has_sufficient_cached_negatives = False

    if not has_sufficient_cached_negatives:
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
        positive_rows = sample_panorama_positive_rows(
            parquet_url=args.panorama_parquet_url,
            parquet_path=panorama_parquet_path,
            content_types=args.content_types,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        few_shot_examples = build_few_shot_examples(
            positive_rows,
            content_types=args.content_types,
            max_examples_per_type=args.few_shot_per_type,
        )
        if args.temperature != 0.8 or args.quality_temperature != 0.2:
            logger.info("Ignoring --temperature / --quality-temperature for OpenAI Responses API calls.")

        logger.info(
            "Each style will generate %d candidate(s); every candidate that passes the PANORAMA-inspired filter will be retained.",
            requested_generation_count,
        )

        pending_futures: Dict[concurrent.futures.Future[Tuple[int, List[Dict], List[Dict]]], Tuple[int, WikiArticle, Path]] = {}
        completed_results: Dict[int, Tuple[List[Dict], List[Dict]]] = {}
        resumed_articles = 0

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
                    resumed_articles += 1
                    if resumed_articles == 1 or resumed_articles % 100 == 0 or resumed_articles == len(sampled_articles):
                        log_progress("Resume progress", resumed_articles, len(sampled_articles))
                except Exception as exc:
                    logger.warning("Could not read cached article payload for %s: %s", article.title, exc)
                continue

        articles_to_process: List[Tuple[int, WikiArticle, Path]] = []
        for i, article in enumerate(sampled_articles, start=1):
            per_article_path = per_article_dir / stable_article_output_name(article)
            if not (args.resume and per_article_path.exists()):
                articles_to_process.append((i, article, per_article_path))

        if args.resume:
            logger.info("Loaded %d cached article result(s)", resumed_articles)
        log_progress("Articles queued for generation", len(articles_to_process), len(sampled_articles))

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
                    few_shot_examples=few_shot_examples,
                )
                pending_futures[future] = (i, article, per_article_path)

            completed_generation = 0
            for future in concurrent.futures.as_completed(pending_futures):
                i, article, per_article_path = pending_futures[future]
                try:
                    article_index, accepted, decisions = future.result()
                    completed_results[article_index] = (accepted, decisions)
                    completed_generation += 1
                    total_completed = resumed_articles + completed_generation
                    logger.info(
                        "Generation finished for '%s' (%d accepted, %d decisions)",
                        article.title,
                        len(accepted),
                        len(decisions),
                    )
                    log_progress("Overall article progress", total_completed, len(sampled_articles))
                except Exception as exc:
                    completed_generation += 1
                    total_completed = resumed_articles + completed_generation
                    logger.exception("Generation failed for '%s': %s", article.title, exc)
                    error_path = per_article_dir / (per_article_path.stem + ".error.txt")
                    error_path.write_text(str(exc), encoding="utf-8")
                    log_progress("Overall article progress", total_completed, len(sampled_articles))

        for i in range(1, len(sampled_articles) + 1):
            result = completed_results.get(i)
            if result is None:
                continue
            accepted, decisions = result
            all_rows.extend(accepted)
            all_decisions.extend(decisions)

        has_sufficient_cached_negatives, sampled_negative_rows, negative_counts = has_enough_cached_negative_rows(
            per_article_dir,
            content_types=requested_labels,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    else:
        logger.info(
            "Skipping synthetic generation because cached negatives contain at least %d rows per requested content type after unique source_title sampling.",
            args.sample_size,
        )

    cached_accepted_rows, cached_decisions = load_cached_results_from_per_article(per_article_dir)
    save_jsonl(generated_jsonl_path, cached_accepted_rows)
    save_jsonl(filter_log_path, cached_decisions)
    all_rows = cached_accepted_rows
    all_decisions = cached_decisions

    if not positive_rows:
        positive_rows = sample_panorama_positive_rows(
            parquet_url=args.panorama_parquet_url,
            parquet_path=panorama_parquet_path,
            content_types=args.content_types,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    binary_rows = build_binary_classification_rows(
        positive_rows=positive_rows,
        negative_rows=sampled_negative_rows,
        seed=args.seed,
    )
    save_jsonl(binary_dataset_path, binary_rows)

    manifest = {
        "dump_url": args.dump_url,
        "dump_path": str(args.dump_path),
        "sample_size_per_content_type": args.sample_size,
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
        "few_shot_per_type": args.few_shot_per_type,
        "force_regenerate": args.force_regenerate,
        "generated_rows_after_filter": len(load_jsonl(generated_jsonl_path)) if generated_jsonl_path.exists() else 0,
        "filter_decision_rows": len(all_decisions),
        "negative_rows_selected_for_binary_dataset": len(sampled_negative_rows),
        "negative_rows_selected_by_content_type": dict(negative_counts),
        "positive_rows_selected_from_panorama": len(positive_rows),
        "binary_dataset_rows": len(binary_rows),
        "panorama_parquet_url": normalize_hf_file_url(args.panorama_parquet_url),
        "panorama_parquet_path": str(panorama_parquet_path),
        "sampled_articles_file": str(sampled_articles_path),
        "aggregate_output_file": str(generated_jsonl_path),
        "filter_log_file": str(filter_log_path),
        "binary_dataset_output_file": str(binary_dataset_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Done. Aggregate output: %s", generated_jsonl_path)
    logger.info("Filter log: %s", filter_log_path)
    logger.info("Binary dataset output: %s", binary_dataset_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
