# `panorama_negative_sampler.py`

`panorama_negative_sampler.py` builds a PANORAMA-aligned negative set for binary document classification. It samples public Wikipedia articles, asks an LLM to rewrite those articles into PANORAMA-like web text styles, filters the results for quality and contamination, and then combines the retained negatives with PANORAMA positives.

The script is not a byte-for-byte reproduction of the PANORAMA pipeline. It is an implementation inspired by the paper's stated goals: realistic web-native text, low contamination, and quality control suitable for PII-related classification experiments.

## Why this script exists

Many public datasets related to personally identifiable information (PII) are designed for token-level named entity recognition, masking or redaction, or span extraction. Those resources are useful for finding and labeling local spans, but they are less suitable for document classification, where the task is to decide whether a whole document should be treated as PII-bearing or privacy-sensitive. This script addresses that gap by constructing a document-level positive/negative dataset with matched content styles.

## What PANORAMA is in this pipeline

In this repository, PANORAMA is the positive dataset source and the style anchor.

The script downloads PANORAMA examples from a gated Hugging Face parquet file and uses them in two ways:

1. It samples positive rows for the final binary classification dataset.
2. It extracts a small number of sanitized few-shot examples per content type so the LLM can imitate PANORAMA-like style without copying identifiers.

The supported PANORAMA-style content types are:

- `social_media`
- `forum_post`
- `online_review`
- `comment`
- `online_ad`

These content types matter because the classifier should learn whether text contains privacy-relevant signals, not just whether it looks like a tweet, a review, or a forum post.

## Why not use an unrelated corpus for negatives

Using negative texts from a corpus that is unrelated to PANORAMA is risky. If the negative set comes from a different source with a different register, domain, or formatting pattern, a classifier may learn superficial stylistic differences instead of actual privacy-related cues. In other words, the model may solve the task by recognizing corpus identity rather than by judging whether the document contains PII-like content.

This script therefore generates negatives with an LLM while conditioning on:

- public Wikipedia facts as the factual source
- PANORAMA-like output formats as the target style
- PANORAMA-derived few-shot examples as stylistic guidance

That design makes the negative samples closer in genre and surface form to the positives. As a result, evaluation is better focused on PII discrimination ability rather than style mismatch.

## End-to-end pipeline

The script performs the following steps.

1. Download a Wikipedia XML dump from Wikimedia if it is not already cached.
2. Stream the compressed dump and keep only eligible namespace-0 articles.
3. Clean the raw wikitext into lightweight plain text.
4. Reservoir-sample `--sample-size` Wikipedia articles.
5. Download the PANORAMA parquet file from Hugging Face.
6. Reservoir-sample PANORAMA positive examples for the requested content types.
7. Build sanitized few-shot style examples from PANORAMA texts.
8. For each sampled Wikipedia article, ask the LLM to generate PANORAMA-style negative candidates.
9. Filter each candidate with both rule-based checks and an LLM judge.
10. Retain up to `--entries-per-type` accepted outputs per article and content type.
11. Re-sample accepted negatives so each requested content type contributes `--sample-size` rows when possible.
12. Merge PANORAMA positives and generated negatives into a shuffled binary classification dataset.

## Detailed behavior of `panorama_negative_sampler.py`

### 1. Wikipedia download and article sampling

The script downloads a Wikimedia dump such as:

- `https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2`

It then streams the dump instead of loading it fully into memory. Only article pages in namespace `0` are kept. It excludes redirects, administrative pages such as `Wikipedia:` or `Template:`, and very short cleaned texts. Sampling is done with reservoir sampling so the script can uniformly sample from very large dumps in one pass.

### 2. Wikitext cleanup

Before generation, the script removes or simplifies common wiki markup:

- comments
- references
- tables
- file/category links
- templates
- HTML tags
- internal and external link markup

The goal is not perfect document reconstruction. The goal is to provide a clean factual article text that an LLM can paraphrase safely.

### 3. PANORAMA positive sampling

For positives, the script reads the PANORAMA parquet file and reservoir-samples rows per requested content type. Each positive row is stored with:

- `label = "positive"`
- PANORAMA content type
- original PANORAMA text
- source metadata

These are later combined with generated negatives for document classification.

### 4. Few-shot style guidance from PANORAMA

The script derives a small number of style examples from PANORAMA and sanitizes them before reuse. The sanitization removes or replaces obvious identifier-like fields such as:

- usernames
- handles
- emails
- phone numbers
- URLs
- dates and times

This matters because PANORAMA is used as a style reference, not as a source to copy PII from.

### 5. LLM-based negative generation

For each sampled Wikipedia article, the script builds a prompt that says:

- the Wikipedia article is the only factual source
- the output must be realistic web-native text
- the text must match PANORAMA-style formats
- the text must not invent synthetic PII
- the output must be returned as JSON

The LLM then generates candidates for one or more target styles such as social media posts, forum posts, online reviews, comments, or online ads.

The result is a set of texts that preserve public facts from Wikipedia while resembling PANORAMA-style web documents.

### 6. Rule-based filtering

Each candidate is checked with heuristics including:

- minimum and maximum length per content type
- invented emails, phone numbers, URLs, or social handles not found in the source article
- approximate contamination against the source article
- long shared n-grams that suggest verbatim copying
- weak style signals for certain content types
- missing reference to the article title

These checks produce a rule score and rule flags.

### 7. LLM-based quality filtering

Each candidate is also judged by a second prompt that evaluates:

- faithfulness to the source article
- style naturalness
- low contamination
- privacy safety
- content-type fit

This stage produces:

- pass/fail
- score
- flags
- rationale

### 8. Combined acceptance decision

The final decision combines rule-based and LLM-based scores. A candidate is retained only if:

- the rules pass
- the LLM judge passes
- the combined score is above `--min-combined-score`

This keeps the negatives realistic and style-matched while reducing copied text and invented identifiers.

### 9. Cached per-article outputs

For each processed Wikipedia article, the script writes a per-article JSON file containing:

- raw generation output
- accepted rows
- filter decisions

This allows resuming runs and later re-sampling accepted negatives without regenerating everything.

### 10. Final binary dataset assembly

After generation, the script creates:

- `generated_negative_samples.jsonl`
- `quality_filter_decisions.jsonl`
- `panorama_positive_negative_dataset.jsonl`
- `manifest.json`

The final binary dataset contains rows shaped like:

- `label`
- `content_type`
- `text`
- `source`
- `source_id`

Positive rows come from PANORAMA. Negative rows come from synthetic generation based on Wikipedia.

## Why this is useful for document classification

The central idea is controlled contrast.

The positive documents are PANORAMA texts that may contain privacy-relevant information or patterns. The negative documents are generated to look similar at the document level while being grounded in public, non-PII Wikipedia content. Because the styles are aligned, the classifier is pushed toward learning privacy-related decision signals instead of relying on obvious source or genre artifacts.

## Requirements

You need:

- Python 3
- `openai`
- `pyarrow`
- `OPENAI_API_KEY`
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`

`HF_TOKEN` is needed because the PANORAMA dataset is gated on Hugging Face.

## Example usage

The example below uses `sample-size 1000` exactly as requested.

```bash
python panorama_negative_sampler.py \
  --dump-url https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2 \
  --sample-size 1000 \
  --content-types social_media forum_post online_review comment online_ad \
  --entries-per-type 2 \
  --model gpt-5-mini \
  --quality-model gpt-5-mini \
  --output-dir ./panorama_negative_output
```

### What this command does

- Samples 1000 PANORAMA positives per requested content type.
- Samples 1000 Wikipedia source articles.
- Generates up to 2 negative candidates per article and style before filtering.
- Retains candidates that pass PANORAMA-inspired filtering.
- Builds a binary positive/negative document classification dataset in `./panorama_negative_output`.

## Output files

Typical outputs under `--output-dir` are:

- `sampled_articles.jsonl`: sampled Wikipedia articles
- `per_article/*.json`: raw generation and filtering results per source article
- `generated_negative_samples.jsonl`: all accepted generated negatives
- `quality_filter_decisions.jsonl`: filter logs for all candidates
- `panorama_positive_negative_dataset.jsonl`: shuffled binary classification dataset
- `manifest.json`: run configuration and counts

## Limitations

- This is a PANORAMA-inspired implementation, not an official reference pipeline.
- Quality filtering is a principled approximation based on the paper's goals, not an exact published second-stage filter.
- The output quality still depends on the generation and judge models.
- If accepted negatives are too sparse for some content types, coverage may depend on generation quality and filtering strictness.

## Summary

`panorama_negative_sampler.py` is a style-matched negative data generator for PII-oriented document classification. It uses PANORAMA as the positive source and stylistic reference, Wikipedia as a public factual source for negatives, and LLM-based rewriting plus filtering to minimize corpus-style leakage. That design makes it more appropriate than many token-level PII datasets when the downstream goal is whole-document classification.
