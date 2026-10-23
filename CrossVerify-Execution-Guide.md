# CrossVerify — Execution Guide

**Project:** CrossVerify: a multimodal retrieval and provenance system for live news archives  
**Scope:** academic information-retrieval project with a reproducible, responsibly collected corpus.

CrossVerify now uses a small live-news corpus rather than downloading a large static archive. The collector reads permitted RSS/Atom feeds, observes `robots.txt`, fetches articles and their primary images, extracts captions, resizes images, computes perceptual hashes, and stores the results in SQLite. This keeps the project manageable while preserving the image–caption–article relationship that retrieval and provenance checks need.

## 1. Current data pipeline

The working corpus lives here:

```text
data/
├── feeds.txt                    # permitted RSS/Atom feed URLs
├── ingest_live.py               # collection pipeline
└── processed/
    ├── crossverify.db           # SQLite metadata database
    └── images/                  # locally resized article images
```

The current database contains `articles` and `images` tables. Their important fields are:

```sql
articles(id, headline, body, date, source, url, fetched_at)
images(id, article_id, filepath, caption, date_taken_estimate,
       phash, caption_source, image_url, width, height)
```

`caption_source` records how the caption was obtained. Prefer `figcaption` pairs for evaluation. Keep `figcaption_unmatched`, `alt`, and `og_description` pairs for exploratory retrieval, but manually review them before using them as ground truth.

## 2. Setup

Use Python 3.10 or later and install the project dependencies:

```powershell
poetry install
```

The live collector also requires packages not yet listed in the base project configuration:

```powershell
poetry run pip install feedparser ImageHash requests trafilatura beautifulsoup4 lxml
```

Before collecting, replace the placeholder contact address in `data/ingest_live.py`:

```python
USER_AGENT = "CrossVerifyResearchBot/0.1 (academic project; contact: your-email@example.com)"
```

Do not treat a failed download as data. In particular, `articles.tar.gz` is a short 404 response, not a usable archive.

## 3. Corpus collection

Only add feeds whose terms allow this academic use. Keep one feed URL per line in `data/feeds.txt`; comments begin with `#`.

Run a small collection pass first:

```powershell
poetry run python data/ingest_live.py `
  --feeds data/feeds.txt `
  --db data/processed/crossverify.db `
  --img-dir data/processed/images `
  --max-per-feed 50 `
  --delay 1.5
```

The collector will:

1. Read the newest entries from each feed.
2. Check `robots.txt` before every site request.
3. Extract the article headline, body, date, source URL, a principal image, and its caption.
4. Discard tiny images, resize accepted images to a maximum dimension of 512 px, and calculate a pHash.
5. Skip a URL already in the database, so normal reruns are safe.

Collection targets should be staged:

| Stage | Target | Purpose |
|---|---:|---|
| Smoke test | 10–50 complete pairs | Verify extraction and retrieval code |
| Development corpus | 200–500 pairs | Build CLIP, FAISS, and BM25 pipelines quickly |
| Evaluation corpus | 1,000–3,000 pairs | Produce stable project metrics |

The RSS feeds expose only recent content, so growing the corpus requires collecting periodically and/or adding permitted diverse sources. Do not chase a 30k corpus unless there is a real evaluation need.

## 4. Quality checks and provenance

Run these checks after every collection batch:

- Confirm each database image path exists on disk.
- Record the number of rows by `source` and `caption_source`.
- Manually inspect a random sample of at least 20 pairs.
- Use `phash` to identify the same or near-identical image attached to multiple articles.
- Retain the article URL, image URL, source, fetched time, and extraction method; they are the provenance record.

Treat the live corpus as a snapshot: news pages can change or disappear. Keep the downloaded 512-pixel image and metadata in the database, but respect publishers' terms and do not redistribute source media outside the academic project.

## 5. Labeled benchmark

Build a separate, versioned benchmark after the development corpus is clean. Store it at `data/labeled_benchmark/benchmark.csv`:

```text
image_path,caption,label,notes
```

Use three labels:

- `consistent`: a real, manually verified image–caption pair, preferably from `figcaption`.
- `mismatched`: an image deliberately paired with an unrelated caption from a different article.
- `reused`: the same/near-duplicate image associated with materially different reporting context.

Suggested initial composition: 150 consistent, 150 mismatched, and 50–100 reused examples. Keep the construction rules, random seed, source record IDs, and reviewer decisions in `LABELING_METHODOLOGY.md`. Never present synthetic mismatches as naturally occurring misinformation.

## 6. Cross-modal retrieval core

1. Load a pretrained CLIP model such as `openai/clip-vit-base-patch32`.
2. Encode every stored image and its caption. Save normalized embeddings with their SQLite IDs.
3. Create FAISS inner-product indexes for image and caption vectors.
4. Implement both directions:
   - text query → ranked images/articles;
   - image query → ranked captions/articles.
5. Build a BM25 index over `articles.headline` and `articles.body` as the lexical baseline.
6. Hold out a small set of known image–caption pairs and report Recall@K.

Done when a query returns ranked results with IDs, scores, captions, source URLs, and provenance fields.

## 7. Query expansion and ranking

Start with WordNet synonyms and KeyBERT keywords. Query each expansion separately, then merge result lists. Compare Recall@K with and without expansion.

For hybrid ranking, fuse CLIP and BM25 using reciprocal-rank fusion:

```text
score(d) = Σ 1 / (60 + rank_i(d))
```

Optionally re-rank the fused top 50 with `cross-encoder/ms-marco-MiniLM-L-6-v2`. Compare BM25-only, CLIP-only, hybrid RRF, and hybrid plus re-ranking with MAP and nDCG@10.

## 8. Consistency and image-reuse checks

Use two complementary signals:

1. **Semantic alignment:** CLIP cosine similarity between the image and caption;
2. **Visual reuse:** group identical or close pHashes, then compare the related article dates, headlines, and captions.

Tune thresholds only on a training partition of the labeled benchmark. Report precision, recall, F1, and ROC-AUC on a held-out partition.

## 9. Feedback, explanations, and interface

Add relevance feedback after baseline retrieval is working. A simple Rocchio update is sufficient:

```text
q_new = α·q_original + β·mean(relevant) − γ·mean(irrelevant)
```

For explainability, show the provenance fields first: publisher, source URL, fetch date, image URL, caption source, similarity score, and pHash matches. Grad-CAM or token saliency is a stretch feature, not a prerequisite for the project.

The final UI can expose:

- text and image search;
- source-linked result cards;
- a consistency badge and pHash reuse warning;
- a provenance panel;
- relevant/not-relevant feedback controls.

## 10. Evaluation and delivery

Keep one evaluation table containing:

| Component | Measure |
|---|---|
| Retrieval | Recall@K, MAP, nDCG@10 |
| Query expansion | Recall@K with vs. without expansion |
| Fusion/re-ranking | MAP and nDCG@10 across ranking variants |
| Consistency | Precision, Recall, F1, ROC-AUC |
| Reuse detection | precision/recall on known duplicate groups |
| Feedback | nDCG before and after one round |

For the report and viva, document the data sources, collection dates, `robots.txt`/terms safeguards, data limitations, benchmark construction, and the distinction between a low image–caption alignment score and a factual claim of misinformation.

## Recommended order of work

1. Grow the live corpus to 200–500 high-quality pairs and run quality checks.
2. Build CLIP embeddings, FAISS retrieval, and a BM25 baseline.
3. Create the labeled benchmark and evaluate retrieval.
4. Add hybrid ranking and the consistency/reuse checker.
5. Add the UI, provenance display, feedback, and optional explanations.
6. Expand the corpus, rerun evaluations, and prepare the final report.
