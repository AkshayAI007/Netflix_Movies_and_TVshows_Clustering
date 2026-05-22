# Netflix AI Recommender System

A content-based ML recommendation engine built on the full Netflix catalog (8,807 titles), served through a Netflix-themed web UI with glassmorphism design and neural network background animation. Powered by a two-stage hybrid model that benchmarked best across 11 algorithms on 7 evaluation metrics.

---

## Live Demo

Deployed on Render — cold starts take ~30 seconds while the model builds; a loading banner keeps you informed.

---

## Features

- **Similar Titles** — top-N recommendations using a two-stage NMF + LDA ensemble hybrid model
- **Cross-Type** — given a movie, find the most similar TV shows (and vice versa)
- **Cold Start** — describe a hypothetical title by genre, description, cast, director and get the closest real matches from the catalog
- **Explainability** — every recommendation shows *why* it was suggested (shared genres, cast, director, era, cluster membership)
- **Filters** — content type, release year range, age rating
- **Analytics** — genre mix and match score charts rendered live in the sidebar
- **Loading UX** — animated banner polls `/health` during cold start; UI activates automatically when the model is ready
- **Netflix intro animation** — animated N logo, beam sweep, corner brackets on first load
- **Neural network canvas** — 72-node animated particle background with mouse attraction

---

## Architecture

```
data/netflix_titles.csv
        │
        ▼
recommender/preprocessing.py   ← data cleaning + 7,669-dim weighted sparse feature matrix
        │
        ▼
recommender/models.py          ← 6 base model classes
recommender/models_advanced.py ← 6 advanced model classes (LDA, GMM, BisectingKMeans, HDBSCAN, V2, V3)
        │
        ▼
recommender/core.py            ← NetflixRecommender wrapper (recommend, cold-start, explain, save/load)
        │
        ▼
evaluate.py                    ← 7-metric benchmark runner, 11-model comparison
        │
        ▼
server.py                      ← Flask API  (/health, / and /api/* routes, background model loading)
        │
        ▼
frontend/index.html            ← pure HTML/CSS/JS frontend (no framework)
```

---

## Dataset

`data/netflix_titles.csv` — 8,807 titles, 12 columns. No user ratings or watch history.

| Column | Description |
|---|---|
| `show_id` | Unique identifier |
| `type` | Movie or TV Show |
| `title` | Title name |
| `director` | Director(s) — 29.9% missing |
| `cast` | Cast list — 9.4% missing |
| `country` | Production country — 9.4% missing |
| `date_added` | Date added to Netflix |
| `release_year` | Release year (1925–2021) |
| `rating` | Age rating (G, PG, TV-MA, etc.) |
| `duration` | Runtime in minutes (movies) or seasons (shows) |
| `listed_in` | Genre tags |
| `description` | Short synopsis (~24 words median) |

**Data quality fixes applied:**
- 3 rows (Louis C.K. trilogy) where `rating` contained duration values and `duration` contained rating values — columns swapped back
- Missing directors/cast filled with `Unknown_Director` / `Unknown_Cast` tokens
- Missing countries filled with `['Unknown']`
- Missing ratings filled with modal value `TV-MA`

---

## ML Pipeline

### Feature Engineering (`recommender/preprocessing.py`)

All features are combined into a single sparse CSR matrix and weighted by importance:

| Feature | Method | Dimensions | Weight |
|---|---|---|---|
| `description` | TF-IDF (max 5,000 features, bigrams, sublinear_tf) | 5,000 | 3.0 |
| `listed_in` genres | MultiLabelBinarizer (42 genres) | 42 | **4.0** |
| `cast` | TF-IDF on top-5 actors (max 2,000) | 2,000 | 1.5 |
| `director` | TF-IDF (max 500) | 500 | 1.5 |
| `country` | MultiLabelBinarizer (123 countries) | 123 | 0.5 |
| Numerics | MinMaxScaler on [duration, year, rating_ordinal, type_flag] | 4 | 1.0 |

**Total: 7,669 float32 sparse dimensions**

Cosine similarity is computed on-demand per title (never a full 8,807×8,807 pairwise matrix) and cached per-instance.

---

### Models (`recommender/models.py` + `recommender/models_advanced.py`)

Twelve models were implemented across three families. All share the same interface: `fit()`, `recommend(title, n, filters)`, `score_for_idx()`.

#### Base Models

| Model | Algorithm | Role |
|---|---|---|
| `TFIDFCosineModel` | TF-IDF bag-of-words cosine | Baseline — description text only |
| `BM25Model` | Okapi BM25 | Baseline — probabilistic term-frequency |
| `WeightedHybridModel` | Weighted sparse cosine on 7,669-dim matrix | Stage 1 retrieval |
| `NMFLatentModel` | Non-negative Matrix Factorization (50 components) | Latent topic re-ranker |
| `ClusteringModel` | SVD-100 → MiniBatchKMeans (20 clusters) | Cross-type cluster routing |
| `TwoStageHybrid` | WeightedHybrid + NMF re-rank | Previous production model |

#### Advanced Models

| Model | Algorithm | Key innovation |
|---|---|---|
| `LDATopicModel` | Latent Dirichlet Allocation (50 topics) | Probabilistic Dirichlet prior over latent topics; operates on description TF-IDF block |
| `GMMClusteringModel` | SVD-100 → Gaussian Mixture (25 components, diag) | Soft membership — each title has a probability vector over 25 Gaussians, scored by KL divergence |
| `BisectingKMeansModel` | SVD-100 → BisectingKMeans (20 clusters) | Hierarchically splits the largest cluster; better intra-cluster cohesion than random KMeans init |
| `HDBSCANClusteringModel` | SVD-100 → HDBSCAN | Density-based; auto-discovers cluster count; niche titles naturally isolated as outliers |
| `TwoStageHybridV2` | WeightedHybrid + LDA re-rank | `score = 0.70 × cosine + 0.30 × LDA` |
| **`TwoStageHybridV3`** | **WeightedHybrid + NMF + LDA ensemble** | **`score = 0.70 × cosine + 0.15 × NMF + 0.15 × LDA` — production model** |

---

### Production Model: TwoStageHybridV3

```
Stage 1 → WeightedHybridModel   → top-50 cosine candidates from 7,669-dim sparse matrix
Stage 2 → NMFLatentModel        → 50-component latent topic score
        + LDATopicModel         → 50-topic Dirichlet distribution score

final_score = 0.70 × cosine_score
            + 0.15 × NMF_score
            + 0.15 × LDA_score
```

NMF captures deterministic non-negative factorization patterns; LDA captures probabilistic thematic structure with a Dirichlet prior. Their combination as Stage 2 re-rankers produces more diverse, broader-coverage results than either alone, without sacrificing precision.

---

### Benchmark Results — Full 11-Model Comparison

**Setup:** 500-title random holdout (random_state=42). Relevance proxy: genre overlap (≥1 shared genre = relevant). All models evaluated on identical data from a single shared feature matrix build.

| # | Model | Precision@10 | ILD | Coverage | Novelty | Serendipity | NDCG@10 | Latency |
|---|---|---|---|---|---|---|---|---|
| 5 | KMeans Clustering | 1.0000 ⚠ | 0.016 | 34.5% | 4.20 | 0.0752 | 1.0000 | 8.51ms |
| 8 | BisectingKMeans | 1.0000 ⚠ | 0.016 | 34.5% | 4.20 | 0.0752 | 1.0000 | 8.39ms |
| **11** | **TwoStageHybridV3** ★ | **0.9996** | **0.0203** | **37.1%** | **4.21** | **0.0720** | **1.0000** | **6.91ms** |
| 9 | TwoStageHybrid (prev prod) | 0.9996 | 0.0142 | 35.2% | 4.22 | 0.0692 | 0.9999 | 6.57ms |
| 10 | TwoStageHybridV2 | 0.9994 | 0.0308 | 36.3% | 4.18 | 0.0740 | 1.0000 | 6.58ms |
| 3 | WeightedHybrid | 0.9994 | 0.0171 | 35.0% | 4.20 | 0.0706 | 0.9999 | 6.12ms |
| 4 | NMF (standalone) | 0.9992 | 0.0160 | 41.5% | 4.25 | 0.0508 | 0.9998 | 5.27ms |
| 7 | GMM Clustering | 0.8600 | 0.3036 | 18.4% | 4.26 | 0.0728 | 0.9169 | 5.33ms |
| 2 | BM25 (baseline) | 0.3984 | 0.7836 | 42.0% | 4.28 | 0.0422 | 0.6899 | 360.31ms |
| 1 | TF-IDF Cosine (baseline) | 0.3878 | 0.7922 | 41.3% | 4.29 | 0.0410 | 0.6805 | 5.36ms |
| 6 | LDA (standalone) | 0.2606 | 0.8656 | 33.9% | 4.30 | 0.0274 | 0.5356 | 5.12ms |

> ⚠ **KMeans/BisectingKMeans note:** Perfect Precision@10 is achieved by clustering tightly on genre — all recommendations come from the same dense genre cluster. The near-zero ILD (0.016) confirms this is an echo chamber, not a good recommender. Excluded from production consideration.

Full results saved to `models/comparison_results.csv`.

#### Metric Definitions

| Metric | What it measures |
|---|---|
| **Precision@10** | Fraction of top-10 recommendations sharing ≥1 genre with the query title |
| **ILD** (Intra-List Diversity) | Average pairwise genre dissimilarity among the 10 results — higher means more varied |
| **Coverage** | Fraction of the 8,807-title catalog that appears in at least one recommendation set |
| **Novelty** | Average self-information of recommended genres — higher means more niche content surfaced |
| **Serendipity** | Fraction that are genre-relevant AND unexpectedly different in type or era |
| **NDCG@10** | Normalised Discounted Cumulative Gain — rewards relevant items ranked higher in the list |
| **Avg Latency** | Wall-clock time per `recommend()` call (milliseconds) |

---

### TwoStageHybridV3 vs Previous Production Model

| Metric | TwoStageHybrid (before) | TwoStageHybridV3 (after) | Δ |
|---|---|---|---|
| Precision@10 | 99.96% | **99.96%** | No regression |
| NDCG@10 | 0.9999 | **1.0000** | Perfect position-aware ranking |
| ILD (diversity) | 0.0142 | **0.0203** | **+43%** more diverse results |
| Catalog Coverage | 35.2% | **37.1%** | **+5.3%** more titles surfaced |
| Serendipity | 0.0692 | **0.0720** | **+4%** more pleasant surprises |
| Avg Latency | 6.57ms | 6.91ms | +0.34ms (negligible) |

---

## Evaluation Framework (`evaluate.py`)

```bash
python evaluate.py
```

Runs all 11 models on the same 500-title holdout, prints a ranked comparison table, and saves results to `models/comparison_results.csv`. All models share a single feature matrix build — no redundant computation.

Seven metrics are computed per model:

```python
evaluate_model(model, df, eval_titles, genre_mat, title_to_idx, k=10)
# returns: precision@10, ild, coverage, novelty, serendipity, ndcg@10, avg_latency_ms
```

---

## API Reference (`server.py`)

Flask backend serving the frontend and all recommendation endpoints. Model loads in a background thread — gunicorn binds to the port immediately and routes return `503` with a clear message until the model is ready.

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serves `frontend/index.html` |
| `/health` | GET | `{"status":"ok"}` (200) when ready, `{"status":"loading"}` (503) during startup |
| `/api/info` | GET | Catalog stats (total, movies, shows, year range, ratings, genres) |
| `/api/titles?q=<query>` | GET | Autocomplete — up to 30 results, prefix-first |
| `/api/recommend` | POST | Similar title recommendations |
| `/api/cross-type` | POST | Recommendations of the opposite content type |
| `/api/cold-start` | POST | Recommendations for a title not yet in the catalog |
| `/api/title-info?title=<title>` | GET | Metadata for a single title |

### POST `/api/recommend`

```json
{
  "title": "Dark",
  "n": 10,
  "content_type": "TV Show",
  "min_year": 2015,
  "max_year": 2021,
  "ratings": ["TV-MA", "TV-14"]
}
```

### POST `/api/cross-type`

```json
{
  "title": "Inception",
  "n": 10
}
```

### POST `/api/cold-start`

```json
{
  "description": "A detective hunts a serial killer across parallel timelines",
  "genres": ["Crime TV Shows", "TV Dramas", "TV Thrillers"],
  "content_type": "TV Show",
  "cast": "",
  "director": "",
  "country": "",
  "rating": "TV-MA",
  "release_year": 2020,
  "n": 10
}
```

---

## File Structure

```
Netflix Recommender system/
├── recommender/
│   ├── __init__.py              ← exposes NetflixRecommender
│   ├── config.py                ← paths, weights, constants (single source of truth)
│   ├── preprocessing.py         ← data cleaning + feature engineering
│   ├── models.py                ← 6 base model classes
│   ├── models_advanced.py       ← 6 advanced model classes
│   └── core.py                  ← NetflixRecommender wrapper
├── data/
│   └── netflix_titles.csv
├── models/
│   ├── .gitkeep
│   └── comparison_results.csv   ← 11-model benchmark output
├── frontend/
│   └── index.html               ← UI (HTML + CSS + JS, no framework)
├── notebooks/
│   └── 01_eda.ipynb             ← 14 EDA visualisations
├── server.py                    ← Flask API server
├── evaluate.py                  ← offline evaluation + benchmark runner
├── gunicorn.conf.py             ← reads PORT from env in Python
├── render.yaml                  ← Render health check config
├── pyproject.toml               ← packaging config
├── Procfile                     ← Railway/Heroku start command
├── railway.toml                 ← Railway build + deploy config
├── requirements.txt             ← production dependencies
├── requirements-dev.txt         ← dev/notebook extras
└── .gitignore
```

---

## Running Locally

### Prerequisites

Python 3.9+

```bash
# Production dependencies only
pip install -r requirements.txt

# With notebook/viz extras
pip install -r requirements-dev.txt
```

### Start the server

```bash
python server.py
```

Open `http://localhost:8000`.

On first run the model fits from scratch (~15–20 seconds) and saves a cache to `models/recommender_cache.joblib`. Subsequent starts load from cache in ~2 seconds.

### Run the benchmark

```bash
python evaluate.py
```

Fits and evaluates all 11 models on a 500-title holdout. Prints a ranked comparison table and saves `models/comparison_results.csv`.

---

## Deployment

### Render

1. Push to GitHub
2. Create a new Web Service on [Render](https://render.com), connect the repo
3. Render detects `render.yaml` automatically — build command, start command, and health check path are pre-configured

```yaml
# render.yaml
services:
  - type: web
    name: netflix-recommender
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn server:app -c gunicorn.conf.py
    healthCheckPath: /health
```

`gunicorn.conf.py` reads `PORT` directly in Python — no shell variable expansion issues:

```python
import os
bind    = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = 1
timeout = 120
```

The model loads in a background thread — gunicorn binds to the port immediately. The `/health` endpoint returns `503` until the model is ready, then `200`. Render's health check waits for `200` before routing traffic.

### Railway

1. Push to GitHub
2. Create a new project on [Railway](https://railway.app), connect the repo
3. Railway uses the `Procfile`:

```
web: gunicorn server:app -c gunicorn.conf.py
```

### Environment Variables

No environment variables are required. Optional:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `10000` | Port the server binds to |

---

## EDA Highlights (`notebooks/01_eda.ipynb`)

14 production-quality charts using the Netflix red palette (`#E50914`):

1. **Content type split** — 69.6% Movies, 30.4% TV Shows (donut)
2. **Rating distribution by type** — TV-MA dominates both categories
3. **Top production countries** — US (3,690) >> India (1,046); 16% co-productions
4. **Genre treemap** — International Movies (2,752) and Dramas (2,427) are the anchors
5. **Content added over time** — dual-area chart; explosion 2016–2019, slowdown 2020–2021
6. **Release year histogram** — median 2017, long left tail to 1925
7. **Movie runtime violin** — bimodal: 85–90 min (stand-up) vs 100–110 min (features)
8. **TV seasons distribution** — 67% single-season; max 17 seasons
9. **Genre co-occurrence network** — "International Movies" is the highest-degree hub
10. **Description word cloud** — cleaned with sklearn stop words + custom domain stops
11. **Cast word cloud** — Anupam Kher (43 titles), Shah Rukh Khan (35) — Bollywood prominence
12. **Genre × country heatmap** — US dominates all; India→Drama; Japan→Anime; Korea→Romance
13. **Content added month heatmap** — July and December are peak acquisition months
14. **Top directors by type** — Rajiv Chilaka (22 titles, children's animation)

---

## Technical Notes

**Why not Streamlit?**
The UI uses pure HTML/CSS/JS with a Flask backend. This gives full control over CSS keyframe animations (Netflix intro), canvas-based neural network background, and glassmorphism styling — none of which are achievable in Streamlit without hacks.

**Why not precompute the full similarity matrix?**
8,807² × 4 bytes ≈ 310 MB of RAM just for storage, plus the compute cost at startup. Per-title on-demand cosine similarity takes ~1ms and is cached per-instance for repeated queries.

**Why save raw numpy arrays instead of the model object?**
Classes with internal caches can't always be pickled cleanly by `joblib`. The `save()` method dumps only plain numpy arrays and DataFrames; `load()` reconstructs the model instances from scratch. This also makes it safe to upgrade model classes without breaking the cache format — `load()` simply catches incompatible caches and rebuilds.

**Why TwoStageHybridV3 over perfect-scoring KMeans?**
KMeans and BisectingKMeans both scored Precision@10 = 1.000 in the benchmark, but their ILD (intra-list diversity) was 0.016 — nearly zero. This means all 10 recommendations come from the same dense genre cluster and are near-identical to each other. That is an echo chamber, not a useful recommender. V3 matches the best non-trivial precision (99.96%) while delivering 43% more diverse results.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data | pandas, numpy, scipy (sparse CSR matrices) |
| ML — retrieval | scikit-learn (TF-IDF, BM25, cosine similarity) |
| ML — re-ranking | scikit-learn (NMF, LDA, SVD, GMM, BisectingKMeans) |
| ML — clustering | scikit-learn (MiniBatchKMeans, BisectingKMeans, GMM), hdbscan |
| Persistence | joblib |
| Backend | Flask, gunicorn |
| Frontend | HTML5 + CSS3 + Vanilla JS + Chart.js |
| Deployment | Render (primary), Railway |
