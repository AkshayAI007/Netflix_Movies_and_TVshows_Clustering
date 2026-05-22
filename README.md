# Netflix AI Recommender System

A content-based ML recommendation engine built on the Netflix catalog, served through a Netflix-themed web UI with glassmorphism design and neural network background animation.

---

## Live Demo

Deploy to Railway in one click — see [Deployment](#deployment) section.

---

## Features

- **Similar Titles** — find the top-N most similar movies or shows using a two-stage hybrid ML model
- **Cross-Type** — given a movie, find similar TV shows (and vice versa)
- **Cold Start** — describe a hypothetical title by genre, description, cast, and get the closest matches from the catalog
- **Explainability** — every recommendation shows *why* it was suggested (shared genres, cast, director, era)
- **Filters** — content type, release year range, age rating
- **Analytics** — genre mix and match score charts rendered live in the sidebar
- **Netflix intro animation** — animated N logo, beam sweep, corner brackets on first load
- **Neural network canvas** — 72-node animated particle network background with mouse attraction

---

## Architecture

```
netflix_titles.csv
       │
       ▼
02_preprocessing.py   ← data cleaning + 7,669-dim weighted sparse feature matrix
       │
       ▼
03_models.py          ← 5 model classes + TwoStageHybrid
       │
       ▼
04_recommender.py     ← NetflixRecommender wrapper (recommend, cold start, explain, save/load)
       │
       ▼
05_evaluation.py      ← 6 evaluation metrics (Precision@10, coverage, novelty, …)
       │
       ▼
server.py             ← Flask API  (/ and /api/* routes)
       │
       ▼
index.html            ← pure HTML/CSS/JS frontend (no framework)
```

---

## Dataset

`netflix_titles.csv` — 8,807 titles, 12 columns (no user ratings or watch history).

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

### Feature Engineering (`02_preprocessing.py`)

All features are combined into a single sparse matrix and weighted by importance:

| Feature | Method | Dimensions | Weight |
|---|---|---|---|
| `description` | TF-IDF (max 5,000 features, bigrams, sublinear_tf) | ~4,800 | 3.0 |
| `listed_in` genres | MultiLabelBinarizer (42 genres) | 42 | **4.0** |
| `cast` | TF-IDF on top-5 actors (max 2,000) | ~1,800 | 1.5 |
| `director` | TF-IDF (max 500) | ~480 | 1.5 |
| `country` | MultiLabelBinarizer (124 countries) | 124 | 0.5 |
| Numerics | MinMaxScaler on [duration, year, rating_ordinal, type] | 4 | 1.0 |

**Total: ~7,669 float32 sparse dimensions**

Cosine similarity is computed on-demand per title (never a full 8,807×8,807 pairwise matrix) and cached with `@lru_cache(maxsize=2000)`.

### Models (`03_models.py`)

| Model | Role | Precision@10 |
|---|---|---|
| `TFIDFCosineModel` | TF-IDF description baseline | ~50% |
| `BM25Model` | Okapi BM25 short-text baseline | ~58% |
| `WeightedHybridModel` | Cosine on full weighted feature matrix | ~93% |
| `NMFLatentModel` | NMF (50 components) latent topic re-ranker | ~83% |
| `ClusteringModel` | KMeans (20 clusters, SVD-100) for cross-type | ~78% |
| **`TwoStageHybrid`** | **Production model** | **99.96%** |

### Production Model: TwoStageHybrid

```
Stage 1 → WeightedHybridModel  → top-50 cosine candidates
Stage 2 → NMFLatentModel        → re-rank using latent topics
final_score = 0.70 × cosine_score + 0.30 × nmf_score
```

### Evaluation Metrics (`05_evaluation.py`)

Evaluated on a 500-title holdout (random_state=42). Relevance is proxied by genre overlap (≥1 shared genre = relevant).

| Metric | Score |
|---|---|
| Precision@10 | **99.96%** |
| Catalog Coverage | 35.17% |
| Novelty | 4.22 |
| Serendipity | 0.069 |
| Avg Query Latency | ~10ms |

---

## API Reference (`server.py`)

Flask backend serving the frontend and all recommendation endpoints.

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serves `index.html` |
| `/api/info` | GET | Catalog stats (total, movies, shows, year range, ratings, genres) |
| `/api/titles?q=<query>` | GET | Autocomplete search — returns up to 30 matching titles, prefix-first |
| `/api/recommend` | POST | Similar title recommendations |
| `/api/cross-type` | POST | Recommendations of the opposite type |
| `/api/cold-start` | POST | Recommendations for a title not in the catalog |
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
├── netflix_titles.csv          ← source dataset
├── 01_eda.ipynb                ← 14 EDA visualizations (Plotly/Matplotlib)
├── 02_preprocessing.py         ← data cleaning + feature engineering
├── 03_models.py                ← all model classes + TwoStageHybrid
├── 04_recommender.py           ← NetflixRecommender wrapper
├── 05_evaluation.py            ← evaluation metrics + benchmark runner
├── server.py                   ← Flask API server
├── index.html                  ← frontend (HTML + CSS + JS, no framework)
├── requirements.txt
├── Procfile                    ← Railway deployment
├── railway.toml                ← Railway build + deploy config
└── .gitignore
```

---

## Running Locally

### Prerequisites

- Python 3.9+
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Start the server

```bash
python server.py
```

Open `http://localhost:8000` in your browser.

On first run the model fits from scratch (~15 seconds) and saves a cache to `recommender_cache.joblib`. Subsequent starts load from cache in ~2 seconds.

### Run evaluation

```bash
python 05_evaluation.py
```

Prints a comparison table of all models across all 6 metrics on the 500-title holdout.

---

## Deployment

### Railway (recommended)

1. Push this repository to GitHub (the `.gitignore` already excludes the cache file and virtual env)
2. Create a new project on [Railway](https://railway.app) and connect the GitHub repo
3. Railway auto-detects `nixpacks` and runs `python server.py` via the `Procfile`
4. The model fits from scratch on first deploy (~15s startup) — Railway free tier handles this fine

The `PORT` environment variable is read automatically:

```python
port = int(os.environ.get('PORT', 8000))
app.run(host='0.0.0.0', port=port)
```

### Environment variables

No environment variables are required. Optional:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port the Flask server listens on |

---

## EDA Highlights (`01_eda.ipynb`)

14 production-quality charts using the Netflix red palette (`#E50914`):

1. **Content type split** — 69.6% Movies, 30.4% TV Shows (donut)
2. **Rating distribution by type** — TV-MA dominates both categories
3. **Top production countries** — US (3,690) >> India (1,046); 16% co-productions
4. **Genre treemap** — International Movies (2,752) and Dramas (2,427) are anchors
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
8,807² × 4 bytes ≈ 310MB of RAM just for storage, plus the compute cost at startup. Per-title on-demand cosine similarity takes ~1ms and is cached with `@lru_cache(maxsize=2000)` for repeated queries.

**Why `importlib` instead of normal imports?**
Python module names must be valid identifiers. Files named `02_preprocessing.py` and `03_models.py` can't be imported with `import 02_preprocessing`. `importlib.util.spec_from_file_location` bypasses this restriction.

**Why save raw components instead of the model object?**
`@lru_cache` decorators and classes loaded via `importlib` can't be pickled by `joblib`. The `save()` method dumps only plain numpy arrays and DataFrames; `load()` reconstructs the class instances from scratch.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data | pandas, numpy, scipy (sparse matrices) |
| ML | scikit-learn (TF-IDF, NMF, KMeans, SVD, MinMaxScaler) |
| Persistence | joblib |
| Backend | Flask |
| Frontend | HTML5 + CSS3 + Vanilla JS + Chart.js |
| Deployment | Railway (nixpacks) |
