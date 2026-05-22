"""
Offline evaluation for the Netflix Recommender System.

Metrics (all proxy relevance via genre overlap — no user ratings available):
  precision@K        — fraction of top-K sharing ≥1 genre with query
  recall@K           — fraction of all relevant catalog titles in top-K
  intra-list diversity (ILD) — average pairwise genre dissimilarity
  catalog coverage   — fraction of catalog appearing in any recommendation
  novelty            — average self-information (niche-ness)
  serendipity        — relevant AND unexpected (type/era divergence)
"""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _genres_of(title: str, df: pd.DataFrame) -> set:
    row = df[df['title'].str.lower() == title.lower()]
    return set(row.iloc[0]['listed_in'].split(', ')) if not row.empty else set()


# ── Metric 1: Precision@K ─────────────────────────────────────────────────────

def precision_at_k(recommended: list, query: str,
                   df: pd.DataFrame, k: int = 10) -> float:
    query_genres = _genres_of(query, df)
    if not query_genres:
        return 0.0
    hits = sum(
        1 for t in recommended[:k] if _genres_of(t, df) & query_genres
    )
    return hits / min(k, len(recommended))


# ── Metric 2: Recall@K ────────────────────────────────────────────────────────

def recall_at_k(recommended: list, query: str,
                df: pd.DataFrame, k: int = 10) -> float:
    query_genres = _genres_of(query, df)
    if not query_genres:
        return 0.0
    all_relevant = df[df['title'] != query]['listed_in'].apply(
        lambda g: bool(set(g.split(', ')) & query_genres)
    ).sum()
    if all_relevant == 0:
        return 0.0
    hits = sum(
        1 for t in recommended[:k] if _genres_of(t, df) & query_genres
    )
    return hits / all_relevant


# ── Metric 3: Intra-List Diversity ────────────────────────────────────────────

def intra_list_diversity(recommended: list, df: pd.DataFrame,
                         genre_mat: np.ndarray,
                         title_to_idx: dict) -> float:
    """Average pairwise genre dissimilarity (high = diverse, low = echo chamber)."""
    indices = [title_to_idx[t] for t in recommended if t in title_to_idx]
    if len(indices) < 2:
        return 0.0
    vecs  = genre_mat[indices]
    sims  = cosine_similarity(vecs)
    K     = len(indices)
    total = sum(1 - sims[i, j] for i in range(K) for j in range(i + 1, K))
    return float(total / (K * (K - 1) / 2))


# ── Metric 4: Catalog Coverage ────────────────────────────────────────────────

def catalog_coverage(all_recommendations: list, catalog_size: int) -> float:
    unique = {t for recs in all_recommendations for t in recs}
    return len(unique) / catalog_size


# ── Metric 5: Novelty ─────────────────────────────────────────────────────────

def novelty_score(recommended: list, df: pd.DataFrame) -> float:
    """Average self-information; higher = more niche recommendations."""
    N = len(df)
    genre_counts = df['listed_in'].str.split(', ').explode().value_counts()
    scores = []
    for title in recommended:
        genres = _genres_of(title, df)
        if not genres:
            continue
        p = min(genre_counts.get(g, 1) for g in genres) / N
        scores.append(-np.log2(p + 1e-9))
    return float(np.mean(scores)) if scores else 0.0


# ── Metric 6: Serendipity ─────────────────────────────────────────────────────

def serendipity_score(recommended: list, query: str,
                      df: pd.DataFrame, k: int = 10) -> float:
    """Fraction that are relevant (shared genre) AND unexpected (type/era differs)."""
    q_row = df[df['title'].str.lower() == query.lower()]
    if q_row.empty:
        return 0.0
    q_genres = set(q_row.iloc[0]['listed_in'].split(', '))
    q_year   = int(q_row.iloc[0]['release_year'])
    q_type   = q_row.iloc[0]['type']

    hits = 0
    for title in recommended[:k]:
        r_row = df[df['title'].str.lower() == title.lower()]
        if r_row.empty:
            continue
        r_genres = set(r_row.iloc[0]['listed_in'].split(', '))
        r_year   = int(r_row.iloc[0]['release_year'])
        r_type   = r_row.iloc[0]['type']
        if (r_genres & q_genres) and (
            r_type != q_type or abs(r_year - q_year) > 15
        ):
            hits += 1
    return hits / min(k, len(recommended))


# ── Full model evaluation ──────────────────────────────────────────────────────

def evaluate_model(model, df: pd.DataFrame, eval_titles: list,
                   genre_mat: np.ndarray, title_to_idx: dict,
                   k: int = 10) -> dict:
    """Run all metrics on eval_titles. model.recommend(title, n=k) must return a
    DataFrame with a 'title' column."""
    p_scores, ild_scores, nov_scores, seren_scores, runtimes = [], [], [], [], []
    all_recs = []

    for title in eval_titles:
        t0 = time.perf_counter()
        try:
            recs_df = model.recommend(title, n=k)
        except Exception:
            continue
        runtimes.append((time.perf_counter() - t0) * 1000)

        rec_titles = recs_df['title'].tolist()
        all_recs.append(rec_titles)
        p_scores.append(precision_at_k(rec_titles, title, df, k))
        ild_scores.append(intra_list_diversity(rec_titles, df, genre_mat, title_to_idx))
        nov_scores.append(novelty_score(rec_titles, df))
        seren_scores.append(serendipity_score(rec_titles, title, df, k))

    return {
        f'precision@{k}': round(float(np.mean(p_scores)), 4),
        'ild':             round(float(np.mean(ild_scores)), 4),
        'coverage':        round(catalog_coverage(all_recs, len(df)), 4),
        'novelty':         round(float(np.mean(nov_scores)), 4),
        'serendipity':     round(float(np.mean(seren_scores)), 4),
        'avg_latency_ms':  round(float(np.mean(runtimes)), 2),
    }


def compare_models(models: dict, df: pd.DataFrame, eval_titles: list,
                   genre_mat: np.ndarray, title_to_idx: dict,
                   k: int = 10) -> pd.DataFrame:
    rows = {}
    for name, model in models.items():
        log.info('Evaluating %s…', name)
        rows[name] = evaluate_model(model, df, eval_titles, genre_mat, title_to_idx, k)
    return pd.DataFrame(rows).T


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import joblib
    from sklearn.preprocessing import MultiLabelBinarizer

    from recommender.preprocessing import load_and_build, parse_genres
    from recommender.models import TwoStageHybrid

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(name)s — %(message)s')

    log.info('Loading data and building features…')
    combined, transformers, df_clean = load_and_build()

    log.info('Fitting TwoStageHybrid…')
    model = TwoStageHybrid()
    model.fit_prebuilt(combined, transformers, df_clean)

    genres_list  = df_clean['listed_in'].apply(parse_genres).tolist()
    mlb          = MultiLabelBinarizer()
    genre_mat    = mlb.fit_transform(genres_list).astype(float)
    title_to_idx = {t: i for i, t in enumerate(df_clean['title'])}

    eval_titles = df_clean.sample(500, random_state=42)['title'].tolist()

    log.info('Evaluating on 500 random titles…')
    results = evaluate_model(model, df_clean, eval_titles, genre_mat, title_to_idx)

    print('\n=== Results ===')
    for metric, value in results.items():
        print(f'  {metric:<20} {value}')
