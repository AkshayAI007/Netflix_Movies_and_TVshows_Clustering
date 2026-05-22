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
    p_scores, ild_scores, nov_scores, seren_scores, ndcg_scores, runtimes = \
        [], [], [], [], [], []
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
        ndcg_scores.append(ndcg_at_k(rec_titles, title, df, k))

    return {
        f'precision@{k}': round(float(np.mean(p_scores)), 4),
        'ild':             round(float(np.mean(ild_scores)), 4),
        'coverage':        round(catalog_coverage(all_recs, len(df)), 4),
        'novelty':         round(float(np.mean(nov_scores)), 4),
        'serendipity':     round(float(np.mean(seren_scores)), 4),
        'ndcg@10':         round(float(np.mean(ndcg_scores)), 4),
        'avg_latency_ms':  round(float(np.mean(runtimes)), 2),
    }


# ── Metric 7: NDCG@K (position-aware) ────────────────────────────────────────

def ndcg_at_k(recommended: list, query: str,
              df: pd.DataFrame, k: int = 10) -> float:
    """
    Normalised Discounted Cumulative Gain.
    Relevance = 1 if shared genre with query else 0.
    Rewards putting relevant items higher in the list.
    """
    query_genres = _genres_of(query, df)
    if not query_genres:
        return 0.0
    rels = [
        1 if _genres_of(t, df) & query_genres else 0
        for t in recommended[:k]
    ]
    dcg  = sum(r / np.log2(i + 2) for i, r in enumerate(rels))
    # Ideal DCG: all relevant items at top
    n_relevant = sum(rels)
    idcg = sum(1 / np.log2(i + 2) for i in range(n_relevant))
    return float(dcg / idcg) if idcg > 0 else 0.0


def compare_models(models: dict, df: pd.DataFrame, eval_titles: list,
                   genre_mat: np.ndarray, title_to_idx: dict,
                   k: int = 10) -> pd.DataFrame:
    rows = {}
    for name, model in models.items():
        log.info('Evaluating %s…', name)
        rows[name] = evaluate_model(model, df, eval_titles, genre_mat, title_to_idx, k)
    return pd.DataFrame(rows).T


# ── Master comparison runner ──────────────────────────────────────────────────

def compare_all(df_clean: pd.DataFrame, combined, transformers: dict,
                eval_titles: list, genre_mat: np.ndarray,
                title_to_idx: dict, k: int = 10,
                save_path: str = None) -> pd.DataFrame:
    """
    Fit and evaluate all 9 models on the same pre-built feature matrix.
    Returns a comparison DataFrame sorted by precision@k descending.
    Optionally saves results to save_path (CSV).
    """
    import os
    from recommender.models import (
        TFIDFCosineModel, BM25Model, WeightedHybridModel,
        NMFLatentModel, ClusteringModel, TwoStageHybrid,
    )
    from recommender.models_advanced import (
        LDATopicModel, GMMClusteringModel, BisectingKMeansModel,
        TwoStageHybridV2, TwoStageHybridV3,
    )

    # Wrappers so each model exposes .recommend(title, n=k)
    class _ClusterWrap:
        def __init__(self, m): self._m = m
        def recommend(self, title, n=10, filters=None):
            return self._m.recommend(title, n=n, same_cluster_only=False)

    log.info('Building all models from shared feature matrix…')

    # Baselines — need df with desc_clean
    tfidf = TFIDFCosineModel().fit(df_clean)
    bm25  = BM25Model().fit(df_clean)

    # Primary models — use pre-built matrix
    hybrid = WeightedHybridModel()
    hybrid.fit_prebuilt(combined, transformers, df_clean)

    nmf = NMFLatentModel()
    nmf.fit(combined, df_clean)

    kmeans = ClusteringModel()
    kmeans.fit(combined, df_clean)

    lda = LDATopicModel()
    lda.fit(combined, df_clean)

    gmm = GMMClusteringModel()
    gmm.fit(combined, df_clean)

    bkm = BisectingKMeansModel()
    bkm.fit(combined, df_clean)

    prod = TwoStageHybrid()
    prod.fit_prebuilt(combined, transformers, df_clean)

    v2 = TwoStageHybridV2()
    v2.fit_prebuilt(combined, transformers, df_clean)

    v3 = TwoStageHybridV3()
    v3.fit_prebuilt(combined, transformers, df_clean)

    model_registry = {
        '1. TFIDFCosine (baseline)':         tfidf,
        '2. BM25 (baseline)':                bm25,
        '3. WeightedHybrid':                 hybrid,
        '4. NMF (standalone)':               nmf,
        '5. KMeans Clustering':              _ClusterWrap(kmeans),
        '6. LDA (standalone)':               lda,
        '7. GMM Clustering':                 _ClusterWrap(gmm),
        '8. BisectingKMeans':                _ClusterWrap(bkm),
        '9. TwoStageHybrid (current prod)':  prod,
        '10. TwoStageHybridV2 (LDA)':        v2,
        '11. TwoStageHybridV3 (NMF+LDA)':   v3,
    }

    results = compare_models(
        model_registry, df_clean, eval_titles, genre_mat, title_to_idx, k
    )
    results = results.sort_values(f'precision@{k}', ascending=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        results.to_csv(save_path)
        log.info('Saved comparison results to %s', save_path)

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from sklearn.preprocessing import MultiLabelBinarizer

    from recommender.preprocessing import load_and_build, parse_genres
    from recommender.config import MODELS_DIR
    import os

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(name)s — %(message)s')

    log.info('Loading data and building features…')
    combined, transformers, df_clean = load_and_build()

    genres_list  = df_clean['listed_in'].apply(parse_genres).tolist()
    mlb          = MultiLabelBinarizer()
    genre_mat    = mlb.fit_transform(genres_list).astype(float)
    title_to_idx = {t: i for i, t in enumerate(df_clean['title'])}
    eval_titles  = df_clean.sample(500, random_state=42)['title'].tolist()

    save_path = os.path.join(MODELS_DIR, 'comparison_results.csv')
    results   = compare_all(
        df_clean, combined, transformers,
        eval_titles, genre_mat, title_to_idx,
        save_path=save_path,
    )

    print('\n' + '='*72)
    print('MODEL COMPARISON — 500-title holdout, genre-overlap relevance proxy')
    print('='*72)
    print(results.to_string())
    print('='*72)
    print(f'\nSaved to: {save_path}')
