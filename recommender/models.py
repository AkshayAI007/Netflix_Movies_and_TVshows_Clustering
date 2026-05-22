"""
ML models for the Netflix Recommender System.

  1. TFIDFCosineModel    — baseline: TF-IDF on description + cosine similarity
  2. BM25Model           — improved baseline (numpy BM25 from scratch)
  3. WeightedHybridModel — PRIMARY: weighted multi-feature cosine similarity
  4. NMFLatentModel      — latent topics via NMF; used for re-ranking
  5. ClusteringModel     — KMeans on SVD-reduced features; enables cross-type queries
  6. TwoStageHybrid      — PRODUCTION: WeightedHybrid candidate pool + NMF re-rank
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from scipy.sparse import issparse

from .config import CUSTOM_STOPS, FEATURE_WEIGHTS
from .preprocessing import build_all_features

log = logging.getLogger(__name__)


# ── Shared utilities ──────────────────────────────────────────────────────────

def _title_index(df: pd.DataFrame, title: str) -> int:
    matches = df.index[df['title'].str.lower() == title.lower()].tolist()
    if not matches:
        raise ValueError(f"Title not found in catalog: '{title}'")
    return matches[0]


def _scores_to_frame(
    scores: np.ndarray,
    df: pd.DataFrame,
    query_idx: int,
    n: int,
    filters: Optional[dict],
) -> pd.DataFrame:
    scores = scores.copy()
    scores[query_idx] = -1
    order = np.argsort(scores)[::-1]

    out_rows = []
    for idx in order:
        if len(out_rows) >= n:
            break
        row = df.iloc[idx]
        if filters:
            if filters.get('type') and row['type'] != filters['type']:
                continue
            if filters.get('min_year') and row['release_year'] < filters['min_year']:
                continue
            if filters.get('max_year') and row['release_year'] > filters['max_year']:
                continue
            if filters.get('ratings') and row['rating'] not in filters['ratings']:
                continue
        out_rows.append({
            'title':        row['title'],
            'type':         row['type'],
            'genres':       row['listed_in'],
            'release_year': int(row['release_year']),
            'rating':       row['rating'],
            'similarity':   round(float(scores[idx]), 4),
        })
    return pd.DataFrame(out_rows)


# ── 1. TF-IDF Cosine Baseline ─────────────────────────────────────────────────

class TFIDFCosineModel:
    """Baseline: TF-IDF on description only."""

    def __init__(self):
        self._tfidf = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            stop_words=list(CUSTOM_STOPS), sublinear_tf=True,
        )
        self._mat = None
        self._df  = None

    def fit(self, df: pd.DataFrame) -> 'TFIDFCosineModel':
        self._df  = df.reset_index(drop=True)
        self._mat = self._tfidf.fit_transform(self._df['desc_clean'].fillna(''))
        return self

    def recommend(self, title: str, n: int = 10,
                  filters: Optional[dict] = None) -> pd.DataFrame:
        idx    = _title_index(self._df, title)
        scores = cosine_similarity(self._mat[idx], self._mat).flatten()
        return _scores_to_frame(scores, self._df, idx, n, filters)


# ── 2. BM25 Model ─────────────────────────────────────────────────────────────

class BM25Model:
    """Okapi BM25 on description (k1=1.5, b=0.75), implemented with numpy."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self._df_    = None
        self._tf_mat = None
        self._idf    = None
        self._dl     = None
        self._avgdl  = None

    def fit(self, df: pd.DataFrame) -> 'BM25Model':
        self._df_ = df.reset_index(drop=True)
        vec = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            stop_words=list(CUSTOM_STOPS),
            sublinear_tf=False, use_idf=False, norm=None,
        )
        tf_sparse    = vec.fit_transform(self._df_['desc_clean'].fillna(''))
        self._tf_mat = tf_sparse.toarray().astype(np.float32)

        N            = self._tf_mat.shape[0]
        df_counts    = (self._tf_mat > 0).sum(axis=0).astype(np.float32)
        self._idf    = np.log((N - df_counts + 0.5) / (df_counts + 0.5) + 1.0)
        self._dl     = self._tf_mat.sum(axis=1)
        self._avgdl  = self._dl.mean()
        return self

    def _score(self, query_idx: int) -> np.ndarray:
        q_mask = self._tf_mat[query_idx] > 0
        k1, b  = self.k1, self.b
        denom  = self._tf_mat + k1 * (1 - b + b * (self._dl[:, None] / self._avgdl))
        numer  = self._tf_mat * (k1 + 1)
        return ((numer / denom) * self._idf * q_mask).sum(axis=1).astype(np.float32)

    def recommend(self, title: str, n: int = 10,
                  filters: Optional[dict] = None) -> pd.DataFrame:
        idx    = _title_index(self._df_, title)
        scores = self._score(idx)
        return _scores_to_frame(scores, self._df_, idx, n, filters)


# ── 3. Weighted Hybrid Model (PRIMARY) ───────────────────────────────────────

class WeightedHybridModel:
    """Weighted multi-feature sparse cosine similarity (~7,600-dim vector)."""

    def __init__(self, weights: Optional[dict] = None):
        self._weights      = weights or FEATURE_WEIGHTS
        self._mat          = None
        self._df           = None
        self._transformers = None
        self._score_cache: dict = {}

    def fit(self, df: pd.DataFrame,
            weights: Optional[dict] = None) -> 'WeightedHybridModel':
        w = weights or self._weights
        combined, transformers, df_clean = build_all_features(df, w)
        return self.fit_prebuilt(combined, transformers, df_clean)

    def fit_prebuilt(self, combined, transformers,
                     df_clean) -> 'WeightedHybridModel':
        self._mat          = combined
        self._df           = df_clean.reset_index(drop=True)
        self._transformers = transformers
        self._score_cache  = {}
        return self

    def _get_scores(self, idx: int) -> np.ndarray:
        if idx not in self._score_cache:
            self._score_cache[idx] = cosine_similarity(
                self._mat[idx], self._mat
            ).flatten()
        return self._score_cache[idx]

    def recommend(self, title: str, n: int = 10,
                  filters: Optional[dict] = None) -> pd.DataFrame:
        idx    = _title_index(self._df, title)
        scores = self._get_scores(idx)
        return _scores_to_frame(scores, self._df, idx, n, filters)

    @property
    def df(self):
        return self._df

    @property
    def matrix(self):
        return self._mat

    @property
    def transformers(self):
        return self._transformers


# ── 4. NMF Latent Model ───────────────────────────────────────────────────────

class NMFLatentModel:
    """NMF on feature matrix → cosine on 50-dim topic factors (re-ranker)."""

    def __init__(self, n_components: int = 50, random_state: int = 42):
        self.n_components = n_components
        self._nmf = NMF(
            n_components=n_components, random_state=random_state,
            max_iter=300, init='nndsvda',
        )
        self._W   = None
        self._df  = None
        self._score_cache: dict = {}

    def fit(self, feature_matrix, df: pd.DataFrame) -> 'NMFLatentModel':
        self._df = df.reset_index(drop=True)
        mat      = feature_matrix.toarray() if issparse(feature_matrix) else feature_matrix
        self._W  = self._nmf.fit_transform(mat).astype(np.float32)
        self._score_cache = {}
        log.info('NMF reconstruction error: %.2f', self._nmf.reconstruction_err_)
        return self

    def _get_scores(self, idx: int) -> np.ndarray:
        if idx not in self._score_cache:
            self._score_cache[idx] = cosine_similarity(
                self._W[idx:idx + 1], self._W
            ).flatten()
        return self._score_cache[idx]

    def recommend(self, title: str, n: int = 10,
                  filters: Optional[dict] = None) -> pd.DataFrame:
        idx    = _title_index(self._df, title)
        scores = self._get_scores(idx)
        return _scores_to_frame(scores, self._df, idx, n, filters)

    def score_for_idx(self, query_idx: int,
                      candidate_indices: np.ndarray) -> np.ndarray:
        return cosine_similarity(
            self._W[query_idx:query_idx + 1], self._W[candidate_indices]
        ).flatten()


# ── 5. Clustering Model ───────────────────────────────────────────────────────

class ClusteringModel:
    """KMeans on SVD-100 compressed features. Enables cross-type queries."""

    def __init__(self, n_clusters: int = 20, svd_components: int = 100,
                 random_state: int = 42):
        self.n_clusters     = n_clusters
        self.svd_components = svd_components
        self._svd     = TruncatedSVD(n_components=svd_components,
                                     random_state=random_state)
        self._km      = MiniBatchKMeans(n_clusters=n_clusters,
                                        random_state=random_state, n_init=10)
        self._reduced = None
        self._labels  = None
        self._df      = None

    def fit(self, feature_matrix, df: pd.DataFrame) -> 'ClusteringModel':
        self._df      = df.reset_index(drop=True)
        self._reduced = normalize(
            self._svd.fit_transform(feature_matrix).astype(np.float32)
        )
        self._labels = self._km.fit_predict(self._reduced)
        log.info(
            'SVD-%d explained variance: %.3f',
            self.svd_components,
            self._svd.explained_variance_ratio_.sum(),
        )
        return self

    def get_cluster_summary(self, cluster_id: int) -> dict:
        mask   = self._labels == cluster_id
        subset = self._df[mask]
        genres = subset['listed_in'].str.split(', ').explode().value_counts().head(5)
        return {
            'size':       int(mask.sum()),
            'top_genres': genres.index.tolist(),
            'types':      subset['type'].value_counts().to_dict(),
        }

    def recommend(self, title: str, n: int = 10,
                  same_cluster_only: bool = True,
                  content_type_filter: Optional[str] = None) -> pd.DataFrame:
        idx        = _title_index(self._df, title)
        cluster_id = self._labels[idx]
        mask       = (self._labels == cluster_id) if same_cluster_only \
                     else np.ones(len(self._df), dtype=bool)
        candidates = np.where(mask)[0]
        scores_all = cosine_similarity(
            self._reduced[idx:idx + 1], self._reduced[candidates]
        ).flatten()

        out_rows = []
        for rank in np.argsort(scores_all)[::-1]:
            c_idx = candidates[rank]
            if c_idx == idx:
                continue
            row = self._df.iloc[c_idx]
            if content_type_filter and row['type'] != content_type_filter:
                continue
            out_rows.append({
                'title':        row['title'],
                'type':         row['type'],
                'genres':       row['listed_in'],
                'release_year': int(row['release_year']),
                'rating':       row['rating'],
                'similarity':   round(float(scores_all[rank]), 4),
                'cluster_id':   int(cluster_id),
            })
            if len(out_rows) >= n:
                break
        return pd.DataFrame(out_rows)

    @property
    def labels(self):
        return self._labels

    @property
    def df(self):
        return self._df


# ── 6. Two-Stage Hybrid (PRODUCTION) ─────────────────────────────────────────

class TwoStageHybrid:
    """
    Stage 1: WeightedHybridModel → top-N cosine candidates
    Stage 2: NMFLatentModel      → re-rank
    final_score = alpha * cosine + (1 - alpha) * nmf
    """

    def __init__(self, alpha: float = 0.70, n_candidates: int = 50,
                 n_clusters: int = 20, svd_components: int = 100):
        self.alpha        = alpha
        self.n_candidates = n_candidates
        self._hybrid  = WeightedHybridModel()
        self._nmf     = NMFLatentModel()
        self._cluster = ClusteringModel(n_clusters=n_clusters,
                                        svd_components=svd_components)
        self._df      = None

    def fit(self, df: pd.DataFrame,
            weights: Optional[dict] = None) -> 'TwoStageHybrid':
        """Build all features then fit all sub-models."""
        combined, transformers, df_clean = build_all_features(
            df, weights or FEATURE_WEIGHTS
        )
        return self.fit_prebuilt(combined, transformers, df_clean)

    def fit_prebuilt(self, combined, transformers,
                     df_clean) -> 'TwoStageHybrid':
        """Fit all sub-models from pre-built feature matrix (avoids double build)."""
        self._df = df_clean.reset_index(drop=True)

        log.info('Fitting WeightedHybridModel…')
        self._hybrid.fit_prebuilt(combined, transformers, df_clean)

        log.info('Fitting NMFLatentModel…')
        self._nmf.fit(combined, df_clean)

        log.info('Fitting ClusteringModel…')
        self._cluster.fit(combined, df_clean)

        log.info('TwoStageHybrid ready.')
        return self

    def recommend(self, title: str, n: int = 10,
                  filters: Optional[dict] = None) -> pd.DataFrame:
        idx = _title_index(self._df, title)

        cosine_scores = self._hybrid._get_scores(idx).copy()
        cosine_scores[idx] = -1
        top_indices = np.argsort(cosine_scores)[::-1][:self.n_candidates]

        nmf_scores = self._nmf.score_for_idx(idx, top_indices)
        final_scores = {
            int(c_idx): self.alpha * float(cosine_scores[c_idx])
                        + (1 - self.alpha) * float(nmf_scores[rank])
            for rank, c_idx in enumerate(top_indices)
        }

        out_rows = []
        for c_idx, score in sorted(final_scores.items(),
                                   key=lambda x: x[1], reverse=True):
            if len(out_rows) >= n:
                break
            row = self._df.iloc[c_idx]
            if filters:
                if filters.get('type') and row['type'] != filters['type']:
                    continue
                if filters.get('min_year') and row['release_year'] < filters['min_year']:
                    continue
                if filters.get('max_year') and row['release_year'] > filters['max_year']:
                    continue
                if filters.get('ratings') and row['rating'] not in filters['ratings']:
                    continue
            out_rows.append({
                'title':        row['title'],
                'type':         row['type'],
                'genres':       row['listed_in'],
                'release_year': int(row['release_year']),
                'rating':       row['rating'],
                'similarity':   round(score, 4),
                'cluster_id':   int(self._cluster.labels[c_idx]),
            })
        return pd.DataFrame(out_rows)

    def recommend_cross_type(self, title: str, n: int = 10) -> pd.DataFrame:
        idx         = _title_index(self._df, title)
        query_type  = self._df.iloc[idx]['type']
        target_type = 'TV Show' if query_type == 'Movie' else 'Movie'

        cosine_scores = self._hybrid._get_scores(idx).copy()
        cosine_scores[idx] = -1
        sorted_indices = np.argsort(cosine_scores)[::-1]

        cross_indices = [
            i for i in sorted_indices
            if self._df.iloc[i]['type'] == target_type
        ]
        top_indices = np.array(cross_indices[:self.n_candidates])
        if len(top_indices) == 0:
            return pd.DataFrame()

        nmf_scores = self._nmf.score_for_idx(idx, top_indices)
        final_scores = {
            int(c_idx): self.alpha * float(cosine_scores[c_idx])
                        + (1 - self.alpha) * float(nmf_scores[rank])
            for rank, c_idx in enumerate(top_indices)
        }

        out_rows = []
        for c_idx, score in sorted(final_scores.items(),
                                   key=lambda x: x[1], reverse=True):
            if len(out_rows) >= n:
                break
            row = self._df.iloc[c_idx]
            out_rows.append({
                'title':        row['title'],
                'type':         row['type'],
                'genres':       row['listed_in'],
                'release_year': int(row['release_year']),
                'rating':       row['rating'],
                'similarity':   round(score, 4),
                'cluster_id':   int(self._cluster.labels[c_idx]),
            })
        return pd.DataFrame(out_rows)

    @property
    def df(self):
        return self._df

    @property
    def hybrid(self):
        return self._hybrid

    @property
    def cluster(self):
        return self._cluster

    @property
    def transformers(self):
        return self._hybrid.transformers
