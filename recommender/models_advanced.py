"""
Advanced ML models for the Netflix Recommender System.

New models (all implement the same interface as models.py):
  1. LDATopicModel          — LDA topic modelling on descriptions (re-ranker alternative to NMF)
  2. GMMClusteringModel     — Gaussian Mixture Model (soft cluster membership)
  3. BisectingKMeansModel   — Hierarchical KMeans (better cluster quality than MiniBatchKMeans)
  4. HDBSCANClusteringModel — Density-based clustering, no K needed
  5. TwoStageHybridV2       — WeightedHybrid + LDA re-rank
  6. TwoStageHybridV3       — WeightedHybrid + NMF + LDA ensemble re-rank
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.cluster import BisectingKMeans
from sklearn.preprocessing import normalize

from .config import FEATURE_WEIGHTS
from .preprocessing import build_all_features
from .models import (
    WeightedHybridModel, NMFLatentModel,
    _title_index, _scores_to_frame,
)

log = logging.getLogger(__name__)


# ── 1. LDA Topic Model ────────────────────────────────────────────────────────

class LDATopicModel:
    """
    LatentDirichletAllocation on TF-IDF description features as a re-ranker.

    LDA requires non-negative count-like input, so we operate on the description
    TF-IDF block (cols 0:5000 of the feature matrix) converted to dense float32.
    Produces 50-dim topic distributions; cosine similarity in topic space.
    """

    def __init__(self, n_components: int = 50, max_iter: int = 20,
                 random_state: int = 42):
        self.n_components = n_components
        self._lda = LatentDirichletAllocation(
            n_components=n_components, max_iter=max_iter,
            random_state=random_state, n_jobs=-1,
        )
        self._H: np.ndarray = None   # (N, n_components) topic distributions
        self._df: pd.DataFrame = None
        self._score_cache: dict = {}
        self._desc_cols: int = 5000  # TF-IDF description block width

    def fit(self, feature_matrix, df: pd.DataFrame) -> 'LDATopicModel':
        self._df = df.reset_index(drop=True)
        # Extract the description TF-IDF block (first 5000 cols)
        mat = feature_matrix[:, :self._desc_cols]
        if issparse(mat):
            mat = mat.toarray()
        mat = np.abs(mat).astype(np.float32)   # ensure non-negative
        self._H = self._lda.fit_transform(mat).astype(np.float32)
        self._score_cache = {}
        log.info('LDA perplexity: %.1f', self._lda.perplexity(mat))
        return self

    def _get_scores(self, idx: int) -> np.ndarray:
        if idx not in self._score_cache:
            self._score_cache[idx] = cosine_similarity(
                self._H[idx:idx + 1], self._H
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
            self._H[query_idx:query_idx + 1], self._H[candidate_indices]
        ).flatten()


# ── 2. GMM Clustering Model ───────────────────────────────────────────────────

class GMMClusteringModel:
    """
    Gaussian Mixture Model on SVD-100 reduced features.

    Soft cluster membership: each title gets a probability vector over
    n_components Gaussians. Recommendations scored by negative KL divergence
    between query and candidate probability vectors (higher = more similar).
    """

    def __init__(self, n_components: int = 25, svd_components: int = 100,
                 random_state: int = 42):
        self.n_components   = n_components
        self.svd_components = svd_components
        self._svd     = TruncatedSVD(n_components=svd_components,
                                     random_state=random_state)
        self._gmm     = GaussianMixture(n_components=n_components,
                                        covariance_type='diag',
                                        random_state=random_state,
                                        max_iter=200)
        self._proba:   np.ndarray = None  # (N, n_components) soft membership
        self._labels:  np.ndarray = None  # (N,) hard labels via argmax
        self._reduced: np.ndarray = None
        self._df:      pd.DataFrame = None

    def fit(self, feature_matrix, df: pd.DataFrame) -> 'GMMClusteringModel':
        self._df      = df.reset_index(drop=True)
        self._reduced = normalize(
            self._svd.fit_transform(feature_matrix).astype(np.float32)
        )
        self._proba  = self._gmm.fit_predict  # placeholder
        self._proba  = self._gmm.fit(self._reduced).predict_proba(
            self._reduced
        ).astype(np.float32)
        self._labels = self._proba.argmax(axis=1)
        log.info(
            'GMM converged=%s  components=%d  SVD variance=%.3f',
            self._gmm.converged_, self.n_components,
            self._svd.explained_variance_ratio_.sum(),
        )
        return self

    def _kl_similarity(self, p: np.ndarray, Q: np.ndarray,
                       eps: float = 1e-9) -> np.ndarray:
        """Negative symmetric KL divergence (higher = more similar)."""
        p  = p + eps
        Qe = Q + eps
        kl = (p * np.log(p / Qe)).sum(axis=1) + (Qe * np.log(Qe / p)).sum(axis=1)
        return -kl   # negate so higher = more similar

    def recommend(self, title: str, n: int = 10,
                  same_cluster_only: bool = False,
                  content_type_filter: Optional[str] = None) -> pd.DataFrame:
        idx      = _title_index(self._df, title)
        q_proba  = self._proba[idx:idx + 1]

        if same_cluster_only:
            cluster_id = int(self._labels[idx])
            mask       = self._labels == cluster_id
            candidates = np.where(mask)[0]
            scores_all = np.full(len(self._df), -np.inf, dtype=np.float32)
            scores_all[candidates] = self._kl_similarity(
                q_proba, self._proba[candidates]
            )
        else:
            scores_all = self._kl_similarity(q_proba, self._proba)

        out_rows = []
        for i in np.argsort(scores_all)[::-1]:
            if i == idx:
                continue
            row = self._df.iloc[i]
            if content_type_filter and row['type'] != content_type_filter:
                continue
            out_rows.append({
                'title':        row['title'],
                'type':         row['type'],
                'genres':       row['listed_in'],
                'release_year': int(row['release_year']),
                'rating':       row['rating'],
                'similarity':   round(float(scores_all[i]), 4),
                'cluster_id':   int(self._labels[i]),
            })
            if len(out_rows) >= n:
                break
        return pd.DataFrame(out_rows)

    def get_cluster_summary(self, cluster_id: int) -> dict:
        mask   = self._labels == cluster_id
        subset = self._df[mask]
        genres = subset['listed_in'].str.split(', ').explode().value_counts().head(5)
        return {
            'size':       int(mask.sum()),
            'top_genres': genres.index.tolist(),
            'types':      subset['type'].value_counts().to_dict(),
        }

    @property
    def labels(self):
        return self._labels

    @property
    def df(self):
        return self._df


# ── 3. Bisecting KMeans Model ─────────────────────────────────────────────────

class BisectingKMeansModel:
    """
    BisectingKMeans (sklearn ≥ 1.1) on SVD-100 reduced features.

    Repeatedly bisects the largest cluster → better intra-cluster cohesion
    than standard KMeans random initialisation. Drop-in for ClusteringModel.
    """

    def __init__(self, n_clusters: int = 20, svd_components: int = 100,
                 random_state: int = 42):
        self.n_clusters     = n_clusters
        self.svd_components = svd_components
        self._svd     = TruncatedSVD(n_components=svd_components,
                                     random_state=random_state)
        self._bkm     = BisectingKMeans(n_clusters=n_clusters,
                                        bisecting_strategy='largest_cluster',
                                        random_state=random_state)
        self._reduced: np.ndarray = None
        self._labels:  np.ndarray = None
        self._df:      pd.DataFrame = None

    def fit(self, feature_matrix, df: pd.DataFrame) -> 'BisectingKMeansModel':
        self._df      = df.reset_index(drop=True)
        self._reduced = normalize(
            self._svd.fit_transform(feature_matrix).astype(np.float32)
        )
        self._labels = self._bkm.fit_predict(self._reduced)
        log.info(
            'BisectingKMeans  clusters=%d  SVD variance=%.3f',
            self.n_clusters, self._svd.explained_variance_ratio_.sum(),
        )
        return self

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

    def get_cluster_summary(self, cluster_id: int) -> dict:
        mask   = self._labels == cluster_id
        subset = self._df[mask]
        genres = subset['listed_in'].str.split(', ').explode().value_counts().head(5)
        return {
            'size':       int(mask.sum()),
            'top_genres': genres.index.tolist(),
            'types':      subset['type'].value_counts().to_dict(),
        }

    @property
    def labels(self):
        return self._labels

    @property
    def df(self):
        return self._df


# ── 4. HDBSCAN Clustering Model ───────────────────────────────────────────────

class HDBSCANClusteringModel:
    """
    HDBSCAN on SVD-100 reduced features.

    Density-based: auto-discovers cluster count, no K to tune.
    Noise points (label = -1) are treated as their own singleton cluster.
    Requires: pip install hdbscan
    """

    def __init__(self, min_cluster_size: int = 15, svd_components: int = 100,
                 random_state: int = 42):
        self.min_cluster_size = min_cluster_size
        self.svd_components   = svd_components
        self._svd     = TruncatedSVD(n_components=svd_components,
                                     random_state=random_state)
        self._reduced: np.ndarray = None
        self._labels:  np.ndarray = None
        self._df:      pd.DataFrame = None

    def fit(self, feature_matrix, df: pd.DataFrame) -> 'HDBSCANClusteringModel':
        try:
            import hdbscan
        except ImportError:
            raise ImportError(
                'hdbscan is required: pip install hdbscan'
            )
        self._df      = df.reset_index(drop=True)
        self._reduced = normalize(
            self._svd.fit_transform(feature_matrix).astype(np.float32)
        )
        clusterer     = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean', core_dist_n_jobs=-1,
        )
        self._labels  = clusterer.fit_predict(self._reduced)
        n_clusters    = len(set(self._labels)) - (1 if -1 in self._labels else 0)
        n_noise       = int((self._labels == -1).sum())
        log.info(
            'HDBSCAN  clusters=%d  noise_points=%d  SVD variance=%.3f',
            n_clusters, n_noise, self._svd.explained_variance_ratio_.sum(),
        )
        return self

    def recommend(self, title: str, n: int = 10,
                  same_cluster_only: bool = True,
                  content_type_filter: Optional[str] = None) -> pd.DataFrame:
        idx        = _title_index(self._df, title)
        cluster_id = int(self._labels[idx])

        if same_cluster_only and cluster_id != -1:
            mask = self._labels == cluster_id
        else:
            mask = np.ones(len(self._df), dtype=bool)

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
                'cluster_id':   cluster_id,
            })
            if len(out_rows) >= n:
                break
        return pd.DataFrame(out_rows)

    def get_cluster_summary(self, cluster_id: int) -> dict:
        mask   = self._labels == cluster_id
        subset = self._df[mask]
        genres = subset['listed_in'].str.split(', ').explode().value_counts().head(5)
        return {
            'size':       int(mask.sum()),
            'top_genres': genres.index.tolist(),
            'types':      subset['type'].value_counts().to_dict(),
            'is_noise':   cluster_id == -1,
        }

    @property
    def labels(self):
        return self._labels

    @property
    def df(self):
        return self._df


# ── 5. TwoStageHybridV2 (LDA as Stage 2) ─────────────────────────────────────

class TwoStageHybridV2:
    """
    Stage 1: WeightedHybridModel  → top-50 cosine candidates  (unchanged)
    Stage 2: LDATopicModel        → re-rank with LDA topic cosine
    final_score = alpha * cosine + (1 - alpha) * LDA_score
    """

    def __init__(self, alpha: float = 0.70, n_candidates: int = 50,
                 n_clusters: int = 20, svd_components: int = 100):
        self.alpha        = alpha
        self.n_candidates = n_candidates
        self._hybrid  = WeightedHybridModel()
        self._lda     = LDATopicModel()
        self._cluster = BisectingKMeansModel(n_clusters=n_clusters,
                                              svd_components=svd_components)
        self._df: pd.DataFrame = None

    def fit_prebuilt(self, combined, transformers,
                     df_clean) -> 'TwoStageHybridV2':
        self._df = df_clean.reset_index(drop=True)
        log.info('V2 — fitting WeightedHybridModel…')
        self._hybrid.fit_prebuilt(combined, transformers, df_clean)
        log.info('V2 — fitting LDATopicModel…')
        self._lda.fit(combined, df_clean)
        log.info('V2 — fitting BisectingKMeansModel…')
        self._cluster.fit(combined, df_clean)
        log.info('TwoStageHybridV2 ready.')
        return self

    def recommend(self, title: str, n: int = 10,
                  filters: Optional[dict] = None) -> pd.DataFrame:
        idx = _title_index(self._df, title)

        cosine_scores = self._hybrid._get_scores(idx).copy()
        cosine_scores[idx] = -1
        top_indices = np.argsort(cosine_scores)[::-1][:self.n_candidates]

        lda_scores   = self._lda.score_for_idx(idx, top_indices)
        final_scores = {
            int(c_idx): self.alpha * float(cosine_scores[c_idx])
                        + (1 - self.alpha) * float(lda_scores[rank])
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
        cross_indices = [
            i for i in np.argsort(cosine_scores)[::-1]
            if self._df.iloc[i]['type'] == target_type
        ]
        top_indices = np.array(cross_indices[:self.n_candidates])
        if len(top_indices) == 0:
            return pd.DataFrame()

        lda_scores   = self._lda.score_for_idx(idx, top_indices)
        final_scores = {
            int(c_idx): self.alpha * float(cosine_scores[c_idx])
                        + (1 - self.alpha) * float(lda_scores[rank])
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


# ── 6. TwoStageHybridV3 (NMF + LDA ensemble) ─────────────────────────────────

class TwoStageHybridV3:
    """
    Stage 1: WeightedHybridModel  → top-50 cosine candidates  (unchanged)
    Stage 2: NMFLatentModel + LDATopicModel, scores averaged
    final_score = alpha * cosine + beta * NMF + gamma * LDA
                = 0.70 * cosine + 0.15 * NMF + 0.15 * LDA
    """

    def __init__(self, alpha: float = 0.70, beta: float = 0.15,
                 gamma: float = 0.15, n_candidates: int = 50,
                 n_clusters: int = 20, svd_components: int = 100):
        self.alpha        = alpha
        self.beta         = beta
        self.gamma        = gamma
        self.n_candidates = n_candidates
        self._hybrid  = WeightedHybridModel()
        self._nmf     = NMFLatentModel()
        self._lda     = LDATopicModel()
        self._cluster = BisectingKMeansModel(n_clusters=n_clusters,
                                              svd_components=svd_components)
        self._df: pd.DataFrame = None

    def fit_prebuilt(self, combined, transformers,
                     df_clean) -> 'TwoStageHybridV3':
        self._df = df_clean.reset_index(drop=True)
        log.info('V3 — fitting WeightedHybridModel…')
        self._hybrid.fit_prebuilt(combined, transformers, df_clean)
        log.info('V3 — fitting NMFLatentModel…')
        self._nmf.fit(combined, df_clean)
        log.info('V3 — fitting LDATopicModel…')
        self._lda.fit(combined, df_clean)
        log.info('V3 — fitting BisectingKMeansModel…')
        self._cluster.fit(combined, df_clean)
        log.info('TwoStageHybridV3 ready.')
        return self

    def recommend(self, title: str, n: int = 10,
                  filters: Optional[dict] = None) -> pd.DataFrame:
        idx = _title_index(self._df, title)

        cosine_scores = self._hybrid._get_scores(idx).copy()
        cosine_scores[idx] = -1
        top_indices = np.argsort(cosine_scores)[::-1][:self.n_candidates]

        nmf_scores = self._nmf.score_for_idx(idx, top_indices)
        lda_scores = self._lda.score_for_idx(idx, top_indices)

        final_scores = {
            int(c_idx): (self.alpha * float(cosine_scores[c_idx])
                         + self.beta  * float(nmf_scores[rank])
                         + self.gamma * float(lda_scores[rank]))
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
        cross_indices = [
            i for i in np.argsort(cosine_scores)[::-1]
            if self._df.iloc[i]['type'] == target_type
        ]
        top_indices = np.array(cross_indices[:self.n_candidates])
        if len(top_indices) == 0:
            return pd.DataFrame()

        nmf_scores = self._nmf.score_for_idx(idx, top_indices)
        lda_scores = self._lda.score_for_idx(idx, top_indices)
        final_scores = {
            int(c_idx): (self.alpha * float(cosine_scores[c_idx])
                         + self.beta  * float(nmf_scores[rank])
                         + self.gamma * float(lda_scores[rank]))
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
