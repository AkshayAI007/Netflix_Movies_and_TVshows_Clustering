"""
NetflixRecommender — production wrapper around TwoStageHybrid.

Exposes:
  recommend()                — standard filtered recommendations
  recommend_cross_type()     — "more like X but as TV Show / Movie"
  recommend_for_cold_start() — new title not yet in catalog
  explain_recommendation()   — human-readable explanation
  save() / load()            — joblib persistence
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import DATA_PATH, CACHE_PATH, MODELS_DIR, FEATURE_WEIGHTS
from .preprocessing import load_and_build, transform_single
from .models import TwoStageHybrid, WeightedHybridModel, NMFLatentModel, ClusteringModel

log = logging.getLogger(__name__)


class NetflixRecommender:
    """
    End-to-end Netflix content recommender.

    Usage
    -----
    rec = NetflixRecommender().fit()      # build all models (~15 s)
    rec.save()                            # cache to disk
    rec = NetflixRecommender.load()       # reload from cache (~2 s)

    rec.recommend('Ozark', n=10)
    rec.recommend('Inception', content_type_filter='TV Show')
    rec.recommend_for_cold_start(description='...', genres=['Crime TV Shows'])
    rec.explain_recommendation('Ozark', 'Narcos')
    """

    def __init__(self, data_path: str = DATA_PATH,
                 weights: dict = None):
        self.data_path = data_path
        self.weights   = weights or FEATURE_WEIGHTS
        self._model: TwoStageHybrid = None
        self._df: pd.DataFrame      = None

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self) -> 'NetflixRecommender':
        log.info('Building features from %s…', self.data_path)
        combined, transformers, df_clean = load_and_build(
            self.data_path, self.weights
        )
        self._model = TwoStageHybrid()
        self._model.fit_prebuilt(combined, transformers, df_clean)
        self._df = self._model.df
        return self

    # ── Core recommendation ───────────────────────────────────────────────────

    def recommend(self, title: str, n: int = 10,
                  content_type_filter: str = None,
                  min_release_year: int = None,
                  max_release_year: int = None,
                  rating_filter: list = None) -> pd.DataFrame:
        """
        Recommend titles similar to `title`.

        Parameters
        ----------
        title               : must exist in the catalog
        n                   : number of results
        content_type_filter : 'Movie' or 'TV Show'
        min_release_year    : filter floor (inclusive)
        max_release_year    : filter ceiling (inclusive)
        rating_filter       : e.g. ['G', 'PG', 'TV-G']
        """
        filters = {}
        if content_type_filter:
            filters['type']     = content_type_filter
        if min_release_year:
            filters['min_year'] = min_release_year
        if max_release_year:
            filters['max_year'] = max_release_year
        if rating_filter:
            filters['ratings']  = set(rating_filter)
        return self._model.recommend(title, n=n, filters=filters or None)

    # ── Cross-type ────────────────────────────────────────────────────────────

    def recommend_cross_type(self, title: str, n: int = 10) -> pd.DataFrame:
        """Return recommendations of the opposite content type."""
        return self._model.recommend_cross_type(title, n)

    # ── Cold start ────────────────────────────────────────────────────────────

    def recommend_for_cold_start(
        self,
        description: str,
        genres: list,
        content_type: str = 'Movie',
        cast: str = '',
        director: str = '',
        country: str = '',
        rating: str = 'TV-MA',
        release_year: int = 2020,
        n: int = 10,
    ) -> pd.DataFrame:
        """Recommend for a title NOT yet in the catalog."""
        row_dict = {
            'description':  description,
            'listed_in':    ', '.join(genres),
            'cast':         cast,
            'director':     director,
            'country':      country,
            'rating':       rating,
            'type':         content_type,
            'release_year': release_year,
            'duration_num': 90,
        }
        vec    = transform_single(row_dict, self._model.transformers)
        scores = cosine_similarity(vec, self._model.hybrid.matrix).flatten()
        order  = np.argsort(scores)[::-1]

        out_rows = []
        for idx in order[:n]:
            row = self._df.iloc[idx]
            out_rows.append({
                'title':        row['title'],
                'type':         row['type'],
                'genres':       row['listed_in'],
                'release_year': int(row['release_year']),
                'rating':       row['rating'],
                'similarity':   round(float(scores[idx]), 4),
            })
        return pd.DataFrame(out_rows)

    # ── Explanation ───────────────────────────────────────────────────────────

    def explain_recommendation(self, query_title: str,
                                rec_title: str) -> dict:
        """Human-readable explanation of why rec_title was recommended."""
        df = self._df
        q_rows = df[df['title'].str.lower() == query_title.lower()]
        r_rows = df[df['title'].str.lower() == rec_title.lower()]

        if q_rows.empty or r_rows.empty:
            return {'error': 'One or both titles not found.'}

        q, r = q_rows.iloc[0], r_rows.iloc[0]

        q_genres = set(q['listed_in'].split(', '))
        r_genres = set(r['listed_in'].split(', '))
        q_cast   = set(str(q['cast']).split(', '))
        r_cast   = set(str(r['cast']).split(', '))
        same_dir = (q['director'] != 'Unknown_Director'
                    and q['director'] == r['director'])

        q_idx = df.index[df['title'].str.lower() == query_title.lower()][0]
        r_idx = df.index[df['title'].str.lower() == rec_title.lower()][0]

        return {
            'shared_genres': sorted(q_genres & r_genres),
            'shared_cast':   sorted((q_cast & r_cast) - {'Unknown_Cast'}),
            'same_director': same_dir,
            'director_name': q['director'] if same_dir else None,
            'rating_match':  q['rating'] == r['rating'],
            'era_match':     abs(int(q['release_year']) - int(r['release_year'])) <= 5,
            'query_year':    int(q['release_year']),
            'rec_year':      int(r['release_year']),
            'same_cluster':  (self._model.cluster.labels[q_idx]
                              == self._model.cluster.labels[r_idx]),
            'cluster_id':    int(self._model.cluster.labels[q_idx]),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = CACHE_PATH) -> None:
        """Save raw model components to avoid class-level pickle issues."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            'df_clean':        self._df,
            'combined':        self._model.hybrid.matrix,
            'transformers':    self._model.transformers,
            'nmf_W':           self._model._nmf._W,
            'cluster_labels':  self._model._cluster._labels,
            'cluster_reduced': self._model._cluster._reduced,
            'weights':         self.weights,
            'alpha':           self._model.alpha,
            'n_candidates':    self._model.n_candidates,
        }
        joblib.dump(payload, path, compress=3)
        log.info('Recommender saved to %s', path)

    @classmethod
    def load(cls, path: str = CACHE_PATH) -> 'NetflixRecommender':
        """Restore NetflixRecommender from saved components."""
        payload = joblib.load(path)

        rec           = cls.__new__(cls)
        rec.data_path = DATA_PATH
        rec.weights   = payload['weights']

        df_clean     = payload['df_clean']
        combined     = payload['combined']
        transformers = payload['transformers']

        hybrid = WeightedHybridModel()
        hybrid.fit_prebuilt(combined, transformers, df_clean)

        nmf      = NMFLatentModel()
        nmf._df  = df_clean.reset_index(drop=True)
        nmf._W   = payload['nmf_W']
        nmf._score_cache = {}

        cluster          = ClusteringModel()
        cluster._df      = df_clean.reset_index(drop=True)
        cluster._labels  = payload['cluster_labels']
        cluster._reduced = payload['cluster_reduced']

        model          = TwoStageHybrid(
            alpha=payload['alpha'],
            n_candidates=payload['n_candidates'],
        )
        model._hybrid  = hybrid
        model._nmf     = nmf
        model._cluster = cluster
        model._df      = df_clean.reset_index(drop=True)

        rec._model = model
        rec._df    = df_clean.reset_index(drop=True)

        log.info('Recommender loaded from %s (%d titles)', path, len(rec._df))
        return rec

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def catalog(self) -> pd.DataFrame:
        return self._df[['title', 'type', 'listed_in',
                          'release_year', 'rating']].copy()

    @property
    def titles(self) -> list:
        return self._df['title'].tolist()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(name)s — %(message)s')

    if os.path.exists(CACHE_PATH):
        rec = NetflixRecommender.load()
    else:
        rec = NetflixRecommender().fit()
        rec.save()

    test_title = 'Breaking Bad'
    print(f'\nTop 10 for "{test_title}":')
    print(rec.recommend(test_title, n=10).to_string(index=False))
    top1 = rec.recommend(test_title, n=1).iloc[0]['title']
    print(f'\nExplanation ({test_title} → {top1}):')
    print(rec.explain_recommendation(test_title, top1))
