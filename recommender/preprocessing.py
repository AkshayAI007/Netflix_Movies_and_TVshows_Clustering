import logging
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

from .config import CUSTOM_STOPS, RATING_ORDER, FEATURE_WEIGHTS, DATA_PATH

log = logging.getLogger(__name__)


# ── Data loading & quality fixes ──────────────────────────────────────────────

def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding='utf-8', encoding_errors='replace')


def fix_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix 3 rows where rating/duration are swapped (Louis C.K. trilogy)
    bad = df['rating'].str.contains('min', na=False)
    df.loc[bad, ['rating', 'duration']] = df.loc[bad, ['duration', 'rating']].values

    df['rating']   = df['rating'].fillna('TV-MA')
    df['director'] = df['director'].fillna('Unknown_Director')
    df['cast']     = df['cast'].fillna('Unknown_Cast')
    df['country']  = df['country'].fillna('Unknown')

    df['date_added'] = df['date_added'].str.strip()
    df['year_added'] = pd.to_datetime(
        df['date_added'], format='%B %d, %Y', errors='coerce'
    ).dt.year
    df['year_added'] = df.groupby('type')['year_added'].transform(
        lambda s: s.fillna(s.median())
    )
    return df


# ── Feature helpers ───────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if t not in CUSTOM_STOPS and len(t) > 2]
    return ' '.join(tokens)


def parse_duration(df: pd.DataFrame) -> pd.Series:
    def _parse(row):
        d = str(row['duration']) if pd.notna(row['duration']) else ''
        if 'Season' in d:
            try:
                return int(d.split()[0]) * 10
            except ValueError:
                return 0
        try:
            return int(d.replace(' min', ''))
        except ValueError:
            return 0
    return df.apply(_parse, axis=1)


def cast_to_tokens(cast_str: str, top_n: int = 5) -> str:
    if pd.isna(cast_str) or cast_str in ('Unknown_Cast', ''):
        return 'unknown_cast'
    actors = [a.strip().lower().replace(' ', '_') for a in cast_str.split(',')][:top_n]
    return ' '.join(actors)


def dir_to_token(director: str) -> str:
    if pd.isna(director) or director == 'Unknown_Director':
        return 'unknown_director'
    return director.strip().lower().replace(' ', '_')


def encode_rating_ordinal(df: pd.DataFrame) -> pd.Series:
    return df['rating'].map(RATING_ORDER).fillna(4).astype(float)


def parse_countries(country_str: str) -> list:
    if pd.isna(country_str) or country_str == 'Unknown':
        return ['Unknown']
    return [c.strip() for c in country_str.split(',') if c.strip()]


def parse_genres(genre_str: str) -> list:
    if pd.isna(genre_str):
        return []
    return [g.strip() for g in genre_str.split(',') if g.strip()]


# ── Full feature pipeline ─────────────────────────────────────────────────────

def build_all_features(df: pd.DataFrame, weights: dict = FEATURE_WEIGHTS):
    """
    Build the weighted sparse feature matrix for all titles.

    Returns (combined, transformers, df_clean):
      combined     — scipy CSR matrix, shape (N, ~7,600)
      transformers — dict of fitted sklearn objects needed for cold-start
      df_clean     — preprocessed DataFrame (index aligned with matrix rows)
    """
    df = df.reset_index(drop=True)

    df['desc_clean'] = df['description'].fillna('').apply(clean_text)
    df['cast_tokens'] = df['cast'].apply(cast_to_tokens)
    df['dir_token']   = df['director'].apply(dir_to_token)
    df['duration_num']   = parse_duration(df)
    df['rating_ordinal'] = encode_rating_ordinal(df)
    df['type_binary']    = (df['type'] == 'Movie').astype(float)

    tfidf_desc = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2),
        stop_words=list(CUSTOM_STOPS), min_df=2, max_df=0.85,
        sublinear_tf=True, dtype=np.float32,
    )
    desc_mat = tfidf_desc.fit_transform(df['desc_clean'])

    tfidf_cast = TfidfVectorizer(
        max_features=2000, token_pattern=r'\S+', min_df=2, dtype=np.float32,
    )
    cast_mat = tfidf_cast.fit_transform(df['cast_tokens'])

    tfidf_dir = TfidfVectorizer(
        max_features=500, token_pattern=r'\S+', min_df=2, dtype=np.float32,
    )
    dir_mat = tfidf_dir.fit_transform(df['dir_token'])

    genres_list = df['listed_in'].apply(parse_genres).tolist()
    mlb_genre = MultiLabelBinarizer()
    genre_mat = mlb_genre.fit_transform(genres_list).astype(np.float32)

    country_list = df['country'].apply(parse_countries).tolist()
    mlb_country = MultiLabelBinarizer()
    country_mat = mlb_country.fit_transform(country_list).astype(np.float32)

    numeric_raw = df[['duration_num', 'release_year', 'rating_ordinal', 'type_binary']].fillna(0)
    scaler = MinMaxScaler()
    numeric_mat = scaler.fit_transform(numeric_raw).astype(np.float32)

    w = weights
    combined = hstack([
        desc_mat                * w['description'],
        csr_matrix(genre_mat)   * w['genre'],
        cast_mat                * w['cast'],
        dir_mat                 * w['director'],
        csr_matrix(country_mat) * w['country'],
        csr_matrix(numeric_mat) * w['numeric'],
    ]).astype(np.float32)

    transformers = {
        'tfidf_desc':  tfidf_desc,
        'tfidf_cast':  tfidf_cast,
        'tfidf_dir':   tfidf_dir,
        'mlb_genre':   mlb_genre,
        'mlb_country': mlb_country,
        'scaler':      scaler,
        'weights':     w,
    }

    log.info(
        'Feature matrix: %s  (desc=%d genre=%d cast=%d dir=%d country=%d numeric=%d)',
        combined.shape,
        desc_mat.shape[1], genre_mat.shape[1], cast_mat.shape[1],
        dir_mat.shape[1], country_mat.shape[1], numeric_mat.shape[1],
    )
    return combined, transformers, df


def transform_single(row_dict: dict, transformers: dict) -> csr_matrix:
    """Vectorise a single new title (cold-start) through fitted transformers."""
    w = transformers['weights']

    desc_clean  = clean_text(row_dict.get('description', ''))
    cast_tokens = cast_to_tokens(row_dict.get('cast', ''))
    dir_token   = dir_to_token(row_dict.get('director', ''))
    genres      = parse_genres(row_dict.get('listed_in', ''))
    countries   = parse_countries(row_dict.get('country', ''))

    rating_ord = RATING_ORDER.get(row_dict.get('rating', 'TV-MA'), 4)
    type_bin   = float(row_dict.get('type', 'Movie') == 'Movie')
    dur_num    = float(row_dict.get('duration_num', 0))
    rel_year   = float(row_dict.get('release_year', 2020))

    desc_v  = transformers['tfidf_desc'].transform([desc_clean])  * w['description']
    cast_v  = transformers['tfidf_cast'].transform([cast_tokens]) * w['cast']
    dir_v   = transformers['tfidf_dir'].transform([dir_token])    * w['director']
    genre_v = csr_matrix(
        transformers['mlb_genre'].transform([genres]).astype(np.float32)
    ) * w['genre']
    country_v = csr_matrix(
        transformers['mlb_country'].transform([countries]).astype(np.float32)
    ) * w['country']
    numeric_raw = np.array([[dur_num, rel_year, rating_ord, type_bin]], dtype=np.float32)
    numeric_v   = csr_matrix(
        transformers['scaler'].transform(numeric_raw).astype(np.float32)
    ) * w['numeric']

    return hstack([desc_v, genre_v, cast_v, dir_v, country_v, numeric_v]).astype(np.float32)


# ── Entry point ───────────────────────────────────────────────────────────────

def load_and_build(path: str = DATA_PATH, weights: dict = FEATURE_WEIGHTS):
    """Load CSV → fix quality → build features. Returns (combined, transformers, df_clean)."""
    df_raw   = load_raw(path)
    df_fixed = fix_data_quality(df_raw)
    return build_all_features(df_fixed, weights)
