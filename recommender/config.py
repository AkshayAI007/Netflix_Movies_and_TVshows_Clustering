import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH  = os.path.join(_ROOT, 'data', 'netflix_titles.csv')
MODELS_DIR = os.path.join(_ROOT, 'models')
CACHE_PATH = os.path.join(MODELS_DIR, 'recommender_cache.joblib')

# ── Feature weights ───────────────────────────────────────────────────────────
FEATURE_WEIGHTS: dict = {
    'description': 3.0,
    'genre':       4.0,
    'cast':        1.5,
    'director':    1.5,
    'country':     0.5,
    'numeric':     1.0,
}

# ── Rating ordinal scale ──────────────────────────────────────────────────────
RATING_ORDER: dict = {
    'G': 1, 'TV-G': 1, 'TV-Y': 1,
    'PG': 2, 'TV-Y7': 2, 'TV-Y7-FV': 2,
    'TV-PG': 3,
    'PG-13': 4, 'TV-14': 4, 'NR': 4, 'UR': 4,
    'R': 5,
    'TV-MA': 6,
    'NC-17': 7,
}

# ── TF-IDF stop words ─────────────────────────────────────────────────────────
CUSTOM_STOPS = ENGLISH_STOP_WORDS.union({
    'film', 'story', 'life', 'find', 'one', 'two', 'new', 'young',
    'man', 'woman', 'comes', 'must', 'world', 'set', 'follows',
    'series', 'show', 'movie', 'episode', 'season', 'netflix',
    'takes', 'place', 'make', 'way', 'gets', 'goes', 'day', 'time',
    'just', 'like', 'three', 'get', 'after', 'back', 'never',
    'when', 'while', 'before', 'until', 'soon', 'already', 'still',
    'help', 'try', 'live', 'turn', 'face', 'discover', 'decide',
    'journey', 'adventure', 'family', 'friends', 'love', 'work',
    'real', 'true', 'old', 'long', 'high', 'big',
})
