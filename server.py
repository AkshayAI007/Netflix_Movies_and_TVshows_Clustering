"""Flask API server for the Netflix Recommender System."""

import logging
import os

from flask import Flask, jsonify, request, send_from_directory
import pandas as pd

from recommender import NetflixRecommender
from recommender.config import CACHE_PATH, DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(name)s — %(message)s',
)
log = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
app   = Flask(__name__, static_folder=os.path.join(_HERE, 'frontend'),
              static_url_path='')

# ── Load recommender once at startup ──────────────────────────────────────────
log.info('Loading recommender…')
if os.path.exists(CACHE_PATH):
    rec = NetflixRecommender.load(CACHE_PATH)
else:
    rec = NetflixRecommender(data_path=DATA_PATH).fit()
    rec.save(CACHE_PATH)

df_cat = rec.catalog
log.info('Ready — %d titles loaded.', len(df_cat))


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/info')
def info():
    return jsonify({
        'total':    len(df_cat),
        'movies':   int((df_cat['type'] == 'Movie').sum()),
        'shows':    int((df_cat['type'] == 'TV Show').sum()),
        'min_year': int(df_cat['release_year'].min()),
        'max_year': int(df_cat['release_year'].max()),
        'ratings':  sorted(df_cat['rating'].dropna().unique().tolist()),
        'genres':   sorted(
            df_cat['listed_in'].str.split(', ').explode().unique().tolist()
        ),
    })


@app.route('/api/titles')
def titles():
    q     = request.args.get('q', '').lower().strip()
    all_t = sorted(df_cat['title'].tolist())
    if not q:
        return jsonify(all_t)
    prefix  = [t for t in all_t if t.lower().startswith(q)]
    contain = [t for t in all_t if q in t.lower() and not t.lower().startswith(q)]
    return jsonify((prefix + contain)[:30])


@app.route('/api/recommend', methods=['POST'])
def recommend():
    d      = request.get_json(force=True)
    title  = d.get('title', '')
    n      = int(d.get('n', 10))
    ct     = d.get('content_type') or None
    min_yr = int(d['min_year']) if d.get('min_year') else None
    max_yr = int(d['max_year']) if d.get('max_year') else None
    ratings = d.get('ratings') or None

    try:
        results = rec.recommend(
            title=title, n=n,
            content_type_filter=ct,
            min_release_year=min_yr,
            max_release_year=max_yr,
            rating_filter=ratings,
        )
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify(_enrich(results, title))


@app.route('/api/cross-type', methods=['POST'])
def cross_type():
    d     = request.get_json(force=True)
    title = d.get('title', '')
    n     = int(d.get('n', 10))

    try:
        results = rec.recommend_cross_type(title, n=n)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    row      = df_cat[df_cat['title'] == title]
    src_type = row.iloc[0]['type'] if not row.empty else ''
    return jsonify({'source_type': src_type, 'results': _enrich(results, title)})


@app.route('/api/cold-start', methods=['POST'])
def cold_start():
    d = request.get_json(force=True)
    try:
        results = rec.recommend_for_cold_start(
            description  = d.get('description', ''),
            genres       = d.get('genres', []),
            content_type = d.get('content_type', 'Movie'),
            cast         = d.get('cast', ''),
            director     = d.get('director', ''),
            country      = d.get('country', ''),
            rating       = d.get('rating', 'TV-MA'),
            release_year = int(d.get('release_year', 2020)),
            n            = int(d.get('n', 10)),
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify(_enrich(results, ''))


@app.route('/api/title-info')
def title_info():
    title = request.args.get('title', '')
    row   = df_cat[df_cat['title'] == title]
    if row.empty:
        return jsonify({}), 404
    r = row.iloc[0]
    return jsonify({
        'title':  r['title'],
        'type':   r['type'],
        'year':   int(r['release_year']),
        'rating': r['rating'],
        'genres': r.get('listed_in', ''),
    })


# ── Helpers ────────────────────────────────────────────────────────────────────

def _enrich(results: pd.DataFrame, query_title: str) -> list:
    """Attach explanation dict to each result row."""
    rows = []
    for _, row in results.iterrows():
        item = row.to_dict()
        item['release_year'] = int(item['release_year'])
        if query_title:
            try:
                item['explanation'] = rec.explain_recommendation(
                    query_title, row['title']
                )
            except Exception:
                item['explanation'] = {}
        else:
            item['explanation'] = {}
        rows.append(item)
    return rows


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
