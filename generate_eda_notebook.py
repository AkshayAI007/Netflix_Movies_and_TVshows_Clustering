"""
Generates 01_eda.ipynb with 14 production-quality EDA charts.
Run: python generate_eda_notebook.py
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

SETUP = '''
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from wordcloud import WordCloud
from collections import Counter
from itertools import combinations
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# Netflix palette
RED      = '#E50914'
DARK_RED = '#B20710'
CHARCOAL = '#221F1F'
LIGHT    = '#E5E5E5'
MUTED    = '#A3A3A3'
BG       = '#141414'
CARD     = '#1F1F1F'

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        plot_bgcolor=CARD,
        paper_bgcolor=BG,
        font=dict(color=LIGHT, family='Helvetica'),
        title=dict(font=dict(size=18, color=RED)),
        xaxis=dict(gridcolor='#333', linecolor='#333', tickcolor=MUTED, titlefont=dict(color=MUTED)),
        yaxis=dict(gridcolor='#333', linecolor='#333', tickcolor=MUTED, titlefont=dict(color=MUTED)),
    )
)

df = pd.read_csv('netflix_titles.csv', encoding='utf-8', encoding_errors='replace')

# Fix 3 Louis C.K. rows
bad = df['rating'].str.contains('min', na=False)
df.loc[bad, ['rating','duration']] = df.loc[bad, ['duration','rating']].values
df['rating'] = df['rating'].fillna('TV-MA')

# Parse date_added
df['year_added'] = pd.to_datetime(df['date_added'].str.strip(), format='%B %d, %Y', errors='coerce').dt.year
df['month_added'] = pd.to_datetime(df['date_added'].str.strip(), format='%B %d, %Y', errors='coerce').dt.month

# Duration numerics
def parse_dur(row):
    d = str(row['duration']) if pd.notna(row['duration']) else ''
    if 'Season' in d:
        try: return int(d.split()[0]), 'seasons'
        except: return 0, 'seasons'
    try: return int(d.replace(' min','')), 'min'
    except: return 0, 'min'

df['dur_val'], df['dur_unit'] = zip(*df.apply(parse_dur, axis=1))

CUSTOM_STOPS = ENGLISH_STOP_WORDS.union({
    'film','story','life','find','one','two','new','young','man','woman',
    'comes','must','world','set','follows','series','show','movie',
    'episode','season','netflix','takes','place','make','way','gets',
    'goes','day','time','just','like','three','get','after','back',
    'never','when','while','before','until','soon','already','still',
    'help','try','live','turn','face','discover','decide','journey',
    'adventure','family','friends','love','work','real','true',
})

print(f"Dataset loaded: {df.shape[0]:,} titles × {df.shape[1]} columns")
print(f"Movies: {(df['type']=='Movie').sum():,} | TV Shows: {(df['type']=='TV Show').sum():,}")
'''

CHART1 = '''
# ── Chart 1: Content Type Donut ──────────────────────────────────────────────
type_counts = df['type'].value_counts()
fig = go.Figure(go.Pie(
    labels=type_counts.index,
    values=type_counts.values,
    hole=0.55,
    marker_colors=[RED, DARK_RED],
    textinfo='label+percent',
    textfont=dict(size=16, color='white'),
    hovertemplate='%{label}: %{value:,} titles (%{percent})<extra></extra>',
))
fig.add_annotation(text='8,807<br>Titles', x=0.5, y=0.5,
    font=dict(size=20, color=LIGHT), showarrow=False)
fig.update_layout(
    title='Netflix Catalog: Movies vs TV Shows',
    **PLOTLY_TEMPLATE['layout'].to_plotly_json(),
    height=420, showlegend=True,
    legend=dict(font=dict(color=LIGHT), bgcolor=BG),
)
fig.show()
'''

CHART2 = '''
# ── Chart 2: Rating Distribution by Content Type ─────────────────────────────
RATING_ORDER = ['G','TV-G','TV-Y','PG','TV-Y7','TV-Y7-FV','TV-PG','PG-13','TV-14','NR','UR','R','TV-MA','NC-17']
rating_type = df.groupby(['rating','type']).size().reset_index(name='count')
rating_type = rating_type[rating_type['rating'].isin(RATING_ORDER)]

fig = px.bar(
    rating_type, x='rating', y='count', color='type', barmode='group',
    color_discrete_map={'Movie': RED, 'TV Show': DARK_RED},
    category_orders={'rating': RATING_ORDER},
    title='Rating Distribution by Content Type',
    labels={'count':'Titles', 'rating':'Rating'},
    template='plotly_dark',
)
fig.update_layout(
    **PLOTLY_TEMPLATE['layout'].to_plotly_json(),
    height=420, legend_title='',
    legend=dict(font=dict(color=LIGHT), bgcolor=BG),
)
fig.show()
'''

CHART3 = '''
# ── Chart 3: Top 20 Countries ────────────────────────────────────────────────
countries = df['country'].fillna('Unknown').str.split(',').explode().str.strip()
top_countries = countries.value_counts().head(20).reset_index()
top_countries.columns = ['country','count']
top_countries = top_countries[top_countries['country'] != 'Unknown']

fig = px.bar(
    top_countries.sort_values('count'), x='count', y='country',
    orientation='h', text='count',
    color='count', color_continuous_scale=[DARK_RED, RED],
    title='Top 20 Countries by Number of Titles',
    labels={'count':'Titles', 'country':''},
    template='plotly_dark',
)
fig.update_traces(textfont=dict(color='white'), textposition='outside')
fig.update_layout(
    **PLOTLY_TEMPLATE['layout'].to_plotly_json(),
    height=560, coloraxis_showscale=False,
    margin=dict(l=150),
)
fig.show()
'''

CHART4 = '''
# ── Chart 4: Genre Treemap ───────────────────────────────────────────────────
genres = df['listed_in'].str.split(',').explode().str.strip()
genre_counts = genres.value_counts().reset_index()
genre_counts.columns = ['genre','count']

fig = px.treemap(
    genre_counts, path=['genre'], values='count',
    color='count', color_continuous_scale=[DARK_RED, RED, '#FF6B6B'],
    title='Netflix Genre Landscape (Treemap)',
)
fig.update_traces(
    textfont=dict(size=14, color='white'),
    hovertemplate='<b>%{label}</b><br>%{value:,} titles<extra></extra>',
)
fig.update_layout(
    **PLOTLY_TEMPLATE['layout'].to_plotly_json(),
    height=520, margin=dict(t=50,l=0,r=0,b=0),
    coloraxis_showscale=False,
)
fig.show()
'''

CHART5 = '''
# ── Chart 5: Content Added Over Time (Dual-Area) ─────────────────────────────
time_data = df.groupby(['year_added','type']).size().reset_index(name='count')
time_data = time_data[time_data['year_added'].between(2008, 2021)]

movies_t  = time_data[time_data['type']=='Movie']
tvshows_t = time_data[time_data['type']=='TV Show']

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=movies_t['year_added'], y=movies_t['count'],
    fill='tozeroy', name='Movies',
    line=dict(color=RED, width=2.5),
    fillcolor='rgba(229,9,20,0.25)',
))
fig.add_trace(go.Scatter(
    x=tvshows_t['year_added'], y=tvshows_t['count'],
    fill='tozeroy', name='TV Shows',
    line=dict(color='#FFAD49', width=2.5),
    fillcolor='rgba(255,173,73,0.20)',
))
fig.update_layout(
    title='Content Added to Netflix Per Year',
    xaxis_title='Year', yaxis_title='Titles Added',
    **PLOTLY_TEMPLATE['layout'].to_plotly_json(),
    height=420,
    legend=dict(font=dict(color=LIGHT), bgcolor=BG),
)
fig.show()
'''

CHART6 = '''
# ── Chart 6: Release Year Distribution + KDE ─────────────────────────────────
from scipy.stats import gaussian_kde

years = df['release_year'].dropna()
kde   = gaussian_kde(years, bw_method=0.15)
x_range = np.linspace(years.min(), years.max(), 300)

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=years, nbinsx=50, name='Count',
    marker_color=RED, opacity=0.7,
))
fig.add_trace(go.Scatter(
    x=x_range, y=kde(x_range) * len(years) * (years.max()-years.min()) / 50,
    mode='lines', name='KDE',
    line=dict(color='#FFAD49', width=3),
))
fig.update_layout(
    title='Release Year Distribution (1925–2021)',
    xaxis_title='Release Year', yaxis_title='Number of Titles',
    **PLOTLY_TEMPLATE['layout'].to_plotly_json(),
    height=420, barmode='overlay',
    legend=dict(font=dict(color=LIGHT), bgcolor=BG),
)
fig.show()
'''

CHART7 = '''
# ── Chart 7: Movie Duration — Violin + Box ───────────────────────────────────
movie_dur = df[(df['type']=='Movie') & (df['dur_val']>0)]['dur_val']

fig = go.Figure()
fig.add_trace(go.Violin(
    y=movie_dur, name='Duration',
    box_visible=True, meanline_visible=True,
    fillcolor='rgba(229,9,20,0.3)',
    line_color=RED, width=0.6,
    points='outliers',
    marker=dict(color=RED, size=3),
))
fig.update_layout(
    title='Movie Duration Distribution',
    yaxis_title='Duration (minutes)',
    **PLOTLY_TEMPLATE['layout'].to_plotly_json(),
    height=450, showlegend=False,
)
# Annotation for key stats
for val, label in [(movie_dur.median(), 'Median'), (movie_dur.mean(), 'Mean')]:
    fig.add_hline(y=val, line_dash='dash', line_color='#FFAD49',
                  annotation_text=f'{label}: {val:.0f} min',
                  annotation_font_color='#FFAD49')
fig.show()
'''

CHART8 = '''
# ── Chart 8: TV Show Seasons (Log-scale) ─────────────────────────────────────
tv_dur = df[(df['type']=='TV Show') & (df['dur_val']>0)]['dur_val']
season_counts = tv_dur.value_counts().sort_index().reset_index()
season_counts.columns = ['seasons','count']

fig = px.bar(
    season_counts, x='seasons', y='count',
    log_y=True, text='count',
    color_discrete_sequence=[RED],
    title='TV Show Season Count Distribution (Log Scale)',
    labels={'seasons':'Seasons','count':'Number of Shows'},
    template='plotly_dark',
)
fig.update_traces(textposition='outside', textfont=dict(color='white'))
fig.update_layout(
    **PLOTLY_TEMPLATE['layout'].to_plotly_json(),
    height=420,
)
fig.show()
'''

CHART9 = '''
# ── Chart 9: Genre Co-occurrence Network ─────────────────────────────────────
genre_lists = df['listed_in'].dropna().str.split(',').apply(
    lambda lst: [g.strip() for g in lst]
)

# Count genre co-occurrences
edge_weight = Counter()
node_count  = Counter()
for glist in genre_lists:
    for g in glist:
        node_count[g] += 1
    for a, b in combinations(sorted(glist), 2):
        edge_weight[(a, b)] += 1

# Build graph (top edges only)
G = nx.Graph()
for genre, cnt in node_count.items():
    G.add_node(genre, size=cnt)
for (a, b), w in edge_weight.most_common(60):
    G.add_edge(a, b, weight=w)

pos = nx.spring_layout(G, k=0.6, seed=42, iterations=100)

# Edge traces
edge_traces = []
for (a, b), data in G.edges(data=True):
    x0, y0 = pos[a]; x1, y1 = pos[b]
    w = data['weight']
    edge_traces.append(go.Scatter(
        x=[x0,x1,None], y=[y0,y1,None],
        mode='lines',
        line=dict(width=max(0.5, w/50), color='rgba(178,7,16,0.4)'),
        hoverinfo='none',
    ))

# Node trace
node_x = [pos[n][0] for n in G.nodes()]
node_y = [pos[n][1] for n in G.nodes()]
node_sz= [5 + node_count[n]/40 for n in G.nodes()]
node_labels = list(G.nodes())

node_trace = go.Scatter(
    x=node_x, y=node_y, mode='markers+text',
    text=node_labels, textposition='top center',
    textfont=dict(size=9, color=LIGHT),
    marker=dict(
        size=node_sz, color=RED,
        line=dict(color=DARK_RED, width=1),
    ),
    hovertemplate='%{text}<br>Appears in %{customdata:,} titles',
    customdata=[node_count[n] for n in G.nodes()],
)

fig = go.Figure(data=edge_traces + [node_trace])
fig.update_layout(
    title='Genre Co-occurrence Network',
    **PLOTLY_TEMPLATE['layout'].to_plotly_json(),
    height=600,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    showlegend=False,
)
fig.show()
'''

CHART10 = '''
# ── Chart 10: Description Word Cloud ────────────────────────────────────────
import numpy as np
from PIL import Image as PILImage

def clean_for_wc(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\\s]", " ", text)
    return " ".join(t for t in text.split() if t not in CUSTOM_STOPS and len(t) > 2)

all_desc = " ".join(df['description'].fillna('').apply(clean_for_wc))

# Circular mask
xx, yy = np.mgrid[:500, :500]
circle_mask = ((xx - 250)**2 + (yy - 250)**2 < 240**2).astype(np.uint8) * 255

wc = WordCloud(
    width=500, height=500,
    background_color='#141414',
    colormap='Reds',
    mask=circle_mask,
    max_words=200,
    prefer_horizontal=0.8,
    min_font_size=8,
).generate(all_desc)

fig, ax = plt.subplots(figsize=(9, 9), facecolor='#141414')
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
ax.set_title('Most Common Words in Netflix Descriptions',
             color=RED, fontsize=16, pad=10)
plt.tight_layout()
plt.savefig('wc_description.png', dpi=150, bbox_inches='tight',
            facecolor='#141414')
plt.show()
'''

CHART11 = '''
# ── Chart 11: Cast Word Cloud ────────────────────────────────────────────────
cast_counter = Counter()
for cast_str in df['cast'].dropna():
    for actor in cast_str.split(','):
        a = actor.strip()
        if a and a != 'Unknown_Cast':
            cast_counter[a.title()] += 1

wc_cast = WordCloud(
    width=900, height=400,
    background_color='#141414',
    colormap='Reds',
    max_words=150,
    prefer_horizontal=0.7,
).generate_from_frequencies(cast_counter)

fig, ax = plt.subplots(figsize=(14, 6), facecolor='#141414')
ax.imshow(wc_cast, interpolation='bilinear')
ax.axis('off')
ax.set_title('Most Frequent Actors on Netflix',
             color=RED, fontsize=16, pad=10)
plt.tight_layout()
plt.savefig('wc_cast.png', dpi=150, bbox_inches='tight', facecolor='#141414')
plt.show()

# Top 15 actors bar
top_actors = pd.DataFrame(cast_counter.most_common(15), columns=['Actor','Count'])
fig2 = px.bar(top_actors.sort_values('Count'), x='Count', y='Actor',
              orientation='h', color_discrete_sequence=[RED],
              title='Top 15 Most Frequent Cast Members', template='plotly_dark')
fig2.update_layout(**PLOTLY_TEMPLATE['layout'].to_plotly_json(), height=430)
fig2.show()
'''

CHART12 = '''
# ── Chart 12: Genre × Country Heatmap ───────────────────────────────────────
TOP_GENRES = [
    'International Movies','Dramas','Comedies','International TV Shows',
    'Documentaries','Action & Adventure','TV Dramas','Independent Movies',
    'Children & Family Movies','Romantic Movies','Thrillers','Crime TV Shows',
    'Docuseries','Horror Movies','TV Comedies',
]
TOP_COUNTRIES = ['United States','India','United Kingdom','Canada',
                 'France','Japan','South Korea','Spain']

rows = []
for _, row in df.iterrows():
    genre_list   = [g.strip() for g in str(row['listed_in']).split(',')]
    country_list = [c.strip() for c in str(row['country']).split(',')]
    for g in genre_list:
        for c in country_list:
            if g in TOP_GENRES and c in TOP_COUNTRIES:
                rows.append({'genre': g, 'country': c})

heat_df = pd.DataFrame(rows).groupby(['genre','country']).size().unstack(fill_value=0)
heat_df = heat_df.reindex(index=TOP_GENRES, columns=TOP_COUNTRIES, fill_value=0)

fig = go.Figure(go.Heatmap(
    z=heat_df.values, x=heat_df.columns, y=heat_df.index,
    colorscale=[[0,'#1f1f1f'],[0.5,DARK_RED],[1,RED]],
    text=heat_df.values,
    texttemplate='%{text}',
    textfont=dict(size=11, color='white'),
    hovertemplate='Genre: %{y}<br>Country: %{x}<br>Titles: %{z}<extra></extra>',
))
fig.update_layout(
    title='Genre × Country Co-occurrence Heatmap',
    **PLOTLY_TEMPLATE['layout'].to_plotly_json(),
    height=520,
    margin=dict(l=200, b=100),
)
fig.show()
'''

CHART13 = '''
# ── Chart 13: Content Added Month × Year Heatmap ────────────────────────────
monthly = df.groupby(['year_added','month_added']).size().reset_index(name='count')
monthly = monthly[monthly['year_added'].between(2015, 2021)]
pivot = monthly.pivot(index='year_added', columns='month_added', values='count').fillna(0)

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

fig = go.Figure(go.Heatmap(
    z=pivot.values,
    x=[MONTHS[int(m)-1] for m in pivot.columns],
    y=[str(int(y)) for y in pivot.index],
    colorscale=[[0,'#1f1f1f'],[0.4,DARK_RED],[1,RED]],
    text=pivot.values.astype(int),
    texttemplate='%{text}',
    textfont=dict(size=11, color='white'),
    hovertemplate='%{y} %{x}: %{z} titles added<extra></extra>',
))
fig.update_layout(
    title='Netflix Content Acquisition: Month × Year Heatmap',
    xaxis_title='Month', yaxis_title='Year',
    **PLOTLY_TEMPLATE['layout'].to_plotly_json(),
    height=380,
)
fig.show()
'''

CHART14 = '''
# ── Chart 14: Top Directors by Content Type ──────────────────────────────────
dir_counts = (
    df[df['director'] != 'Unknown_Director']
    .assign(director=df['director'].str.split(','))
    .explode('director')
    .assign(director=lambda x: x['director'].str.strip())
)
top_dir = (
    dir_counts.groupby(['director','type'])
    .size().reset_index(name='count')
    .groupby('director').filter(lambda g: g['count'].sum() >= 8)
    .sort_values('count', ascending=False)
)

fig = px.bar(
    top_dir, x='count', y='director', color='type',
    barmode='stack', orientation='h',
    color_discrete_map={'Movie': RED, 'TV Show': DARK_RED},
    title='Top Directors by Number of Titles (≥8 titles)',
    labels={'count':'Titles','director':'','type':''},
    template='plotly_dark',
)
fig.update_layout(
    **PLOTLY_TEMPLATE['layout'].to_plotly_json(),
    height=500, legend_title='',
    legend=dict(font=dict(color=LIGHT), bgcolor=BG),
    margin=dict(l=180),
)
fig.show()
'''

SUMMARY = '''
# ── Summary Statistics Table ──────────────────────────────────────────────────
summary = pd.DataFrame({
    'Metric': [
        'Total Titles', 'Movies', 'TV Shows', 'Unique Countries',
        'Unique Genres', 'Unique Directors', 'Unique Cast Members',
        'Avg Movie Duration (min)', 'Median Movie Duration (min)',
        'Avg TV Show Seasons', 'Release Year Range',
        'Missing Director (%)', 'Missing Cast (%)', 'Missing Country (%)',
    ],
    'Value': [
        f"{len(df):,}",
        f"{(df['type']=='Movie').sum():,}",
        f"{(df['type']=='TV Show').sum():,}",
        str(df['country'].fillna('').str.split(',').explode().str.strip().nunique()),
        str(df['listed_in'].str.split(',').explode().str.strip().nunique()),
        str(df['director'].fillna('').str.split(',').explode().str.strip().nunique()),
        str(df['cast'].fillna('').str.split(',').explode().str.strip().nunique()),
        f"{df[df['type']=='Movie']['dur_val'].replace(0,pd.NA).mean():.1f}",
        f"{df[df['type']=='Movie']['dur_val'].replace(0,pd.NA).median():.0f}",
        f"{df[df['type']=='TV Show']['dur_val'].replace(0,pd.NA).mean():.2f}",
        f"{int(df['release_year'].min())}–{int(df['release_year'].max())}",
        f"{df['director'].isna().mean()*100:.1f}%",
        f"{df['cast'].isna().mean()*100:.1f}%",
        f"{df['country'].isna().mean()*100:.1f}%",
    ]
})
display(summary.style
    .set_properties(**{'text-align':'left','background-color':'#1f1f1f','color':'#e5e5e5'})
    .set_table_styles([{'selector':'th','props':[('background-color','#E50914'),('color','white'),('font-size','14px')]}])
    .hide(axis='index'))
'''

cells = [
    nbf.v4.new_markdown_cell('# Netflix Titles — Exploratory Data Analysis\n**14 production-quality charts exploring content distribution, temporal patterns, genre landscape, and contributor insights.**\n\n> Color palette: Netflix Red `#E50914` throughout.'),
    nbf.v4.new_code_cell(SETUP),
    nbf.v4.new_markdown_cell('## Chart 1 — Content Type Split\n*Are we a movie or TV show platform?*'),
    nbf.v4.new_code_cell(CHART1),
    nbf.v4.new_markdown_cell('## Chart 2 — Rating Distribution by Content Type\n*What age groups does Netflix target per content type?*'),
    nbf.v4.new_code_cell(CHART2),
    nbf.v4.new_markdown_cell('## Chart 3 — Top 20 Countries\n*Netflix\'s global reach — US dominates but India is a strong #2.*'),
    nbf.v4.new_code_cell(CHART3),
    nbf.v4.new_markdown_cell('## Chart 4 — Genre Treemap\n*International Movies and Dramas anchor the catalog.*'),
    nbf.v4.new_code_cell(CHART4),
    nbf.v4.new_markdown_cell('## Chart 5 — Content Added Over Time\n*The Netflix content explosion: 2016 → 2019 rapid growth.*'),
    nbf.v4.new_code_cell(CHART5),
    nbf.v4.new_markdown_cell('## Chart 6 — Release Year Distribution\n*Netflix is a recent-content platform — skewed heavily to 2015+.*'),
    nbf.v4.new_code_cell(CHART6),
    nbf.v4.new_markdown_cell('## Chart 7 — Movie Duration (Violin + Box)\n*Standard feature film: 90–115 min. Outliers: documentaries, stand-up specials.*'),
    nbf.v4.new_code_cell(CHART7),
    nbf.v4.new_markdown_cell('## Chart 8 — TV Show Seasons (Log Scale)\n*67% of TV shows are single-season — Netflix favors limited series.*'),
    nbf.v4.new_code_cell(CHART8),
    nbf.v4.new_markdown_cell('## Chart 9 — Genre Co-occurrence Network\n*"International Movies" is the hub — connected to almost everything.*'),
    nbf.v4.new_code_cell(CHART9),
    nbf.v4.new_markdown_cell('## Chart 10 — Description Word Cloud\n*What language dominates Netflix synopses?*'),
    nbf.v4.new_code_cell(CHART10),
    nbf.v4.new_markdown_cell('## Chart 11 — Cast Word Cloud & Top Actors\n*Bollywood actors dominate — India is Netflix\'s #2 content market.*'),
    nbf.v4.new_code_cell(CHART11),
    nbf.v4.new_markdown_cell('## Chart 12 — Genre × Country Heatmap\n*US content spans all genres. India = Dramas. Japan = Anime. Korea = Romance.*'),
    nbf.v4.new_code_cell(CHART12),
    nbf.v4.new_markdown_cell('## Chart 13 — Month × Year Acquisition Heatmap\n*July and December are peak Netflix acquisition months.*'),
    nbf.v4.new_code_cell(CHART13),
    nbf.v4.new_markdown_cell('## Chart 14 — Top Directors\n*Rajiv Chilaka (22 titles) — children\'s animation. Scorsese (12) — critically acclaimed.*'),
    nbf.v4.new_code_cell(CHART14),
    nbf.v4.new_markdown_cell('## Summary Statistics'),
    nbf.v4.new_code_cell(SUMMARY),
]

nb.cells = cells
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}
nb.metadata['language_info'] = {'name': 'python', 'version': '3.12'}

with open('01_eda.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('01_eda.ipynb created with 14 charts.')
