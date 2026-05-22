"""
Netflix AI Recommender — Ultra UI
WebAI theme · glassmorphism · neural network background · Netflix intro animation
Color palette: #050505 (void) · #E50914 (Netflix) · #00D4FF (AI cyan) · #7B2FBE (AI purple)
"""

import os
import importlib.util
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title='Netflix · AI',
    page_icon='🎬',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if 'intro_played' not in st.session_state:
    st.session_state.intro_played = True
    SHOW_INTRO = True
else:
    SHOW_INTRO = False

if 'results'      not in st.session_state: st.session_state.results      = None
if 'query_title'  not in st.session_state: st.session_state.query_title  = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FONTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900'
    '&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GLOBAL CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
/* ── Variables ─────────────────────────────────────────────────── */
:root {
  --void:       #050505;
  --surface:    rgba(12,12,20,0.85);
  --surface2:   rgba(18,18,30,0.9);
  --red:        #E50914;
  --red-dark:   #B20710;
  --red-glow:   rgba(229,9,20,0.45);
  --cyan:       #00D4FF;
  --cyan-glow:  rgba(0,212,255,0.35);
  --purple:     #7B2FBE;
  --purple-glow:rgba(123,47,190,0.35);
  --orange:     #FF6B35;
  --white:      #FFFFFF;
  --text:       #E8EAF0;
  --muted:      #8892A4;
  --faint:      rgba(255,255,255,0.06);
  --border:     rgba(255,255,255,0.07);
  --border-glow:rgba(0,212,255,0.18);
  --radius:     14px;
  --radius-sm:  8px;
  --font:       'Outfit', 'Inter', sans-serif;
}

/* ── Reset & Base ──────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
  background: var(--void) !important;
  font-family: var(--font) !important;
  color: var(--text) !important;
  overflow-x: hidden;
}

/* ── Scrollbar ─────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: rgba(229,9,20,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--red); }

/* ── Animated Background ───────────────────────────────────────── */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background:
    radial-gradient(ellipse 80% 60% at 15% 85%, rgba(123,47,190,0.11) 0%, transparent 60%),
    radial-gradient(ellipse 60% 50% at 85% 10%, rgba(229,9,20,0.07) 0%, transparent 55%),
    radial-gradient(ellipse 70% 60% at 50% 50%, rgba(0,212,255,0.04) 0%, transparent 70%),
    radial-gradient(ellipse 50% 40% at 80% 75%, rgba(123,47,190,0.07) 0%, transparent 50%),
    radial-gradient(ellipse 40% 35% at 5%  20%, rgba(0,212,255,0.05) 0%, transparent 50%);
  animation: bgBreath 12s ease-in-out infinite alternate;
}
@keyframes bgBreath {
  0%   { opacity: 0.6; transform: scale(1)    rotate(0deg); }
  50%  { opacity: 1;   transform: scale(1.03) rotate(0.5deg); }
  100% { opacity: 0.7; transform: scale(0.98) rotate(-0.3deg); }
}

/* ── AI Grid Overlay ───────────────────────────────────────────── */
.ai-grid {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(0,212,255,0.018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.018) 1px, transparent 1px);
  background-size: 72px 72px;
  animation: gridPulse 6s ease-in-out infinite alternate;
}
@keyframes gridPulse {
  from { opacity: 0.4; }
  to   { opacity: 1; }
}

/* ── Floating Orbs ──────────────────────────────────────────────── */
.orb {
  position: fixed; border-radius: 50%; pointer-events: none; z-index: 0;
  filter: blur(80px); animation: orbFloat linear infinite;
}
.orb-1 { width: 300px; height: 300px; background: rgba(229,9,20,0.08);
          top: 10%; left: 5%; animation-duration: 18s; }
.orb-2 { width: 400px; height: 400px; background: rgba(0,212,255,0.06);
          top: 60%; right: 5%; animation-duration: 22s; animation-delay: -8s; }
.orb-3 { width: 250px; height: 250px; background: rgba(123,47,190,0.09);
          top: 30%; left: 60%; animation-duration: 15s; animation-delay: -4s; }
@keyframes orbFloat {
  0%,100% { transform: translate(0,0) scale(1); }
  25%     { transform: translate(30px,-20px) scale(1.05); }
  50%     { transform: translate(-15px,25px) scale(0.97); }
  75%     { transform: translate(20px,10px) scale(1.03); }
}

/* ── Streamlit Core Overrides ───────────────────────────────────── */
#MainMenu, header, footer { visibility: hidden !important; }
.stDeployButton { display: none !important; }

[data-testid="stAppViewContainer"] {
  background: transparent !important;
  position: relative; z-index: 1;
}
.main .block-container {
  padding: 2rem 2.5rem 4rem !important;
  max-width: 1400px !important;
  position: relative; z-index: 1;
}

/* ── Sidebar ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: rgba(8,8,16,0.92) !important;
  backdrop-filter: blur(24px) !important;
  border-right: 1px solid rgba(0,212,255,0.1) !important;
  box-shadow: 4px 0 40px rgba(0,0,0,0.6) !important;
}
[data-testid="stSidebar"] > div { padding: 2rem 1.5rem !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebarCollapseButton"] svg { fill: var(--cyan) !important; }

/* ── Tabs ──────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 6px;
  background: rgba(255,255,255,0.025);
  border-radius: var(--radius);
  padding: 5px;
  border: 1px solid var(--border);
  backdrop-filter: blur(10px);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px !important;
  padding: 10px 28px !important;
  color: var(--muted) !important;
  font-family: var(--font) !important;
  font-weight: 500 !important;
  font-size: 14px !important;
  transition: all 0.35s cubic-bezier(0.4,0,0.2,1) !important;
  border: 1px solid transparent !important;
  background: transparent !important;
  letter-spacing: 0.02em !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--white) !important;
  background: rgba(229,9,20,0.08) !important;
  border-color: rgba(229,9,20,0.2) !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg,rgba(229,9,20,0.25),rgba(178,7,16,0.15)) !important;
  color: var(--white) !important;
  border-color: rgba(229,9,20,0.5) !important;
  box-shadow: 0 0 20px rgba(229,9,20,0.2), inset 0 1px 0 rgba(255,255,255,0.1) !important;
  font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

/* ── Buttons ───────────────────────────────────────────────────── */
.stButton > button {
  font-family: var(--font) !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  letter-spacing: 0.04em !important;
  border-radius: var(--radius-sm) !important;
  padding: 10px 28px !important;
  transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
  position: relative !important;
  overflow: hidden !important;
  border: none !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #E50914 0%, #B20710 100%) !important;
  color: white !important;
  box-shadow: 0 4px 24px rgba(229,9,20,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(229,9,20,0.55) !important;
}
.stButton > button[kind="primary"]:active {
  transform: translateY(0) !important;
}
.stButton > button[kind="secondary"] {
  background: rgba(255,255,255,0.04) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}
.stButton > button[kind="secondary"]:hover {
  background: rgba(229,9,20,0.08) !important;
  border-color: rgba(229,9,20,0.35) !important;
  transform: translateY(-1px) !important;
}

/* Ripple on buttons */
.stButton > button::after {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at center, rgba(255,255,255,0.12) 0%, transparent 70%);
  opacity: 0;
  transition: opacity 0.3s;
}
.stButton > button:active::after { opacity: 1; }

/* ── Selectbox ─────────────────────────────────────────────────── */
.stSelectbox > label,
.stMultiSelect > label,
.stSlider > label,
.stTextArea > label,
.stTextInput > label,
.stNumberInput > label,
.stRadio > label {
  font-family: var(--font) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  margin-bottom: 6px !important;
}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
  transition: all 0.3s ease !important;
}
[data-baseweb="select"]:focus-within > div,
[data-baseweb="input"]:focus-within > div {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 2px rgba(0,212,255,0.12), 0 0 20px rgba(0,212,255,0.08) !important;
  background: rgba(255,255,255,0.06) !important;
}
[data-baseweb="select"] svg { color: var(--muted) !important; }
[data-baseweb="popover"] { background: #0d0d1a !important; border: 1px solid var(--border-glow) !important; border-radius: var(--radius) !important; }
[data-baseweb="menu-item"] { color: var(--text) !important; font-family: var(--font) !important; }
[data-baseweb="menu-item"]:hover { background: rgba(229,9,20,0.12) !important; }
[data-baseweb="tag"] { background: rgba(229,9,20,0.2) !important; border: 1px solid rgba(229,9,20,0.4) !important; color: var(--white) !important; }

/* ── Text Inputs ────────────────────────────────────────────────── */
textarea, input[type="text"], input[type="number"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
  font-size: 14px !important;
}
textarea:focus, input:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 2px rgba(0,212,255,0.12) !important;
  outline: none !important;
}

/* ── Slider ─────────────────────────────────────────────────────── */
[data-testid="stSlider"] [class*="thumb"] {
  background: var(--red) !important;
  box-shadow: 0 0 10px var(--red-glow) !important;
}
[data-testid="stSlider"] [class*="track-inner"] {
  background: var(--red) !important;
}

/* ── Radio (sidebar mode) ───────────────────────────────────────── */
.stRadio [data-testid="stMarkdownContainer"] p { font-size: 14px !important; }
.stRadio > div { gap: 4px !important; }
.stRadio > div label {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  padding: 8px 12px !important;
  cursor: pointer !important;
  transition: all 0.25s ease !important;
  width: 100%;
}
.stRadio > div label:has(input:checked) {
  background: rgba(229,9,20,0.12) !important;
  border-color: rgba(229,9,20,0.4) !important;
}

/* ── Divider ─────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Alert boxes ─────────────────────────────────────────────────── */
.stAlert {
  background: rgba(229,9,20,0.06) !important;
  border: 1px solid rgba(229,9,20,0.25) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
}

/* ── Spinner ─────────────────────────────────────────────────────── */
[data-testid="stSpinner"] { color: var(--cyan) !important; }

/* ── Custom Components ──────────────────────────────────────────── */

/* Hero header */
.hero-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 2rem;
  padding: 0 0 1.5rem;
  border-bottom: 1px solid var(--border);
}
.hero-brand {
  font-size: 28px;
  font-weight: 900;
  font-style: italic;
  color: var(--red);
  text-shadow: 0 0 30px rgba(229,9,20,0.6);
  letter-spacing: -0.02em;
  line-height: 1;
}
.hero-ai-badge {
  background: linear-gradient(135deg, var(--cyan), var(--purple));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  border: 1px solid rgba(0,212,255,0.25);
  padding: 3px 10px;
  border-radius: 20px;
  background: rgba(0,212,255,0.06);
  color: var(--cyan) !important;
  -webkit-text-fill-color: var(--cyan) !important;
}
.hero-subtitle {
  margin-left: auto;
  font-size: 12px;
  color: var(--muted);
  font-weight: 400;
}
.hero-stat {
  font-size: 12px;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 6px;
}
.hero-stat span {
  color: var(--white);
  font-weight: 600;
}

/* Query meta pill */
.query-meta {
  display: inline-flex;
  align-items: center;
  gap: 12px;
  background: rgba(229,9,20,0.07);
  border: 1px solid rgba(229,9,20,0.2);
  border-radius: 40px;
  padding: 8px 20px;
  margin: 1rem 0;
  font-size: 13px;
  color: var(--text);
}
.query-meta b { color: var(--white); }
.query-meta .sep { color: rgba(255,255,255,0.2); }
.query-tag {
  background: rgba(229,9,20,0.15);
  color: var(--red);
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
}

/* Section header */
.section-label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* Recommendation card */
.rec-card {
  position: relative;
  display: flex;
  gap: 0;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 12px;
  backdrop-filter: blur(16px);
  transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
  cursor: default;
}
.rec-card:hover {
  transform: translateY(-3px) translateX(1px);
  border-color: rgba(229,9,20,0.35);
  box-shadow: 0 8px 40px rgba(0,0,0,0.5), 0 0 30px rgba(229,9,20,0.12);
  background: var(--surface2);
}

/* Left accent strip */
.rec-accent {
  width: 5px;
  flex-shrink: 0;
  transition: width 0.3s ease;
}
.rec-card:hover .rec-accent { width: 6px; }

/* Rank badge */
.rec-rank {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 52px;
  flex-shrink: 0;
  font-size: 22px;
  font-weight: 800;
  color: rgba(255,255,255,0.08);
  font-style: italic;
  user-select: none;
  transition: color 0.3s;
}
.rec-card:hover .rec-rank { color: rgba(229,9,20,0.25); }

/* Card body */
.rec-body {
  flex: 1;
  padding: 14px 16px 14px 4px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}
.rec-title {
  font-size: 15px;
  font-weight: 700;
  color: var(--white);
  line-height: 1.3;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  letter-spacing: 0.01em;
}
.rec-meta {
  font-size: 12px;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.rec-meta-dot { color: rgba(255,255,255,0.15); }
.rating-chip {
  border: 1px solid rgba(255,255,255,0.15);
  padding: 1px 7px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  color: rgba(255,255,255,0.6);
}
.rec-genres {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-top: 2px;
}

/* Genre badge */
.gbadge {
  display: inline-flex;
  align-items: center;
  padding: 2px 9px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 500;
  white-space: nowrap;
  border: 1px solid;
  transition: all 0.2s ease;
}

/* Progress bar row */
.rec-bar-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 4px;
}
.rec-bar-track {
  flex: 1;
  height: 3px;
  background: rgba(255,255,255,0.06);
  border-radius: 2px;
  overflow: hidden;
}
.rec-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 1.2s cubic-bezier(0.4,0,0.2,1);
  background: linear-gradient(90deg, var(--red-dark), var(--red), var(--orange));
  box-shadow: 0 0 8px rgba(229,9,20,0.5);
}

/* Explanation chip */
.why-chip {
  font-size: 11px;
  color: rgba(0,212,255,0.75);
  display: flex;
  align-items: center;
  gap: 5px;
  margin-top: 1px;
}
.why-chip::before {
  content: '◈';
  font-size: 9px;
  color: var(--cyan);
}

/* Score display */
.rec-score-block {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 14px 18px 14px 8px;
  min-width: 70px;
}
.rec-score-num {
  font-size: 24px;
  font-weight: 800;
  line-height: 1;
  letter-spacing: -0.03em;
}
.rec-score-unit {
  font-size: 11px;
  color: var(--muted);
  font-weight: 500;
  margin-top: 2px;
}

/* Cluster badge */
.cluster-badge {
  font-size: 10px;
  color: rgba(123,47,190,0.8);
  border: 1px solid rgba(123,47,190,0.25);
  padding: 1px 6px;
  border-radius: 8px;
  font-weight: 600;
}

/* Stats bar */
.stats-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin: 1.5rem 0;
}
.stat-chip {
  display: flex;
  flex-direction: column;
  align-items: center;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 20px;
  backdrop-filter: blur(12px);
  min-width: 100px;
  flex: 1;
  transition: all 0.3s ease;
}
.stat-chip:hover {
  border-color: var(--border-glow);
  box-shadow: 0 0 20px rgba(0,212,255,0.08);
}
.stat-val {
  font-size: 26px;
  font-weight: 800;
  background: linear-gradient(135deg, var(--white), var(--muted));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1;
}
.stat-val.red { background: linear-gradient(135deg, var(--red), var(--orange)); -webkit-background-clip: text; background-clip: text; }
.stat-val.cyan { background: linear-gradient(135deg, var(--cyan), var(--purple)); -webkit-background-clip: text; background-clip: text; }
.stat-label { font-size: 11px; color: var(--muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }

/* Chart card */
.chart-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  backdrop-filter: blur(12px);
  transition: border-color 0.3s ease;
}
.chart-card:hover { border-color: var(--border-glow); }

/* Cold start form */
.form-glass {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  backdrop-filter: blur(16px);
  margin-bottom: 16px;
}

/* Sidebar label */
.sidebar-section {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: rgba(0,212,255,0.5);
  padding: 16px 0 8px;
  border-bottom: 1px solid rgba(0,212,255,0.08);
  margin-bottom: 12px;
}

/* Empty state */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 40px;
  color: var(--muted);
  text-align: center;
  gap: 12px;
}
.empty-icon { font-size: 48px; opacity: 0.4; }
.empty-title { font-size: 16px; font-weight: 600; color: rgba(255,255,255,0.4); }
.empty-sub { font-size: 13px; color: var(--muted); max-width: 300px; line-height: 1.5; }

/* Cross-type result table override */
.dataframe-container [data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
}

/* Plotly chart container */
[data-testid="stPlotlyChart"] {
  background: transparent !important;
}
[data-testid="stPlotlyChart"] > div {
  border-radius: var(--radius) !important;
  overflow: hidden;
}

/* Fade-in animation */
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
.anim-in { animation: fadeSlideUp 0.5s cubic-bezier(0.4,0,0.2,1) both; }

/* Staggered card animation delay */
.card-delay-0 { animation-delay: 0.05s; }
.card-delay-1 { animation-delay: 0.10s; }
.card-delay-2 { animation-delay: 0.15s; }
.card-delay-3 { animation-delay: 0.20s; }
.card-delay-4 { animation-delay: 0.25s; }
.card-delay-5 { animation-delay: 0.30s; }
.card-delay-6 { animation-delay: 0.35s; }
.card-delay-7 { animation-delay: 0.40s; }
.card-delay-8 { animation-delay: 0.45s; }
.card-delay-9 { animation-delay: 0.50s; }

/* Results count badge */
.results-count {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--muted);
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  padding: 4px 12px;
  border-radius: 20px;
  margin-left: auto;
}
.results-count b { color: var(--white); }

/* App content fade in after intro */
.main-content-wrapper {
  animation: appReveal 0.7s 3.4s cubic-bezier(0.4,0,0.2,1) both;
}
@keyframes appReveal {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
.no-intro-wrapper { opacity: 1; }
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKGROUND ELEMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(
    '<div class="ai-grid"></div>'
    '<div class="orb orb-1"></div>'
    '<div class="orb orb-2"></div>'
    '<div class="orb orb-3"></div>',
    unsafe_allow_html=True
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NETFLIX INTRO (plays once per session)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if SHOW_INTRO:
    st.markdown("""
<style>
/* ── Intro overlay ──────────────────────────────────────── */
#nf-intro {
  position: fixed; inset: 0;
  background: #000;
  z-index: 99999;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0;
  overflow: hidden;
  animation: introExit 0.9s 3.4s cubic-bezier(0.8,0,0.2,1) forwards;
}
@keyframes introExit {
  0%   { opacity:1; transform:scale(1);    clip-path:inset(0%); }
  60%  { opacity:1; transform:scale(1.04); clip-path:inset(0%); }
  100% { opacity:0; transform:scale(1.08); clip-path:inset(0%); pointer-events:none; }
}

/* Red scan lines on intro */
#nf-intro::before {
  content:'';
  position:absolute; inset:0;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 3px,
    rgba(229,9,20,0.03) 3px, rgba(229,9,20,0.03) 4px
  );
  pointer-events:none; z-index:1;
  animation: scanMove 8s linear infinite;
}
@keyframes scanMove {
  from { background-position: 0 0; }
  to   { background-position: 0 100px; }
}

/* Radial burst on intro */
#nf-intro::after {
  content:'';
  position:absolute;
  width:600px; height:600px;
  border-radius:50%;
  background: radial-gradient(circle, rgba(229,9,20,0.15) 0%, transparent 70%);
  animation: burstPulse 1.2s 0.6s ease-out forwards;
  opacity:0;
}
@keyframes burstPulse {
  0%   { transform:scale(0.2); opacity:0; }
  40%  { opacity:1; }
  100% { transform:scale(2.5); opacity:0; }
}

/* The N letterform */
.intro-n-wrap {
  position:relative; z-index:10;
  display:flex; flex-direction:column; align-items:center; gap:0;
}
.intro-n {
  font-family: Georgia, 'Times New Roman', serif;
  font-style: italic;
  font-weight: 900;
  font-size: clamp(120px, 20vw, 220px);
  line-height: 1;
  color: #E50914;
  letter-spacing: -0.03em;
  text-shadow:
    0 0 40px  rgba(229,9,20,0.9),
    0 0 80px  rgba(229,9,20,0.6),
    0 0 160px rgba(229,9,20,0.3),
    0 0 300px rgba(229,9,20,0.1);
  animation: nReveal 1.0s 0.2s cubic-bezier(0.16,1,0.3,1) both;
  transform-origin: center bottom;
}
@keyframes nReveal {
  0%   { opacity:0; transform:scale(0.4) translateY(30px); filter:blur(20px); }
  60%  { opacity:1; transform:scale(1.06) translateY(-4px); filter:blur(0); }
  80%  { transform:scale(0.98) translateY(2px); }
  100% { opacity:1; transform:scale(1) translateY(0); filter:blur(0); }
}

/* Underline glow bar */
.intro-n-bar {
  width: 0;
  height: 3px;
  background: linear-gradient(90deg, transparent, #E50914 20%, #FF6B35 50%, #E50914 80%, transparent);
  border-radius: 2px;
  box-shadow: 0 0 20px rgba(229,9,20,0.8);
  animation: barGrow 0.5s 1.1s ease-out forwards;
  margin-top: -4px;
}
@keyframes barGrow {
  from { width: 0; opacity: 0; }
  to   { width: 80%; opacity: 1; }
}

/* Subtitle */
.intro-sub {
  font-family: 'Outfit', sans-serif;
  font-size: clamp(11px, 1.5vw, 14px);
  font-weight: 600;
  letter-spacing: 0.35em;
  text-transform: uppercase;
  color: rgba(255,255,255,0);
  margin-top: 20px;
  animation: subReveal 0.6s 1.5s ease-out forwards;
}
@keyframes subReveal {
  0%  { color: rgba(255,255,255,0); letter-spacing: 0.6em; }
  100%{ color: rgba(255,255,255,0.45); letter-spacing: 0.35em; }
}

/* Horizontal beam sweep */
.intro-beam {
  position:absolute;
  top:50%; left:-100%;
  width:100%; height:1px;
  background: linear-gradient(90deg,
    transparent 0%,
    rgba(229,9,20,0.6) 30%,
    rgba(255,255,255,0.9) 50%,
    rgba(229,9,20,0.6) 70%,
    transparent 100%
  );
  box-shadow:
    0  2px 12px rgba(229,9,20,0.7),
    0 -2px 12px rgba(229,9,20,0.7),
    0  0   30px rgba(255,107,53,0.5);
  animation: beamSweep 0.55s 1.9s cubic-bezier(0.4,0,0.8,1) forwards;
}
@keyframes beamSweep {
  0%   { left:-100%; opacity:0; }
  5%   { opacity:1; }
  100% { left:100%; opacity:0; }
}

/* Corner brackets - AI aesthetic */
.intro-bracket {
  position:absolute;
  width:40px; height:40px;
  opacity:0;
  animation: bracketFade 0.4s 2.3s ease-out forwards;
}
.intro-bracket::before, .intro-bracket::after {
  content:''; position:absolute; background: rgba(0,212,255,0.6); border-radius:1px;
}
.intro-bracket.tl { top:12%; left:12%; }
.intro-bracket.tr { top:12%; right:12%; transform: scaleX(-1); }
.intro-bracket.bl { bottom:12%; left:12%; transform: scaleY(-1); }
.intro-bracket.br { bottom:12%; right:12%; transform: scale(-1,-1); }
.intro-bracket::before { top:0; left:0; width:100%; height:2px; }
.intro-bracket::after  { top:0; left:0; width:2px; height:100%; }
@keyframes bracketFade {
  from { opacity:0; transform: scale(0.7); }
  to   { opacity:1; transform: scale(1); }
}
.intro-bracket.tr { transform-origin: right top; }
.intro-bracket.bl { transform-origin: left bottom; }
.intro-bracket.br { transform-origin: right bottom; }

/* Loading dots at bottom */
.intro-dots {
  position:absolute; bottom:10%; left:50%; transform:translateX(-50%);
  display:flex; gap:8px; z-index:10;
  animation: dotsReveal 0.4s 2.1s ease-out both;
}
.intro-dot {
  width:5px; height:5px; border-radius:50%;
  background: rgba(229,9,20,0.6);
  animation: dotBlink 0.6s ease-in-out infinite alternate;
}
.intro-dot:nth-child(2) { animation-delay:0.15s; background:rgba(0,212,255,0.5); }
.intro-dot:nth-child(3) { animation-delay:0.30s; background:rgba(123,47,190,0.5); }
.intro-dot:nth-child(4) { animation-delay:0.45s; }
@keyframes dotBlink {
  from { opacity:0.3; transform:scale(0.8); }
  to   { opacity:1;   transform:scale(1.2); box-shadow:0 0 8px currentColor; }
}
@keyframes dotsReveal {
  from { opacity:0; transform:translateX(-50%) translateY(10px); }
  to   { opacity:1; transform:translateX(-50%) translateY(0); }
}
</style>

<div id="nf-intro">
  <div class="intro-beam"></div>
  <div class="intro-bracket tl"></div>
  <div class="intro-bracket tr"></div>
  <div class="intro-bracket bl"></div>
  <div class="intro-bracket br"></div>

  <div class="intro-n-wrap">
    <div class="intro-n">N</div>
    <div class="intro-n-bar"></div>
    <div class="intro-sub">AI Recommender Engine</div>
  </div>

  <div class="intro-dots">
    <div class="intro-dot"></div>
    <div class="intro-dot"></div>
    <div class="intro-dot"></div>
    <div class="intro-dot"></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD RECOMMENDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource(show_spinner=False)
def _load_recommender():
    here = os.path.dirname(os.path.abspath(__file__))
    def _load_mod(fname):
        spec = importlib.util.spec_from_file_location(
            fname.replace('.py',''), os.path.join(here, fname))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    cache_path = os.path.join(here, 'recommender_cache.joblib')
    mod = _load_mod('04_recommender.py')
    NR = mod.NetflixRecommender
    if os.path.exists(cache_path):
        return NR.load(cache_path)
    rec = NR(data_path=os.path.join(here, 'netflix_titles.csv'))
    rec.fit()
    rec.save(cache_path)
    return rec

with st.spinner(''):
    rec = _load_recommender()

df_cat    = rec.catalog
all_titles = sorted(df_cat['title'].tolist())
all_ratings = sorted(df_cat['rating'].dropna().unique().tolist())
min_yr    = int(df_cat['release_year'].min())
max_yr    = int(df_cat['release_year'].max())
n_movies  = int((df_cat['type']=='Movie').sum())
n_shows   = int((df_cat['type']=='TV Show').sum())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENRE_PALETTE = {
    'Dramas':                  ('#E50914','#B20710'),
    'Crime TV Shows':          ('#C0392B','#962d22'),
    'Thrillers':               ('#E74C3C','#b83c2f'),
    'Action & Adventure':      ('#E67E22','#b86319'),
    'Comedies':                ('#F39C12','#c27d0e'),
    'Romantic Movies':         ('#FF6B9D','#cc5680'),
    'Horror Movies':           ('#8E44AD','#712d8a'),
    'Sci-Fi & Fantasy':        ('#2980B9','#2066a0'),
    'Documentaries':           ('#27AE60','#1e8a4d'),
    'International Movies':    ('#00D4FF','#00a8cc'),
    'International TV Shows':  ('#16A085','#117a65'),
    'Children & Family Movies':('#3498DB','#2980b9'),
    'Anime Series':            ('#9B59B6','#7d4890'),
    'Stand-Up Comedy':         ('#F1C40F','#c19b0c'),
    'TV Dramas':               ('#E55A14','#b84710'),
    'TV Comedies':             ('#F3751F','#c25e19'),
    'Reality TV':              ('#1ABC9C','#17967d'),
    'Music & Musicals':        ('#FF6B35','#cc562a'),
    'Sports Movies':           ('#2ECC71','#25a35a'),
}

def genre_color(genre_str: str):
    if not genre_str:
        return ('#E50914', '#B20710')
    first = genre_str.split(',')[0].strip()
    return GENRE_PALETTE.get(first, ('#E50914', '#B20710'))

def score_color(pct: float) -> str:
    if pct >= 85: return 'var(--cyan)'
    if pct >= 70: return 'var(--red)'
    return 'var(--muted)'

PLOTLY_DARK = dict(
    template='plotly_dark',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Outfit, Inter, sans-serif', color='#8892A4', size=12),
    margin=dict(l=0, r=0, t=28, b=0),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', linecolor='rgba(255,255,255,0.06)',
               tickcolor='rgba(255,255,255,0.2)', titlefont=dict(color='#8892A4')),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', linecolor='rgba(255,255,255,0.06)',
               tickcolor='rgba(255,255,255,0.2)', titlefont=dict(color='#8892A4')),
    hoverlabel=dict(bgcolor='#0d0d1a', bordercolor='rgba(0,212,255,0.3)',
                    font=dict(family='Outfit', color='#e8eaf0')),
)

def render_rec_card(row, rank: int, query_title: str, delay: int, show_explain: bool = True):
    g1, g2 = genre_color(row['genres'])
    score_pct = int(row['similarity'] * 100)
    score_col = score_color(score_pct)
    genres = [g.strip() for g in row['genres'].split(',')[:3]]

    genre_badges_html = ''
    for g in genres:
        c1, _ = GENRE_PALETTE.get(g, ('#E50914','#B20710'))
        genre_badges_html += (
            f'<span class="gbadge" style="color:{c1};border-color:{c1}33;'
            f'background:{c1}12">{g}</span>'
        )

    # Why explanation
    why_html = ''
    if show_explain:
        try:
            ex = rec.explain_recommendation(query_title, row['title'])
            parts = []
            sg = ex.get('shared_genres', [])
            if sg:   parts.append(f"Genres: {', '.join(sg[:2])}")
            if ex.get('same_director'): parts.append(f"Director: {ex['director_name']}")
            if ex.get('era_match'):     parts.append('Similar era')
            why_txt = ' · '.join(parts) if parts else 'Similar content profile'
            why_html = f'<div class="why-chip">{why_txt}</div>'
        except Exception:
            pass

    cluster_html = ''
    if 'cluster_id' in row:
        cluster_html = f'<span class="cluster-badge">C{int(row["cluster_id"])}</span>'

    type_icon = '🎬' if row['type'] == 'Movie' else '📺'
    rank_display = rank + 1

    st.markdown(f"""
<div class="rec-card anim-in card-delay-{min(delay,9)}">
  <div class="rec-accent" style="background:linear-gradient(180deg,{g1},{g2})"></div>
  <div class="rec-rank">{rank_display}</div>
  <div class="rec-body">
    <div class="rec-title">{row['title']}</div>
    <div class="rec-meta">
      <span>{type_icon} {row['type']}</span>
      <span class="rec-meta-dot">·</span>
      <span>{row['release_year']}</span>
      <span class="rec-meta-dot">·</span>
      <span class="rating-chip">{row['rating']}</span>
      {cluster_html}
    </div>
    <div class="rec-genres">{genre_badges_html}</div>
    <div class="rec-bar-row">
      <div class="rec-bar-track">
        <div class="rec-bar-fill" style="width:{score_pct}%"></div>
      </div>
    </div>
    {why_html}
  </div>
  <div class="rec-score-block">
    <div class="rec-score-num" style="color:{score_col}">{score_pct}</div>
    <div class="rec-score-unit">match</div>
  </div>
</div>""", unsafe_allow_html=True)


def render_charts(results: pd.DataFrame, query_title: str):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        # Genre distribution
        all_g = results['genres'].str.split(', ').explode()
        gcnt  = all_g.value_counts().head(8).reset_index()
        gcnt.columns = ['Genre','Count']
        colors = [GENRE_PALETTE.get(g, ('#E50914','#B20710'))[0] for g in gcnt['Genre']]

        fig = go.Figure(go.Bar(
            x=gcnt['Count'], y=gcnt['Genre'], orientation='h',
            marker_color=colors,
            text=gcnt['Count'], textposition='outside',
            textfont=dict(color='rgba(255,255,255,0.5)', size=11),
            hovertemplate='<b>%{y}</b><br>%{x} results<extra></extra>',
        ))
        fig.update_layout(height=260, title_text='Genre Mix',
                          title_font=dict(color='rgba(255,255,255,0.5)', size=12),
                          yaxis=dict(categoryorder='total ascending'),
                          **{k:v for k,v in PLOTLY_DARK.items()
                             if k not in ('template','margin')},
                          margin=dict(l=0,r=30,t=36,b=0))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        # Similarity scores horizontal
        top = results.sort_values('similarity', ascending=True).tail(10)
        norm_scores = top['similarity']
        bar_colors = [f'rgba(229,9,20,{0.4 + 0.6*s})' for s in
                      (norm_scores - norm_scores.min()) /
                      (norm_scores.max() - norm_scores.min() + 1e-9)]

        fig2 = go.Figure(go.Bar(
            x=norm_scores, y=top['title'].str[:22], orientation='h',
            marker=dict(color=bar_colors,
                        line=dict(color='rgba(229,9,20,0.3)', width=0.5)),
            text=[f'{int(s*100)}%' for s in norm_scores],
            textposition='outside',
            textfont=dict(color='rgba(255,255,255,0.45)', size=11),
            hovertemplate='<b>%{y}</b><br>Match: %{x:.1%}<extra></extra>',
        ))
        fig2.update_layout(height=260, title_text='Match Scores',
                           title_font=dict(color='rgba(255,255,255,0.5)', size=12),
                           xaxis=dict(range=[0,1], tickformat='.0%',
                                      gridcolor='rgba(255,255,255,0.04)'),
                           **{k:v for k,v in PLOTLY_DARK.items()
                              if k not in ('template','margin','xaxis')},
                           margin=dict(l=0,r=50,t=36,b=0))
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Year distribution of results
    st.markdown('<div class="chart-card" style="margin-top:12px">', unsafe_allow_html=True)
    yr_cnt = results['release_year'].value_counts().sort_index().reset_index()
    yr_cnt.columns = ['Year','Count']

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=yr_cnt['Year'], y=yr_cnt['Count'],
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#E50914', width=2.5),
        fillcolor='rgba(229,9,20,0.08)',
        marker=dict(color='#E50914', size=6,
                    line=dict(color='#FF6B35', width=1.5)),
        hovertemplate='<b>%{x}</b><br>%{y} results<extra></extra>',
    ))
    fig3.update_layout(height=130,
                       title_text='Release Year of Recommendations',
                       title_font=dict(color='rgba(255,255,255,0.5)', size=12),
                       **{k:v for k,v in PLOTLY_DARK.items()
                          if k not in ('template','margin')},
                       margin=dict(l=0,r=0,t=36,b=0))
    st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar':False})
    st.markdown('</div>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:1.5rem">
  <span style="font-size:24px;font-weight:900;font-style:italic;color:#E50914;
               text-shadow:0 0 20px rgba(229,9,20,0.7);line-height:1">N</span>
  <div>
    <div style="font-size:13px;font-weight:700;color:#fff;letter-spacing:0.05em">
      RECOMMENDER</div>
    <div style="font-size:10px;color:rgba(0,212,255,0.7);letter-spacing:0.15em;
                text-transform:uppercase">AI ENGINE</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Filters</div>', unsafe_allow_html=True)

    content_type_raw = st.selectbox('Content Type', ['Any', 'Movie', 'TV Show'])
    content_type = None if content_type_raw == 'Any' else content_type_raw

    year_range = st.slider('Release Year', min_value=min_yr, max_value=max_yr,
                           value=(2000, max_yr))

    rating_filter = st.multiselect('Age Rating', options=all_ratings, default=[])

    n_results = st.slider('Results Count', min_value=5, max_value=20, value=10)

    st.markdown('<div class="sidebar-section">Catalog</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:1rem">
  <div style="background:rgba(229,9,20,0.06);border:1px solid rgba(229,9,20,0.15);
              border-radius:10px;padding:10px;text-align:center">
    <div style="font-size:18px;font-weight:800;color:#E50914">{n_movies:,}</div>
    <div style="font-size:10px;color:#8892A4;text-transform:uppercase;letter-spacing:0.1em">Movies</div>
  </div>
  <div style="background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.12);
              border-radius:10px;padding:10px;text-align:center">
    <div style="font-size:18px;font-weight:800;color:#00D4FF">{n_shows:,}</div>
    <div style="font-size:10px;color:#8892A4;text-transform:uppercase;letter-spacing:0.1em">TV Shows</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-size:11px;color:#8892A4;line-height:1.8">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
    <span style="color:rgba(255,255,255,0.4)">Architecture</span>
    <span style="color:#00D4FF">Two-Stage Hybrid</span>
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
    <span style="color:rgba(255,255,255,0.4)">Stage 1</span>
    <span>Weighted Cosine</span>
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
    <span style="color:rgba(255,255,255,0.4)">Stage 2</span>
    <span>NMF Re-rank</span>
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
    <span style="color:rgba(255,255,255,0.4)">Precision@10</span>
    <span style="color:#E50914;font-weight:700">99.96%</span>
  </div>
  <div style="display:flex;justify-content:space-between">
    <span style="color:rgba(255,255,255,0.4)">Latency</span>
    <span style="color:#27AE60">~10ms</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN CONTENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
wrapper_cls = 'main-content-wrapper' if SHOW_INTRO else 'no-intro-wrapper'
st.markdown(f'<div class="{wrapper_cls}">', unsafe_allow_html=True)

# ── Hero Header ──────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-header">
  <div style="display:flex;align-items:center;gap:12px">
    <div class="hero-brand">N</div>
    <div>
      <div style="font-size:16px;font-weight:700;color:#fff;letter-spacing:0.04em">
        RECOMMENDER</div>
      <div class="hero-ai-badge">AI ENGINE</div>
    </div>
  </div>
  <div style="margin-left:auto;display:flex;align-items:center;gap:24px">
    <div class="hero-stat">
      <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
                   background:#E50914;box-shadow:0 0 8px #E50914"></span>
      <span><b>{len(all_titles):,}</b> titles indexed</span>
    </div>
    <div class="hero-stat">
      <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
                   background:#00D4FF;box-shadow:0 0 8px #00D4FF"></span>
      <span>Precision <b>99.96%</b></span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Navigation Tabs ───────────────────────────────────────────────
tab_sim, tab_cross, tab_cold = st.tabs([
    '⬡  Similar Titles',
    '⇄  Cross-Type',
    '◈  Cold Start',
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: SIMILAR TITLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_sim:
    s_col1, s_col2 = st.columns([3, 1])
    with s_col1:
        query_title = st.selectbox(
            'Select a title',
            all_titles,
            label_visibility='collapsed',
            key='sim_query',
            placeholder='Search titles...',
        )
    with s_col2:
        run_btn = st.button('Find Similar', type='primary',
                            use_container_width=True, key='sim_run')

    if run_btn:
        with st.spinner(''):
            try:
                results = rec.recommend(
                    title=query_title, n=n_results,
                    content_type_filter=content_type,
                    min_release_year=year_range[0],
                    max_release_year=year_range[1],
                    rating_filter=rating_filter if rating_filter else None,
                )
                st.session_state.results     = results
                st.session_state.query_title = query_title
            except ValueError as e:
                st.error(str(e))

    if st.session_state.results is not None and st.session_state.query_title is not None:
        results     = st.session_state.results
        query_title = st.session_state.query_title

        if results.empty:
            st.markdown("""
<div class="empty-state">
  <div class="empty-icon">⬡</div>
  <div class="empty-title">No matches found</div>
  <div class="empty-sub">Try relaxing the filters in the sidebar</div>
</div>""", unsafe_allow_html=True)
        else:
            # Query pill
            qr = df_cat[df_cat['title'] == query_title]
            if not qr.empty:
                r = qr.iloc[0]
                g1, _ = genre_color(r.get('listed_in',''))
                st.markdown(f"""
<div class="query-meta">
  <b>{query_title}</b>
  <span class="sep">·</span>
  <span>{r['type']}</span>
  <span class="sep">·</span>
  <span>{int(r['release_year'])}</span>
  <span class="sep">·</span>
  <span class="rating-chip">{r['rating']}</span>
  <span class="results-count"><b>{len(results)}</b>&nbsp;matches</span>
</div>""", unsafe_allow_html=True)

            left_col, right_col = st.columns([3, 2])

            with left_col:
                st.markdown(
                    '<div class="section-label">Recommendations</div>',
                    unsafe_allow_html=True
                )
                for i, (_, row) in enumerate(results.iterrows()):
                    render_rec_card(row, i, query_title, i, show_explain=True)

            with right_col:
                st.markdown(
                    '<div class="section-label">Analytics</div>',
                    unsafe_allow_html=True
                )
                render_charts(results, query_title)

    else:
        st.markdown("""
<div class="empty-state">
  <div class="empty-icon">🎬</div>
  <div class="empty-title">Find your next watch</div>
  <div class="empty-sub">Select any title above and click Find Similar to discover recommendations powered by a two-stage ML model</div>
</div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: CROSS-TYPE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_cross:
    st.markdown("""
<div style="font-size:13px;color:#8892A4;margin-bottom:1rem;line-height:1.6">
  Loved a <b style="color:#E50914">Movie</b>? Find the best
  <b style="color:#00D4FF">TV Show</b> equivalent — and vice versa.
  The model searches across content types using semantic feature matching.
</div>""", unsafe_allow_html=True)

    cx_col1, cx_col2 = st.columns([3, 1])
    with cx_col1:
        cx_query = st.selectbox('Select a title', all_titles,
                                label_visibility='collapsed', key='cx_query')
    with cx_col2:
        cx_btn = st.button('Convert', type='primary',
                           use_container_width=True, key='cx_run')

    # Show what type it is
    qrow = df_cat[df_cat['title'] == cx_query]
    if not qrow.empty:
        qt = qrow.iloc[0]['type']
        tgt = 'TV Show' if qt == 'Movie' else 'Movie'
        icon_src = '🎬' if qt == 'Movie' else '📺'
        icon_tgt = '📺' if qt == 'Movie' else '🎬'
        st.markdown(f"""
<div style="display:inline-flex;align-items:center;gap:12px;
            background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
            border-radius:40px;padding:8px 20px;margin:0.5rem 0;font-size:13px">
  <span>{icon_src} <b style="color:#fff">{qt}</b></span>
  <span style="color:rgba(0,212,255,0.5);font-size:18px">→</span>
  <span>{icon_tgt} <b style="color:#00D4FF">{tgt}</b></span>
</div>""", unsafe_allow_html=True)

    if cx_btn:
        with st.spinner(''):
            try:
                cx_results = rec.recommend_cross_type(cx_query, n=n_results)
                st.session_state['cx_results']  = cx_results
                st.session_state['cx_query']     = cx_query
            except ValueError as e:
                st.error(str(e))

    if 'cx_results' in st.session_state and st.session_state['cx_results'] is not None:
        cx_results = st.session_state['cx_results']
        if cx_results.empty:
            st.markdown('<div class="empty-state"><div class="empty-icon">⇄</div>'
                        '<div class="empty-title">No cross-type results</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-label">Cross-Type Matches</div>',
                        unsafe_allow_html=True)
            for i, (_, row) in enumerate(cx_results.iterrows()):
                render_rec_card(row, i, st.session_state['cx_query'], i, show_explain=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: COLD START
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_cold:
    st.markdown("""
<div style="font-size:13px;color:#8892A4;margin-bottom:1.5rem;line-height:1.6">
  Describe any title — even one that doesn't exist yet. The AI vectorises your
  description and finds the closest matches in the catalog.
</div>""", unsafe_allow_html=True)

    g1_cs, g2_cs = st.columns(2)

    with g1_cs:
        st.markdown('<div class="form-glass">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Content Description</div>', unsafe_allow_html=True)
        cs_desc   = st.text_area('Plot description', height=110,
                                  placeholder='A chemistry teacher diagnosed with terminal cancer begins producing methamphetamine to secure his family\'s future...', key='cs_desc')
        cs_genres = st.multiselect('Genres', key='cs_genres',
                                    options=sorted(df_cat['listed_in'].str.split(', ').explode().unique()))
        cs_type   = st.selectbox('Content Type', ['Movie', 'TV Show'], key='cs_type')
        cs_rating = st.selectbox('Rating', all_ratings, key='cs_rating',
                                  index=all_ratings.index('TV-MA') if 'TV-MA' in all_ratings else 0)
        st.markdown('</div>', unsafe_allow_html=True)

    with g2_cs:
        st.markdown('<div class="form-glass">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Contributors</div>', unsafe_allow_html=True)
        cs_cast     = st.text_input('Cast (comma-separated)', key='cs_cast',
                                     placeholder='Bryan Cranston, Aaron Paul')
        cs_director = st.text_input('Director', key='cs_director',
                                     placeholder='Vince Gilligan')
        cs_country  = st.text_input('Country', key='cs_country',
                                     placeholder='United States')
        cs_year     = st.number_input('Release Year', min_value=1950,
                                       max_value=2025, value=2020, key='cs_year')
        st.markdown('</div>', unsafe_allow_html=True)

    cs_btn = st.button('Analyze & Find Matches', type='primary',
                        use_container_width=True, key='cs_run')

    if cs_btn:
        if not cs_desc or not cs_genres:
            st.warning('Please add a description and at least one genre.')
        else:
            with st.spinner(''):
                cs_results = rec.recommend_for_cold_start(
                    description=cs_desc, genres=cs_genres,
                    content_type=cs_type, cast=cs_cast,
                    director=cs_director, country=cs_country,
                    rating=cs_rating, release_year=cs_year,
                    n=n_results,
                )
                st.session_state['cs_results'] = cs_results

    if 'cs_results' in st.session_state and st.session_state['cs_results'] is not None:
        cs_results = st.session_state['cs_results']
        st.markdown('<div class="section-label">Catalog Matches</div>', unsafe_allow_html=True)
        for i, (_, row) in enumerate(cs_results.iterrows()):
            render_rec_card(row, i, '', i, show_explain=False)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:3rem 0 1rem;
            font-size:11px;color:rgba(255,255,255,0.12);letter-spacing:0.1em">
  NETFLIX AI RECOMMENDER &nbsp;·&nbsp; TWO-STAGE HYBRID MODEL &nbsp;·&nbsp;
  TF-IDF + NMF + COSINE SIMILARITY &nbsp;·&nbsp; 8,807 TITLES
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close wrapper
