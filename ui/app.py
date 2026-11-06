import os
import sys
import json
import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
import streamlit as st

load_dotenv(override=True)

# ── Constants ─────────────────────────────────────────────────────────────────
HISTORY_FILE = PROJECT_ROOT / "data" / "history.json"
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
AUTH_FILE = PROJECT_ROOT / "data" / "auth_profile.json"

# ── Cached Resources (loaded ONCE per process) ────────────────────────────────
@st.cache_resource(show_spinner="Connecting to vector store...")
def get_qdrant_client():
    from vector_store.qdrant_store import get_shared_local_client
    return get_shared_local_client("data/qdrant_db")

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    from embeddings.embedder import CodeEmbedder
    return CodeEmbedder()

from ingestion.indexer import RepositoryIndexer, sanitize_collection_name
from retrieval.rag_pipeline import RAGPipeline
from vector_store.qdrant_store import QdrantStore

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CodeLens AI - Codebase Intelligence",
    page_icon="CL",
    layout="wide",
    initial_sidebar_state="expanded"
)

_client = get_qdrant_client()
_embedder = get_embedder()

# ══════════════════════════════════════════════════════════════════════════════
# CRISP MONOCHROME WHITE & NICHE BENTO GRID DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --bg-base: #f8fafc;
  --bg-surface: #ffffff;
  --border-subtle: #e2e8f0;
  --border-medium: #cbd5e1;
  --border-dark: #0f172a;
  --text-main: #0f172a;
  --text-secondary: #475569;
  --text-muted: #64748b;
  --accent-cyan: #0284c7;
  --accent-emerald: #16a34a;
  --accent-amber: #d97706;
  --accent-rose: #dc2626;
}

/* Global resets & typography */
*, html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
  box-sizing: border-box;
}

code, pre, .mono {
  font-family: 'JetBrains Mono', monospace !important;
}

/* Main background with subtle dot grid on crisp off-white canvas */
.stApp {
  background-color: var(--bg-base) !important;
  background-image: radial-gradient(#cbd5e1 0.85px, transparent 0.85px);
  background-size: 20px 20px;
  background-position: 10px 10px;
  color: var(--text-main) !important;
}

#MainMenu, footer, header {
  visibility: hidden;
}

.block-container {
  padding: 1.6rem 2.8rem 4rem !important;
  max-width: 1560px !important;
}

/* ── Clean White Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background-color: #ffffff !important;
  border-right: 1px solid #e2e8f0 !important;
  box-shadow: 1px 0 10px rgba(0, 0, 0, 0.02) !important;
}

[data-testid="stSidebar"] > div:first-child {
  padding-top: 0.6rem;
}

[data-testid="stSidebar"] .block-container {
  padding: 0 0.8rem 2rem !important;
}

/* Sidebar navigation buttons */
[data-testid="stSidebar"] .stButton > button {
  background: transparent !important;
  color: #475569 !important;
  border: 1px solid transparent !important;
  border-radius: 8px !important;
  padding: 0.55rem 0.85rem !important;
  font-weight: 600 !important;
  font-size: 0.86rem !important;
  text-align: left !important;
  display: flex !important;
  align-items: center !important;
  gap: 0.5rem !important;
  transition: all 0.16s ease !important;
  margin-bottom: 0.25rem !important;
  box-shadow: none !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
  background: #f1f5f9 !important;
  color: #0f172a !important;
  border-color: #e2e8f0 !important;
  transform: translateX(2px) !important;
}

/* Active navigation button highlight */
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button:focus:not(:hover) {
  background: #f1f5f9 !important;
  color: #0f172a !important;
  border-left: 3px solid #0f172a !important;
  border-color: #e2e8f0 !important;
  font-weight: 700 !important;
}
/* Override global white-text rule for sidebar active nav buttons */
[data-testid="stSidebar"] .stButton > button[kind="primary"] *,
[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
[data-testid="stSidebar"] .stButton > button[kind="primary"] span {
  color: #0f172a !important;
}

/* ── Primary Action Buttons & Form Submit Buttons ───────────────────────────── */
.stButton > button,
[data-testid="stFormSubmitButton"] > button,
button[kind="primaryFormSubmit"],
button[data-testid="stBaseButton-primaryFormSubmit"],
button[data-testid="baseButton-primary"],
button[kind="primary"] {
  background: #0f172a !important;
  color: #ffffff !important;
  border: 1px solid #0f172a !important;
  border-radius: 8px !important;
  padding: 0.6rem 1.25rem !important;
  font-weight: 700 !important;
  font-size: 0.88rem !important;
  letter-spacing: -0.01em !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}

/* Force ALL text inside primary/submit buttons to be PURE WHITE */
.stButton > button *,
[data-testid="stFormSubmitButton"] > button *,
button[kind="primaryFormSubmit"] *,
button[data-testid="stBaseButton-primaryFormSubmit"] *,
button[data-testid="baseButton-primary"] *,
button[kind="primary"] * {
  color: #ffffff !important;
  font-weight: 700 !important;
}

/* Hover: lift + shadow animation only — colours stay the same */
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover,
button[kind="primaryFormSubmit"]:hover,
button[data-testid="stBaseButton-primaryFormSubmit"]:hover,
button[data-testid="baseButton-primary"]:hover,
button[kind="primary"]:hover {
  background: #0f172a !important;
  color: #ffffff !important;
  border-color: #0f172a !important;
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.26) !important;
  transform: translateY(-2px) scale(1.015) !important;
}
.stButton > button:hover *,
[data-testid="stFormSubmitButton"] > button:hover *,
button[kind="primaryFormSubmit"]:hover *,
button[data-testid="stBaseButton-primaryFormSubmit"]:hover *,
button[data-testid="baseButton-primary"]:hover *,
button[kind="primary"]:hover * {
  color: #ffffff !important;
}

.stButton > button:active,
[data-testid="stFormSubmitButton"] > button:active {
  transform: translateY(0px) scale(1) !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
}

/* Secondary button variation */
button[kind="secondary"],
button[data-testid="baseButton-secondary"],
button[data-testid="stBaseButton-secondary"] {
  background: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
  color: #0f172a !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}
button[kind="secondary"] *,
button[data-testid="baseButton-secondary"] *,
button[data-testid="stBaseButton-secondary"] * {
  color: #0f172a !important;
}
/* Hover: lift animation only — white bg & dark text stay */
button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover {
  background: #ffffff !important;
  border-color: #94a3b8 !important;
  color: #0f172a !important;
  box-shadow: 0 7px 18px rgba(0, 0, 0, 0.11) !important;
  transform: translateY(-2px) scale(1.015) !important;
}
button[kind="secondary"]:hover *,
button[data-testid="baseButton-secondary"]:hover *,
button[data-testid="stBaseButton-secondary"]:hover * {
  color: #0f172a !important;
}

/* ── Modern Form Controls & Crisp White Inputs ────────────────────────────── */
.stTextInput input, .stTextArea textarea, .stNumberInput input {
  background: #ffffff !important;
  color: #0f172a !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 8px !important;
  padding: 0.62rem 0.95rem !important;
  font-size: 0.9rem !important;
  transition: all 0.15s ease !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) inset !important;
}

.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
  border-color: #0f172a !important;
  box-shadow: 0 0 0 3px rgba(15, 23, 42, 0.08) !important;
  background: #ffffff !important;
}

.stTextInput input::placeholder, .stTextArea textarea::placeholder {
  color: #94a3b8 !important;
}

/* ── Selectbox & Model Chooser - Full White Theme with High Contrast ──────── */
div[data-testid="stSelectbox"],
div[data-testid="stSelectbox"] > div,
div[data-testid="stSelectbox"] > div > div,
div[data-baseweb="select"],
div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div > div {
  background-color: #ffffff !important;
  background: #ffffff !important;
  color: #0f172a !important;
  border-color: #cbd5e1 !important;
}

div[data-baseweb="select"] > div {
  border: 1px solid #cbd5e1 !important;
  border-radius: 8px !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
}

div[data-baseweb="select"] * {
  color: #0f172a !important;
  fill: #0f172a !important;
}

/* The selected value label inside selectbox */
div[data-baseweb="select"] div[aria-selected],
div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
  color: #0f172a !important;
  background: transparent !important;
  font-weight: 600 !important;
}

/* Dropdown menu list & popover */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"],
ul[role="listbox"] {
  background-color: #ffffff !important;
  background: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 8px !important;
  box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1) !important;
}

ul[role="listbox"] li,
li[role="option"] {
  background-color: #ffffff !important;
  color: #0f172a !important;
}

ul[role="listbox"] li:hover,
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
  background-color: #f1f5f9 !important;
  color: #0f172a !important;
}

label, [data-testid="stWidgetLabel"] p {
  color: #334155 !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
  margin-bottom: 0.35rem !important;
}

/* ── Bento Grid Core Architecture (White Theme) ───────────────────────────── */
.bento-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 1.25rem 1.45rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02);
  transition: all 0.2s ease;
  position: relative;
  overflow: hidden;
}

.bento-card:hover {
  border-color: #cbd5e1;
  box-shadow: 0 6px 20px -2px rgba(0, 0, 0, 0.06), 0 2px 6px -1px rgba(0, 0, 0, 0.03);
  transform: translateY(-1.5px);
}

.bento-hero {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 2.2rem 2.6rem;
  position: relative;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
  margin-bottom: 1.6rem;
}

.bento-hero-title {
  font-size: clamp(2rem, 3.8vw, 2.8rem);
  font-weight: 800;
  color: #0f172a !important;
  letter-spacing: -0.04em;
  line-height: 1.15;
  margin: 0 0 0.6rem;
}

.bento-hero-title span {
  color: #64748b;
}

.bento-hero-sub {
  color: #64748b !important;
  font-size: 0.98rem;
  max-width: 650px;
  line-height: 1.6;
  margin: 0;
}

/* Niche Bento Metric Tiles */
.stat-tile {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 1.2rem 1.35rem;
  position: relative;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
  transition: all 0.2s ease;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.stat-tile:hover {
  border-color: #cbd5e1;
  transform: translateY(-1.5px);
}

.stat-tile.hero-metric {
  border: 1px solid #cbd5e1;
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
}

.stat-tile-num {
  font-size: 2.25rem;
  font-weight: 800;
  color: #0f172a !important;
  letter-spacing: -0.05em;
  line-height: 1;
  font-family: 'Plus Jakarta Sans', sans-serif;
}

.stat-tile-num.cyan {
  color: #0284c7 !important;
}

.stat-tile-lbl {
  font-size: 0.72rem;
  font-weight: 700;
  color: #64748b !important;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-top: 0.35rem;
}

.stat-tag {
  display: inline-flex;
  align-items: center;
  font-size: 0.68rem;
  font-weight: 600;
  padding: 0.18rem 0.55rem;
  border-radius: 6px;
  letter-spacing: 0.04em;
  font-family: 'JetBrains Mono', monospace;
  margin-top: 0.4rem;
}

.tag-cyan { background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }
.tag-indigo { background: #f1f5f9; color: #0f172a; border: 1px solid #cbd5e1; }
.tag-emerald { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
.tag-amber { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }

/* ── Section Titles & Badges ──────────────────────────────────────────────── */
.sec-title {
  font-size: 0.76rem;
  font-weight: 700;
  color: #475569 !important;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.9rem;
}

.sec-title::after {
  content: "";
  flex: 1;
  height: 1px;
  background: #e2e8f0;
}

.chip-pill {
  display: inline-flex;
  align-items: center;
  background: #f1f5f9;
  color: #334155 !important;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 0.2rem 0.55rem;
  font-size: 0.72rem;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
}

/* ── Conversation & Reasoning Studio Styles ────────────────────────────────── */
.chat-q-box {
  background: #0f172a;
  color: #ffffff !important;
  border: 1px solid #1e293b;
  border-radius: 12px 12px 2px 12px;
  padding: 1.1rem 1.35rem;
  margin: 0.8rem 0 0.8rem 8%;
  font-size: 0.93rem;
  line-height: 1.6;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
}

.chat-a-box {
  background: #ffffff;
  color: #0f172a !important;
  border: 1px solid #e2e8f0;
  border-left: 3px solid #0f172a;
  border-radius: 2px 12px 12px 12px;
  padding: 1.25rem 1.45rem;
  margin: 0.6rem 8% 1.2rem 0;
  font-size: 0.92rem;
  line-height: 1.75;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
}

.chat-role-badge {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #94a3b8;
  margin-bottom: 0.5rem;
}

.source-item {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 0.75rem 1rem;
  margin: 0.45rem 0;
  transition: border-color 0.15s ease;
}

.source-item:hover {
  border-color: #cbd5e1;
}

.score-badge {
  display: inline-flex;
  align-items: center;
  background: #e0f2fe;
  color: #0369a1 !important;
  border: 1px solid #bae6fd;
  border-radius: 5px;
  padding: 0.15rem 0.45rem;
  font-size: 0.68rem;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
}

/* ── Minimalist Status Dots (Pure CSS) ────────────────────────────────────── */
.dot-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
  margin-right: 7px;
  vertical-align: middle;
}
.dot-indicator.green { background: #16a34a; }
.dot-indicator.indigo { background: #0f172a; }
.dot-indicator.red { background: #dc2626; }
.dot-indicator.amber { background: #d97706; }

/* ── Code Blocks & Expanders (White Theme) ────────────────────────────────── */
.stCodeBlock {
  border: 1px solid #e2e8f0 !important;
  border-radius: 8px !important;
  background: #f8fafc !important;
}

.stCodeBlock code {
  color: #0f172a !important;
}

[data-testid="stExpander"] {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 8px !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
}

[data-testid="stExpander"]:hover {
  border-color: #cbd5e1 !important;
}

[data-testid="stExpander"] summary span {
  color: #0f172a !important;
  font-weight: 600 !important;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: #f1f5f9 !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 8px !important;
  padding: 0.25rem !important;
  gap: 0.35rem !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
  border-radius: 6px !important;
  color: #64748b !important;
  font-weight: 600 !important;
  font-size: 0.84rem !important;
  padding: 0.5rem 1rem !important;
}

[data-testid="stTabs"] [aria-selected="true"] {
  background: #ffffff !important;
  color: #0f172a !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
}

[data-testid="stAlert"] {
  border-radius: 8px !important;
  border: 1px solid #e2e8f0 !important;
  background: #f8fafc !important;
}

[data-testid="stProgressBar"] > div > div {
  background: #0f172a !important;
  border-radius: 999px !important;
}

body, .stApp, .block-container {
  color: #0f172a;
}

h1, h2, h3, h4 {
  color: #0f172a !important;
}

.stMarkdown p, .stMarkdown li {
  color: #0f172a;
}

/* Repo Name Badge in Knowledge Bank Vault - Black Box with Crisp White Text */
.repo-name-badge {
  display: inline-flex;
  align-items: center;
  background: #0f172a !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  font-size: 0.88rem !important;
  padding: 0.25rem 0.68rem !important;
  border-radius: 6px !important;
  letter-spacing: -0.01em !important;
}

.repo-name-badge * {
  color: #ffffff !important;
}

code {
  background: #f1f5f9 !important;
  color: #0f172a !important;
  border: 1px solid #e2e8f0 !important;
  padding: 0.12rem 0.4rem !important;
  border-radius: 5px !important;
  font-family: 'JetBrains Mono', monospace !important;
}

/* ── Responsive adjustments ──────────────────────────────────────────────── */
@media (max-width: 900px) {
  .block-container { padding: 1rem 1rem 3rem !important; }
  .bento-hero { padding: 1.6rem 1.4rem; }
  .chat-q-box { margin-left: 2%; }
  .chat-a-box { margin-right: 2%; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION & SINGLE PROFILE PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_DEMO_USER = {
    "email": "demo@codelens.ai",
    "password": "codelens-demo",
    "name": "Demo Engineer",
    "role": "Lead Architect",
    "user_type": "demo",
    "is_guest": False,
    "remember_me": True
}

def load_auth_profile():
    if AUTH_FILE.exists():
        try:
            return json.loads(AUTH_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None

def save_auth_profile(profile_data):
    AUTH_FILE.write_text(json.dumps(profile_data, indent=2), encoding="utf-8")

def clear_auth_profile():
    if AUTH_FILE.exists():
        try:
            AUTH_FILE.unlink()
        except Exception:
            pass

def get_user_initials(name_or_email):
    if not name_or_email:
        return "CL"
    parts = str(name_or_email).strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    name = parts[0].split("@")[0]
    return name[:2].upper() if len(name) >= 2 else (name + "X")[:2].upper()


# Initialize Authentication State
saved_profile = load_auth_profile()
if "authenticated" not in st.session_state:
    if saved_profile and saved_profile.get("remember_me"):
        st.session_state.authenticated = True
        st.session_state.user = saved_profile
    else:
        st.session_state.authenticated = False
        st.session_state.user = None


# ══════════════════════════════════════════════════════════════════════════════
# DEMO LOGIN + GUEST ACCESS PORTAL (SHOWN WHEN NOT AUTHENTICATED)
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.get("authenticated", False):
    st.markdown("""
    <div style='margin-bottom: 1.8rem; text-align: left;'>
        <div style='display:inline-flex; align-items:center; gap:0.6rem; margin-bottom: 0.6rem;'>
            <div style='width: 36px; height: 36px; border-radius: 8px; background: #0f172a; color: #ffffff; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; font-weight: 800; font-family: "JetBrains Mono", monospace;'>
                CL
            </div>
            <span style='font-size: 0.72rem; color: #64748b; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;'>CODELENS AI WORKBENCH</span>
        </div>
        <h1 style='font-size: 2.2rem; font-weight: 800; color: #0f172a; letter-spacing: -0.04em; margin: 0 0 0.4rem;'>
            Access Codebase Intelligence
        </h1>
        <p style='color: #64748b; font-size: 0.95rem; margin: 0;'>
            Sign in with your email and password, launch instant Demo access, or explore as Guest. Your single profile and login details can be saved locally.
        </p>
    </div>
    """, unsafe_allow_html=True)

    auth_l, auth_r = st.columns([1.1, 1.3])

    with auth_l:
        st.markdown("<div class='sec-title'>Fast Instant Access</div>", unsafe_allow_html=True)

        # Demo Account Card
        st.markdown("""
        <div class='bento-card' style='margin-bottom: 1rem; border-left: 3px solid #0f172a;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div style='font-weight: 800; font-size: 0.95rem; color: #0f172a;'>Demo Account</div>
                <span class='chip-pill' style='font-size:0.65rem;'>PRE-CONFIGURED</span>
            </div>
            <div style='font-size: 0.82rem; color: #475569; margin: 0.5rem 0 0.8rem; line-height: 1.6;'>
                Email: <code style='color:#0f172a;'>demo@codelens.ai</code><br>
                Password: <code style='color:#0f172a;'>codelens-demo</code><br>
                Role: <b>Lead Architect</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Instant Demo Sign In", key="btn_instant_demo", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.user = DEFAULT_DEMO_USER.copy()
            save_auth_profile(DEFAULT_DEMO_USER)
            st.rerun()

        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

        # Guest Access Card
        st.markdown("""
        <div class='bento-card' style='margin-bottom: 1rem; border-left: 3px solid #64748b;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div style='font-weight: 800; font-size: 0.95rem; color: #0f172a;'>Guest Access</div>
                <span class='chip-pill' style='font-size:0.65rem;'>SANDBOX</span>
            </div>
            <div style='font-size: 0.82rem; color: #475569; margin: 0.5rem 0 0.8rem; line-height: 1.6;'>
                Explore full codebase intelligence and RAG reasoning without persistent credentials.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Continue as Guest", key="btn_guest_access", use_container_width=True, type="secondary"):
            guest_user = {
                "email": "guest@codelens.ai",
                "password": "",
                "name": "Guest Explorer",
                "role": "Guest Sandbox",
                "user_type": "guest",
                "is_guest": True,
                "remember_me": False
            }
            st.session_state.authenticated = True
            st.session_state.user = guest_user
            st.rerun()

    with auth_r:
        st.markdown("<div class='sec-title'>Credential Sign In & Profile Setup</div>", unsafe_allow_html=True)
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)

        init_email = saved_profile.get("email", "") if saved_profile else ""
        init_pass = saved_profile.get("password", "") if (saved_profile and saved_profile.get("remember_me")) else ""
        init_name = saved_profile.get("name", "") if saved_profile else ""

        with st.form("auth_login_form", clear_on_submit=False):
            f_email = st.text_input("Email Address", value=init_email, placeholder="engineer@organization.com")
            f_pass = st.text_input("Password", value=init_pass, type="password", placeholder="Enter your password")
            f_name = st.text_input("Display Name (for your single profile)", value=init_name, placeholder="Your Name or Alias")
            f_remember = st.checkbox("Remember login details and single profile on this device", value=True)

            st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
            submitted_login = st.form_submit_button("Sign In / Save Profile", use_container_width=True)

        if submitted_login:
            if not f_email.strip() or not f_pass.strip():
                st.warning("Please provide both email and password to proceed.")
            else:
                email_clean = f_email.strip()
                pass_clean = f_pass.strip()

                # Check if matches demo credentials
                if email_clean.lower() == "demo@codelens.ai" and pass_clean == "codelens-demo":
                    profile = DEFAULT_DEMO_USER.copy()
                    profile["remember_me"] = f_remember
                else:
                    name_clean = f_name.strip() or email_clean.split("@")[0].capitalize()
                    profile = {
                        "email": email_clean,
                        "password": pass_clean,
                        "name": name_clean,
                        "role": "Software Engineer",
                        "user_type": "member",
                        "is_guest": False,
                        "remember_me": f_remember,
                        "last_login": datetime.datetime.now().strftime("%b %d, %Y %H:%M")
                    }

                if f_remember:
                    save_auth_profile(profile)
                else:
                    clear_auth_profile()

                st.session_state.authenticated = True
                st.session_state.user = profile
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Stop rendering the rest of the application until authenticated
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            pass
    return {"projects": [], "qa": []}

def save_history(h):
    HISTORY_FILE.write_text(json.dumps(h, indent=2))

def add_project(h, repo, coll, files, chunks):
    entry = {
        "id": datetime.datetime.now().isoformat(),
        "repo": repo,
        "collection": coll,
        "files": files,
        "chunks": chunks,
        "date": datetime.datetime.now().strftime("%b %d, %Y %H:%M"),
        "ts": datetime.datetime.now().timestamp()
    }
    h["projects"] = [p for p in h["projects"] if p["collection"] != coll]
    h["projects"].insert(0, entry)
    h["projects"] = h["projects"][:30]
    save_history(h)

def add_qa(h, coll, question, answer):
    h["qa"].insert(0, {
        "id": datetime.datetime.now().isoformat(),
        "collection": coll,
        "question": question,
        "answer": answer,
        "date": datetime.datetime.now().strftime("%b %d %H:%M"),
        "ts": datetime.datetime.now().timestamp()
    })
    h["qa"] = h["qa"][:200]
    save_history(h)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE (AUTHENTICATED SESSION)
# ══════════════════════════════════════════════════════════════════════════════
PAGES = ["Dashboard", "Index", "Ask", "History", "Settings", "About"]

defaults = {
    "page": "Dashboard",
    "history": load_history(),
    "chat": [],
    "collection": None,
    "dark_code": False,
    "top_k": 5,
    "nav_history": [],
    "_last_page": "Dashboard",
    "q_preset": ""
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

H = st.session_state.history
SS = st.session_state

if SS.page != SS._last_page:
    SS.nav_history.append(SS._last_page)
    SS.nav_history = SS.nav_history[-30:]
    SS._last_page = SS.page

def navigate_back():
    if SS.nav_history:
        previous_page = SS.nav_history.pop()
        SS.page = previous_page
        SS._last_page = previous_page


# ══════════════════════════════════════════════════════════════════════════════
# ENV / CONFIG
# ══════════════════════════════════════════════════════════════════════════════
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
GROQ_KEY = os.getenv("GROQ_API_KEY") or ""
QDRANT_URL = os.getenv("QDRANT_URL", "local").strip()

has_gemini = bool(GEMINI_KEY)
has_groq = bool(GROQ_KEY)
is_local_q = QDRANT_URL.lower() in ("local", ":memory:", "") or "your-cluster-url" in QDRANT_URL

provider_opts = ["Gemini", "Groq"]

GEMINI_MODELS = ["gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-2.5-pro"]
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "allam-2-7b"]

try:
    _store = QdrantStore(client=_client)
    COLLECTIONS = _store.list_collections()
except Exception:
    COLLECTIONS = []


# ══════════════════════════════════════════════════════════════════════════════
# MINIMALIST MONOCHROME SIDEBAR WITH USER PROFILE BADGE
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Clean Brand Block
    st.markdown("""
    <div style="padding: 1.1rem 0.6rem 0.75rem; border-bottom: 1px solid #e2e8f0;">
        <div style="display:flex; align-items:center; gap: 0.75rem;">
            <div style="width: 34px; height: 34px; border-radius: 8px; background: #0f172a; color: #ffffff; display: flex; align-items: center; justify-content: center; font-size: 0.82rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; letter-spacing: -0.05em;">
                CL
            </div>
            <div>
                <div style="font-size: 1.02rem; font-weight: 800; color: #0f172a; letter-spacing: -0.03em;">CodeLens AI</div>
                <div style="font-size: 0.65rem; color: #64748b; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">AST &bull; RAG REASONING</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # User Profile Block
    curr_user = st.session_state.get("user") or {}
    u_name = curr_user.get("name", "Active User")
    u_email = curr_user.get("email", "user@codelens.ai")
    u_is_guest = curr_user.get("is_guest", False)
    u_is_demo = (curr_user.get("user_type") == "demo")
    u_badge = "DEMO USER" if u_is_demo else ("GUEST ACCESS" if u_is_guest else "SAVED PROFILE")
    u_initials = get_user_initials(u_name or u_email)

    st.markdown(f"""
    <div style="margin: 0.7rem 0.4rem 0.5rem; padding: 0.75rem 0.85rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 9px; display: flex; align-items: center; gap: 0.65rem;">
        <div style="width: 32px; height: 32px; border-radius: 8px; background: #0f172a; color: #ffffff; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; flex-shrink: 0;">
            {u_initials}
        </div>
        <div style="overflow: hidden; flex: 1;">
            <div style="font-size: 0.82rem; font-weight: 700; color: #0f172a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{u_name}</div>
            <div style="font-size: 0.68rem; color: #64748b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{u_email}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    prof_col1, prof_col2 = st.columns([1.2, 1])
    with prof_col1:
        st.markdown(f"<div style='padding-top:0.3rem;'><span class='chip-pill' style='font-size:0.62rem; padding:0.18rem 0.45rem;'>{u_badge}</span></div>", unsafe_allow_html=True)
    with prof_col2:
        if st.button("Sign Out", key="sb_btn_signout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Navigation menu
    st.markdown("<div style='padding: 0.2rem 0.6rem 0.4rem; font-size: 0.68rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em;'>WORKBENCH</div>", unsafe_allow_html=True)
    for p in PAGES:
        is_active = (SS.page == p)
        btn_type = "primary" if is_active else "secondary"
        if st.button(p, key=f"nav_btn_{p}", use_container_width=True, type=btn_type):
            SS.page = p
            st.rerun()

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # Active Collection HUD Card
    st.markdown("<div style='padding: 0.2rem 0.6rem 0.4rem; font-size: 0.68rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; border-top: 1px solid #e2e8f0; padding-top: 0.8rem;'>ACTIVE KNOWLEDGE BASE</div>", unsafe_allow_html=True)
    if SS.collection:
        st.markdown(f"""
        <div style='margin: 0.2rem 0.4rem 0.5rem; padding: 0.8rem 0.95rem; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 9px;'>
            <div style='display:flex; align-items:center; justify-content:space-between; margin-bottom: 0.25rem;'>
                <span style='font-size: 0.65rem; color: #475569; font-weight: 700; letter-spacing: 0.08em;'>MOUNTED REPO</span>
                <span class='dot-indicator green'></span>
            </div>
            <div style='font-size: 0.85rem; color: #0f172a; font-family: "JetBrains Mono", monospace; font-weight: 700; word-break: break-all;'>
                {SS.collection}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='margin: 0.2rem 0.4rem 0.5rem; padding: 0.75rem 0.9rem; background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; color: #64748b; font-size: 0.78rem;'>
            No repository mounted.<br><span style='color:#0f172a; font-weight:600;'>Index or switch below.</span>
        </div>
        """, unsafe_allow_html=True)

    # Collection switcher dropdown
    if COLLECTIONS:
        opts = ["-- Select / Switch Repository --"] + COLLECTIONS
        selected_idx = (opts.index(SS.collection) if SS.collection in opts else 0)
        picked = st.selectbox("Mount Collection", opts, index=selected_idx, label_visibility="collapsed", key="sb_coll_pick")
        if picked != "-- Select / Switch Repository --" and picked != SS.collection:
            SS.collection = picked
            SS.chat = []
            st.rerun()

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # Live Engine Telemetry
    st.markdown("<div style='padding: 0.2rem 0.6rem 0.4rem; font-size: 0.68rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; border-top: 1px solid #e2e8f0; padding-top: 0.8rem;'>ENGINE TELEMETRY</div>", unsafe_allow_html=True)
    def dot_html(status):
        return "<span class='dot-indicator green'></span>" if status else "<span class='dot-indicator red'></span>"

    st.markdown(f"""
    <div style='padding: 0.3rem 0.6rem; font-size: 0.78rem; color: #475569; line-height: 2;'>
        <div>{dot_html(has_gemini)} Gemini API <span style='float:right; color:#64748b; font-weight:600;'>{"READY" if has_gemini else "NO KEY"}</span></div>
        <div>{dot_html(has_groq)} Groq API <span style='float:right; color:#64748b; font-weight:600;'>{"READY" if has_groq else "OFFLINE"}</span></div>
        <div><span class='dot-indicator indigo'></span> Vector Store <span style='float:right; color:#0f172a; font-weight:600;'>{("Embedded" if is_local_q else "Cloud")}</span></div>
        <div><span class='dot-indicator green'></span> Repositories <span style='float:right; color:#0284c7; font-weight:600;'>{len(COLLECTIONS)} active</span></div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TOP COMMAND BAR & NICHE ASYMMETRIC HEADER
# ══════════════════════════════════════════════════════════════════════════════
page = SS.page

top_c1, top_c2, top_c3 = st.columns([1.2, 5.8, 3])
with top_c1:
    if SS.nav_history and page != "Dashboard":
        if st.button("<- Back", key="top_nav_back", use_container_width=True):
            navigate_back()
            st.rerun()
with top_c2:
    st.markdown(f"""
    <div style='padding: 0.45rem 0; display:flex; align-items:center; gap: 0.6rem;'>
        <span style='color: #64748b; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;'>CODELENS</span>
        <span style='color: #cbd5e1;'>/</span>
        <span style='color: #0f172a; font-size: 0.75rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase;'>{page}</span>
    </div>
    """, unsafe_allow_html=True)
with top_c3:
    if SS.collection:
        st.markdown(f"""
        <div style='text-align: right; padding: 0.35rem 0;'>
            <span class='chip-pill'>
                <span class='dot-indicator green' style='margin-right:4px;'></span>
                MOUNTED: {SS.collection}
            </span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD (BENTO GRID ARCHITECTURE · WHITE THEME)
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    # ── Bento Hero Section: Asymmetric 5:3 Split ──────────────────────────────
    hero_l, hero_r = st.columns([5, 3])
    with hero_l:
        st.markdown("""
        <div class='bento-hero' style='margin-bottom:0; min-height: 220px; display:flex; flex-direction:column; justify-content:center;'>
            <div style='display:inline-flex; align-items:center; gap:0.4rem; background:#f1f5f9; border:1px solid #e2e8f0; border-radius:999px; padding:0.2rem 0.65rem; font-size:0.72rem; color:#475569; font-weight:700; margin-bottom:0.8rem; width:fit-content;'>
                <span class='dot-indicator green' style='margin:0;'></span> AST-AWARE CODE INTELLIGENCE
            </div>
            <div class='bento-hero-title'>
                Query Any Codebase.<br><span>Zero Hallucinations.</span>
            </div>
            <p class='bento-hero-sub'>
                Tree-sitter chunking, dense vector similarity, and LLM reasoning pipeline engineered for deep codebase exploration.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with hero_r:
        active_repo_label = SS.collection if SS.collection else "None Mounted"
        st.markdown(f"""
        <div class='bento-card' style='height: 100%; min-height: 220px; display: flex; flex-direction: column; justify-content: space-between;'>
            <div>
                <div class='sec-title'>Live Mission HUD</div>
                <div style='font-size: 0.85rem; color: #475569; line-height: 1.8;'>
                    <div style='display:flex; justify-content:space-between; border-bottom:1px solid #f1f5f9; padding-bottom:0.35rem;'>
                        <span>Active Collection:</span>
                        <span style='color:#0f172a; font-family:"JetBrains Mono",monospace; font-weight:700;'>{active_repo_label}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between; padding:0.35rem 0; border-bottom:1px solid #f1f5f9;'>
                        <span>Vector Index Mode:</span>
                        <span style='color:#0284c7; font-weight:600;'>{("Local Embedded Qdrant" if is_local_q else "Qdrant Cloud Cluster")}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between; padding-top:0.35rem;'>
                        <span>Embedding Model:</span>
                        <span style='color:#16a34a; font-weight:600;'>all-MiniLM-L6-v2</span>
                    </div>
                </div>
            </div>
            <div style='margin-top:0.8rem;'>
                <span class='chip-pill' style='width:100%; justify-content:center; padding: 0.35rem;'>
                    READY FOR REASONING TASKS
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── Niche Bento Grid: Fibonacci Metric Tiles (2.4 : 1.3 : 1.3 : 1) ───────
    total_chunks = sum(p.get("chunks", 0) for p in H.get("projects", []))
    total_files = sum(p.get("files", 0) for p in H.get("projects", []))
    m1, m2, m3, m4 = st.columns([2.4, 1.3, 1.3, 1.0])

    with m1:
        st.markdown(f"""
        <div class='stat-tile hero-metric'>
            <div>
                <div class='stat-tile-num cyan'>{total_chunks:,}</div>
                <div class='stat-tile-lbl'>Indexed Code Chunks</div>
            </div>
            <div>
                <span class='stat-tag tag-cyan'>Dense Vectors ({total_files} files)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class='stat-tile'>
            <div>
                <div class='stat-tile-num'>{len(H.get("projects", []))}</div>
                <div class='stat-tile-lbl'>Repositories In Vault</div>
            </div>
            <div>
                <span class='stat-tag tag-indigo'>AST Parsed</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class='stat-tile'>
            <div>
                <div class='stat-tile-num'>{len(H.get("qa", []))}</div>
                <div class='stat-tile-lbl'>Inquiries Resolved</div>
            </div>
            <div>
                <span class='stat-tag tag-emerald'>RAG Verified</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class='stat-tile'>
            <div>
                <div class='stat-tile-num'>{len(COLLECTIONS)}</div>
                <div class='stat-tile-lbl'>Collections</div>
            </div>
            <div>
                <span class='stat-tag tag-amber'>Qdrant DB</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

    # ── Quick Action Command Bar ──────────────────────────────────────────────
    st.markdown("<div class='sec-title'>Quick Command Launchpad</div>", unsafe_allow_html=True)
    act1, act2, act3, act4 = st.columns([1.5, 1.5, 1.5, 1.5])
    with act1:
        if st.button("Index New Repository", use_container_width=True):
            SS.page = "Index"
            st.rerun()
    with act2:
        if st.button("Launch Query Studio", use_container_width=True):
            SS.page = "Ask"
            st.rerun()
    with act3:
        if st.button("Inspect Audit History", use_container_width=True):
            SS.page = "History"
            st.rerun()
    with act4:
        if st.button("Configure Engine", use_container_width=True):
            SS.page = "Settings"
            st.rerun()

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Asymmetric Split Workbench: Repositories (5) & Telemetry Stream (3) ────
    wb_left, wb_right = st.columns([5, 3])

    with wb_left:
        st.markdown("<div class='sec-title'>Codebase Knowledge Vault</div>", unsafe_allow_html=True)
        if not H.get("projects"):
            st.markdown("""
            <div class='bento-card' style='text-align:center; padding: 2.8rem 1rem;'>
                <div style='color:#0f172a; font-weight:700;'>Knowledge Vault is currently empty</div>
                <div style='color:#64748b; font-size:0.82rem; margin-top:0.3rem;'>Index a GitHub repository or local codebase to populate semantic vectors.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for proj in H["projects"][:5]:
                repo_name = proj["repo"].rstrip("/").split("/")[-1] if "/" in proj["repo"] else Path(proj["repo"]).name
                is_active = (SS.collection == proj["collection"])
                card_border = "border: 1px solid #0f172a; background: #ffffff;" if is_active else ""

                c_info, c_btn = st.columns([4.2, 1.2])
                with c_info:
                    st.markdown(f"""
                    <div class='bento-card' style='padding: 0.95rem 1.2rem; margin-bottom: 0.6rem; {card_border}'>
                        <div style='display:flex; align-items:center; justify-content:space-between;'>
                            <div class='repo-name-badge'>
                                {repo_name}
                            </div>
                            <span class='chip-pill'>{proj["collection"]}</span>
                        </div>
                        <div style='font-size: 0.76rem; color: #64748b; margin-top: 0.45rem; display:flex; gap:0.9rem;'>
                            <span>{proj["files"]} files</span>
                            <span>{proj["chunks"]} chunks</span>
                            <span>{proj["date"]}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with c_btn:
                    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
                    btn_label = "Mounted" if is_active else "Mount & Ask"
                    if st.button(btn_label, key=f"dash_load_{proj['id']}", use_container_width=True, disabled=is_active):
                        SS.collection = proj["collection"]
                        SS.chat = []
                        SS.page = "Ask"
                        st.rerun()

    with wb_right:
        st.markdown("<div class='sec-title'>Live Intelligence Stream</div>", unsafe_allow_html=True)
        if not H.get("qa"):
            st.markdown("""
            <div class='bento-card' style='text-align:center; padding: 2.8rem 1rem;'>
                <div style='color:#0f172a; font-weight:700;'>No recent inquiries</div>
                <div style='color:#64748b; font-size:0.82rem; margin-top:0.3rem;'>Launch Query Studio to ask architectural questions.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for qa in H["qa"][:5]:
                q_text = qa["question"][:65] + ("..." if len(qa["question"]) > 65 else "")
                st.markdown(f"""
                <div class='bento-card' style='padding: 0.8rem 1rem; margin-bottom: 0.55rem;'>
                    <div style='font-size: 0.82rem; color: #0f172a; font-weight: 600; line-height: 1.4;'>
                        "{q_text}"
                    </div>
                    <div style='font-size: 0.7rem; color: #64748b; margin-top: 0.35rem; display:flex; justify-content:space-between;'>
                        <span style='color: #0f172a; font-family:"JetBrains Mono",monospace; font-weight:600;'>{qa.get("collection","")}</span>
                        <span>{qa.get("date","")}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# INDEX (INGESTION REACTOR · ASYMMETRIC WORKBENCH)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Index":
    st.markdown("""
    <div style='margin-bottom: 1.4rem;'>
        <h2 style='color:#0f172a; font-size:1.8rem; font-weight:800; letter-spacing:-0.03em; margin:0;'>
            Repository Ingestion Reactor
        </h2>
        <div style='color:#64748b; font-size:0.88rem; margin-top:0.25rem;'>
            Clone, filter, AST-chunk, embed, and upload source code into dense Qdrant vector spaces.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Asymmetric Ingestion Layout: Left Ingestion Deck (5.5) / Right Vault (4.5)
    idx_col1, idx_col2 = st.columns([5.5, 4.5])

    with idx_col1:
        st.markdown("<div class='sec-title'>Reactor Controls</div>", unsafe_allow_html=True)
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)

        repo_input = st.text_input(
            "Source Repository URL or Local Directory Path",
            placeholder="https://github.com/fastapi/fastapi  or  /path/to/project",
            help="Supports GitHub URLs (cloned via GitPython) and local folders."
        )

        auto_name = ""
        if repo_input.strip():
            stem = Path(repo_input.rstrip("/\\")).stem
            auto_name = sanitize_collection_name(f"{stem}_codebase")

        col_name = st.text_input(
            "Target Vector Collection Name",
            value=auto_name,
            placeholder="auto-generated identifier",
            help="Alphanumeric, underscores, and hyphens only."
        )

        opt_c1, opt_c2 = st.columns([3, 2])
        with opt_c1:
            recreate = st.checkbox("Overwrite if collection exists", value=False)
        with opt_c2:
            st.markdown("<div style='height:0.1rem'></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        go_btn = st.button("Initialize and Index Repository", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if go_btn:
            if not repo_input.strip():
                st.warning("Please specify a valid repository URL or local directory path.")
            else:
                target = sanitize_collection_name(col_name.strip() or auto_name or "codebase")
                bar = st.progress(0, text="Initializing clone and AST engine...")
                try:
                    bar.progress(15, text="Cloning and scanning filesystem...")
                    indexer = RepositoryIndexer(collection_name=target, client=_client)
                    bar.progress(35, text="Parsing AST grammar and segmenting code chunks...")
                    result = indexer.index_repository(repo_input.strip(), recreate_collection=recreate)
                    bar.progress(80, text="Generating embeddings and pushing to Qdrant...")
                    SS.collection = target
                    SS.chat = []
                    add_project(H, repo_input.strip(), target, result["files"], result["chunks"])
                    bar.progress(100, text="Knowledge Ingestion Complete!")

                    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
                    st.success(f"Indexed {target} into vector memory successfully.")

                    # Ingestion Telemetry Bento
                    st.markdown(f"""
                    <div style='display:grid; grid-template-columns: 1fr 1fr 1.5fr; gap: 10px; margin: 0.8rem 0;'>
                        <div class='stat-tile' style='padding:0.9rem;'>
                            <div class='stat-tile-num' style='font-size:1.6rem;'>{result["files"]}</div>
                            <div class='stat-tile-lbl'>Files Scanned</div>
                        </div>
                        <div class='stat-tile' style='padding:0.9rem;'>
                            <div class='stat-tile-num cyan' style='font-size:1.6rem;'>{result["chunks"]}</div>
                            <div class='stat-tile-lbl'>Chunks Embedded</div>
                        </div>
                        <div class='stat-tile' style='padding:0.9rem;'>
                            <div style='font-family:"JetBrains Mono",monospace; font-weight:700; color:#0f172a; font-size:0.95rem; word-break:break-all;'>{target}</div>
                            <div class='stat-tile-lbl'>Active Target</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if result["chunks"] == 0:
                        st.warning("0 chunks indexed: Verify repo contains supported source files (.py, .js, .ts, .md, .go, etc.).")

                    if st.button("Open in Query Studio ->", key="jump_to_ask"):
                        SS.page = "Ask"
                        st.rerun()

                except Exception as e:
                    bar.empty()
                    import traceback
                    st.error(f"Ingestion failed: {e}")
                    with st.expander("Diagnostic Traceback"):
                        st.code(traceback.format_exc())

    with idx_col2:
        st.markdown("<div class='sec-title'>Knowledge Bank Vault</div>", unsafe_allow_html=True)
        if not H.get("projects"):
            st.markdown("""
            <div class='bento-card' style='text-align:center; padding: 3rem 1rem;'>
                <div style='color:#64748b; font-size:0.88rem;'>No previously indexed repositories found.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for proj in H["projects"][:8]:
                repo_short = proj["repo"].rstrip("/").split("/")[-1] if "/" in proj["repo"] else Path(proj["repo"]).name
                is_active = (SS.collection == proj["collection"])
                border_style = "border-color: #0f172a; background: #ffffff;" if is_active else ""

                st.markdown(f"""
                <div class='bento-card' style='padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; {border_style}'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <div class='repo-name-badge'>
                            {repo_short}
                        </div>
                        <span class='chip-pill'>{proj["collection"]}</span>
                    </div>
                    <div style='font-size:0.75rem; color:#64748b; margin-top:0.45rem;'>
                        {proj["files"]} files &bull; {proj["chunks"]} chunks &bull; {proj["date"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if not is_active:
                    if st.button("Mount & Query", key=f"idx_load_{proj['id']}", use_container_width=True):
                        SS.collection = proj["collection"]
                        SS.chat = []
                        SS.page = "Ask"
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ASK (QUERY & REASONING STUDIO · ASYMMETRIC IDE HUD)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Ask":
    # ── Top Control HUD Strip: Asymmetric 4-Part Toolbar ──────────────────────
    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([2.5, 2.2, 1.3, 1.2])

    with ctrl_c1:
        provider = st.selectbox("AI Model", ["Gemini", "Groq"], index=0, key="ask_provider", label_visibility="collapsed")
    with ctrl_c2:
        models = GEMINI_MODELS if provider == "Gemini" else GROQ_MODELS
        default_m = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite") if provider == "Gemini" else os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        idx = models.index(default_m) if default_m in models else 0
        model = st.selectbox("Model", models, index=idx, key="ask_model", label_visibility="collapsed")
    with ctrl_c3:
        top_k = st.number_input("Top-K Snippets", min_value=1, max_value=20, value=SS.top_k, key="ask_top_k", label_visibility="collapsed")
    with ctrl_c4:
        if st.button("Clear Session", use_container_width=True):
            SS.chat = []
            st.rerun()

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── Asymmetric IDE Layout: Left Context Inspector (1) : Right Canvas (2.8) ─
    ask_left, ask_right = st.columns([1.0, 2.8])

    with ask_left:
        st.markdown("<div class='sec-title'>Context Inspector</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='bento-card' style='padding: 1rem 1.15rem; margin-bottom: 0.9rem;'>
            <div style='font-size: 0.68rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;'>Active Knowledge Base</div>
            <div style='font-size: 0.92rem; color: #0f172a; font-family:"JetBrains Mono",monospace; font-weight: 700; margin-top: 0.25rem; word-break:break-all;'>
                {SS.collection if SS.collection else "None Mounted"}
            </div>
            <div style='margin-top: 0.8rem; font-size: 0.76rem; color: #475569; line-height: 1.8;'>
                <div>Engine: <span style='color:#0f172a; font-weight:700;'>{provider} ({model})</span></div>
                <div>Retriever: <span style='color:#0284c7; font-weight:700;'>K={top_k} vectors</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Prompt Inspiration Chips
        st.markdown("<div class='sec-title'>Prompt Inspiration</div>", unsafe_allow_html=True)
        preset_prompts = [
            ("System Architecture", "Explain the high-level architecture and component flow."),
            ("Authentication and Security", "How does authentication, auth middleware, and token validation work?"),
            ("Performance Hotspots", "Identify computational bottlenecks or optimization points."),
            ("Test Coverage and Mocks", "How are unit tests structured and how are external dependencies mocked?"),
            ("Data Flow and Pipeline", "Trace the end-to-end data ingestion and vector indexing pipeline.")
        ]
        for label, p_text in preset_prompts:
            if st.button(label, key=f"pchip_{label}", use_container_width=True):
                SS.q_preset = p_text
                st.rerun()

    with ask_right:
        if not SS.collection:
            st.markdown("""
            <div class='bento-card' style='text-align:center; padding: 4rem 1.5rem;'>
                <div style='font-size: 1.15rem; font-weight: 800; color: #0f172a;'>No Repository Mounted</div>
                <div style='color: #64748b; font-size: 0.88rem; max-width: 440px; margin: 0.5rem auto 1.4rem;'>
                    Mount an existing collection from the sidebar, or navigate to Index to ingest a new codebase.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go to Ingestion Reactor", use_container_width=True):
                SS.page = "Index"
                st.rerun()
        else:
            # Welcome banner if empty chat
            if not SS.chat:
                st.markdown(f"""
                <div class='bento-card' style='padding: 1.8rem 2rem; margin-bottom: 1.2rem; border-left: 3px solid #0f172a;'>
                    <div style='font-size: 1.1rem; font-weight: 800; color: #0f172a;'>Querying {SS.collection}</div>
                    <div style='color: #64748b; font-size: 0.88rem; margin-top: 0.4rem;'>
                        Ask targeted questions about architecture, functions, logic, dependencies, or edge cases.
                        Every answer retrieves verified code snippets with exact line citations.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Chat Message Stream
            for msg in SS.chat:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class='chat-q-box'>
                        <div class='chat-role-badge'>DEVELOPER INQUIRY</div>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='chat-a-box'>
                        <div class='chat-role-badge'>CODELENS REASONING ENGINE</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(msg['content'])

                    if msg.get("sources"):
                        with st.expander(f"Verified Source Citations ({len(msg['sources'])} snippets)"):
                            for s in msg["sources"]:
                                score = f"{s.get('score', 0):.3f}" if isinstance(s.get("score"), (int, float)) else "--"
                                fpath = s.get('file_path', 'unknown')
                                st.markdown(f"""
                                <div class='source-item'>
                                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                                        <div>
                                            <span class='chip-pill'>{fpath}</span>
                                            <span style='color: #64748b; font-size: 0.75rem; margin-left: 0.4rem;'>
                                                {s.get('type','snippet')} &bull; {s.get('name','')} &bull; L{s.get('start_line','?')}--{s.get('end_line','?')}
                                            </span>
                                        </div>
                                        <span class='score-badge'>Score: {score}</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                if s.get("content"):
                                    ext = Path(fpath).suffix.lstrip(".")
                                    st.code(s["content"], language=ext or "python")

            # Input Console Form
            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            with st.form("ask_form", clear_on_submit=False):
                preset_val = SS.q_preset
                q_text = st.text_input(
                    "Query Input",
                    value=preset_val,
                    placeholder="E.g., How does the token refresher work? / Explain the index_repository pipeline.",
                    label_visibility="collapsed"
                )

                f_c1, f_c2 = st.columns([5, 1])
                with f_c1:
                    st.caption("Press Enter or click Submit to query dense vector store and generate cited answer.")
                with f_c2:
                    submitted = st.form_submit_button("Ask ->", use_container_width=True)

            if submitted and q_text.strip():
                SS.q_preset = ""  # reset preset
                key_needed = "GEMINI_API_KEY" if provider == "Gemini" else "GROQ_API_KEY"
                has_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) if provider == "Gemini" else bool(os.getenv("GROQ_API_KEY"))
                if not has_key:
                    st.error(f"Missing {provider} API key: Please add `{key_needed}` in your `.env` file.")
                else:
                    with st.spinner(f"Retrieving context and generating answer with {provider}..."):
                        try:
                            rag = RAGPipeline(
                                collection_name=SS.collection,
                                model=model,
                                provider=provider.lower(),
                                client=_client
                            )
                            res = rag.answer(question=q_text, top_k=int(top_k))
                            SS.top_k = int(top_k)
                            SS.chat.append({"role": "user", "content": q_text})
                            SS.chat.append({"role": "assistant", "content": res["answer"], "sources": res["sources"]})
                            add_qa(H, SS.collection, q_text, res["answer"])
                            st.rerun()
                        except Exception as e:
                            st.error(f"{provider} API request failed: {e}")
                            with st.expander("Details"):
                                import traceback
                                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY (AUDIT ARCHIVE · ASYMMETRIC TABS)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "History":
    st.markdown("""
    <div style='margin-bottom: 1.4rem;'>
        <h2 style='color:#0f172a; font-size:1.8rem; font-weight:800; letter-spacing:-0.03em; margin:0;'>
            Audit Archive and Knowledge Ledger
        </h2>
        <div style='color:#64748b; font-size:0.88rem; margin-top:0.25rem;'>
            Full chronological record of indexed repositories, vector dimensions, and Q&A dialogues.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_proj, tab_qa = st.tabs(["  Indexed Repositories Ledger  ", "  Q&A Intelligence Records  "])

    with tab_proj:
        if not H.get("projects"):
            st.info("No repositories indexed yet.")
        else:
            for proj in H["projects"]:
                repo_name = proj["repo"].rstrip("/").split("/")[-1] if "/" in proj["repo"] else Path(proj["repo"]).name
                with st.expander(f"{repo_name}  |  {proj['date']}"):
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        st.markdown(f"**Collection Identifier:** `{proj['collection']}`")
                        st.markdown(f"**Source Repository:** `{proj['repo']}`")
                    with c2:
                        st.metric("Files Scanned", proj["files"])
                        st.metric("Chunks Stored", proj["chunks"])

                    if SS.collection != proj["collection"]:
                        if st.button("Mount into Active Workspace ->", key=f"hist_load_{proj['id']}"):
                            SS.collection = proj["collection"]
                            SS.chat = []
                            SS.page = "Ask"
                            st.rerun()

    with tab_qa:
        qa_colls = list({q["collection"] for q in H["qa"]}) if H.get("qa") else []
        if qa_colls:
            fcoll = st.selectbox("Filter records by collection", ["All Collections"] + qa_colls, key="hist_filter_coll")
        else:
            fcoll = "All Collections"

        fqa = [q for q in H.get("qa", []) if fcoll == "All Collections" or q.get("collection") == fcoll]
        if not fqa:
            st.info("No recorded questions matching this filter.")
        else:
            for qa in fqa[:60]:
                q_preview = qa["question"][:85] + ("..." if len(qa["question"]) > 85 else "")
                with st.expander(f"Question: {q_preview}  |  {qa.get('date','')}"):
                    st.markdown(f"**Target Collection:** `{qa.get('collection','')}`")
                    st.markdown(f"**Question:** {qa['question']}")
                    st.divider()
                    st.markdown(qa["answer"])

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    if st.button("Purge Entire History Ledger"):
        SS.history = {"projects": [], "qa": []}
        save_history(SS.history)
        st.success("History purged.")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS (ENVIRONMENT, USER PROFILE & TOPOLOGY MATRIX)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Settings":
    st.markdown("""
    <div style='margin-bottom: 1.4rem;'>
        <h2 style='color:#0f172a; font-size:1.8rem; font-weight:800; letter-spacing:-0.03em; margin:0;'>
            Engine Configuration and Topology
        </h2>
        <div style='color:#64748b; font-size:0.88rem; margin-top:0.25rem;'>
            API authentication vaults, user profile management, default reasoning parameters, and vector database topology.
        </div>
    </div>
    """, unsafe_allow_html=True)

    env_path = PROJECT_ROOT / ".env"
    env_text = env_path.read_text() if env_path.exists() else ""

    def get_env_val(key):
        for line in env_text.splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
        return ""

    # ── User Profile and Saved Credentials Bento Card ─────────────────────────
    st.markdown("<div class='sec-title'>User Profile & Saved Credentials</div>", unsafe_allow_html=True)
    st.markdown("<div class='bento-card' style='margin-bottom: 1.4rem;'>", unsafe_allow_html=True)

    prof_u1, prof_u2 = st.columns([1, 1])
    with prof_u1:
        edit_name = st.text_input("Profile Display Name", value=curr_user.get("name", ""), key="edit_prof_name")
        edit_email = st.text_input("User Email Address", value=curr_user.get("email", ""), key="edit_prof_email")
    with prof_u2:
        edit_role = st.text_input("Role / Title", value=curr_user.get("role", "Software Engineer"), key="edit_prof_role")
        saved_status_desc = "Saved locally in data/auth_profile.json" if (AUTH_FILE.exists() and not curr_user.get("is_guest")) else "Temporary Guest Session"
        st.text_input("Profile Storage Status", value=saved_status_desc, disabled=True, key="edit_prof_status")

    prof_act1, prof_act2 = st.columns([2.5, 1])
    with prof_act1:
        if st.button("Update Profile and Save Credentials", key="btn_update_save_prof"):
            curr_user["name"] = edit_name.strip() or curr_user.get("name", "Active User")
            curr_user["email"] = edit_email.strip() or curr_user.get("email", "user@codelens.ai")
            curr_user["role"] = edit_role.strip() or "Software Engineer"
            curr_user["remember_me"] = True
            curr_user["is_guest"] = False
            save_auth_profile(curr_user)
            st.session_state.user = curr_user
            st.success("User profile and login details saved to data/auth_profile.json successfully.")
            st.rerun()
    with prof_act2:
        if st.button("Sign Out of Profile", key="btn_settings_signout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Asymmetric 2-Column Settings Matrix ────────────────────────────────────
    set_col1, set_col2 = st.columns([1, 1])

    with set_col1:
        st.markdown("<div class='sec-title'>Authentication Vault</div>", unsafe_allow_html=True)
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        g_key = st.text_input(
            "Gemini API Key",
            value=get_env_val("GEMINI_API_KEY"),
            type="password",
            key="cfg_gemini"
        )
        st.caption("Obtain key at [aistudio.google.com](https://aistudio.google.com)")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        gr_key = st.text_input(
            "Groq API Key",
            value=get_env_val("GROQ_API_KEY"),
            type="password",
            key="cfg_groq"
        )
        st.caption("Obtain key at [console.groq.com](https://console.groq.com)")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

        st.markdown("<div class='sec-title'>Vector Store Topology</div>", unsafe_allow_html=True)
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        vs_mode = st.radio(
            "Storage Topology Mode",
            ["Local (Embedded disk DB)", "Cloud Cluster (Qdrant Cloud)"],
            index=0 if is_local_q else 1,
            key="cfg_vsmode"
        )
        if vs_mode == "Cloud Cluster (Qdrant Cloud)":
            qurl = st.text_input("Qdrant Cluster URL", value=get_env_val("QDRANT_URL"), placeholder="https://xxx.cloud.qdrant.io:6333")
            qkey = st.text_input("Qdrant Cluster API Key", value=get_env_val("QDRANT_API_KEY"), type="password")
        else:
            qurl = "local"
            qkey = ""
        st.markdown("</div>", unsafe_allow_html=True)

    with set_col2:
        st.markdown("<div class='sec-title'>Default Reasoning Models</div>", unsafe_allow_html=True)
        st.markdown("<div class='bento-card'>", unsafe_allow_html=True)
        dm_g = st.selectbox(
            "Default Gemini Model",
            GEMINI_MODELS,
            index=GEMINI_MODELS.index(get_env_val("GEMINI_MODEL")) if get_env_val("GEMINI_MODEL") in GEMINI_MODELS else 0,
            key="cfg_gmodel"
        )
        dm_gr = st.selectbox(
            "Default Groq Model",
            GROQ_MODELS,
            index=GROQ_MODELS.index(get_env_val("GROQ_MODEL")) if get_env_val("GROQ_MODEL") in GROQ_MODELS else 0,
            key="cfg_grmodel"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

        st.markdown("<div class='sec-title' style='color:#dc2626 !important;'>Collection Pruning Zone</div>", unsafe_allow_html=True)
        st.markdown("<div class='bento-card' style='border-color: #fecaca;'>", unsafe_allow_html=True)
        if COLLECTIONS:
            del_coll = st.selectbox("Select collection to permanently remove", COLLECTIONS, key="cfg_del_coll")
            if st.button("Permanently Delete Collection", type="secondary"):
                try:
                    _client.delete_collection(del_coll)
                    if SS.collection == del_coll:
                        SS.collection = None
                    st.success(f"Deleted {del_coll}.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        else:
            st.caption("No vector collections currently exist.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    if st.button("Save Environment Configuration"):
        new_lines = []
        for line in env_text.splitlines():
            k = line.split("=")[0].strip()
            if k == "GEMINI_API_KEY": new_lines.append(f"GEMINI_API_KEY={g_key}")
            elif k == "GROQ_API_KEY": new_lines.append(f"GROQ_API_KEY={gr_key}")
            elif k == "GEMINI_MODEL": new_lines.append(f"GEMINI_MODEL={dm_g}")
            elif k == "GROQ_MODEL": new_lines.append(f"GROQ_MODEL={dm_gr}")
            elif k == "QDRANT_URL": new_lines.append(f"QDRANT_URL={qurl}")
            elif k == "QDRANT_API_KEY": new_lines.append(f"QDRANT_API_KEY={qkey}")
            else: new_lines.append(line)

        # Ensure critical keys exist
        keys_present = {line.split("=")[0].strip() for line in new_lines}
        if "GEMINI_API_KEY" not in keys_present: new_lines.append(f"GEMINI_API_KEY={g_key}")
        if "GROQ_API_KEY" not in keys_present: new_lines.append(f"GROQ_API_KEY={gr_key}")
        if "GEMINI_MODEL" not in keys_present: new_lines.append(f"GEMINI_MODEL={dm_g}")
        if "GROQ_MODEL" not in keys_present: new_lines.append(f"GROQ_MODEL={dm_gr}")
        if "QDRANT_URL" not in keys_present: new_lines.append(f"QDRANT_URL={qurl}")

        env_path.write_text("\n".join(new_lines))
        st.success(".env updated successfully.")


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT (ENGINE BLUEPRINT & ASYMMETRIC SPECS)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("""
    <div class='bento-hero' style='margin-bottom:1.5rem;'>
        <div class='bento-hero-title'>CodeLens AI <span>Engine Blueprint</span></div>
        <p class='bento-hero-sub'>
            High-performance Retrieval-Augmented Generation platform built specifically for AST-level understanding of complex software architectures.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Asymmetric Bento Features Grid (Unequal weights: 2:1 and 1:2) ────────
    st.markdown("<div class='sec-title'>Core Capabilities and Architecture</div>", unsafe_allow_html=True)
    f_row1_a, f_row1_b = st.columns([1.8, 1.2])
    with f_row1_a:
        st.markdown("""
        <div class='bento-card' style='height:100%;'>
            <div style='font-size:0.75rem; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem;'>SYNTACTIC PARSING</div>
            <div style='font-size:1.05rem; font-weight:800; color:#0f172a;'>Tree-sitter AST and Dense Semantic Retrieval</div>
            <div style='font-size:0.85rem; color:#475569; margin-top:0.4rem; line-height:1.6;'>
                Unlike naive line chunkers that split functions in half, CodeLens inspects syntactic Abstract Syntax Trees (AST) to extract logical classes, functions, and module boundaries intact.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f_row1_b:
        st.markdown("""
        <div class='bento-card' style='height:100%;'>
            <div style='font-size:0.75rem; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem;'>VECTOR TRANSFORM</div>
            <div style='font-size:1.05rem; font-weight:800; color:#0f172a;'>Sub-Millisecond Embeddings</div>
            <div style='font-size:0.85rem; color:#475569; margin-top:0.4rem; line-height:1.6;'>
                Powered by <code>all-MiniLM-L6-v2</code> running locally on CPU or CUDA. Transforms code blocks into 384-dimensional dense vectors with zero API latency.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    f_row2_a, f_row2_b = st.columns([1.2, 1.8])
    with f_row2_a:
        st.markdown("""
        <div class='bento-card' style='height:100%;'>
            <div style='font-size:0.75rem; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem;'>STORAGE TOPOLOGY</div>
            <div style='font-size:1.05rem; font-weight:800; color:#0f172a;'>Local or Cloud Qdrant DB</div>
            <div style='font-size:0.85rem; color:#475569; margin-top:0.4rem; line-height:1.6;'>
                Scalable vector database supporting in-memory, embedded disk persistence, or high-throughput remote cloud clusters.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f_row2_b:
        st.markdown("""
        <div class='bento-card' style='height:100%;'>
            <div style='font-size:0.75rem; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.3rem;'>INTELLIGENCE LAYER</div>
            <div style='font-size:1.05rem; font-weight:800; color:#0f172a;'>Dual LLM Synthesis (Gemini + Groq)</div>
            <div style='font-size:0.85rem; color:#475569; margin-top:0.4rem; line-height:1.6;'>
                Retrieved code snippets are injected into strict prompt frames. The LLM generates verified answers and includes file citations and line ranges.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Pipeline Architecture Terminal ────────────────────────────────────────
    st.markdown("<div class='sec-title'>Ingestion and Retrieval Pipeline</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='bento-card' style='padding: 1.4rem 1.6rem; background:#f8fafc; border:1px solid #cbd5e1;'>
        <div style='display:flex; align-items:center; gap:0.4rem; margin-bottom: 0.8rem;'>
            <span class='dot-indicator red'></span>
            <span class='dot-indicator amber'></span>
            <span class='dot-indicator green'></span>
            <span style='font-size: 0.72rem; color: #475569; font-family:"JetBrains Mono",monospace; font-weight:700; margin-left: 0.5rem;'>PIPELINE_SCHEMATIC.SYS</span>
        </div>
        <pre style='color: #0f172a; font-family: "JetBrains Mono", monospace; font-size: 0.82rem; line-height: 1.7; margin:0; overflow-x:auto;'>
  Git Repository / Local Directory
        |
        v
  +--------------+     +--------------+     +-----------------+
  | GitHubLoader |---->| RepoScanner  |---->| CodeChunker     |
  | (git clone)  |     |(filter noise)|     |(AST + Fallback) |
  +--------------+     +--------------+     +--------+--------+
                                                     | code chunks
                                                     v
                                            +-----------------+
                                            | CodeEmbedder    |
                                            | (MiniLM-L6-v2)  |
                                            +--------+--------+
                                                     | 384d vectors
                                                     v
                                            +-----------------+
                                            | Qdrant DB       |
                                            | (Cosine Index)  |
                                            +--------+--------+
                                                     |
  User Prompt ----> MiniLM Embed ----> Similarity Search +
                                                     | top-k snippets
                                                     v
                                            +-----------------+
                                            | Context Builder |
                                            +--------+--------+
                                                     |
                                                     v
                                            +-----------------+
                                            | Gemini / Groq   |
                                            +--------+--------+
                                                     |
                                                     v
                                            Synthesized Answer + Citations
        </pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Tech Stack Matrix ─────────────────────────────────────────────────────
    st.markdown("<div class='sec-title'>Technology Stack</div>", unsafe_allow_html=True)
    ts1, ts2, ts3, ts4 = st.columns([1, 1, 1, 1])
    with ts1:
        st.markdown("""
        <div class='bento-card'>
            <div style='font-weight:700; color:#0f172a; font-size:0.9rem;'>Frontend Studio</div>
            <div style='font-size:0.78rem; color:#64748b; margin-top:0.3rem;'>
                Streamlit 1.41+<br>White Bento System<br>Plus Jakarta & JetBrains
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ts2:
        st.markdown("""
        <div class='bento-card'>
            <div style='font-weight:700; color:#0f172a; font-size:0.9rem;'>AI and Embeddings</div>
            <div style='font-size:0.78rem; color:#64748b; margin-top:0.3rem;'>
                Google Gemini API<br>Groq Ultra-Fast API<br>sentence-transformers
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ts3:
        st.markdown("""
        <div class='bento-card'>
            <div style='font-weight:700; color:#0f172a; font-size:0.9rem;'>Vector Storage</div>
            <div style='font-size:0.78rem; color:#64748b; margin-top:0.3rem;'>
                Qdrant Vector DB<br>Local Embedded Engine<br>Cloud Cluster Client
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ts4:
        st.markdown("""
        <div class='bento-card'>
            <div style='font-weight:700; color:#0f172a; font-size:0.9rem;'>AST and Ingestion</div>
            <div style='font-size:0.78rem; color:#64748b; margin-top:0.3rem;'>
                Tree-sitter Language Pack<br>GitPython Core Loader<br>Pathlib & Regex
            </div>
        </div>
        """, unsafe_allow_html=True)
