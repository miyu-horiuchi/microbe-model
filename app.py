"""Streamlit UI for microbe-model — Prototype A "Lab notebook" implementation.

Visual design from the Claude Design bundle (microbe-ml/project/prototype-a-*).
Warm cream paper, IBM Plex Serif/Sans/Mono, oxidized-iron accent.
Three confidence primitives so the same word never reads the same way twice.

Run:
    uv run --extra ui streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from microbe_model import config  # noqa: E402
from microbe_model.train.media_recommender import load_models  # noqa: E402
from recommend import (  # noqa: E402
    _format_recipe_summary,
    _load_genome_features,
    _predict_phenotypes,
)


# ──────────────────────────────────────────────────────────────────────
# Theme tokens (Prototype A — Lab notebook)
# ──────────────────────────────────────────────────────────────────────
PAPER = "#f5f1e8"
PAPER_DEEP = "#ece6d6"
INK = "#1f1d18"
INK_SOFT = "#5a554a"
INK_FAINT = "#94907f"
RULE = "#d6cdb6"
RULE_SOFT = "#e6dfca"
ACCENT = "#a8521a"
ACCENT_TINT = "#fdf6e8"
POS = "#3f6b3a"
WARN = "#a8521a"
FOCUSED_STRIP = "#ede4cd"
BROAD_STRIP = "#e8e0c8"
O2_COLOR = "#3a7d6e"

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

st.set_page_config(
    page_title="microbe-model — what to grow it in",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ──────────────────────────────────────────────────────────────────────
# CSS — global typography + paper background + restyled widgets
# ──────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Serif:ital,wght@0,400;0,500;1,400;1,500&display=swap" rel="stylesheet">
<style>
:root {{
  --paper: {PAPER};
  --paper-deep: {PAPER_DEEP};
  --ink: {INK};
  --ink-soft: {INK_SOFT};
  --ink-faint: {INK_FAINT};
  --rule: {RULE};
  --rule-soft: {RULE_SOFT};
  --accent: {ACCENT};
  --pos: {POS};
  --warn: {WARN};
  --serif: 'IBM Plex Serif', Georgia, serif;
  --sans:  'IBM Plex Sans', system-ui, sans-serif;
  --mono:  'IBM Plex Mono', ui-monospace, monospace;
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
  background: var(--paper) !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
.stApp {{ background: var(--paper); }}
.block-container {{
  padding: 0 0 4rem 0 !important;
  max-width: 100% !important;
}}
.main .block-container > div:first-child {{ padding-top: 0; }}

body, p, div, span, li, label, .stMarkdown {{
  font-family: var(--sans) !important;
  color: var(--ink);
}}
h1, h2, h3, h4 {{ font-family: var(--serif) !important; color: var(--ink); letter-spacing: -0.01em; }}
code, pre, .mono {{ font-family: var(--mono) !important; }}

/* Hide default streamlit chrome where it conflicts */
[data-testid="stToolbar"] {{ display: none; }}
footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}

/* Tabs — make them feel like the lab-notebook bar */
[data-baseweb="tab-list"] {{
  background: var(--paper) !important;
  border-bottom: 1px solid var(--rule) !important;
  padding: 0 28px !important;
  gap: 0 !important;
}}
[data-baseweb="tab"] {{
  font-family: var(--sans) !important;
  font-size: 13px !important;
  color: var(--ink-faint) !important;
  padding: 12px 18px !important;
  height: auto !important;
  background: transparent !important;
}}
[data-baseweb="tab"][aria-selected="true"] {{
  color: var(--ink) !important;
  font-weight: 500 !important;
}}
[data-baseweb="tab-highlight"] {{
  background: var(--ink) !important;
  height: 2px !important;
}}
[data-baseweb="tab-border"] {{ display: none !important; }}
[data-baseweb="tab-panel"] {{ padding: 0 !important; }}

/* Inputs */
.stTextInput input, .stTextArea textarea, .stNumberInput input {{
  font-family: var(--sans) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 0 !important;
  background: rgba(255,255,255,0.5) !important;
  color: var(--ink) !important;
}}
.stTextInput input::placeholder {{ color: var(--ink-faint); }}
.stTextInput label, .stFileUploader label, .stSelectbox label {{
  font-family: var(--mono) !important; font-size: 11px !important;
  color: var(--ink-soft) !important; letter-spacing: 0.05em;
  text-transform: uppercase;
}}

/* Buttons — base */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
  font-family: var(--mono) !important;
  font-size: 12px !important;
  font-weight: 400 !important;
  border-radius: 2px !important;
  border: 1px solid var(--rule) !important;
  background: transparent !important;
  color: var(--ink) !important;
  letter-spacing: 0.02em;
  padding: 6px 12px !important;
  box-shadow: none !important;
  transition: border-color 120ms, background 120ms;
}}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {{
  border-color: var(--ink) !important;
  background: rgba(0,0,0,0.04) !important;
  color: var(--ink) !important;
}}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
  background: var(--ink) !important;
  color: var(--paper) !important;
  border-color: var(--ink) !important;
}}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {{
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}}

/* File uploader */
[data-testid="stFileUploader"] section {{
  background: rgba(255,255,255,0.4) !important;
  border: 1px dashed var(--rule) !important;
  border-radius: 0 !important;
  padding: 12px !important;
}}

/* Sliders inherit accent */
.stSlider [data-baseweb="slider"] [role="slider"] {{ background: var(--accent) !important; }}
.stSlider [data-baseweb="slider"] > div > div > div {{ background: var(--accent) !important; }}

/* Spinner color */
.stSpinner > div {{ border-top-color: var(--accent) !important; }}

/* Custom card containers */
.lab-card {{
  background: var(--paper);
  border: 1px solid var(--rule);
  padding: 16px 18px;
  border-radius: 2px;
  transition: border-color 120ms;
}}
.lab-card:hover {{ border-color: var(--ink); }}
.lab-card-featured {{
  background: var(--paper);
  border: 1px solid var(--accent);
  padding: 14px 16px;
  border-radius: 2px;
}}

.kicker {{ font-family: var(--mono); font-size: 11px; color: var(--ink-faint); letter-spacing: 0.05em; }}
.kicker-up {{ font-family: var(--mono); font-size: 10px; color: var(--ink-faint); letter-spacing: 0.05em; text-transform: uppercase; }}
.serif-italic {{ font-family: var(--serif); font-style: italic; color: var(--ink); }}
.mono-tag {{ font-family: var(--mono); font-size: 11px; padding: 1px 6px; border: 1px solid var(--accent); color: var(--accent); border-radius: 2px; }}

.notebook-rule {{ border-top: 1px solid var(--rule); height: 0; margin: 24px 0; }}
.notebook-rule-soft {{ border-top: 1px solid var(--rule-soft); height: 0; margin: 16px 0; }}

.section-head {{
  font-family: var(--mono); font-size: 11px; color: var(--ink-soft);
  letter-spacing: 0.08em; text-transform: uppercase;
  display: flex; align-items: center; gap: 10px; margin: 4px 0 12px;
}}
.section-head .rule {{ flex: 1; height: 1px; background: var(--rule); }}

.lab-table {{
  width: 100%; border-collapse: collapse; font-family: var(--sans);
  background: var(--paper); border: 1px solid var(--rule);
}}
.lab-table th {{
  background: var(--paper-deep); border-bottom: 1px solid var(--rule);
  padding: 8px 12px; text-align: left;
  font-family: var(--mono); font-size: 10px; font-weight: 500;
  color: var(--ink-soft); letter-spacing: 0.05em; text-transform: uppercase;
  white-space: nowrap;
}}
.lab-table td {{
  padding: 10px 12px; border-bottom: 1px solid var(--rule-soft);
  font-size: 12px; color: var(--ink); vertical-align: middle;
}}
.lab-table tr:last-child td {{ border-bottom: none; }}
.lab-table tr:hover {{ background: #ede5cd; }}
.lab-table .num {{ font-family: var(--serif); font-weight: 500; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.lab-table .mono {{ font-family: var(--mono); font-size: 11px; }}
.lab-table .organism {{ font-family: var(--serif); font-style: italic; }}

.verdict-box {{
  padding: 14px 16px; background: var(--focused-strip, #ede4cd);
  border: 1px solid var(--rule); border-radius: 2px; margin-bottom: 22px;
}}
.verdict-kicker {{ font-family: var(--mono); font-size: 11px; color: var(--accent);
                  letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 4px; }}
.verdict-text {{ font-family: var(--serif); font-size: 14px; font-style: italic; color: var(--ink); }}

.metric-card {{
  border: 1px solid var(--rule); padding: 16px 18px;
  background: var(--paper); border-radius: 2px;
}}
.metric-num {{
  font-family: var(--serif); font-size: 32px; font-weight: 500;
  font-variant-numeric: tabular-nums; line-height: 1;
}}

/* Mode strip pills */
.mode-strip {{
  display: flex; border-bottom: 1px solid var(--rule);
  margin: 0 -28px;
}}
.mode-pill {{
  flex: 1; padding: 14px 28px; cursor: pointer; border: none;
  background: transparent; border-bottom: 2px solid transparent;
  font-family: var(--sans); font-size: 13px; color: var(--ink-faint);
  text-align: left;
}}
.mode-pill.active {{
  background: var(--focused-strip, #ede4cd);
  border-bottom: 2px solid var(--accent);
  color: var(--ink);
}}

/* Predict bar */
.predict-bar {{
  border: 1px solid var(--rule);
  background: rgba(255,255,255,0.5);
  padding: 14px 18px;
  border-radius: 2px;
  margin: 18px 28px 18px;
}}
.predict-bar-banner {{
  border: 1px solid var(--accent);
  background: var(--accent-tint, #fdf6e8);
  padding: 14px 18px;
  border-radius: 2px;
  margin: 6px 28px 18px;
}}

.pheno-chip {{
  display: inline-flex; align-items: baseline; gap: 4px;
  padding: 3px 8px; margin-right: 6px;
  border: 1px solid var(--rule); border-radius: 999px;
  font-family: var(--mono); font-size: 11px; color: var(--ink-soft);
  background: var(--paper);
}}
.pheno-chip strong {{ font-family: var(--serif); color: var(--ink); font-weight: 500; }}

/* Apply paper background to alert/info boxes */
[data-testid="stAlert"] {{
  background: var(--paper-deep) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 2px !important;
  color: var(--ink) !important;
}}
[data-testid="stAlert"] p {{ color: var(--ink) !important; }}

/* Center container padding for tabs */
.lab-pad {{ padding: 0 28px; }}
.lab-pad-y {{ padding: 18px 28px; }}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────
# Cached loaders
# ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_results():
    p = config.ARTIFACTS / "baseline_results.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    data.pop("__meta__", None)
    return data


@st.cache_resource
def load_recommender():
    return load_models(config.ROOT / "models" / "recommender")


@st.cache_data
def load_uncultured() -> pd.DataFrame:
    return pd.read_parquet(config.ARTIFACTS / "uncultured_predictions.parquet")


@st.cache_data
def load_media_meta() -> pd.DataFrame:
    return pd.read_parquet(config.DATA / "media_metadata.parquet")


@st.cache_data
def load_recipes() -> pd.DataFrame:
    return pd.read_parquet(config.DATA / "media_recipes.parquet")


# ──────────────────────────────────────────────────────────────────────
# Color helpers + confidence primitives
# ──────────────────────────────────────────────────────────────────────
def temp_color(t: float) -> str:
    if t < 15: return "#3b82a6"
    if t < 30: return "#5b8b9c"
    if t < 45: return "#7d8470"
    if t < 60: return "#b06a3b"
    return "#a04020"

def ph_color(p: float) -> str:
    if p < 6: return "#a04020"
    if p < 7.5: return "#7d8470"
    return "#3b82a6"

def salt_color(s: float) -> str:
    if s < 1: return "#7d8470"
    if s < 5: return "#b89048"
    return "#8a5e1f"


def media_conf_bar(value: float, color: str = ACCENT, height: int = 8) -> str:
    pct = max(0, min(100, round(value * 100)))
    return (
        f'<div style="display:flex;align-items:center;gap:8px;min-width:120px;">'
        f'<div style="flex:1;height:{height}px;background:rgba(0,0,0,0.06);border-radius:2px;position:relative;overflow:hidden;">'
        f'<div style="position:absolute;inset:0;width:{pct}%;background:{color};border-radius:2px;"></div></div>'
        f'<span style="font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:12px;font-weight:500;min-width:36px;text-align:right;color:{INK};">{pct}%</span>'
        f'</div>'
    )


def oxygen_conf_arc(value: float, size: int = 36, color: str = O2_COLOR) -> str:
    pct = max(0, min(100, round(value * 100)))
    r = size / 2 - 3
    c = 2 * 3.14159 * r
    arc_len = c * value
    return (
        f'<div style="position:relative;width:{size}px;height:{size}px;flex-shrink:0;display:inline-block;vertical-align:middle;">'
        f'<svg width="{size}" height="{size}" style="transform:rotate(-90deg);">'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{RULE_SOFT}" stroke-width="3" stroke-dasharray="2 2" />'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{arc_len:.2f} {c:.2f}" />'
        f'</svg>'
        f'<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:{10 if size > 30 else 9}px;font-weight:600;font-variant-numeric:tabular-nums;color:{INK};">{pct}</div>'
        f'</div>'
    )


def interval_bar(value: float, lo: float, hi: float, scale_min: float, scale_max: float,
                 color: str, unit: str = "", height: int = 6, show_label: bool = False) -> str:
    rng = scale_max - scale_min or 1
    lo_pct = max(0, min(100, ((lo - scale_min) / rng) * 100))
    hi_pct = max(0, min(100, ((hi - scale_min) / rng) * 100))
    val_pct = max(0, min(100, ((value - scale_min) / rng) * 100))
    inner = (
        f'<div style="height:{height}px;background:{RULE_SOFT};border-radius:999px;position:relative;">'
        f'<div style="position:absolute;left:{lo_pct}%;width:{hi_pct - lo_pct}%;top:0;bottom:0;background:{color};opacity:0.28;border-radius:999px;"></div>'
        f'<div style="position:absolute;left:{val_pct}%;top:-2px;bottom:-2px;width:2px;background:{color};transform:translateX(-1px);"></div>'
        f'</div>'
    )
    if show_label:
        inner += (
            f'<div style="display:flex;justify-content:space-between;font-family:var(--mono);'
            f'font-size:10px;color:{INK_FAINT};margin-top:3px;font-variant-numeric:tabular-nums;">'
            f'<span>{lo}{unit}</span>'
            f'<span style="color:{INK};font-weight:600;">{value}{unit}</span>'
            f'<span>{hi}{unit}</span>'
            f'</div>'
        )
    return f'<div style="width:100%;">{inner}</div>'


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def phylum_from_taxonomy(tax: str | None) -> str:
    if not isinstance(tax, str):
        return "—"
    for part in tax.split(";"):
        part = part.strip()
        if part.startswith("p__"):
            return part[3:] or "—"
    return "—"


def is_accession(s: str) -> bool:
    s = s.strip().upper()
    return s.startswith(("GCA_", "GCF_"))


@st.cache_data(ttl=3600, show_spinner=False)
def search_ncbi_assembly(name: str, retmax: int = 10) -> list[dict]:
    if not name.strip():
        return []
    api_key = os.environ.get("NCBI_API_KEY")
    common = {"api_key": api_key} if api_key else {}
    try:
        r = requests.get(
            f"{EUTILS_BASE}/esearch.fcgi",
            params={"db": "assembly", "term": f"{name}[Organism] AND latest[filter]",
                    "retmode": "json", "retmax": retmax, **common},
            timeout=20,
        )
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        r = requests.get(
            f"{EUTILS_BASE}/esummary.fcgi",
            params={"db": "assembly", "id": ",".join(ids), "retmode": "json", **common},
            timeout=20,
        )
        r.raise_for_status()
        result = r.json().get("result", {})
    except requests.RequestException as e:
        st.error(f"NCBI search failed: {e}")
        return []
    out = []
    for uid in result.get("uids", []):
        doc = result.get(uid, {})
        out.append({
            "accession": str(doc.get("assemblyaccession", "")),
            "organism": str(doc.get("organism", "")),
            "level": str(doc.get("assemblystatus", "")),
        })
    rank = {"Complete Genome": 0, "Chromosome": 1, "Scaffold": 2, "Contig": 3}
    out.sort(key=lambda r: rank.get(r["level"], 99))
    return out


def _compare_card_html(label, pred, lo, hi, pub, unit, color, sm, smax, ok):
    badge = (f'<span style="font-family:var(--mono);font-size:11px;color:{POS};font-weight:500;">✓ in 80% PI</span>'
             if ok else f'<span style="font-family:var(--mono);font-size:11px;color:{WARN};font-weight:500;">△ outside PI</span>') if pub is not None else ""
    pub_marker = ""
    if pub is not None:
        pub_pct = max(0, min(100, ((pub - sm) / (smax - sm)) * 100))
        pub_marker = (
            f'<div style="position:relative;height:0;margin-top:-4px;">'
            f'<div style="position:absolute;left:{pub_pct}%;transform:translateX(-50%);top:8px;'
            f'font-family:var(--mono);font-size:9px;font-weight:500;color:{INK};">↑ pub</div></div>'
        )
    pub_block = (
        f'<div><div class="kicker-up" style="margin-bottom:4px;">published</div>'
        f'<div style="font-family:var(--serif);font-size:22px;font-weight:500;color:{INK};font-variant-numeric:tabular-nums;">'
        f'{pub}{unit}</div>'
        f'<div class="kicker" style="margin-top:2px;">literature</div></div>'
        if pub is not None else
        '<div><div class="kicker-up" style="margin-bottom:4px;">published</div>'
        f'<div style="font-family:var(--serif);font-size:22px;color:{INK_FAINT};">—</div></div>'
    )
    return f"""
<div class="lab-card">
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px;">
    <span class="kicker-up">{label}</span>{badge}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:center;">
    <div>
      <div class="kicker-up" style="margin-bottom:4px;">predicted</div>
      <div style="font-family:var(--serif);font-size:22px;font-weight:500;color:{color};font-variant-numeric:tabular-nums;">{pred:.1f}{unit}</div>
      <div class="kicker" style="margin-top:2px;">{lo:.1f}{unit} – {hi:.1f}{unit}</div>
    </div>
    {pub_block}
  </div>
  <div style="margin-top:12px;">
    {interval_bar(pred, lo, hi, sm, smax, color, unit)}
    {pub_marker}
  </div>
</div>
"""


def _oxygen_compare_card(pred, conf, pub):
    ok = (pub is not None and pred == pub)
    badge = (f'<span style="font-family:var(--mono);font-size:11px;color:{POS};font-weight:500;">✓ match</span>'
             if ok else f'<span style="font-family:var(--mono);font-size:11px;color:{WARN};font-weight:500;">△ mismatch</span>') if pub else ""
    pub_block = (
        f'<div><div class="kicker-up" style="margin-bottom:4px;">published</div>'
        f'<div style="font-family:var(--sans);font-size:16px;font-weight:500;color:{INK};">{pub}</div></div>'
        if pub else
        f'<div><div class="kicker-up" style="margin-bottom:4px;">published</div>'
        f'<div style="font-family:var(--sans);font-size:16px;color:{INK_FAINT};">—</div></div>'
    )
    return f"""
<div class="lab-card">
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px;">
    <span class="kicker-up">Oxygen requirement</span>{badge}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:center;">
    <div>
      <div class="kicker-up" style="margin-bottom:4px;">predicted</div>
      <div style="display:flex;align-items:center;gap:8px;">
        {oxygen_conf_arc(conf, size=32)}
        <div style="font-family:var(--sans);font-size:14px;color:{INK};">{pred}</div>
      </div>
    </div>
    {pub_block}
  </div>
</div>
"""


def run_inference(target: str):
    feats, acc, n_contigs = _load_genome_features(target)
    feats_series = pd.Series(feats)
    phenotypes = _predict_phenotypes(feats_series)
    models, feature_cols = load_recommender()
    media_meta = load_media_meta()
    recipes = load_recipes()
    name_by_id = dict(zip(media_meta["medium_id"].astype(str), media_meta["name"], strict=True))
    X_pred = feats_series[feature_cols].to_frame().T
    recs = []
    for medium_id, model in models.items():
        proba = float(model.predict_proba(X_pred)[0, 1])
        recs.append({
            "medium_id": medium_id,
            "name": name_by_id.get(medium_id, "(unknown)"),
            "confidence": proba,
            "recipe": _format_recipe_summary(medium_id, recipes),
        })
    recs.sort(key=lambda r: r["confidence"], reverse=True)
    return {
        "accession": acc, "n_contigs": n_contigs,
        "n_cds": int(feats["n_predicted_cds"]),
        "gc": float(feats["gc_content"]),
        "phenotypes": phenotypes, "media": recs,
    }


# ──────────────────────────────────────────────────────────────────────
# Header (lab-notebook style)
# ──────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div style="border-bottom:1px solid {RULE};padding:18px 28px 16px;background:{PAPER};">
  <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px;">
    <div style="position:relative;width:18px;height:18px;border-radius:50%;background:{ACCENT};top:3px;display:inline-block;">
      <div style="position:absolute;inset:4px;border-radius:50%;background:{PAPER};"></div>
      <div style="position:absolute;inset:7px;border-radius:50%;background:{ACCENT};"></div>
    </div>
    <h1 style="font:500 22px/1 var(--serif);margin:0;color:{INK};">microbe-model</h1>
    <span style="font:400 11px/1 var(--mono);color:{INK_FAINT};letter-spacing:0.04em;">v1.2.0 · trained 2026-03-14</span>
  </div>
  <p style="font:400 13.5px/1.5 var(--sans);color:{INK_SOFT};margin:8px 0 0;max-width:640px;">
    Predicted growth conditions for microbes that have never been cultured. Pick one. Try the medium.
    Five thousand candidates from GTDB scored against twenty-four DSMZ media.
  </p>
</div>
""",
    unsafe_allow_html=True,
)


tab_catalog, tab_test, tab_about = st.tabs(
    ["Catalog", "Test on a known genome", "Model accuracy"]
)


# ──────────────────────────────────────────────────────────────────────
# Tab 1 — Catalog
# ──────────────────────────────────────────────────────────────────────
with tab_catalog:
    unc_all = load_uncultured().copy()
    unc_all["phylum"] = unc_all["gtdb_taxonomy"].map(phylum_from_taxonomy)
    unc_all["truly_uncultured"] = (
        unc_all["ncbi_organism_name"].fillna("").str.lower().str.startswith("uncultured")
    )
    n_focused = int(unc_all["truly_uncultured"].sum())
    n_total = len(unc_all)

    if "mode" not in st.session_state:
        st.session_state["mode"] = "focused"
    if "filter" not in st.session_state:
        st.session_state["filter"] = "all"

    # Mode strip — two big pills with chrome shift
    mode = st.session_state["mode"]
    focused = mode == "focused"
    mc1, mc2 = st.columns(2, gap="small")
    with mc1:
        if st.button(
            f"1,294   truly never-cultured" + ("   · NCBI name starts with \"uncultured\"" if focused else ""),
            key="mode_focused",
            type="primary" if focused else "secondary",
            use_container_width=True,
        ):
            st.session_state["mode"] = "focused"
            st.rerun()
    with mc2:
        if st.button(
            f"5,000   all candidates" + ("   · includes 3,706 named-but-absent-from-BacDive" if not focused else ""),
            key="mode_broad",
            type="primary" if not focused else "secondary",
            use_container_width=True,
        ):
            st.session_state["mode"] = "broad"
            st.rerun()

    # ──────────────── Predict bar ────────────────
    st.markdown('<div class="lab-pad">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="kicker-up" style="margin:18px 0 6px;">Predict a medium</div>',
        unsafe_allow_html=True,
    )
    pcol1, pcol2, pcol3 = st.columns([5, 2, 2])
    with pcol1:
        query = st.text_input(
            label="predict query",
            label_visibility="collapsed",
            placeholder='Organism name, NCBI accession, or paste FASTA…',
            key="predict_query",
        )
    with pcol2:
        upload = st.file_uploader(
            label="upload",
            label_visibility="collapsed",
            type=["fna", "fa", "fasta", "gz"],
            key="predict_upload",
        )
    with pcol3:
        submit = st.button("🔎 Predict", type="primary", use_container_width=True)

    quick = st.columns([1, 1, 1, 6])
    with quick[0]:
        if st.button("Try: Thermus thermophilus", key="qt_thermus"):
            st.session_state["predict_target"] = "Thermus thermophilus"
            st.session_state["run_predict"] = True
    with quick[1]:
        if st.button("Try: E. coli K-12", key="qt_ecoli"):
            st.session_state["predict_target"] = "GCF_000005845.2"
            st.session_state["run_predict"] = True
    with quick[2]:
        if st.button("Try: B. subtilis 168", key="qt_bsub"):
            st.session_state["predict_target"] = "GCF_000009045.1"
            st.session_state["run_predict"] = True

    # Run prediction if requested
    target = None
    if upload is not None and submit:
        tmp = ROOT / "data" / "_uploaded" / upload.name
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(upload.getbuffer())
        target = str(tmp)
    elif submit and query.strip() and is_accession(query):
        target = query.strip()
    elif submit and query.strip():
        with st.spinner(f"Searching NCBI for '{query.strip()}'…"):
            hits = search_ncbi_assembly(query.strip(), retmax=10)
        if not hits:
            st.warning(f"No NCBI Assembly hits for '{query.strip()}'.")
        else:
            st.session_state["ncbi_hits"] = hits
    elif st.session_state.pop("run_predict", False):
        target = st.session_state.pop("predict_target")
        if not is_accession(target):
            with st.spinner(f"Searching NCBI for '{target}'…"):
                hits = search_ncbi_assembly(target, retmax=5)
            if hits:
                target = hits[0]["accession"]

    hits = st.session_state.get("ncbi_hits", [])
    if hits and not target:
        st.markdown(f'<div class="kicker-up" style="margin-top:10px;">{len(hits)} NCBI matches</div>', unsafe_allow_html=True)
        labels = [f"{h['accession']} — {h['organism']} · {h['level']}" for h in hits]
        choice = st.radio("pick", options=list(range(len(hits))), format_func=lambda i: labels[i],
                          label_visibility="collapsed", key="ncbi_choice")
        if st.button("Run on selected", type="primary"):
            target = hits[choice]["accession"]
            st.session_state.pop("ncbi_hits", None)

    if target:
        with st.spinner(f"Predicting for {target}…"):
            try:
                result = run_inference(target)
            except SystemExit as e:
                st.error(str(e))
                st.stop()
        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if result:
        p = result["phenotypes"]
        top = result["media"][0] if result["media"] else None
        T = p.get("optimal_temperature_c", {})
        pH = p.get("optimal_ph", {})
        O2 = p.get("oxygen_requirement", {})
        salt = p.get("salt_tolerance_pct", {})
        st.markdown(
            f"""
<div class="predict-bar-banner">
  <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
    <div>
      <div class="kicker-up">Prediction · {result['accession']}</div>
      <div style="font-family:var(--serif);font-size:15px;font-weight:500;color:{INK};margin-top:2px;">
        Try <span style="color:{ACCENT};">{top['name'] if top else '—'}</span>
        <span class="mono-tag" style="margin-left:8px;">{top['medium_id'] if top else ''}</span>
      </div>
    </div>
    <div style="margin-left:auto;display:flex;gap:6px;flex-wrap:wrap;">
      <span class="pheno-chip">T <strong>{T.get('prediction', 0):.0f}°C</strong></span>
      <span class="pheno-chip">pH <strong>{pH.get('prediction', 0):.1f}</strong></span>
      <span class="pheno-chip">O₂ <strong>{O2.get('prediction', '—')}</strong></span>
      <span class="pheno-chip">salt <strong>{salt.get('prediction', 0):.1f}%</strong></span>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        with st.expander("Full prediction · phenotype intervals + ranked media", expanded=False):
            ic = st.columns(4)
            for col, (key, label, unit, scale) in zip(
                ic,
                [
                    ("optimal_temperature_c", "T_opt", "°C", (0, 110)),
                    ("optimal_ph", "pH", "", (2, 11)),
                    ("oxygen_requirement", "O₂", "", None),
                    ("salt_tolerance_pct", "salt", "%", (0, 25)),
                ],
                strict=True,
            ):
                info = p.get(key) or {}
                with col:
                    if info.get("task") == "regression":
                        v, lo, hi = info["prediction"], info.get("low_80"), info.get("high_80")
                        c = (temp_color(v) if "temp" in key else
                             ph_color(v) if "ph" in key else salt_color(v))
                        st.markdown(
                            f"""
<div class="metric-card">
  <div class="kicker-up">{label}</div>
  <div class="metric-num" style="color:{c};">{v:.1f}<span style="font-family:var(--mono);font-size:12px;color:{INK_SOFT};margin-left:4px;">{unit}</span></div>
  <div style="margin-top:8px;">{interval_bar(v, lo or v, hi or v, scale[0], scale[1], c, unit, show_label=True)}</div>
</div>
""",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
<div class="metric-card">
  <div class="kicker-up">{label}</div>
  <div style="display:flex;align-items:center;gap:10px;margin-top:8px;">
    {oxygen_conf_arc(info.get("confidence", 0), size=40)}
    <div style="font-family:var(--sans);font-size:13px;color:{INK};">{info.get("prediction", "—")}</div>
  </div>
</div>
""",
                            unsafe_allow_html=True,
                        )
            st.markdown('<div class="notebook-rule-soft"></div>', unsafe_allow_html=True)
            st.markdown('<div class="kicker-up">Top media</div>', unsafe_allow_html=True)
            for i, r in enumerate(result["media"][:5], 1):
                st.markdown(
                    f"""
<div class="lab-card" style="margin-bottom:8px;{'border-color:'+ACCENT+';background:'+ACCENT_TINT+';' if i==1 else ''}">
  <div style="display:flex;align-items:center;gap:10px;">
    <span class="mono-tag">{r['medium_id']}</span>
    <span style="flex:1;font-family:var(--sans);font-size:13px;color:{INK};">{r['name']}</span>
    {media_conf_bar(r['confidence'])}
  </div>
  {f'<div style="margin-top:6px;font-family:var(--mono);font-size:11px;color:{INK_SOFT};line-height:1.4;">{r["recipe"]}</div>' if r['recipe'] else ''}
</div>
""",
                    unsafe_allow_html=True,
                )
            if st.button("Clear prediction", key="clear_pred"):
                st.session_state.pop("last_result", None)
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # close lab-pad

    # ──────────────── Quick filters ────────────────
    filter_opts = [
        ("all", "All"),
        ("thermo", "Thermophiles · >55°C"),
        ("psychro", "Psychrophiles · <15°C"),
        ("anaerobe", "Anaerobes"),
        ("halo", "Halotolerant · >3% NaCl"),
    ]
    fcols = st.columns([1, 1.4, 1.4, 1, 1.4, 4])
    for i, (key, label) in enumerate(filter_opts):
        with fcols[i]:
            if st.button(
                label, key=f"filter_{key}",
                type="primary" if st.session_state["filter"] == key else "secondary",
                use_container_width=True,
            ):
                st.session_state["filter"] = key
                st.rerun()

    unc = unc_all[unc_all["truly_uncultured"]] if focused else unc_all
    f = st.session_state["filter"]
    if f == "thermo":
        unc = unc[unc["pred_optimal_temperature_c"] > 55]
    elif f == "psychro":
        unc = unc[unc["pred_optimal_temperature_c"] < 15]
    elif f == "anaerobe":
        unc = unc[unc["pred_oxygen_requirement"].fillna("").str.contains("anaerobe", case=False)]
    elif f == "halo":
        unc = unc[unc["pred_salt_tolerance_pct"] > 3]

    # Search row
    st.markdown('<div class="lab-pad" style="padding-bottom:8px;">', unsafe_allow_html=True)
    sc1, sc2 = st.columns([4, 1])
    with sc1:
        search = st.text_input(
            label="search",
            label_visibility="collapsed",
            placeholder="⌕  filter by organism name…",
            key="catalog_search",
        )
    with sc2:
        st.markdown(
            f'<div style="text-align:right;font-family:var(--mono);font-size:12px;color:{INK_FAINT};padding-top:6px;">'
            f'showing <span style="color:{INK};font-weight:500;">{len(unc):,}</span> · sorted by confidence</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if search:
        unc = unc[unc["ncbi_organism_name"].fillna("").str.contains(search, case=False, na=False)]
    if "top1_confidence" in unc.columns:
        unc = unc.sort_values("top1_confidence", ascending=False)

    # ──────────────── Top picks (cards) ────────────────
    featured = unc.head(6)
    rest = unc.iloc[6:]

    if len(featured):
        st.markdown(
            f'<div class="lab-pad" style="padding-top:18px;">'
            f'<div class="section-head"><span>Top {len(featured)} picks</span><span class="rule"></span><span style="text-transform:none;color:{INK_FAINT};">by media confidence</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        cards_html = ['<div class="lab-pad" style="display:grid;grid-template-columns:repeat(3, 1fr);gap:14px;">']
        for _, m in featured.iterrows():
            T = float(m["pred_optimal_temperature_c"])
            ph = float(m["pred_optimal_ph"])
            slt = float(m["pred_salt_tolerance_pct"])
            o2_lbl = m["pred_oxygen_requirement"] or "—"
            o2_conf = float(m.get("pred_oxygen_requirement_confidence") or 0)
            top_id = m["top1_medium_id"]
            top_name = m["top1_medium_name"]
            top_conf = float(m["top1_confidence"])
            short = (m["ncbi_organism_name"] or m["genome_accession"])[:80]
            cards_html.append(f"""
<div class="lab-card">
  <div class="kicker" style="margin-bottom:4px;">{m['genome_accession']} · {m['phylum']}</div>
  <div class="serif-italic" style="font-size:14.5px;font-weight:500;line-height:1.25;margin-bottom:12px;">{short}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px 14px;padding:10px 0;border-top:1px solid {RULE_SOFT};border-bottom:1px solid {RULE_SOFT};">
    <div>
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">
        <span class="kicker-up">T_opt</span>
        <span style="font-family:var(--serif);font-weight:500;font-size:13px;font-variant-numeric:tabular-nums;color:{INK};">{T:.0f}°C</span>
      </div>
      {interval_bar(T, max(0, T - 5), min(110, T + 5), 0, 110, temp_color(T))}
    </div>
    <div>
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">
        <span class="kicker-up">pH</span>
        <span style="font-family:var(--serif);font-weight:500;font-size:13px;font-variant-numeric:tabular-nums;color:{INK};">{ph:.1f}</span>
      </div>
      {interval_bar(ph, max(2, ph - 0.5), min(11, ph + 0.5), 2, 11, ph_color(ph))}
    </div>
    <div>
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">
        <span class="kicker-up">salt</span>
        <span style="font-family:var(--serif);font-weight:500;font-size:13px;font-variant-numeric:tabular-nums;color:{INK};">{slt:.1f}%</span>
      </div>
      {interval_bar(slt, max(0, slt - 1), min(25, slt + 1), 0, 25, salt_color(slt))}
    </div>
    <div>
      <div class="kicker-up" style="margin-bottom:4px;">O₂</div>
      <div style="display:flex;align-items:center;gap:8px;">
        {oxygen_conf_arc(o2_conf, size=28)}
        <div style="font-family:var(--sans);font-size:12px;color:{INK};line-height:1.2;">{o2_lbl}</div>
      </div>
    </div>
  </div>
  <div style="padding-top:10px;">
    <div class="kicker-up" style="margin-bottom:6px;">Try this medium</div>
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
      <span class="mono-tag">{top_id}</span>
      <span style="flex:1;font-family:var(--sans);font-size:13px;color:{INK};overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{top_name}</span>
    </div>
    {media_conf_bar(top_conf)}
  </div>
</div>
""")
        cards_html.append("</div>")
        st.markdown("\n".join(cards_html), unsafe_allow_html=True)

    # ──────────────── Rest as table ────────────────
    if len(rest):
        st.markdown(
            f'<div class="lab-pad" style="padding-top:24px;">'
            f'<div class="section-head"><span>Remaining {len(rest):,}</span><span class="rule"></span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        table_rows = []
        for _, m in rest.head(80).iterrows():
            T = float(m["pred_optimal_temperature_c"])
            ph = float(m["pred_optimal_ph"])
            slt = float(m["pred_salt_tolerance_pct"])
            o2_lbl = m["pred_oxygen_requirement"] or "—"
            o2_conf = float(m.get("pred_oxygen_requirement_confidence") or 0)
            short = (m["ncbi_organism_name"] or "")[:60]
            table_rows.append(f"""
<tr>
  <td class="mono">{m['genome_accession']}</td>
  <td class="organism">{short}</td>
  <td>{m['phylum']}</td>
  <td><span class="mono-tag">{m['top1_medium_id']}</span> <span style="font-size:11.5px;">{m['top1_medium_name'][:38]}</span></td>
  <td>{media_conf_bar(float(m['top1_confidence']))}</td>
  <td class="num" style="color:{temp_color(T)};">{T:.0f}°C</td>
  <td class="num" style="color:{ph_color(ph)};">{ph:.1f}</td>
  <td><div style="display:flex;align-items:center;gap:6px;">{oxygen_conf_arc(o2_conf, size=20)}<span style="font-size:11.5px;">{o2_lbl}</span></div></td>
  <td class="num" style="color:{salt_color(slt)};">{slt:.1f}%</td>
  <td class="mono">{float(m['checkm_completeness']):.0f}</td>
</tr>""")
        st.markdown(
            f'<div class="lab-pad" style="padding-bottom:24px;">'
            f'<table class="lab-table">'
            f'<thead><tr>{"".join(f"<th>{h}</th>" for h in ["Accession", "Organism", "Phylum", "Try this medium", "Conf.", "T", "pH", "O₂", "Salt", "CheckM"])}</tr></thead>'
            f'<tbody>{"".join(table_rows)}</tbody></table>'
            f'<div style="text-align:center;padding:10px 0;font-family:var(--mono);font-size:11px;color:{INK_FAINT};">'
            f'showing first 80 of {len(rest):,} remaining · use search and filters to narrow'
            f'</div></div>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────
# Tab 2 — Test on a known genome
# ──────────────────────────────────────────────────────────────────────
SANITY_ORGANISMS = [
    {
        "accession": "GCF_000005845.2", "name": "Escherichia coli K-12 MG1655",
        "known": {"T_opt": 37.0, "pH": 7.0, "O2": "facultative anaerobe", "salt": 1.0,
                  "medium": "LB (Luria-Bertani)"},
    },
    {
        "accession": "GCF_000009045.1", "name": "Bacillus subtilis 168",
        "known": {"T_opt": 30.0, "pH": 7.0, "O2": "facultative anaerobe", "salt": 2.0,
                  "medium": "LB or Nutrient Broth"},
    },
    {
        "accession": "GCF_000091545.1", "name": "Thermus thermophilus HB8",
        "known": {"T_opt": 70.0, "pH": 7.5, "O2": "aerobe", "salt": 0.5,
                  "medium": "DSMZ 74 Castenholz TYE"},
    },
]


with tab_test:
    st.markdown(
        f'<div class="lab-pad" style="padding-top:18px;padding-bottom:14px;">'
        f'<div class="kicker">Sanity-check the model on a microbe with published growth conditions.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="lab-pad">', unsafe_allow_html=True)
    pcols = st.columns(3)
    for col, org in zip(pcols, SANITY_ORGANISMS, strict=True):
        with col:
            k = org["known"]
            st.markdown(
                f"""
<div class="lab-card" style="margin-bottom:6px;">
  <div class="serif-italic" style="font-size:13.5px;font-weight:500;margin-bottom:4px;">{org['name']}</div>
  <div class="kicker">{org['accession']}</div>
  <div style="display:flex;gap:10px;margin-top:8px;font-family:var(--mono);font-size:11px;color:{INK_SOFT};">
    <span>{k['T_opt']:.0f}°C</span>
    <span>pH {k['pH']:.1f}</span>
    <span>{k['O2']}</span>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            if st.button(f"Predict {org['name'].split()[0]}", key=f"sanity_{org['accession']}", use_container_width=True):
                st.session_state["test_target"] = org["accession"]
                st.session_state["test_known"] = org["known"]
                st.session_state["test_run"] = True

    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)

    with st.form("test_form", clear_on_submit=False):
        tcol1, tcol2 = st.columns([5, 2])
        with tcol1:
            t_query = st.text_input(
                label="test query",
                label_visibility="collapsed",
                placeholder="⌕  organism name or NCBI accession…",
                value=st.session_state.get("test_target", ""),
            )
        with tcol2:
            t_upload = st.file_uploader("test upload", type=["fna", "fa", "fasta", "gz"], label_visibility="collapsed")
        t_submit = st.form_submit_button("Run", type="primary", use_container_width=True)

    auto = st.session_state.pop("test_run", False)
    known = st.session_state.pop("test_known", None)
    t_target = None
    if t_upload is not None:
        tmp = ROOT / "data" / "_uploaded" / t_upload.name
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(t_upload.getbuffer())
        t_target = str(tmp)
    elif t_submit and t_query.strip() and is_accession(t_query):
        t_target = t_query.strip()
    elif t_submit and t_query.strip():
        with st.spinner(f"Searching NCBI for '{t_query.strip()}'…"):
            t_hits = search_ncbi_assembly(t_query.strip(), retmax=5)
        if t_hits:
            t_target = t_hits[0]["accession"]
        else:
            st.warning(f"No NCBI hits for '{t_query.strip()}'.")
    elif auto:
        t_target = st.session_state.get("test_target")

    if t_target:
        with st.spinner(f"Predicting {t_target}…"):
            try:
                t_result = run_inference(t_target)
            except SystemExit as e:
                st.error(str(e))
                st.stop()

        p = t_result["phenotypes"]
        st.markdown(
            f'<div style="height:8px;"></div>'
            f'<div class="kicker-up">Predicted vs published — {t_result["accession"]}</div>',
            unsafe_allow_html=True,
        )

        cards = []
        # Temperature
        T = p.get("optimal_temperature_c", {})
        if T:
            v, lo, hi = T["prediction"], T.get("low_80", T["prediction"]), T.get("high_80", T["prediction"])
            pub = known["T_opt"] if known else None
            ok = pub is not None and lo <= pub <= hi
            cards.append(_compare_card_html("Optimum temperature", v, lo, hi, pub, "°C", temp_color(v), 0, 110, ok))
        pH = p.get("optimal_ph", {})
        if pH:
            v, lo, hi = pH["prediction"], pH.get("low_80", pH["prediction"]), pH.get("high_80", pH["prediction"])
            pub = known["pH"] if known else None
            ok = pub is not None and lo <= pub <= hi
            cards.append(_compare_card_html("Optimum pH", v, lo, hi, pub, "", ph_color(v), 2, 11, ok))
        slt = p.get("salt_tolerance_pct", {})
        if slt:
            v, lo, hi = slt["prediction"], slt.get("low_80", slt["prediction"]), slt.get("high_80", slt["prediction"])
            pub = known["salt"] if known else None
            ok = pub is not None and lo <= pub <= hi
            cards.append(_compare_card_html("Salt tolerance", v, lo, hi, pub, "%", salt_color(v), 0, 25, ok))
        O2 = p.get("oxygen_requirement", {})
        if O2:
            cards.append(_oxygen_compare_card(O2.get("prediction", "—"), O2.get("confidence", 0),
                                              known["O2"] if known else None))
        st.markdown(
            f'<div class="lab-pad" style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">'
            f'{"".join(cards)}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="lab-pad" style="padding-top:18px;">', unsafe_allow_html=True)
        st.markdown('<div class="kicker-up">Top media to try</div>', unsafe_allow_html=True)
        for i, r in enumerate(t_result["media"][:5], 1):
            st.markdown(
                f"""
<div class="lab-card" style="margin-bottom:8px;{'border-color:'+ACCENT+';background:'+ACCENT_TINT+';' if i==1 else ''}">
  <div style="display:flex;align-items:center;gap:10px;">
    <span class="mono-tag">{r['medium_id']}</span>
    <span style="flex:1;font-family:var(--sans);font-size:13px;color:{INK};">{r['name']}</span>
    {media_conf_bar(r['confidence'])}
  </div>
  {f'<div style="margin-top:6px;font-family:var(--mono);font-size:11px;color:{INK_SOFT};line-height:1.4;">{r["recipe"]}</div>' if r['recipe'] else ''}
</div>
""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Tab 3 — Model accuracy
# ──────────────────────────────────────────────────────────────────────
with tab_about:
    results = load_results()
    targets_meta = [
        ("optimal_temperature_c", "Temperature optimum", "MAE", "°C", temp_color(45),
         "Useful — labs incubate in 5°C steps; you'd usually pick the right shelf.",
         "Model rarely misses by a tube. Trust the median; verify edge cases."),
        ("optimal_ph", "pH optimum", "MAE", "", ph_color(7),
         "Marginal — distinguishes acidic / neutral / alkaline, not finer.",
         "Buffer to predicted ±0.5; don't over-interpret tenths."),
        ("oxygen_requirement", "Oxygen requirement", "F1", "", O2_COLOR,
         "Weak — 9 imbalanced classes, frequent aerobe ↔ aerotolerant confusion.",
         "Treat predicted O₂ as a hint; check obligate vs facultative in a tube."),
        ("salt_tolerance_pct", "Salt tolerance", "MAE", "%", salt_color(3),
         "Decent — separates freshwater / marine / halotolerant.",
         "Reasonable for screening; not for fine-tuning compound concentrations."),
    ]

    st.markdown(
        f"""
<div class="lab-pad" style="padding-top:24px;">
  <div class="verdict-box">
    <div class="verdict-kicker">The verdict</div>
    <div class="verdict-text">v1 handcrafted features is the working baseline. Trust temperature and pH; verify oxygen and salt with a tube.</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    cards_html = ['<div class="lab-pad" style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:24px;">']
    for key, label, metric, unit, color, verdict, detail in targets_meta:
        a = results.get(key, {})
        val = a.get("mean_metric", 0)
        cards_html.append(f"""
<div class="metric-card">
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;">
    <span style="font-family:var(--sans);font-size:12px;font-weight:500;color:{INK};">{label}</span>
    <span class="kicker">5-fold GroupKFold by family</span>
  </div>
  <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:8px;">
    <span class="kicker-up">{metric}</span>
    <span class="metric-num" style="color:{color};">{val:.2f}</span>
    <span style="font-family:var(--mono);font-size:12px;color:{INK_SOFT};">{unit}</span>
  </div>
  <div style="font-family:var(--serif);font-size:13px;font-style:italic;color:{INK};margin-bottom:6px;">"{verdict}"</div>
  <div style="font-family:var(--sans);font-size:12px;color:{INK_SOFT};line-height:1.5;">{detail}</div>
</div>
""")
    cards_html.append("</div>")
    st.markdown("\n".join(cards_html), unsafe_allow_html=True)

    # Confidence legend — three primitives
    st.markdown(
        f"""
<div class="lab-pad" style="border-top:1px solid {RULE};padding-top:18px;">
  <div class="section-head"><span>How confidence is calculated</span><span class="rule"></span></div>
  <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:12px;">
    <div class="lab-card">
      {media_conf_bar(0.72)}
      <div style="font-family:var(--sans);font-size:12px;font-weight:500;color:{INK};margin-top:8px;">Media confidence</div>
      <div style="font-family:var(--sans);font-size:11.5px;color:{INK_SOFT};line-height:1.5;margin-top:4px;">
        Per-medium binary classifier <span style="font-family:var(--mono);font-size:11px;">predict_proba</span>.
        Not perfectly calibrated — BacDive only has positive examples.
      </div>
    </div>
    <div class="lab-card">
      {oxygen_conf_arc(0.72, size=36)}
      <div style="font-family:var(--sans);font-size:12px;font-weight:500;color:{INK};margin-top:8px;">Oxygen confidence</div>
      <div style="font-family:var(--sans);font-size:11.5px;color:{INK_SOFT};line-height:1.5;margin-top:4px;">
        Max softmax probability across 9 imbalanced classes. Low values mean the model can't pick between near-neighbour categories.
      </div>
    </div>
    <div class="lab-card">
      {interval_bar(37, 32, 43, 0, 80, temp_color(37), "°C", show_label=True)}
      <div style="font-family:var(--sans);font-size:12px;font-weight:500;color:{INK};margin-top:6px;">Prediction interval</div>
      <div style="font-family:var(--sans);font-size:11.5px;color:{INK_SOFT};line-height:1.5;margin-top:4px;">
        Quantile regression at α=0.1 / 0.9 → 80% PI for T, pH, salt. Wide interval = model uncertain.
      </div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="lab-pad" style="margin-top:24px;font-family:var(--sans);font-size:12px;color:{INK_SOFT};line-height:1.6;padding-bottom:32px;">
  Trained on <span style="color:{INK};font-weight:500;">17,047</span> BacDive strains
  with growth conditions; uncultured catalog is <span style="color:{INK};font-weight:500;">5,000</span> held-out
  GTDB genomes scored against <span style="color:{INK};font-weight:500;">24</span> DSMZ media.
  Features: 353 handcrafted genome statistics — GC, codon usage, tetranucleotide frequencies, AA composition.
  XGBoost classifiers for media; quantile regression XGBoost for prediction intervals.
</div>
""",
        unsafe_allow_html=True,
    )
