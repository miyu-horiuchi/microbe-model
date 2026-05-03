"""FastAPI backend for microbe-model.

Endpoints:
  GET  /api/health           — sanity check
  GET  /api/catalog          — full uncultured catalog as JSON (cached, served gzip-encoded by middleware)
  POST /api/predict          — { target: "GCA_..." | "Thermus thermophilus" | ">my_fasta\\nACGT..." }
                              → predicted phenotypes + ranked media

Static React build is mounted at /. Run:
    uvicorn api.main:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from microbe_model import config  # noqa: E402
from microbe_model.train.media_recommender import load_models  # noqa: E402
from recommend import (  # noqa: E402
    _format_recipe_summary,
    _load_genome_features,
    _predict_phenotypes,
)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

app = FastAPI(title="microbe-model API", version="2.0.0")
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────
# Cached resources (loaded once at startup)
# ──────────────────────────────────────────────────────────────────────
_state: dict[str, Any] = {}


def _phylum(tax: str | None) -> str:
    if not isinstance(tax, str):
        return "—"
    for part in tax.split(";"):
        part = part.strip()
        if part.startswith("p__"):
            return part[3:] or "—"
    return "—"


@app.on_event("startup")
def _load_resources() -> None:
    print("Loading recommender models...")
    _state["models"], _state["feature_cols"] = load_models(ROOT / "models" / "recommender")
    print(f"  → {len(_state['models'])} per-medium classifiers loaded")

    media_meta = pd.read_parquet(config.DATA / "media_metadata.parquet")
    _state["recipes"] = pd.read_parquet(config.DATA / "media_recipes.parquet")
    _state["name_by_id"] = dict(
        zip(media_meta["medium_id"].astype(str), media_meta["name"], strict=True)
    )

    unc = pd.read_parquet(config.ARTIFACTS / "uncultured_predictions.parquet")
    unc["phylum"] = unc["gtdb_taxonomy"].map(_phylum)
    unc["truly_uncultured"] = (
        unc["ncbi_organism_name"].fillna("").str.lower().str.startswith("uncultured")
    )
    _state["catalog"] = unc
    print(f"  → {len(unc):,} catalog rows ({int(unc['truly_uncultured'].sum()):,} truly uncultured)")


# ──────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    target: str  # accession, organism name, or FASTA string starting with > or ACGT
    top_k: int = 8


class CatalogRow(BaseModel):
    accession: str
    name: str
    phylum: str
    completeness: float
    truly_uncultured: bool
    T_opt: float
    pH: float
    O2: str
    O2_conf: float
    salt: float
    top_medium_id: str
    top_medium_name: str
    top_confidence: float
    top2_medium_id: str | None = None
    top2_medium_name: str | None = None
    top2_confidence: float | None = None
    top3_medium_id: str | None = None
    top3_medium_name: str | None = None
    top3_confidence: float | None = None


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "models_loaded": len(_state.get("models", {})),
        "catalog_rows": len(_state.get("catalog", [])),
    }


def _safe_float(v, ndigits=3, default=0.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if pd.isna(f) or f != f:
        return default
    return round(f, ndigits)


def _safe_str(v, default=None):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    return str(v)


@app.get("/api/catalog")
def catalog():
    """Return the full uncultured catalog as a list of dicts. Gzipped by middleware."""
    df = _state["catalog"]
    rows = []
    for _, m in df.iterrows():
        rows.append({
            "accession": _safe_str(m["genome_accession"], "—"),
            "name": _safe_str(m["ncbi_organism_name"]) or _safe_str(m["genome_accession"], "—"),
            "phylum": _safe_str(m["phylum"], "—"),
            "completeness": _safe_float(m["checkm_completeness"], 1),
            "truly_uncultured": bool(m["truly_uncultured"]),
            "T_opt": _safe_float(m["pred_optimal_temperature_c"], 1),
            "pH": _safe_float(m["pred_optimal_ph"], 2),
            "O2": _safe_str(m["pred_oxygen_requirement"], "—"),
            "O2_conf": _safe_float(m.get("pred_oxygen_requirement_confidence"), 3, 0.0),
            "salt": _safe_float(m["pred_salt_tolerance_pct"], 2),
            "top_medium_id": _safe_str(m["top1_medium_id"], "—"),
            "top_medium_name": _safe_str(m["top1_medium_name"], "—"),
            "top_confidence": _safe_float(m["top1_confidence"], 4),
            "top2_medium_id": _safe_str(m.get("top2_medium_id")),
            "top2_medium_name": _safe_str(m.get("top2_medium_name")),
            "top2_confidence": _safe_float(m.get("top2_confidence"), 4) if pd.notna(m.get("top2_confidence")) else None,
            "top3_medium_id": _safe_str(m.get("top3_medium_id")),
            "top3_medium_name": _safe_str(m.get("top3_medium_name")),
            "top3_confidence": _safe_float(m.get("top3_confidence"), 4) if pd.notna(m.get("top3_confidence")) else None,
        })
    return {"count": len(rows), "rows": rows}


@app.get("/api/ncbi-search")
def ncbi_search(q: str, retmax: int = 10):
    """Resolve an organism name to NCBI assembly accessions."""
    if not q.strip():
        return {"hits": []}
    api_key = os.environ.get("NCBI_API_KEY")
    common = {"api_key": api_key} if api_key else {}
    try:
        r = requests.get(
            f"{EUTILS_BASE}/esearch.fcgi",
            params={"db": "assembly", "term": f"{q}[Organism] AND latest[filter]",
                    "retmode": "json", "retmax": retmax, **common},
            timeout=20,
        )
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"hits": []}
        r = requests.get(
            f"{EUTILS_BASE}/esummary.fcgi",
            params={"db": "assembly", "id": ",".join(ids), "retmode": "json", **common},
            timeout=20,
        )
        r.raise_for_status()
        result = r.json().get("result", {})
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"NCBI search failed: {e}") from e
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
    return {"hits": out}


@app.post("/api/predict")
def predict(req: PredictRequest):
    target = req.target.strip()
    if not target:
        raise HTTPException(status_code=400, detail="empty target")

    tmp_path = None
    try:
        # FASTA inline?
        if target.startswith(">") or (len(target) > 200 and set(target.upper()) <= set("ACGTNRYKMSWBDHV>\n\r ")):
            tmp = NamedTemporaryFile(suffix=".fasta", delete=False, mode="w")
            tmp.write(target)
            tmp.close()
            tmp_path = tmp.name
            feats, accession, n_contigs = _load_genome_features(tmp.name)
        else:
            feats, accession, n_contigs = _load_genome_features(target)

        feats_series = pd.Series(feats)
        phenotypes = _predict_phenotypes(feats_series)

        models = _state["models"]
        feature_cols = _state["feature_cols"]
        recipes = _state["recipes"]
        name_by_id = _state["name_by_id"]

        X_pred = feats_series[feature_cols].to_frame().T
        recs = []
        for medium_id, model in models.items():
            proba = float(model.predict_proba(X_pred)[0, 1])
            recs.append({
                "medium_id": str(medium_id),
                "name": name_by_id.get(str(medium_id), "(unknown)"),
                "confidence": round(proba, 4),
                "recipe": _format_recipe_summary(str(medium_id), recipes),
            })
        recs.sort(key=lambda r: r["confidence"], reverse=True)

        return {
            "accession": accession,
            "n_contigs": n_contigs,
            "n_cds": int(feats["n_predicted_cds"]),
            "gc": round(float(feats["gc_content"]), 4),
            "phenotypes": phenotypes,
            "media": recs[: req.top_k],
        }
    except SystemExit as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────────
# Static frontend (mounted last so /api/* routes win)
# ──────────────────────────────────────────────────────────────────────
WEB_BUILD = ROOT / "web" / "dist"
if WEB_BUILD.exists():
    app.mount("/assets", StaticFiles(directory=WEB_BUILD / "assets"), name="assets")

    @app.get("/")
    def root():
        return FileResponse(WEB_BUILD / "index.html")

    @app.get("/{path:path}")
    def spa(path: str):
        # SPA fallback — serve index.html for any non-API route
        target = WEB_BUILD / path
        if target.is_file():
            return FileResponse(target)
        return FileResponse(WEB_BUILD / "index.html")
