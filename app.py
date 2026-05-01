"""Streamlit UI for microbe-model.

Catalog-first design: the primary view is the 5,000 never-cultured candidates,
each annotated with a recommended medium. Secondary tabs let you verify the model
on known organisms or run on a custom genome.

Run:
    uv run --extra ui streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import os  # noqa: E402

import requests  # noqa: E402

from microbe_model import config  # noqa: E402
from microbe_model.train.media_recommender import load_models  # noqa: E402
from recommend import (  # noqa: E402
    _format_recipe_summary,
    _load_genome_features,
    _predict_phenotypes,
)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _is_accession(s: str) -> bool:
    s = s.strip().upper()
    return s.startswith(("GCA_", "GCF_"))


@st.cache_data(ttl=3600, show_spinner=False)
def search_ncbi_assembly(name: str, retmax: int = 10) -> list[dict]:
    """Look up assemblies for an organism name via NCBI E-utilities (JSON REST)."""
    if not name.strip():
        return []
    api_key = os.environ.get("NCBI_API_KEY")
    common_params = {"api_key": api_key} if api_key else {}
    try:
        r = requests.get(
            f"{EUTILS_BASE}/esearch.fcgi",
            params={
                "db": "assembly",
                "term": f"{name}[Organism] AND latest[filter]",
                "retmode": "json",
                "retmax": retmax,
                **common_params,
            },
            timeout=20,
        )
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        r = requests.get(
            f"{EUTILS_BASE}/esummary.fcgi",
            params={
                "db": "assembly",
                "id": ",".join(ids),
                "retmode": "json",
                **common_params,
            },
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
            "size_mb": float(doc.get("total_length") or doc.get("assemblylength") or 0) / 1e6,
            "submitter": str(doc.get("submitterorganization", "")),
        })
    # Prefer complete genomes first
    level_rank = {"Complete Genome": 0, "Chromosome": 1, "Scaffold": 2, "Contig": 3}
    out.sort(key=lambda r: level_rank.get(r["level"], 99))
    return out

st.set_page_config(
    page_title="microbe-model — what to grow it in",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="collapsed",
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
# Known organisms for sanity-check (predicted vs published)
# ──────────────────────────────────────────────────────────────────────
SANITY_ORGANISMS = [
    {
        "accession": "GCF_000005845.2",
        "name": "Escherichia coli K-12 MG1655",
        "known": {
            "optimal_temperature_c": 37.0,
            "optimal_ph": 7.0,
            "oxygen_requirement": "facultative anaerobe",
            "salt_tolerance_pct": 1.0,
            "medium": "LB (Luria-Bertani)",
        },
    },
    {
        "accession": "GCF_000009045.1",
        "name": "Bacillus subtilis 168",
        "known": {
            "optimal_temperature_c": 30.0,
            "optimal_ph": 7.0,
            "oxygen_requirement": "facultative anaerobe",
            "salt_tolerance_pct": 2.0,
            "medium": "LB or Nutrient Broth",
        },
    },
    {
        "accession": "GCF_000091545.1",
        "name": "Thermus thermophilus HB8",
        "known": {
            "optimal_temperature_c": 70.0,
            "optimal_ph": 7.5,
            "oxygen_requirement": "aerobe",
            "salt_tolerance_pct": 0.5,
            "medium": "DSMZ 74 Castenholz TYE",
        },
    },
]


def _phylum_from_taxonomy(tax: str | None) -> str:
    if not isinstance(tax, str):
        return "(unknown)"
    for part in tax.split(";"):
        part = part.strip()
        if part.startswith("p__"):
            return part[3:] or "(unclassified)"
    return "(unknown)"


def _run_inference(target: str):
    """Resolve a genome (accession or path), predict phenotypes + media. Returns dict."""
    feats, acc, n_contigs = _load_genome_features(target)
    feats_series = pd.Series(feats)
    phenotypes = _predict_phenotypes(feats_series)

    models, feature_cols = load_recommender()
    media_meta = load_media_meta()
    recipes = load_recipes()
    name_by_id = dict(
        zip(media_meta["medium_id"].astype(str), media_meta["name"], strict=True)
    )
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
        "accession": acc,
        "n_contigs": n_contigs,
        "n_cds": int(feats["n_predicted_cds"]),
        "gc": float(feats["gc_content"]),
        "phenotypes": phenotypes,
        "media": recs,
    }


# ──────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────
st.title("🦠 microbe-model")
st.markdown("##### Pick a never-cultured microbe and see what to grow it in")
st.caption(
    "5,000 microbes from GTDB that have never been grown in a lab — each one "
    "scored against 24 standard DSMZ media. Browse, filter, pick something to try. "
    "All predictions come from a model trained on 17,000 BacDive strains "
    "(5-fold cross-validation by family)."
)


tab_catalog, tab_test, tab_about = st.tabs(
    ["🦠 Catalog of uncultured microbes", "🔬 Test on a known genome", "📊 How accurate is it?"]
)


# ──────────────────────────────────────────────────────────────────────
# Tab 1 — Catalog (the main product)
# ──────────────────────────────────────────────────────────────────────
with tab_catalog:
    unc_all = load_uncultured().copy()
    unc_all["phylum"] = unc_all["gtdb_taxonomy"].map(_phylum_from_taxonomy)
    unc_all["is_genuinely_uncultured"] = (
        unc_all["ncbi_organism_name"].fillna("").str.lower().str.startswith("uncultured")
    )
    n_genuine = int(unc_all["is_genuinely_uncultured"].sum())

    only_uncultured = st.toggle(
        f"Only show genuinely never-cultured microbes ({n_genuine:,} of {len(unc_all):,})",
        value=True,
        help="GTDB lists 5,000 candidates that aren't in BacDive, but many — like "
             "Mycobacterium clinical isolates — have actually been cultured (just not by "
             "BacDive). When this toggle is on, we restrict to genomes whose NCBI "
             "organism name explicitly starts with 'uncultured'. These are the ones "
             "with no published cultivation conditions, where this tool is genuinely useful.",
    )
    unc = unc_all[unc_all["is_genuinely_uncultured"]] if only_uncultured else unc_all
    unc = unc.copy()

    # Quick-filter pills
    st.markdown("**What kind of microbe are you looking for?**")
    pcols = st.columns(5)
    if "catalog_preset" not in st.session_state:
        st.session_state["catalog_preset"] = "all"
    if pcols[0].button("All 5,000", use_container_width=True):
        st.session_state["catalog_preset"] = "all"
    if pcols[1].button("🔥 Thermophiles (>55°C)", use_container_width=True):
        st.session_state["catalog_preset"] = "thermo"
    if pcols[2].button("❄️ Psychrophiles (<15°C)", use_container_width=True):
        st.session_state["catalog_preset"] = "psychro"
    if pcols[3].button("🚫 Anaerobes", use_container_width=True):
        st.session_state["catalog_preset"] = "anaerobe"
    if pcols[4].button("🧂 Halotolerant (>3% salt)", use_container_width=True):
        st.session_state["catalog_preset"] = "halo"

    preset = st.session_state["catalog_preset"]

    mask = pd.Series(True, index=unc.index)
    if preset == "thermo":
        mask &= unc["pred_optimal_temperature_c"] > 55
    elif preset == "psychro":
        mask &= unc["pred_optimal_temperature_c"] < 15
    elif preset == "anaerobe":
        mask &= unc["pred_oxygen_requirement"].isin(
            ["anaerobe", "obligate anaerobe", "facultative anaerobe"]
        )
    elif preset == "halo":
        mask &= unc["pred_salt_tolerance_pct"] > 3

    with st.expander("⚙️ More filters", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            phyla = sorted(unc["phylum"].dropna().unique().tolist())
            sel_phyla = st.multiselect("Phylum", phyla, default=[])
            min_completeness = st.slider("Min CheckM completeness (%)", 50, 100, 90)
        with c2:
            search = st.text_input("Search organism name", "")

    if sel_phyla:
        mask &= unc["phylum"].isin(sel_phyla)
    mask &= unc["checkm_completeness"] >= min_completeness
    if search:
        mask &= unc["ncbi_organism_name"].fillna("").str.contains(
            search, case=False, na=False
        )

    filtered = unc.loc[mask].copy()
    if "top1_confidence" in filtered.columns:
        filtered = filtered.sort_values("top1_confidence", ascending=False)

    st.markdown(f"**{len(filtered):,}** of {len(unc):,} candidates match.")

    # Compact, focused table — what to grow it in is the headline column
    display = filtered[[
        "genome_accession",
        "ncbi_organism_name",
        "phylum",
        "top1_medium_name",
        "top1_confidence",
        "pred_optimal_temperature_c",
        "pred_optimal_ph",
        "pred_oxygen_requirement",
        "pred_salt_tolerance_pct",
        "checkm_completeness",
    ]].copy()
    # ProgressColumn displays the raw value, so scale 0-1 → 0-100 for percent rendering
    display["top1_confidence"] = (display["top1_confidence"] * 100).round(1)

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        height=520,
        column_config={
            "genome_accession": "Accession",
            "ncbi_organism_name": "Organism",
            "phylum": "Phylum",
            "top1_medium_name": st.column_config.TextColumn("👉 Try this medium", width="medium"),
            "top1_confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=100, format="%.1f%%",
                help="Probability the trained classifier assigns to this medium. "
                     "Higher = the model is more sure. Uncultured strains close to "
                     "well-studied groups (e.g. uncultured Mycobacterium → MIDDLEBROOK) "
                     "score very high; phylogenetically isolated genomes score lower.",
            ),
            "pred_optimal_temperature_c": st.column_config.NumberColumn("T (°C)", format="%.0f"),
            "pred_optimal_ph": st.column_config.NumberColumn("pH", format="%.1f"),
            "pred_oxygen_requirement": "O₂",
            "pred_salt_tolerance_pct": st.column_config.NumberColumn("Salt %", format="%.1f"),
            "checkm_completeness": st.column_config.NumberColumn("CheckM %", format="%.0f"),
        },
    )

    st.caption(
        "📋 **How to read this**: the model predicts each row's optimal growth conditions, "
        "then ranks 24 DSMZ media by predicted growth probability. The "
        "*'👉 Try this medium'* column is its top pick. Confidence is the classifier's "
        "predicted probability. Click a row's accession and paste it in **🔬 Test on a known "
        "genome** to see the full ranked list with recipes and uncertainty intervals."
    )

    csv = filtered.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download filtered set as CSV",
        csv,
        file_name="microbe_model_uncultured_candidates.csv",
        mime="text/csv",
    )


# ──────────────────────────────────────────────────────────────────────
# Tab 2 — Test on a genome (sanity check + custom)
# ──────────────────────────────────────────────────────────────────────
with tab_test:
    st.markdown("### Verify the model on organisms with known biology")
    st.caption(
        "These three organisms have well-published growth conditions. Click "
        "to predict — if the predictions match the literature, the model is working."
    )

    scols = st.columns(len(SANITY_ORGANISMS))
    for col, org in zip(scols, SANITY_ORGANISMS, strict=True):
        with col:
            with st.container(border=True):
                st.markdown(f"**{org['name']}**")
                k = org["known"]
                st.caption(
                    f"Known: {k['optimal_temperature_c']:.0f}°C, "
                    f"pH {k['optimal_ph']:.1f}, "
                    f"{k['oxygen_requirement']}, "
                    f"~{k['salt_tolerance_pct']}% salt → *{k['medium']}*"
                )
                if st.button(f"Predict {org['name'].split()[0]}", key=f"sanity_{org['accession']}", use_container_width=True):
                    st.session_state["test_target"] = org["accession"]
                    st.session_state["test_known"] = org["known"]
                    st.session_state["test_run"] = True

    st.markdown("---")
    st.markdown("### Or test on any organism")
    st.caption(
        "Type an organism name (e.g. *Thermus thermophilus*) or paste an "
        "NCBI assembly accession. We'll resolve names through NCBI Assembly automatically."
    )

    with st.form("name_or_accession_form"):
        query = st.text_input(
            "Organism name or NCBI accession",
            value=st.session_state.get("test_target", ""),
            placeholder='e.g. "Thermus thermophilus" or GCF_000005845.2',
        )
        uploaded = st.file_uploader(
            "…or upload a FASTA file directly",
            type=["fna", "fa", "fasta", "gz"],
        )
        top_k = st.slider("Number of media to show", 3, 15, 5)
        submit = st.form_submit_button("🔎 Search / 🚀 Run", type="primary", use_container_width=True)

    auto = st.session_state.pop("test_run", False)
    known = st.session_state.pop("test_known", None)

    target = None
    if uploaded is not None:
        tmp = ROOT / "data" / "_uploaded" / uploaded.name
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(uploaded.getbuffer())
        target = str(tmp)
    elif submit and query.strip() and _is_accession(query):
        target = query.strip()
    elif submit and query.strip():
        # It's a name — do NCBI search and let the user pick
        with st.spinner(f"Searching NCBI Assembly for '{query.strip()}'…"):
            hits = search_ncbi_assembly(query.strip(), retmax=10)
        if not hits:
            st.warning(
                f"No NCBI Assembly hits for '{query.strip()}'. Try a different "
                "spelling, a broader name (e.g. genus only), or paste an accession directly."
            )
        else:
            st.session_state["ncbi_hits"] = hits
    elif auto:
        target = st.session_state.get("test_target") or None

    # If we have NCBI hits from a previous form submission, render the picker
    hits = st.session_state.get("ncbi_hits", [])
    if hits and target is None:
        st.markdown(f"**{len(hits)} NCBI matches** — pick one to predict:")
        labels = [
            f"**{h['accession']}** — {h['organism']} · {h['level']} · "
            f"{h['size_mb']:.2f} Mb · {h['submitter'] or '—'}"
            for h in hits
        ]
        choice = st.radio(
            "Select assembly",
            options=list(range(len(hits))),
            format_func=lambda i: labels[i],
            label_visibility="collapsed",
        )
        if st.button("🚀 Run on selected assembly", type="primary"):
            target = hits[choice]["accession"]
            st.session_state.pop("ncbi_hits", None)

    if not target and submit and not query.strip() and uploaded is None:
        st.error("Provide a name, accession, or FASTA file.")

    if target:
        with st.spinner(f"Resolving and running on `{target}`…"):
            try:
                result = _run_inference(target)
            except SystemExit as e:
                st.error(str(e))
                st.stop()

        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Genome", result["accession"])
            m2.metric("Contigs", result["n_contigs"])
            m3.metric("Predicted CDS", f"{result['n_cds']:,}")
            m4.metric("GC content", f"{result['gc']:.1%}")

        # If we have known values, show predicted-vs-known table
        if known:
            st.markdown("### Predicted vs published")
            rows = []
            p = result["phenotypes"]
            pt = p.get("optimal_temperature_c", {})
            if pt:
                in_range = (pt.get("low_80", 0) <= known["optimal_temperature_c"] <= pt.get("high_80", 1e9))
                rows.append({
                    "Property": "Optimal temperature",
                    "Predicted": f"{pt['prediction']:.1f}°C ({pt.get('low_80', 0):.1f}–{pt.get('high_80', 0):.1f})",
                    "Published": f"{known['optimal_temperature_c']:.0f}°C",
                    "Check": "✅ within 80% interval" if in_range else "⚠️ outside 80% interval",
                })
            pph = p.get("optimal_ph", {})
            if pph:
                in_range = (pph.get("low_80", 0) <= known["optimal_ph"] <= pph.get("high_80", 1e9))
                rows.append({
                    "Property": "Optimal pH",
                    "Predicted": f"{pph['prediction']:.2f} ({pph.get('low_80', 0):.2f}–{pph.get('high_80', 0):.2f})",
                    "Published": f"{known['optimal_ph']:.1f}",
                    "Check": "✅ within 80% interval" if in_range else "⚠️ outside 80% interval",
                })
            pox = p.get("oxygen_requirement", {})
            if pox:
                match = (pox["prediction"] == known["oxygen_requirement"])
                rows.append({
                    "Property": "Oxygen requirement",
                    "Predicted": f"{pox['prediction']} ({pox['confidence']:.0%})",
                    "Published": known["oxygen_requirement"],
                    "Check": "✅ match" if match else "⚠️ mismatch",
                })
            ps = p.get("salt_tolerance_pct", {})
            if ps:
                in_range = (ps.get("low_80", 0) <= known["salt_tolerance_pct"] <= ps.get("high_80", 1e9))
                rows.append({
                    "Property": "Salt tolerance",
                    "Predicted": f"{ps['prediction']:.2f}% ({ps.get('low_80', 0):.2f}–{ps.get('high_80', 0):.2f})",
                    "Published": f"~{known['salt_tolerance_pct']:.1f}%",
                    "Check": "✅ within 80% interval" if in_range else "⚠️ outside 80% interval",
                })
            if result["media"]:
                top1 = result["media"][0]
                rows.append({
                    "Property": "Top medium",
                    "Predicted": f"{top1['name']} ({top1['confidence']:.0%})",
                    "Published": known["medium"],
                    "Check": "—",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption(
                "✅ in the Check column means predicted and published agree at the "
                "model's stated 80% confidence level. ⚠️ means the model disagrees "
                "with the literature for this organism."
            )
        else:
            st.markdown("### Predicted growth conditions")
            pcols = st.columns(4)
            phen = result["phenotypes"]
            for col, (key, label, unit) in zip(
                pcols,
                [
                    ("optimal_temperature_c", "Temperature", "°C"),
                    ("optimal_ph", "pH", ""),
                    ("oxygen_requirement", "Oxygen", ""),
                    ("salt_tolerance_pct", "Salt tolerance", "%"),
                ],
                strict=True,
            ):
                info = phen.get(key)
                with col:
                    if info is None:
                        st.metric(label, "—")
                    elif info["task"] == "regression":
                        st.metric(label, f"{info['prediction']:.2f}{unit}")
                        low = info.get("low_80")
                        high = info.get("high_80")
                        if low is not None and high is not None:
                            st.caption(f"80% CI: {low:.2f}–{high:.2f}{unit}")
                    else:
                        st.metric(label, info["prediction"])
                        st.caption(f"Confidence: {info['confidence']:.0%}")

        st.markdown(f"### Top {top_k} recommended media")
        for i, r in enumerate(result["media"][:top_k], 1):
            with st.container(border=True):
                a, b = st.columns([3, 1])
                with a:
                    st.markdown(f"**{i}. DSMZ Medium {r['medium_id']} — {r['name']}**")
                    if r["recipe"]:
                        st.caption(f"Top compounds: {r['recipe']}")
                with b:
                    st.progress(min(max(r["confidence"], 0), 1), text=f"{r['confidence']:.0%}")


# ──────────────────────────────────────────────────────────────────────
# Tab 3 — About / accuracy
# ──────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("### How accurate is the model?")
    st.info(
        "Cross-validation by family (the model never sees the same family in train + test). "
        "Numbers below are mean across 5 folds."
    )
    results = load_results()
    if not results:
        st.warning("No results found.")
    else:
        cards = [
            ("Temperature", f"MAE = {results.get('optimal_temperature_c', {}).get('mean_metric', 0):.2f}°C",
             "n = 17,007 strains. Useful: most labs incubate in 5°C steps (25/30/37/45). "
             "An MAE of ~3°C means you'd usually pick the right shelf."),
            ("pH", f"MAE = {results.get('optimal_ph', {}).get('mean_metric', 0):.2f}",
             "n = 4,652 strains. Marginal. Distinguishes ‘acidic’ vs ‘neutral’ vs ‘alkaline’ "
             "but not finer than that."),
            ("Oxygen", f"F1 = {results.get('oxygen_requirement', {}).get('mean_metric', 0):.2f}",
             "n = 10,426 strains, 9 imbalanced classes. Weak — frequent confusion between "
             "aerobe ↔ aerotolerant. Use predictions as a coarse hint, not a definitive answer."),
            ("Salt tolerance", f"MAE = {results.get('salt_tolerance_pct', {}).get('mean_metric', 0):.2f}%",
             "n = 4,793 strains. Decent. Distinguishes freshwater (<1%) from marine (~2.5%) "
             "from halotolerant (>5%)."),
        ]
        ccols = st.columns(4)
        for col, (label, metric, note) in zip(ccols, cards, strict=True):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{label}**")
                    st.markdown(f"##### {metric}")
                    st.caption(note)

    st.markdown("### Methodology")
    st.markdown(
        """
- **Training set:** 17,047 bacterial strains from BacDive with at least one phenotype label
  and a public NCBI genome.
- **Features:** 353 genome statistics — GC content, codon usage, tetranucleotide
  frequencies, amino-acid composition, gene density, predicted-CDS stats.
- **Model:** XGBoost (one model per phenotype target). Quantile regression at α = 0.1, 0.5, 0.9
  for regression targets to produce 80% prediction intervals.
- **Media recommender:** 24 binary classifiers, one per DSMZ medium, trained on
  strain↔medium links from MediaDive.
- **Cross-validation:** 5-fold **GroupKFold by family** — folds split at the family
  level so the model never sees the same family in train and test. This is the
  honest test for whether a prediction generalizes to a phylogenetically novel genome.
- **Uncultured candidates:** 5,000 GTDB MAGs that are not in BacDive, scored against
  the trained models. Genomes filtered to ≥50% CheckM completeness.

We also tested ESM-2 (8M-parameter protein language model) embeddings as
features, alone and combined with v1. Results: ESM-2 alone underperforms v1;
combining wins everywhere but only meaningfully on oxygen requirement (+5% F1).
The remaining error is likely a data ceiling — BacDive only records *successful*
cultivations, never failures.
        """
    )

    eval_path = config.ARTIFACTS / "eval_report.md"
    if eval_path.exists():
        with st.expander("📄 Full v1 eval report (per-family error, correlations)"):
            st.markdown(eval_path.read_text())

    cmp_path = config.ARTIFACTS / "v1_vs_v2_comparison.md"
    if cmp_path.exists():
        with st.expander("📄 v1 vs v2 ESM-2 comparison"):
            st.markdown(cmp_path.read_text())
