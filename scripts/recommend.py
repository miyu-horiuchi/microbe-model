"""Predict cultivation conditions + ranked media recommendations for a single genome.

Usage:
    # By NCBI accession (downloaded on demand)
    uv run python scripts/recommend.py GCA_001580615

    # By local FASTA path
    uv run python scripts/recommend.py path/to/genome.fna.gz

    # Top 10 instead of default 5
    uv run python scripts/recommend.py GCA_001580615 --top-k 10

The script:
  1. Resolves the genome (download from NCBI Datasets if accession given,
     try .1, .2, .3, .4 if no version suffix).
  2. Runs pyrodigal + amino-acid composition + tetranucleotide + codon-usage
     feature extraction (same pipeline as training).
  3. Loads the trained per-medium classifiers from models/recommender/.
  4. Predicts probability for each medium, sorts, prints top-K.
  5. Trains phenotype heads on the fly (T_opt, pH, oxygen, salt) using all of
     BacDive and predicts for the input genome.

Output is human-readable to stdout. Use --json to get machine-readable JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from microbe_model import config
from microbe_model.features.genome import extract_features_from_seqs, read_fasta_records
from microbe_model.pipeline import _fetch_fasta_bytes
from microbe_model.train.media_recommender import load_models


def _load_genome_features(target: str) -> tuple[dict, str, int]:
    """Resolve `target` (path or accession) into a feature dict + accession + n_contigs."""
    p = Path(target)
    if p.exists() and (p.suffix in {".fna", ".fa", ".fasta"} or p.name.endswith((".fna.gz", ".fa.gz", ".fasta.gz"))):
        contigs = list(read_fasta_records(p))
        accession = p.stem
    else:
        # Treat as NCBI accession
        contigs = _fetch_fasta_bytes(target)
        if not contigs:
            sys.exit(f"Could not fetch genome for accession: {target}\n"
                     f"(Tried .1, .2, .3, .4. Check the accession and try again.)")
        accession = target
    feats = extract_features_from_seqs(contigs)
    return feats, accession, len(contigs)


def _format_recipe_summary(medium_id: str, recipes: pd.DataFrame, max_compounds: int = 5) -> str:
    """Render the top-N compounds of a medium for display."""
    sub = recipes[recipes["medium_id"].astype(str) == str(medium_id)]
    sub = sub.sort_values("g_l", ascending=False, na_position="last")
    if sub.empty:
        return ""
    parts = []
    for _, row in sub.head(max_compounds).iterrows():
        compound = row["compound"]
        amount = row.get("amount")
        unit = row.get("unit")
        if pd.notna(amount) and pd.notna(unit):
            parts.append(f"{compound} {amount}{unit}")
        else:
            parts.append(str(compound))
    return ", ".join(parts)


def _predict_phenotypes(features_vec: pd.Series) -> dict[str, dict]:
    """Train fresh phenotype heads on all of BacDive, predict for the input.

    For regression targets: trains three models at quantiles 0.1, 0.5, 0.9 to
    emit a median + 80% prediction interval. The interval width is the model's
    own uncertainty estimate — wide intervals mean "I don't know."

    For classification: predict_proba gives a calibration-free confidence.
    """
    pheno = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet")
    feats = pd.read_parquet(config.DATA / "features.parquet")
    df = pheno.merge(feats, on=["bacdive_id", "genome_accession"], how="inner")
    feature_cols = [c for c in features_vec.index if c in df.columns]
    X_pred = features_vec[feature_cols].to_frame().T

    out: dict[str, dict] = {}
    for target, task in config.PHENOTYPE_TARGETS.items():
        if target not in df.columns:
            continue
        labeled = df[df[target].notna()]
        if len(labeled) < 100:
            continue
        X_train = labeled[feature_cols]
        if task == "regression":
            y_train = labeled[target].to_numpy(dtype=float)
            preds: dict[float, float] = {}
            for alpha in (0.1, 0.5, 0.9):
                model = xgb.XGBRegressor(
                    n_estimators=400, max_depth=5, learning_rate=0.05,
                    tree_method="hist", n_jobs=-1,
                    objective="reg:quantileerror", quantile_alpha=alpha,
                )
                model.fit(X_train, y_train)
                preds[alpha] = float(model.predict(X_pred)[0])
            out[target] = {
                "prediction": preds[0.5],
                "low_80": preds[0.1],
                "high_80": preds[0.9],
                "task": "regression",
            }
        else:
            encoder = LabelEncoder()
            y_train_enc = encoder.fit_transform(labeled[target].astype(str))
            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                tree_method="hist", n_jobs=-1, eval_metric="mlogloss",
            )
            model.fit(X_train, y_train_enc)
            pred_idx = int(model.predict(X_pred)[0])
            proba = model.predict_proba(X_pred)[0]
            out[target] = {
                "prediction": str(encoder.inverse_transform([pred_idx])[0]),
                "confidence": float(proba.max()),
                "task": "classification",
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target",
                        help="NCBI assembly accession (e.g. GCA_001580615) "
                             "or path to a FASTA file (.fna / .fna.gz / .fa)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top media to recommend (default 5)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of human-readable")
    args = parser.parse_args()

    models_dir = config.ROOT / "models" / "recommender"
    if not models_dir.exists():
        sys.exit(f"Missing {models_dir}.\n"
                 f"Run scripts/10_train_media_recommender.py first to train and persist models.")

    print(f"Resolving genome: {args.target}", file=sys.stderr)
    feats, accession, n_contigs = _load_genome_features(args.target)
    feats_series = pd.Series(feats)
    print(f"  → {n_contigs} contigs, {feats['n_predicted_cds']:.0f} predicted CDS, "
          f"GC={feats['gc_content']:.3f}", file=sys.stderr)

    # Media recommendations
    print("Loading recommender models...", file=sys.stderr)
    models, feature_cols = load_models(models_dir)

    media_meta = pd.read_parquet(config.DATA / "media_metadata.parquet")
    name_by_id = dict(zip(media_meta["medium_id"].astype(str), media_meta["name"], strict=True))
    recipes = pd.read_parquet(config.DATA / "media_recipes.parquet")

    X_pred = feats_series[feature_cols].to_frame().T
    recommendations = []
    for medium_id, model in models.items():
        proba = float(model.predict_proba(X_pred)[0, 1])
        recommendations.append({
            "medium_id": medium_id,
            "name": name_by_id.get(medium_id, "(unknown)"),
            "confidence": proba,
            "recipe_summary": _format_recipe_summary(medium_id, recipes),
        })
    recommendations.sort(key=lambda r: r["confidence"], reverse=True)
    top = recommendations[: args.top_k]

    # Phenotype predictions
    print("Training phenotype heads on BacDive (~30s)...", file=sys.stderr)
    phenotypes = _predict_phenotypes(feats_series)

    if args.json:
        print(json.dumps({
            "accession": accession,
            "n_contigs": n_contigs,
            "n_predicted_cds": int(feats["n_predicted_cds"]),
            "gc_content": feats["gc_content"],
            "phenotypes": phenotypes,
            "top_media": top,
        }, indent=2))
        return

    print()
    print(f"Recommendations for {accession}")
    print(f"  ({n_contigs} contigs, {int(feats['n_predicted_cds'])} CDS, "
          f"GC={feats['gc_content']:.1%})")
    print()
    print(f"Top {args.top_k} recommended media:")
    for i, rec in enumerate(top, 1):
        print(f"  {i}. DSMZ Medium {rec['medium_id']:<6}  "
              f"confidence={rec['confidence']:.2f}  {rec['name']}")
        if rec["recipe_summary"]:
            print(f"     ↳ {rec['recipe_summary']}")
    print()
    print("Predicted phenotypes:")
    for target, info in phenotypes.items():
        if info["task"] == "regression":
            unit = "°C" if "temperature" in target else ("%" if "salt" in target else "")
            low = info.get("low_80")
            high = info.get("high_80")
            if low is not None and high is not None:
                print(f"  {target:<30s} {info['prediction']:.2f}{unit}  "
                      f"(80% interval: {low:.2f}-{high:.2f}{unit})")
            else:
                print(f"  {target:<30s} {info['prediction']:.2f}{unit}")
        else:
            print(f"  {target:<30s} {info['prediction']}  "
                  f"(confidence={info['confidence']:.2f})")


if __name__ == "__main__":
    main()
