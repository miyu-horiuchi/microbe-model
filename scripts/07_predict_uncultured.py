"""Apply the trained v1 model to uncultured (non-BacDive) GTDB representatives.

Pipeline:
  1. Read data/gtdb_candidates.parquet (created by scripts/06_fetch_gtdb_candidates.py)
  2. Stream-fetch + featurize each candidate (writes data/uncultured_features.jsonl)
  3. Re-train? No — load the existing baseline_results.json + reload an XGBoost model
     trained on the BacDive corpus.

The output: artifacts/uncultured_predictions.parquet — one row per genome with all
four phenotype targets predicted from genome features alone. This is the v0 research
deliverable: predicted cultivation conditions for genomes that have never been cultured.
"""
from __future__ import annotations

import argparse

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from microbe_model import config
from microbe_model.pipeline import stream_fetch_and_featurize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None,
                        help="Cap how many candidates to process (default: all).")
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()

    candidates_path = config.DATA / "gtdb_candidates.parquet"
    if not candidates_path.exists():
        raise SystemExit(f"Missing {candidates_path}. Run scripts/06_fetch_gtdb_candidates.py first.")

    candidates = pd.read_parquet(candidates_path)
    if args.max:
        candidates = candidates.head(args.max)
    print(f"Predicting on {len(candidates):,} GTDB representatives not in BacDive")

    # Use a synthetic int "fake_id" since these aren't BacDive strains
    # (the streaming pipeline expects int IDs but doesn't care what they are)
    candidates = candidates.reset_index(drop=True)
    candidates["fake_id"] = candidates.index + 1_000_000_000  # avoid collision with BacDive IDs

    tasks = list(zip(
        candidates["fake_id"].astype(int),
        candidates["genome_accession"].astype(str),
        strict=True,
    ))
    feats_out = config.DATA / "uncultured_features.jsonl"
    print(f"Output: {feats_out}\n")

    with tqdm(total=len(tasks), desc="featurize uncultured", unit="strain") as bar:
        def progress(c, s, t):
            bar.n = c
            bar.set_postfix(success=s, fail=c - s)
            bar.refresh()

        stream_fetch_and_featurize(
            tasks, out_path=feats_out, n_workers=args.workers, on_progress=progress,
        )

    # Materialize parquet
    feats = pd.read_json(feats_out, lines=True)
    feats_parquet = config.DATA / "uncultured_features.parquet"
    feats.to_parquet(feats_parquet, index=False)
    print(f"\nFeaturized {len(feats):,} genomes")

    # Now re-train one final XGBoost on ALL of BacDive (all folds combined) and predict.
    # We do this here (not loading a saved model) so the prediction surface uses every
    # labeled BacDive strain, not just one fold's training set.
    pheno = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet")
    bacdive_feats = pd.read_parquet(config.DATA / "features.parquet")
    bacdive = pheno.merge(bacdive_feats, on=["bacdive_id", "genome_accession"], how="inner")

    feature_cols = [c for c in feats.columns if c not in {"bacdive_id", "genome_accession"}]
    common_cols = [c for c in feature_cols if c in bacdive.columns]
    print(f"Using {len(common_cols)} features common to BacDive + uncultured")

    output_rows = candidates[[
        "genome_accession", "gtdb_taxonomy", "ncbi_organism_name", "checkm_completeness",
    ]].copy()

    # Index uncultured features by accession for lookup
    feats_by_acc = feats.set_index("genome_accession")

    for target, task in config.PHENOTYPE_TARGETS.items():
        labeled = bacdive[bacdive[target].notna()]
        if len(labeled) < 100:
            continue
        X_train = labeled[common_cols]
        y_train_raw = labeled[target]

        if task == "regression":
            model = xgb.XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                tree_method="hist", n_jobs=-1,
            )
            y_train = y_train_raw.to_numpy(dtype=float)
            model.fit(X_train, y_train)
            X_pred = feats_by_acc.reindex(candidates["genome_accession"])[common_cols]
            preds = model.predict(X_pred.fillna(0))
            output_rows[f"pred_{target}"] = preds
        else:
            encoder = LabelEncoder()
            y_train_enc = encoder.fit_transform(y_train_raw.astype(str))
            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                tree_method="hist", n_jobs=-1, eval_metric="mlogloss",
            )
            model.fit(X_train, y_train_enc)
            X_pred = feats_by_acc.reindex(candidates["genome_accession"])[common_cols]
            preds = model.predict(X_pred.fillna(0))
            preds_proba = model.predict_proba(X_pred.fillna(0))
            output_rows[f"pred_{target}"] = encoder.inverse_transform(preds)
            output_rows[f"pred_{target}_confidence"] = preds_proba.max(axis=1)

        print(f"  {target}: trained on {len(labeled):,} BacDive strains, predicted {len(output_rows)}")

    out_path = config.ARTIFACTS / "uncultured_predictions.parquet"
    output_rows.to_parquet(out_path, index=False)
    print(f"\nWrote {len(output_rows)} predictions to {out_path}")

    # Quick summary
    print("\nPredicted T_opt distribution:")
    if "pred_optimal_temperature_c" in output_rows:
        s = output_rows["pred_optimal_temperature_c"].dropna()
        print(f"  mean={s.mean():.1f}  std={s.std():.1f}  "
              f"p10={s.quantile(0.1):.1f}  p90={s.quantile(0.9):.1f}")
        n_thermo = (s >= 50).sum()
        n_psychro = (s <= 15).sum()
        print(f"  predicted thermophiles (T_opt >= 50°C): {n_thermo}")
        print(f"  predicted psychrophiles (T_opt <= 15°C): {n_psychro}")


if __name__ == "__main__":
    main()
