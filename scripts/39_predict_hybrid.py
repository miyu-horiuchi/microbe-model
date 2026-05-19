"""Hybrid phenotype predictor: tabular heads for regressions, LoRA for oxygen.

This script consumes prepared artifacts instead of raw FASTA. The feature table
feeds the saved XGBoost phenotype heads for temperature, pH, and salt. The marker
sequence JSONL feeds the selected LoRA checkpoint for oxygen classification.

Example:
    PYTHONPATH=src uv run --python 3.11 --extra dev --extra embeddings python scripts/39_predict_hybrid.py \
      --features data/training_table.parquet \
      --marker-sequences data/marker_sequences.jsonl \
      --limit 25 \
      --output artifacts/hybrid_predictions.parquet

For uncultured genomes, first prepare a marker-sequence JSONL with the same schema
as data/marker_sequences.jsonl and matching genome_accession values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb

from microbe_model import config
from microbe_model.train.lora_model import OXYGEN_CLASSES

REGRESSION_TARGETS = ("optimal_temperature_c", "optimal_ph", "salt_tolerance_pct")
DEFAULT_OUTPUT_COLUMNS = (
    "bacdive_id",
    "genome_accession",
    "pred_optimal_temperature_c",
    "pred_optimal_temperature_c_low_80",
    "pred_optimal_temperature_c_high_80",
    "pred_optimal_ph",
    "pred_optimal_ph_low_80",
    "pred_optimal_ph_high_80",
    "pred_salt_tolerance_pct",
    "pred_salt_tolerance_pct_low_80",
    "pred_salt_tolerance_pct_high_80",
    "pred_oxygen_requirement",
    "pred_oxygen_requirement_confidence",
    "pred_oxygen_requirement_source",
)


def read_table(path: Path) -> pd.DataFrame:
    """Read parquet, CSV, JSON, or JSONL into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported table format: {path}")


def read_marker_sequences(path: Path) -> pd.DataFrame:
    """Read LoRA marker-sequence JSONL rows."""
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def join_features_and_sequences(
    features: pd.DataFrame,
    sequences: pd.DataFrame,
    *,
    key: str = "genome_accession",
    how: str = "inner",
) -> pd.DataFrame:
    """Join feature rows with LoRA marker sequences on a stable identifier."""
    if key not in features.columns:
        raise ValueError(f"Feature table is missing join key: {key}")
    if key not in sequences.columns:
        raise ValueError(f"Marker sequence table is missing join key: {key}")
    if "by_category" not in sequences.columns:
        raise ValueError("Marker sequence table is missing required column: by_category")

    seq_cols = [key, "by_category"]
    if "category_counts" in sequences.columns:
        seq_cols.append("category_counts")
    seq = sequences[seq_cols].drop_duplicates(key, keep="first")
    return features.merge(seq, on=key, how=how, validate="many_to_one")


def _load_regressor(path: Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    return model


def predict_tabular_regressions(
    rows: pd.DataFrame,
    *,
    model_dir: Path,
    targets: tuple[str, ...] = REGRESSION_TARGETS,
) -> pd.DataFrame:
    """Predict tabular regression phenotypes with saved quantile XGBoost heads."""
    feature_cols_path = model_dir / "feature_cols.json"
    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Missing tabular feature column list: {feature_cols_path}")
    feature_cols = json.loads(feature_cols_path.read_text())
    x_pred = rows.reindex(columns=feature_cols)

    out = pd.DataFrame(index=rows.index)
    for target in targets:
        preds: dict[str, pd.Series] = {}
        for tag in ("q10", "q50", "q90"):
            path = model_dir / f"{target}_{tag}.ubj"
            if not path.exists():
                raise FileNotFoundError(f"Missing tabular model: {path}")
            model = _load_regressor(path)
            preds[tag] = pd.Series(model.predict(x_pred), index=rows.index)

        out[f"pred_{target}"] = preds["q50"]
        out[f"pred_{target}_low_80"] = preds["q10"]
        out[f"pred_{target}_high_80"] = preds["q90"]
    return out


def predict_lora_oxygen(
    rows: pd.DataFrame,
    *,
    checkpoint_path: Path,
    batch_size: int,
    device_name: str | None,
) -> pd.DataFrame:
    """Predict oxygen class probabilities with the LoRA checkpoint."""
    import torch

    from microbe_model.train.lora_model import LoraModelConfig, PhenoLoRAModel

    if "by_category" not in rows.columns:
        raise ValueError("Rows must include by_category for LoRA prediction")
    lora_rows = rows[rows["by_category"].notna()].copy()
    out = pd.DataFrame(index=rows.index)
    if lora_rows.empty:
        return out

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = LoraModelConfig(**checkpoint["model_cfg"])
    try:
        model = PhenoLoRAModel(model_cfg).to(device)
    except ModuleNotFoundError as exc:
        if exc.name in {"peft", "torch", "transformers"}:
            raise RuntimeError(
                "LoRA prediction requires the embeddings extra. Run with "
                "`uv run --extra embeddings ...`."
            ) from exc
        raise
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    by_category = lora_rows["by_category"].tolist()
    probs_by_row: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(by_category), batch_size):
            chunk = by_category[start : start + batch_size]
            preds = model(chunk, device=device)
            probs = torch.softmax(preds["oxy"], dim=-1).detach().cpu().float().numpy()
            probs_by_row.extend(probs.tolist())

    probs_df = pd.DataFrame(
        probs_by_row,
        index=lora_rows.index,
        columns=[f"pred_oxygen_requirement_prob_{cls}" for cls in OXYGEN_CLASSES],
    )
    pred_indices = probs_df.to_numpy().argmax(axis=1)
    probs_df["pred_oxygen_requirement"] = [OXYGEN_CLASSES[i] for i in pred_indices]
    probs_df["pred_oxygen_requirement_confidence"] = probs_df[
        [f"pred_oxygen_requirement_prob_{cls}" for cls in OXYGEN_CLASSES]
    ].max(axis=1)
    probs_df["pred_oxygen_requirement_source"] = "lora"
    return out.join(probs_df, how="left")


def build_hybrid_predictions(
    joined_rows: pd.DataFrame,
    *,
    tabular_predictions: pd.DataFrame,
    oxygen_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble identifier columns, tabular predictions, and LoRA oxygen predictions."""
    id_cols = [c for c in ("bacdive_id", "genome_accession") if c in joined_rows.columns]
    out = joined_rows[id_cols].copy()
    out = out.join(tabular_predictions)
    out = out.join(oxygen_predictions)
    ordered = [c for c in DEFAULT_OUTPUT_COLUMNS if c in out.columns]
    oxygen_prob_cols = [
        f"pred_oxygen_requirement_prob_{cls}"
        for cls in OXYGEN_CLASSES
        if f"pred_oxygen_requirement_prob_{cls}" in out.columns
    ]
    extra_cols = [c for c in out.columns if c not in set(ordered + oxygen_prob_cols)]
    return out[ordered + oxygen_prob_cols + extra_cols]


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write predictions based on the output suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".jsonl":
        df.to_json(path, orient="records", lines=True)
    elif suffix == ".json":
        path.write_text(json.dumps(df.to_dict(orient="records"), indent=2) + "\n")
    else:
        raise ValueError(f"Unsupported output format: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=config.DATA / "training_table.parquet")
    parser.add_argument("--marker-sequences", type=Path, default=config.DATA / "marker_sequences.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=config.ARTIFACTS / "lora" / "fold0_best.pt")
    parser.add_argument("--phenotype-model-dir", type=Path, default=config.ROOT / "models" / "phenotype")
    parser.add_argument("--output", type=Path, default=config.ARTIFACTS / "hybrid_predictions.parquet")
    parser.add_argument("--join-key", default="genome_accession")
    parser.add_argument("--join", choices=("inner", "left"), default="inner")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default=None, help="Defaults to cuda when available, else cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = read_table(args.features)
    sequences = read_marker_sequences(args.marker_sequences)
    joined = join_features_and_sequences(features, sequences, key=args.join_key, how=args.join)
    if args.limit is not None:
        joined = joined.head(args.limit).copy()
    if joined.empty:
        raise SystemExit(
            "No rows matched between feature rows and marker sequences. "
            "Use a marker-sequence JSONL prepared for the same genome_accession values."
        )

    missing_lora = int(joined["by_category"].isna().sum())
    if missing_lora:
        print(f"[hybrid] {missing_lora:,}/{len(joined):,} rows have no LoRA marker sequences")
    print(f"[hybrid] predicting {len(joined):,} rows")

    tabular = predict_tabular_regressions(joined, model_dir=args.phenotype_model_dir)
    oxygen = predict_lora_oxygen(
        joined,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    predictions = build_hybrid_predictions(
        joined,
        tabular_predictions=tabular,
        oxygen_predictions=oxygen,
    )
    write_table(predictions, args.output)
    print(f"[hybrid] wrote {len(predictions):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
