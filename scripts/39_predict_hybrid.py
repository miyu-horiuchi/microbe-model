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


def reuse_existing_tabular_predictions(
    rows: pd.DataFrame,
    *,
    targets: tuple[str, ...] = REGRESSION_TARGETS,
) -> pd.DataFrame:
    """Reuse already-materialized tabular prediction columns from an input table."""
    out = pd.DataFrame(index=rows.index)
    for target in targets:
        pred_col = f"pred_{target}"
        if pred_col not in rows.columns:
            raise ValueError(
                f"--reuse-existing-tabular requires input column: {pred_col}"
            )
        out[pred_col] = rows[pred_col]
        for suffix in ("low_80", "high_80"):
            col = f"pred_{target}_{suffix}"
            if col in rows.columns:
                out[col] = rows[col]
    return out


def predict_lora_oxygen(
    rows: pd.DataFrame,
    *,
    checkpoint_path: Path,
    batch_size: int,
    device_name: str | None,
    progress_every: int | None = None,
    progress_label: str = "lora",
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
            done = min(start + batch_size, len(by_category))
            if progress_every and (done == len(by_category) or done % progress_every == 0):
                print(f"[{progress_label}] predicted {done:,}/{len(by_category):,} LoRA rows", flush=True)

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
    if "pred_oxygen_requirement" in joined_rows.columns:
        if "pred_oxygen_requirement" not in out.columns:
            out["pred_oxygen_requirement"] = pd.NA
        fallback_mask = out["pred_oxygen_requirement"].isna() & joined_rows[
            "pred_oxygen_requirement"
        ].notna()
        if fallback_mask.any():
            out.loc[fallback_mask, "pred_oxygen_requirement"] = joined_rows.loc[
                fallback_mask, "pred_oxygen_requirement"
            ]
            if "pred_oxygen_requirement_confidence" in joined_rows.columns:
                if "pred_oxygen_requirement_confidence" not in out.columns:
                    out["pred_oxygen_requirement_confidence"] = pd.NA
                out.loc[fallback_mask, "pred_oxygen_requirement_confidence"] = joined_rows.loc[
                    fallback_mask, "pred_oxygen_requirement_confidence"
                ]
            if "pred_oxygen_requirement_source" not in out.columns:
                out["pred_oxygen_requirement_source"] = pd.NA
            out.loc[fallback_mask, "pred_oxygen_requirement_source"] = "tabular"
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


def prediction_output_for_rows(
    rows: pd.DataFrame,
    *,
    args: argparse.Namespace,
    progress_label: str,
) -> pd.DataFrame:
    """Predict all hybrid outputs for one already-joined slice."""
    if args.reuse_existing_tabular:
        tabular = reuse_existing_tabular_predictions(rows)
    else:
        tabular = predict_tabular_regressions(rows, model_dir=args.phenotype_model_dir)
    oxygen = predict_lora_oxygen(
        rows,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device_name=args.device,
        progress_every=args.progress_every,
        progress_label=progress_label,
    )
    return build_hybrid_predictions(
        rows,
        tabular_predictions=tabular,
        oxygen_predictions=oxygen,
    )


def chunk_output_path(base_output: Path, chunk_dir: Path, start: int, stop: int) -> Path:
    """Return a stable chunk path using the final output suffix."""
    suffix = base_output.suffix or ".parquet"
    return chunk_dir / f"{base_output.stem}_{start:06d}_{stop:06d}{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=config.DATA / "training_table.parquet")
    parser.add_argument("--marker-sequences", type=Path, default=config.DATA / "marker_sequences.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=config.ARTIFACTS / "lora" / "fold0_best.pt")
    parser.add_argument("--phenotype-model-dir", type=Path, default=config.ROOT / "models" / "phenotype")
    parser.add_argument("--output", type=Path, default=config.ARTIFACTS / "hybrid_predictions.parquet")
    parser.add_argument("--join-key", default="genome_accession")
    parser.add_argument("--join", choices=("inner", "left"), default="inner")
    parser.add_argument(
        "--reuse-existing-tabular",
        action="store_true",
        help="Reuse pred_temperature/pH/salt columns from --features instead of recomputing XGBoost heads.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Write chunk files and combine them into --output when all chunks finish.",
    )
    parser.add_argument(
        "--chunk-output-dir",
        type=Path,
        default=config.ARTIFACTS / "hybrid_chunks",
        help="Directory for per-chunk outputs when --chunk-size is set.",
    )
    parser.add_argument(
        "--resume-chunks",
        action="store_true",
        help="Skip existing chunk files and combine all expected chunks at the end.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print LoRA progress after this many sequence rows. Use 0 to disable.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default=None, help="Defaults to cuda when available, else cpu.")
    args = parser.parse_args()
    if args.offset < 0:
        parser.error("--offset must be >= 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.chunk_size is not None and args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")
    if args.progress_every is not None and args.progress_every < 1:
        args.progress_every = None
    return args


def main() -> None:
    args = parse_args()
    features = read_table(args.features)
    sequences = read_marker_sequences(args.marker_sequences)
    joined = join_features_and_sequences(features, sequences, key=args.join_key, how=args.join)
    if args.offset:
        joined = joined.iloc[args.offset :].copy()
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

    if args.chunk_size:
        args.chunk_output_dir.mkdir(parents=True, exist_ok=True)
        chunk_paths: list[Path] = []
        for rel_start in range(0, len(joined), args.chunk_size):
            rel_stop = min(rel_start + args.chunk_size, len(joined))
            absolute_start = args.offset + rel_start
            absolute_stop = args.offset + rel_stop
            chunk_path = chunk_output_path(
                args.output,
                args.chunk_output_dir,
                absolute_start,
                absolute_stop,
            )
            chunk_paths.append(chunk_path)
            if args.resume_chunks and chunk_path.exists():
                print(f"[hybrid] skipping existing chunk {chunk_path}", flush=True)
                continue
            chunk_rows = joined.iloc[rel_start:rel_stop].copy()
            print(
                f"[hybrid] chunk {absolute_start:,}-{absolute_stop:,}: "
                f"predicting {len(chunk_rows):,} rows",
                flush=True,
            )
            chunk_predictions = prediction_output_for_rows(
                chunk_rows,
                args=args,
                progress_label=f"lora {absolute_start:,}-{absolute_stop:,}",
            )
            write_table(chunk_predictions, chunk_path)
            print(
                f"[hybrid] chunk {absolute_start:,}-{absolute_stop:,}: "
                f"wrote {len(chunk_predictions):,} rows to {chunk_path}",
                flush=True,
            )

        predictions = pd.concat([read_table(path) for path in chunk_paths], ignore_index=True)
    else:
        predictions = prediction_output_for_rows(
            joined,
            args=args,
            progress_label="lora",
        )
    write_table(predictions, args.output)
    print(f"[hybrid] wrote {len(predictions):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
