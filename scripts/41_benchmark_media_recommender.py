"""Dry-lab benchmark for genome-to-medium recommendation.

The production media recommender is trained as one binary classifier per medium.
This script evaluates the more practical ranking question:

    if we hide a strain's known MediaDive medium links, does the model rank at
    least one true medium in its top-k recommendations?

It compares the model against simple popularity baselines under family-grouped
splits by default.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

from microbe_model import config
from microbe_model.train.media_recommender import build_training_table


def load_recommender_features() -> pd.DataFrame:
    """Load the same feature stack used by scripts/10_train_media_recommender.py."""
    feats = pd.read_parquet(config.DATA / "features.parquet")

    hmm_path = config.DATA / "hmm_features.parquet"
    if hmm_path.exists():
        hmm = pd.read_parquet(hmm_path)
        feats = feats.merge(hmm, on="genome_accession", how="left")

    kegg_path = config.DATA / "kegg_modules.parquet"
    if kegg_path.exists():
        kegg = pd.read_parquet(kegg_path)
        feats = feats.merge(kegg, on="genome_accession", how="left")

    iso_meta_path = config.DATA / "isolation_metadata.parquet"
    if iso_meta_path.exists():
        iso_meta = pd.read_parquet(iso_meta_path)
        iso_meta["bacdive_id"] = iso_meta["bacdive_id"].astype(int)
        feats["bacdive_id"] = feats["bacdive_id"].astype(int)
        keep = ["bacdive_id", "iso_lat", "iso_lon", "iso_collection_year"]
        keep += [
            c
            for c in iso_meta.columns
            if c.startswith(("iso_continent_", "iso_country_", "iso_host_kingdom_"))
        ]
        feats = feats.merge(iso_meta[keep], on="bacdive_id", how="left")

    return feats


def group_labels(pheno: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Return stable taxonomic groups with family, then genus, then species fallback."""
    tax = pheno.set_index("bacdive_id").reindex(index)
    groups = tax["family"].copy()
    groups = groups.fillna(tax["genus"]).fillna(tax["species"]).fillna("__unknown__")
    return groups.astype(str)


def topk_metrics(y_true: np.ndarray, scores: np.ndarray, *, ks: tuple[int, ...]) -> dict[str, float]:
    """Compute strain-level top-k recovery metrics for multi-label medium rankings."""
    valid = y_true.sum(axis=1) > 0
    y = y_true[valid].astype(bool)
    s = scores[valid]
    n = int(len(y))
    if n == 0:
        return {"n_eval": 0}

    order = np.argsort(-s, axis=1)
    true_counts = y.sum(axis=1)
    out: dict[str, float] = {"n_eval": float(n), "mean_true_media": float(true_counts.mean())}

    reciprocal_ranks: list[float] = []
    for row_idx in range(n):
        ranked_truth = y[row_idx, order[row_idx]]
        hits = np.flatnonzero(ranked_truth)
        reciprocal_ranks.append(0.0 if len(hits) == 0 else 1.0 / float(hits[0] + 1))
    out["mrr"] = float(np.mean(reciprocal_ranks))

    for k in ks:
        top = order[:, :k]
        top_truth = np.take_along_axis(y, top, axis=1)
        hit = top_truth.any(axis=1)
        out[f"hit_at_{k}"] = float(hit.mean())
        out[f"recall_at_{k}"] = float((top_truth.sum(axis=1) / true_counts).mean())
        out[f"precision_at_{k}"] = float(top_truth.sum(axis=1).mean() / k)
    return out


def per_medium_auc(y_true: np.ndarray, scores: np.ndarray, medium_ids: list[str]) -> pd.DataFrame:
    """Compute PR-AUC and ROC-AUC per medium where both classes are present."""
    rows = []
    for j, mid in enumerate(medium_ids):
        y = y_true[:, j]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        rows.append({
            "medium_id": str(mid),
            "n_pos": int(y.sum()),
            "n_neg": int((y == 0).sum()),
            "pr_auc": float(average_precision_score(y, scores[:, j])),
            "roc_auc": float(roc_auc_score(y, scores[:, j])),
        })
    return pd.DataFrame(rows)


def taxonomy_popularity_scores(
    y_train: pd.DataFrame,
    tax_train: pd.DataFrame,
    tax_test: pd.DataFrame,
    global_scores: np.ndarray,
) -> np.ndarray:
    """Score by same-genus popularity, else same-family popularity, else global popularity."""
    genus_scores = {
        str(genus): y_train.loc[idx].mean(axis=0).to_numpy(dtype=np.float32)
        for genus, idx in tax_train.groupby("genus").groups.items()
        if pd.notna(genus) and len(idx) >= 3
    }
    family_scores = {
        str(family): y_train.loc[idx].mean(axis=0).to_numpy(dtype=np.float32)
        for family, idx in tax_train.groupby("family").groups.items()
        if pd.notna(family) and len(idx) >= 3
    }
    scores = np.tile(global_scores, (len(tax_test), 1)).astype(np.float32)
    for i, (_, row) in enumerate(tax_test.iterrows()):
        genus = str(row.get("genus")) if pd.notna(row.get("genus")) else None
        family = str(row.get("family")) if pd.notna(row.get("family")) else None
        if genus and genus in genus_scores:
            scores[i] = genus_scores[genus]
        elif family and family in family_scores:
            scores[i] = family_scores[family]
    return scores


def train_fold_scores(
    X: pd.DataFrame,
    y_matrix: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    n_estimators: int,
    max_depth: int,
    seed: int,
) -> np.ndarray:
    """Train per-medium classifiers for one fold and return test-row score matrix."""
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y_matrix.iloc[train_idx]
    global_scores = y_train.mean(axis=0).to_numpy(dtype=np.float32)
    scores = np.tile(global_scores, (len(test_idx), 1)).astype(np.float32)

    for j, medium_id in enumerate(y_matrix.columns):
        y = y_train[medium_id].to_numpy()
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        if n_pos < 5 or n_neg < 5:
            continue
        scale_pos_weight = n_neg / max(1, n_pos)
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            tree_method="hist",
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=seed,
        )
        model.fit(X_train, y)
        scores[:, j] = model.predict_proba(X_test)[:, 1].astype(np.float32)
    return scores


def write_report(
    *,
    path: Path,
    payload: dict[str, Any],
    auc_table: pd.DataFrame,
    medium_names: dict[str, str],
) -> None:
    lines = [
        "# Media Recommender Dry-Lab Benchmark",
        "",
        "This benchmark hides known BacDive/MediaDive strain-medium links and asks",
        "whether the genome-only recommender recovers at least one known medium in",
        "the top-k ranked recommendations.",
        "",
        "## Setup",
        "",
        f"- Split mode: `{payload['split_mode']}`",
        f"- Folds: {payload['n_splits']}",
        f"- Evaluation strains: {int(payload['model']['n_eval'])}",
        f"- Media labels: {payload['n_media']}",
        f"- Feature columns: {payload['n_features']}",
        f"- XGBoost trees per medium per fold: {payload['n_estimators']}",
        "",
        "## Ranking Metrics",
        "",
        "| Method | MRR | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Precision@5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("model", "XGBoost medium recommender"),
        ("global_popularity", "Global popularity baseline"),
        ("taxonomy_popularity", "Taxonomic popularity baseline"),
    ]:
        m = payload[key]
        lines.append(
            f"| {label} | {m['mrr']:.3f} | {m['hit_at_1']:.3f} | {m['hit_at_3']:.3f} | "
            f"{m['hit_at_5']:.3f} | {m['recall_at_5']:.3f} | {m['precision_at_5']:.3f} |"
        )

    valid_auc = auc_table.dropna(subset=["pr_auc", "roc_auc"]) if not auc_table.empty else auc_table
    lines += [
        "",
        "## Per-Medium AUC",
        "",
        f"- Valid media with both classes: {len(valid_auc)}",
        f"- Median ROC-AUC: {valid_auc['roc_auc'].median():.3f}" if len(valid_auc) else "- Median ROC-AUC: n/a",
        f"- Median PR-AUC: {valid_auc['pr_auc'].median():.3f}" if len(valid_auc) else "- Median PR-AUC: n/a",
        "",
        "Top media by PR-AUC:",
        "",
        "| Medium | Positives | PR-AUC | ROC-AUC |",
        "|---|---:|---:|---:|",
    ]
    if len(valid_auc):
        for row in valid_auc.sort_values("pr_auc", ascending=False).head(10).itertuples():
            name = medium_names.get(str(row.medium_id), "")
            lines.append(f"| {row.medium_id} {name} | {row.n_pos} | {row.pr_auc:.3f} | {row.roc_auc:.3f} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "Hit@k is the easiest practical readout: it measures whether at least one",
        "known growth medium appears in the top-k suggestions. PR-AUC is expected",
        "to be much lower than ROC-AUC because medium labels are sparse and heavily",
        "imbalanced; a high ROC-AUC with modest PR-AUC means the model is useful for",
        "ranking candidates, not for guaranteeing growth.",
        "",
    ]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-mode", choices=("family", "random"), default="family")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-media", type=int, default=None, help="Debug option: only score first N media.")
    parser.add_argument("--out-json", type=Path, default=config.ARTIFACTS / "media_recommender_drylab_benchmark.json")
    parser.add_argument("--out-md", type=Path, default=config.ARTIFACTS / "media_recommender_drylab_benchmark.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    pheno = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet")
    feats = load_recommender_features()
    strain_media = pd.read_parquet(config.DATA / "strain_media.parquet")
    media_meta = pd.read_parquet(config.DATA / "media_metadata.parquet")
    medium_names = dict(zip(media_meta["medium_id"].astype(str), media_meta["name"], strict=True))

    X, y_matrix, medium_ids = build_training_table(feats, strain_media, pheno)
    if args.max_media is not None:
        medium_ids = medium_ids[: args.max_media]
        y_matrix = y_matrix[medium_ids]
    X = X.fillna(0)
    tax = pheno.set_index("bacdive_id").reindex(X.index)[["family", "genus", "species"]]
    groups = group_labels(pheno, X.index)

    if args.split_mode == "family":
        splitter = GroupKFold(n_splits=min(args.n_splits, groups.nunique()))
        splits = list(splitter.split(X, y_matrix, groups))
    else:
        splitter = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(X))

    model_scores = np.zeros(y_matrix.shape, dtype=np.float32)
    global_scores = np.zeros(y_matrix.shape, dtype=np.float32)
    taxonomy_scores = np.zeros(y_matrix.shape, dtype=np.float32)
    y_all = y_matrix.to_numpy(dtype=np.uint8)

    print(
        f"Benchmark: {len(X):,} strains x {X.shape[1]:,} features x "
        f"{len(medium_ids):,} media, split={args.split_mode}"
    )
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_t0 = time.time()
        y_train = y_matrix.iloc[train_idx]
        global_prevalence = y_train.mean(axis=0).to_numpy(dtype=np.float32)
        global_scores[test_idx] = np.tile(global_prevalence, (len(test_idx), 1))
        taxonomy_scores[test_idx] = taxonomy_popularity_scores(
            y_train,
            tax.iloc[train_idx],
            tax.iloc[test_idx],
            global_prevalence,
        )
        model_scores[test_idx] = train_fold_scores(
            X,
            y_matrix,
            train_idx,
            test_idx,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            seed=args.seed + fold_idx,
        )
        print(
            f"  fold {fold_idx}: {len(train_idx):,} train / {len(test_idx):,} test "
            f"in {time.time() - fold_t0:.1f}s",
            flush=True,
        )

    ks = (1, 3, 5, 10)
    payload: dict[str, Any] = {
        "split_mode": args.split_mode,
        "n_splits": len(splits),
        "n_features": int(X.shape[1]),
        "n_media": int(len(medium_ids)),
        "n_estimators": args.n_estimators,
        "elapsed_s": time.time() - t0,
        "model": topk_metrics(y_all, model_scores, ks=ks),
        "global_popularity": topk_metrics(y_all, global_scores, ks=ks),
        "taxonomy_popularity": topk_metrics(y_all, taxonomy_scores, ks=ks),
    }
    auc = per_medium_auc(y_all, model_scores, [str(m) for m in medium_ids])
    if not auc.empty:
        payload["model"]["median_roc_auc"] = float(auc["roc_auc"].median())
        payload["model"]["median_pr_auc"] = float(auc["pr_auc"].median())
        payload["per_medium_auc"] = auc.to_dict(orient="records")
    else:
        payload["per_medium_auc"] = []

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    write_report(path=args.out_md, payload=payload, auc_table=auc, medium_names=medium_names)

    print(f"\nWrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(json.dumps({
        "model_hit_at_1": payload["model"]["hit_at_1"],
        "model_hit_at_3": payload["model"]["hit_at_3"],
        "model_hit_at_5": payload["model"]["hit_at_5"],
        "global_hit_at_5": payload["global_popularity"]["hit_at_5"],
        "taxonomy_hit_at_5": payload["taxonomy_popularity"]["hit_at_5"],
        "median_roc_auc": payload["model"].get("median_roc_auc"),
        "median_pr_auc": payload["model"].get("median_pr_auc"),
    }, indent=2))


if __name__ == "__main__":
    main()
