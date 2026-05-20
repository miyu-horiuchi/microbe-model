"""Prepare apples-to-apples external benchmark inputs.

This script pins the same BacDive/MediaDive held-out strains used by
``scripts/41_benchmark_media_recommender.py`` and checks whether the local
machine can run external baselines such as GenomeSPOT, CarveMe, and gapseq.

It deliberately separates preparation from heavy external execution: those tools
need raw genome FASTA files and optional third-party databases that are much
larger than the repository.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from microbe_model import config
from microbe_model.pipeline import _fetch_fasta_bytes
from microbe_model.train.media_recommender import build_training_table


TOOL_CANDIDATES = {
    "GenomeSPOT": ("genomespot", "genome-spot", "genome_spot"),
    "CarveMe": ("carve",),
    "gapseq": ("gapseq",),
}


def load_recommender_features() -> pd.DataFrame:
    """Load the same feature stack used by the media recommender."""
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


def assign_folds(
    X: pd.DataFrame,
    y_matrix: pd.DataFrame,
    pheno: pd.DataFrame,
    *,
    split_mode: str,
    n_splits: int,
    seed: int,
) -> pd.Series:
    """Assign the same fold IDs used by the dry-lab recommender benchmark."""
    if split_mode == "family":
        groups = group_labels(pheno, X.index)
        splitter = GroupKFold(n_splits=min(n_splits, groups.nunique()))
        splits = list(splitter.split(X, y_matrix, groups))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(X))

    fold_by_id = pd.Series(index=X.index, dtype="int64")
    for fold_idx, (_, test_idx) in enumerate(splits):
        fold_by_id.iloc[test_idx] = fold_idx
    return fold_by_id.astype(int)


def build_manifest(*, split_mode: str, n_splits: int, seed: int) -> tuple[pd.DataFrame, list[str]]:
    """Build the external benchmark manifest and return selected medium IDs."""
    pheno = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet")
    feats = load_recommender_features()
    strain_media = pd.read_parquet(config.DATA / "strain_media.parquet")
    media_meta = pd.read_parquet(config.DATA / "media_metadata.parquet")

    X, y_matrix, medium_ids = build_training_table(feats, strain_media, pheno)
    folds = assign_folds(X, y_matrix, pheno, split_mode=split_mode, n_splits=n_splits, seed=seed)

    labels = pheno.set_index("bacdive_id").reindex(X.index)
    medium_names = dict(zip(media_meta["medium_id"].astype(str), media_meta["name"], strict=True))

    rows: list[dict[str, Any]] = []
    for bacdive_id in X.index:
        y_row = y_matrix.loc[bacdive_id]
        true_ids = [str(mid) for mid, value in y_row.items() if int(value) == 1]
        rows.append(
            {
                "bacdive_id": int(bacdive_id),
                "fold": int(folds.loc[bacdive_id]),
                "genome_accession": str(labels.loc[bacdive_id, "genome_accession"]),
                "species": _clean_str(labels.loc[bacdive_id, "species"]),
                "genus": _clean_str(labels.loc[bacdive_id, "genus"]),
                "family": _clean_str(labels.loc[bacdive_id, "family"]),
                "optimal_temperature_c": _clean_float(labels.loc[bacdive_id, "optimal_temperature_c"]),
                "optimal_ph": _clean_float(labels.loc[bacdive_id, "optimal_ph"]),
                "salt_tolerance_pct": _clean_float(labels.loc[bacdive_id, "salt_tolerance_pct"]),
                "oxygen_requirement": _clean_str(labels.loc[bacdive_id, "oxygen_requirement"]),
                "true_media_ids": "|".join(true_ids),
                "true_media_names": "|".join(medium_names.get(mid, "") for mid in true_ids),
                "n_true_media": len(true_ids),
            }
        )

    manifest = pd.DataFrame(rows).sort_values(["fold", "bacdive_id"]).reset_index(drop=True)
    return manifest, [str(mid) for mid in medium_ids]


def _clean_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _clean_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def detect_tools() -> dict[str, dict[str, str | None]]:
    """Detect external command-line tools without installing anything."""
    out: dict[str, dict[str, str | None]] = {}
    for label, candidates in TOOL_CANDIDATES.items():
        found_name = None
        found_path = None
        for candidate in candidates:
            path = shutil.which(candidate)
            if path:
                found_name = candidate
                found_path = path
                break
        if label == "GenomeSPOT" and found_path is None:
            local_source = config.DATA / "external_tools" / "GenomeSPOT-main"
            if (local_source / "genome_spot" / "genome_spot.py").exists() and (local_source / "models").exists():
                found_name = "uv run python -m genome_spot.genome_spot"
                found_path = str(local_source.relative_to(config.ROOT))
        if label == "CarveMe" and found_path is None and shutil.which("diamond"):
            found_name = "uv run --with carveme carve"
            found_path = shutil.which("diamond")
        out[label] = {"command": found_name, "path": found_path}
    return out


def local_fasta_path(fasta_dir: Path, accession: str) -> Path | None:
    """Return an existing FASTA path for an accession, if present."""
    for suffix in (".fna", ".fna.gz", ".fa", ".fa.gz", ".fasta", ".fasta.gz"):
        candidate = fasta_dir / f"{accession}{suffix}"
        if candidate.exists():
            return candidate
    return None


def fasta_coverage(manifest: pd.DataFrame, fasta_dir: Path) -> dict[str, Any]:
    """Count how many manifest accessions already have local FASTA files."""
    accessions = manifest["genome_accession"].dropna().astype(str).unique().tolist()
    present = [acc for acc in accessions if local_fasta_path(fasta_dir, acc)]
    return {
        "fasta_dir": str(fasta_dir),
        "unique_accessions": len(accessions),
        "present_fastas": len(present),
        "missing_fastas": len(accessions) - len(present),
        "coverage_pct": 0.0 if not accessions else 100.0 * len(present) / len(accessions),
    }


def download_fastas(manifest: pd.DataFrame, fasta_dir: Path, *, limit: int) -> dict[str, Any]:
    """Download a small number of missing genome FASTAs from NCBI for smoke tests."""
    fasta_dir.mkdir(parents=True, exist_ok=True)
    accessions = manifest["genome_accession"].dropna().astype(str).drop_duplicates().tolist()
    missing = [acc for acc in accessions if local_fasta_path(fasta_dir, acc) is None]
    if limit <= 0:
        return {"attempted": 0, "downloaded": 0, "failed": 0}

    attempted = 0
    downloaded = 0
    failed = 0
    for accession in missing[:limit]:
        attempted += 1
        contigs = _fetch_fasta_bytes(accession)
        if not contigs:
            failed += 1
            continue
        out_path = fasta_dir / f"{accession}.fna.gz"
        with gzip.open(out_path, "wt") as handle:
            for contig_id, sequence in contigs:
                handle.write(f">{contig_id}\n")
                for i in range(0, len(sequence), 80):
                    handle.write(sequence[i : i + 80] + "\n")
        downloaded += 1
    return {"attempted": attempted, "downloaded": downloaded, "failed": failed}


def write_status_report(
    *,
    path: Path,
    manifest: pd.DataFrame,
    medium_ids: list[str],
    tools: dict[str, dict[str, str | None]],
    coverage: dict[str, Any],
    download: dict[str, Any],
    out_manifest: Path,
) -> None:
    """Write a human-readable status report for the external baseline run."""
    label_counts = {
        "temperature": int(manifest["optimal_temperature_c"].notna().sum()),
        "ph": int(manifest["optimal_ph"].notna().sum()),
        "salt": int(manifest["salt_tolerance_pct"].notna().sum()),
        "oxygen": int((manifest["oxygen_requirement"] != "").sum()),
        "medium": int((manifest["n_true_media"] > 0).sum()),
    }
    fold_counts = manifest["fold"].value_counts().sort_index().to_dict()

    lines = [
        "# External Tool Benchmark Status",
        "",
        "This file tracks the apples-to-apples benchmark setup for external tools",
        "on the same held-out BacDive/MediaDive strains used by the dry-lab media",
        "recommender benchmark.",
        "",
        "## Held-Out Manifest",
        "",
        f"- Manifest: `{display_path(out_manifest)}`",
        f"- Rows: {len(manifest):,}",
        f"- Unique genome accessions: {coverage['unique_accessions']:,}",
        f"- Media labels retained: {len(medium_ids):,}",
        f"- Fold counts: {json.dumps({str(k): int(v) for k, v in fold_counts.items()})}",
        "",
        "Label coverage:",
        "",
        "| Target | Labeled rows |",
        "|---|---:|",
        f"| Temperature | {label_counts['temperature']:,} |",
        f"| pH | {label_counts['ph']:,} |",
        f"| Salt | {label_counts['salt']:,} |",
        f"| Oxygen | {label_counts['oxygen']:,} |",
        f"| Medium | {label_counts['medium']:,} |",
        "",
        "## Local Requirements",
        "",
        f"- FASTA directory: `{display_path(Path(str(coverage['fasta_dir'])))}`",
        f"- FASTAs present: {coverage['present_fastas']:,} / {coverage['unique_accessions']:,} "
        f"({coverage['coverage_pct']:.2f}%)",
        f"- FASTA download smoke run: {json.dumps(download)}",
        "",
        "| Tool | Local command | Status |",
        "|---|---|---|",
    ]
    for tool, info in tools.items():
        command = info["command"] or ""
        status = "available" if info["path"] else "missing"
        lines.append(f"| {tool} | `{command}` | {status} |")

    lines += [
        "",
        "## Verdict",
        "",
    ]
    if all(info["path"] for info in tools.values()) and coverage["present_fastas"] == coverage["unique_accessions"]:
        lines.append("External baseline execution is ready on this machine.")
    else:
        lines.append(
            "External baseline execution is not ready on this machine yet: the full "
            "held-out FASTA set and one or more external tool binaries/databases are missing."
        )

    lines += [
        "",
        "## Next Commands",
        "",
        "Use the manifest to run each external tool against the same rows and folds.",
        "The medium-feasibility tools should be scored by whether at least one known",
        "MediaDive medium is feasible or closest among the tool's predicted feasible",
        "media/metabolite environments.",
        "",
        "```bash",
        "PYTHONPATH=src uv run --python 3.11 python scripts/42_prepare_external_benchmarks.py \\",
        "  --download-fastas 10",
        "```",
        "",
        "For the full benchmark, download the complete FASTA set into the FASTA",
        "directory above, install the external tools plus their databases, then run",
        "tool-specific inference using the `bacdive_id`, `fold`, and",
        "`genome_accession` columns from the manifest.",
        "",
    ]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-mode", choices=("family", "random"), default="family")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fasta-dir", type=Path, default=config.DATA / "external_benchmark_fastas")
    parser.add_argument(
        "--manifest-parquet",
        type=Path,
        default=config.ARTIFACTS / "external_benchmark_manifest.parquet",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=config.ARTIFACTS / "external_benchmark_manifest.csv",
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        default=config.ARTIFACTS / "external_benchmark_status.json",
    )
    parser.add_argument(
        "--status-md",
        type=Path,
        default=config.ARTIFACTS / "external_benchmark_status.md",
    )
    parser.add_argument(
        "--download-fastas",
        type=int,
        default=0,
        help="Download this many missing FASTAs from NCBI for smoke testing. Default: 0.",
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    """Format project-local paths relative to the repository root."""
    try:
        return str(path.resolve().relative_to(config.ROOT.resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    t0 = time.time()
    manifest, medium_ids = build_manifest(split_mode=args.split_mode, n_splits=args.n_splits, seed=args.seed)
    args.manifest_parquet.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.manifest_parquet, index=False)
    manifest.to_csv(args.manifest_csv, index=False)

    download = download_fastas(manifest, args.fasta_dir, limit=args.download_fastas)
    tools = detect_tools()
    coverage = fasta_coverage(manifest, args.fasta_dir)

    payload = {
        "split_mode": args.split_mode,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "elapsed_s": time.time() - t0,
        "manifest_parquet": display_path(args.manifest_parquet),
        "manifest_csv": display_path(args.manifest_csv),
        "rows": int(len(manifest)),
        "media_labels": len(medium_ids),
        "tools": tools,
        "fasta_coverage": {**coverage, "fasta_dir": display_path(Path(str(coverage["fasta_dir"])))},
        "download": download,
    }
    args.status_json.write_text(json.dumps(payload, indent=2))
    write_status_report(
        path=args.status_md,
        manifest=manifest,
        medium_ids=medium_ids,
        tools=tools,
        coverage=coverage,
        download=download,
        out_manifest=args.manifest_parquet,
    )

    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.manifest_parquet}")
    print(f"Wrote {args.manifest_csv}")
    print(f"Wrote {args.status_json}")
    print(f"Wrote {args.status_md}")


if __name__ == "__main__":
    main()
