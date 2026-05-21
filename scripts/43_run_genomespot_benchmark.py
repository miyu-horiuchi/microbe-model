"""Run GenomeSPOT on the prepared held-out benchmark manifest.

The full external comparison requires thousands of genome FASTAs. This runner is
therefore limit-aware: it can smoke-test a few exact held-out rows locally, and
the same command can be scaled on a larger disk by raising ``--limit``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microbe_model import config
from microbe_model.features.genome import predict_genes
from microbe_model.pipeline import _fetch_fasta_bytes


GENOMESPOT_UV_DEPS = [
    "--with",
    "numpy==1.24.4",
    "--with",
    "scipy==1.10.1",
    "--with",
    "pandas==2.0.3",
    "--with",
    "scikit-learn==1.2.2",
    "--with",
    "hmmlearn==0.3.0",
    "--with",
    "biopython>=1.83",
]


def write_fasta_gz(path: Path, records: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        for record_id, sequence in records:
            handle.write(f">{record_id}\n")
            for i in range(0, len(sequence), 80):
                handle.write(sequence[i : i + 80] + "\n")


def ensure_inputs(row: pd.Series, fasta_dir: Path) -> tuple[Path | None, Path | None, str | None]:
    """Fetch contigs and generate proteins for one manifest row if needed."""
    accession = str(row["genome_accession"])
    contigs_path = fasta_dir / f"{accession}.fna.gz"
    proteins_path = fasta_dir / f"{accession}.faa.gz"
    if contigs_path.exists() and proteins_path.exists():
        return contigs_path, proteins_path, None

    contigs = _fetch_fasta_bytes(accession)
    if not contigs:
        return None, None, "fasta_download_failed"

    try:
        proteins, _cds, _total_nt = predict_genes(contigs)
    except Exception as exc:
        return None, None, f"protein_prediction_failed: {exc}"
    if not proteins:
        return None, None, "protein_prediction_empty"

    write_fasta_gz(contigs_path, contigs)
    protein_records = [(f"{accession}_cds_{i + 1}", protein) for i, protein in enumerate(proteins)]
    write_fasta_gz(proteins_path, protein_records)
    return contigs_path, proteins_path, None


def genomespot_command(
    *,
    genome_spot_dir: Path,
    contigs_path: Path,
    proteins_path: Path,
    output_prefix: Path,
) -> list[str]:
    """Build a pinned GenomeSPOT uv command."""
    return [
        "uv",
        "run",
        "--python",
        "3.11",
        "--isolated",
        "--with",
        str(genome_spot_dir),
        *GENOMESPOT_UV_DEPS,
        "python",
        "-m",
        "genome_spot.genome_spot",
        "--models",
        str(genome_spot_dir / "models"),
        "--contigs",
        str(contigs_path),
        "--proteins",
        str(proteins_path),
        "--output-prefix",
        str(output_prefix),
    ]


def run_one(row: pd.Series, *, genome_spot_dir: Path, fasta_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run GenomeSPOT for one row and return status plus parsed predictions."""
    bacdive_id = int(row["bacdive_id"])
    accession = str(row["genome_accession"])
    output_prefix = output_dir / accession
    pred_path = Path(f"{output_prefix}.predictions.tsv")
    contigs_path, proteins_path, input_error = ensure_inputs(row, fasta_dir)
    if input_error:
        return {"bacdive_id": bacdive_id, "genome_accession": accession, "status": "skipped", "error": input_error}

    if pred_path.exists():
        parsed = parse_prediction(pred_path)
        return {
            "bacdive_id": bacdive_id,
            "genome_accession": accession,
            "fold": int(row["fold"]),
            "status": "ok",
            "elapsed_s": 0.0,
            "cached": True,
            "true_temperature_c": _maybe_float(row.get("optimal_temperature_c")),
            "true_ph": _maybe_float(row.get("optimal_ph")),
            "true_salt_pct": _maybe_float(row.get("salt_tolerance_pct")),
            "true_oxygen": str(row.get("oxygen_requirement") or ""),
            **parsed,
        }

    cmd = genomespot_command(
        genome_spot_dir=genome_spot_dir,
        contigs_path=contigs_path,
        proteins_path=proteins_path,
        output_prefix=output_prefix,
    )
    started = time.time()
    result = subprocess.run(cmd, cwd=config.ROOT, text=True, capture_output=True, check=False)
    elapsed_s = time.time() - started
    if result.returncode != 0:
        return {
            "bacdive_id": bacdive_id,
            "genome_accession": accession,
            "status": "failed",
            "error": result.stderr[-2000:] or result.stdout[-2000:],
            "elapsed_s": elapsed_s,
        }

    if not pred_path.exists():
        return {
            "bacdive_id": bacdive_id,
            "genome_accession": accession,
            "status": "failed",
            "error": f"missing output {pred_path}",
            "elapsed_s": elapsed_s,
        }

    parsed = parse_prediction(pred_path)
    return {
        "bacdive_id": bacdive_id,
        "genome_accession": accession,
        "fold": int(row["fold"]),
        "status": "ok",
        "elapsed_s": elapsed_s,
        "true_temperature_c": _maybe_float(row.get("optimal_temperature_c")),
        "true_ph": _maybe_float(row.get("optimal_ph")),
        "true_salt_pct": _maybe_float(row.get("salt_tolerance_pct")),
        "true_oxygen": str(row.get("oxygen_requirement") or ""),
        **parsed,
    }


def parse_prediction(path: Path) -> dict[str, Any]:
    """Parse GenomeSPOT's TSV dataframe output into flat fields."""
    table = pd.read_csv(path, sep="\t", index_col=0)

    def get(condition: str, column: str) -> Any:
        if condition not in table.index or column not in table.columns:
            return None
        value = table.loc[condition, column]
        if pd.isna(value):
            return None
        return value

    return {
        "genomespot_temperature_c": _maybe_float(get("temperature_optimum", "value")),
        "genomespot_temperature_error": _maybe_float(get("temperature_optimum", "error")),
        "genomespot_ph": _maybe_float(get("ph_optimum", "value")),
        "genomespot_ph_error": _maybe_float(get("ph_optimum", "error")),
        "genomespot_salt_pct": _maybe_float(get("salinity_optimum", "value")),
        "genomespot_salt_error": _maybe_float(get("salinity_optimum", "error")),
        "genomespot_oxygen": str(get("oxygen", "value") or ""),
        "genomespot_oxygen_probability": _maybe_float(get("oxygen", "error")),
    }


def _maybe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in results if row.get("status") == "ok"]

    def mae(true_key: str, pred_key: str) -> float | None:
        pairs = [
            (row[true_key], row[pred_key])
            for row in ok
            if row.get(true_key) is not None and row.get(pred_key) is not None
        ]
        if not pairs:
            return None
        return float(np.mean([abs(t - p) for t, p in pairs]))

    return {
        "n_requested": len(results),
        "n_ok": len(ok),
        "n_failed_or_skipped": len(results) - len(ok),
        "temperature_mae_c": mae("true_temperature_c", "genomespot_temperature_c"),
        "ph_mae": mae("true_ph", "genomespot_ph"),
        "salt_mae_pct": mae("true_salt_pct", "genomespot_salt_pct"),
        "mean_elapsed_s": None if not ok else float(np.mean([row["elapsed_s"] for row in ok])),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# GenomeSPOT Held-Out Benchmark",
        "",
        "GenomeSPOT was run on rows selected from the same held-out manifest used",
        "by the microbe-model media benchmark. The manifest and limit define",
        "whether this is a smoke run, a representative subset, or the full run.",
        "",
        "## Setup",
        "",
        f"- Manifest: `{payload['manifest']}`",
        f"- Limit: {payload['limit']}",
        f"- Required labels: {', '.join(payload['required_labels']) or 'none'}",
        f"- GenomeSPOT source: `{payload['genome_spot_dir']}`",
        f"- FASTA directory: `{payload['fasta_dir']}`",
        "",
        "## Results",
        "",
        f"- OK: {summary['n_ok']} / {summary['n_requested']}",
        f"- Failed/skipped: {summary['n_failed_or_skipped']}",
        f"- Mean runtime per OK genome: {summary['mean_elapsed_s']:.2f}s" if summary["mean_elapsed_s"] is not None else "- Mean runtime per OK genome: n/a",
        f"- Temperature MAE: {summary['temperature_mae_c']:.3f} C" if summary["temperature_mae_c"] is not None else "- Temperature MAE: n/a",
        f"- pH MAE: {summary['ph_mae']:.3f}" if summary["ph_mae"] is not None else "- pH MAE: n/a",
        f"- Salt MAE: {summary['salt_mae_pct']:.3f}%" if summary["salt_mae_pct"] is not None else "- Salt MAE: n/a",
        "",
        "## Notes",
        "",
        "GenomeSPOT oxygen is a tolerant/not-tolerant label, while microbe-model",
        "uses BacDive oxygen categories. The smoke report keeps raw labels rather",
        "than forcing an evaluation mapping that may hide label-definition mismatch.",
        "",
    ]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=config.ARTIFACTS / "external_benchmark_manifest.parquet")
    parser.add_argument("--genome-spot-dir", type=Path, default=config.DATA / "external_tools" / "GenomeSPOT-main")
    parser.add_argument("--fasta-dir", type=Path, default=config.DATA / "external_benchmark_fastas")
    parser.add_argument("--output-dir", type=Path, default=config.ARTIFACTS / "genomespot_predictions")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument(
        "--require-label",
        action="append",
        choices=("temperature", "ph", "salt", "oxygen", "medium"),
        default=[],
        help="Keep only rows with this label. Can be repeated.",
    )
    parser.add_argument("--out-json", type=Path, default=config.ARTIFACTS / "genomespot_smoke_benchmark.json")
    parser.add_argument("--out-md", type=Path, default=config.ARTIFACTS / "genomespot_smoke_benchmark.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_parquet(args.manifest)
    if args.fold is not None:
        manifest = manifest[manifest["fold"] == args.fold]
    for label in args.require_label:
        if label == "temperature":
            manifest = manifest[manifest["optimal_temperature_c"].notna()]
        elif label == "ph":
            manifest = manifest[manifest["optimal_ph"].notna()]
        elif label == "salt":
            manifest = manifest[manifest["salt_tolerance_pct"].notna()]
        elif label == "oxygen":
            manifest = manifest[manifest["oxygen_requirement"].fillna("") != ""]
        elif label == "medium":
            manifest = manifest[manifest["n_true_media"] > 0]
    manifest = manifest.head(args.limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for _, row in manifest.iterrows():
        result = run_one(row, genome_spot_dir=args.genome_spot_dir, fasta_dir=args.fasta_dir, output_dir=args.output_dir)
        results.append(result)
        print(json.dumps(result), flush=True)

    payload = {
        "manifest": str(args.manifest.relative_to(config.ROOT) if args.manifest.is_relative_to(config.ROOT) else args.manifest),
        "genome_spot_dir": str(
            args.genome_spot_dir.relative_to(config.ROOT)
            if args.genome_spot_dir.is_relative_to(config.ROOT)
            else args.genome_spot_dir
        ),
        "fasta_dir": str(args.fasta_dir.relative_to(config.ROOT) if args.fasta_dir.is_relative_to(config.ROOT) else args.fasta_dir),
        "limit": args.limit,
        "fold": args.fold,
        "required_labels": args.require_label,
        "summary": summarize(results),
        "results": results,
    }
    args.out_json.write_text(json.dumps(payload, indent=2))
    write_report(args.out_md, payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
