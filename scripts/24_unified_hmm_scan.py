"""Unified HMMER scan — phenotype + medium markers, all genomes.

Scans every unique genome accession that appears in features.parquet against
the verified marker library in microbe_model.features.markers.

For each genome, writes one row with three columns per marker:
  - hmm_<name>_n        : hit count above e-value 1e-5
  - hmm_<name>_score    : top bitscore among the hits
  - hmm_<name>_present  : 0/1 binary

Output: data/hmm_features.parquet (one row per unique genome_accession).
Streaming to data/hmm_features.jsonl, resumable.

Usage:
    python scripts/24_unified_hmm_scan.py --workers 8
    python scripts/24_unified_hmm_scan.py --limit 500 --workers 4   # sanity-check first
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import pyhmmer
import pyhmmer.easel
import pyhmmer.plan7
import requests
from tqdm import tqdm

from microbe_model import config
from microbe_model.features.genome import predict_genes
from microbe_model.features.markers import all_markers
from microbe_model.pipeline import _fetch_fasta_bytes

INTERPRO_HMM_URL = "https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/{pfam}/?annotation=hmm"
MARKER_DIR = config.DATA / "markers" / "unified"
MARKER_LIB = MARKER_DIR / "unified_markers.hmm"
EVALUE_THRESHOLD = 1e-5


def download_markers(markers: dict[str, tuple[str, str]]) -> Path:
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    if MARKER_LIB.exists():
        text = MARKER_LIB.read_text()
        if all(name for name, _ in markers.values() if name in text):
            return MARKER_LIB

    parts: list[str] = []
    for pfam_id, (friendly, _role) in markers.items():
        cached = MARKER_DIR / f"{pfam_id}.hmm"
        if not cached.exists():
            url = INTERPRO_HMM_URL.format(pfam=pfam_id)
            print(f"  downloading {pfam_id} ({friendly}) ...", flush=True)
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            raw = resp.content
            try:
                hmm_text = gzip.decompress(raw).decode("ascii")
            except gzip.BadGzipFile:
                hmm_text = raw.decode("ascii")
            lines = hmm_text.splitlines()
            for i, line in enumerate(lines):
                if line.startswith("NAME "):
                    lines[i] = f"NAME  {friendly}"
                    break
            cached.write_text("\n".join(lines) + "\n")
        parts.append(cached.read_text().rstrip() + "\n")
    MARKER_LIB.write_text("\n".join(parts))
    print(f"  wrote {MARKER_LIB} ({len(markers)} HMMs)")
    return MARKER_LIB


def _load_hmms(lib_path: Path) -> list[pyhmmer.plan7.HMM]:
    with pyhmmer.plan7.HMMFile(str(lib_path)) as fh:
        return list(fh)


def scan_proteins(
    proteins: list[str],
    hmms: list[pyhmmer.plan7.HMM],
    alphabet: pyhmmer.easel.Alphabet,
    marker_names: set[str],
) -> dict[str, dict[str, float]]:
    seqs: list[pyhmmer.easel.DigitalSequence] = []
    for i, prot in enumerate(proteins):
        if not prot:
            continue
        ts = pyhmmer.easel.TextSequence(name=f"p{i}".encode(), sequence=prot)
        seqs.append(ts.digitize(alphabet))

    summary: dict[str, dict[str, float]] = {
        name: {"n_hits": 0.0, "top_bitscore": 0.0, "top_evalue": 1.0}
        for name in marker_names
    }
    if not seqs:
        return summary

    for top_hits in pyhmmer.hmmer.hmmsearch(hmms, seqs, E=EVALUE_THRESHOLD):
        raw_name = top_hits.query.name
        marker = raw_name.decode() if isinstance(raw_name, bytes) else raw_name
        if marker not in summary:
            continue
        n = 0
        best_score = 0.0
        best_evalue = 1.0
        for hit in top_hits:
            if hit.evalue > EVALUE_THRESHOLD:
                continue
            n += 1
            if hit.score > best_score:
                best_score = float(hit.score)
                best_evalue = float(hit.evalue)
        summary[marker] = {"n_hits": float(n), "top_bitscore": best_score, "top_evalue": best_evalue}
    return summary


def _process_one(args: tuple[str, str, list[str]]) -> dict[str, Any] | None:
    accession, lib_path, marker_names = args
    contigs = _fetch_fasta_bytes(accession)
    if not contigs:
        return None
    try:
        proteins, _cds, _nt = predict_genes(contigs)
    except Exception:
        return None
    if not proteins:
        return None
    alphabet = pyhmmer.easel.Alphabet.amino()
    hmms = _load_hmms(Path(lib_path))
    summary = scan_proteins(proteins, hmms, alphabet, set(marker_names))
    row: dict[str, Any] = {"genome_accession": accession}
    for marker, stats in summary.items():
        row[f"hmm_{marker}_n"] = stats["n_hits"]
        row[f"hmm_{marker}_score"] = stats["top_bitscore"]
        row[f"hmm_{marker}_present"] = float(stats["n_hits"] > 0)
    return row


def _existing_accessions(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    seen: set[str] = set()
    with open(jsonl_path) as fh:
        for line in fh:
            try:
                seen.add(str(json.loads(line)["genome_accession"]))
            except Exception:
                continue
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap genomes (default: all unique accessions in features.parquet)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    markers = all_markers()
    marker_names = [name for name, _ in markers.values()]
    print(f"Loaded {len(markers)} verified markers from microbe_model.features.markers")

    print("\nStep 1: ensuring HMM library is present")
    lib_path = download_markers(markers)
    n_hmms = len(_load_hmms(lib_path))
    print(f"  loaded {n_hmms} HMMs from {lib_path}")
    if n_hmms != len(markers):
        raise SystemExit(f"  expected {len(markers)} HMMs, got {n_hmms}")

    print("\nStep 2: collecting unique genome accessions")
    feats = pd.read_parquet(config.DATA / "features.parquet")
    unique_accs = feats["genome_accession"].dropna().astype(str).unique().tolist()
    if args.limit:
        unique_accs = unique_accs[: args.limit]
    print(f"  {len(unique_accs):,} unique genome accessions to scan")

    out_jsonl = config.DATA / "hmm_features.jsonl"
    out_parquet = config.DATA / "hmm_features.parquet"
    done = _existing_accessions(out_jsonl)
    pending = [(acc, str(lib_path), marker_names) for acc in unique_accs if acc not in done]
    print(f"  {len(done):,} cached, {len(pending):,} new tasks")

    print(f"\nStep 3: streaming fetch + predict + scan ({args.workers} workers)")
    t0 = time.time()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "a") as log, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, t): t for t in pending}
        with tqdm(total=len(pending), unit="genome") as bar:
            n_ok = 0
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                except Exception:
                    result = None
                bar.update(1)
                if result is None:
                    continue
                log.write(json.dumps(result) + "\n")
                log.flush()
                n_ok += 1
                bar.set_postfix(ok=n_ok)
    elapsed = time.time() - t0
    print(f"  scan finished in {elapsed/60:.1f} min")

    print("\nStep 4: materializing parquet")
    rows = []
    with open(out_jsonl) as fh:
        for line in fh:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    print(f"  wrote {out_parquet} ({len(df):,} rows × {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
