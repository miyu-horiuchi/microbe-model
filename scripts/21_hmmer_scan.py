"""HMMER pre-filter — Phase 1 oxygen-marker scan.

Tests whether per-genome counts of oxygen-relevant Pfam families add signal
beyond mean-pooled composition features. Streams genomes (no disk caching),
runs pyrodigal + pyhmmer, writes one row per genome to
data/hmm_features_oxygen.parquet.

Usage:
    python scripts/21_hmmer_scan.py --limit 100
    python scripts/21_hmmer_scan.py --limit 100 --workers 4

The first run downloads 10 marker HMMs from InterPro into data/markers/oxygen/.
Subsequent runs reuse the cached library.
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
import requests
import pyhmmer.easel
import pyhmmer.plan7
from tqdm import tqdm

from microbe_model import config
from microbe_model.features.genome import predict_genes
from microbe_model.pipeline import _fetch_fasta_bytes

# Pfam families: 7 aerobic markers + 3 anaerobic markers. Names below become
# the column suffixes in the output parquet.
OXYGEN_MARKERS: dict[str, str] = {
    "PF00115": "COX1_aerobic",          # heme-Cu terminal oxidase, subunit I
    "PF02790": "COX2_aerobic",          # cytochrome c oxidase, subunit II
    "PF00116": "COX3_aerobic",          # cytochrome c oxidase, subunit III
    "PF00199": "Catalase_aerobic",      # H2O2 detoxification
    "PF00081": "SOD_FeMn_aerobic",      # iron/manganese superoxide dismutase
    "PF00080": "SOD_CuZn_aerobic",      # Cu/Zn superoxide dismutase
    "PF00355": "Rieske_aerobic",        # Rieske 2Fe-2S in cytochrome bc1
    "PF02906": "FeFe_hyd_anaerobic",    # [FeFe]-hydrogenase large subunit C
    "PF00890": "FAD_binding_2",         # fumarate reductase / succinate DH
    "PF00037": "Fer4_anaerobic",        # 4Fe-4S ferredoxin
}

INTERPRO_HMM_URL = "https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/{pfam}/?annotation=hmm"
MARKER_DIR = config.DATA / "markers" / "oxygen"
MARKER_LIB = MARKER_DIR / "oxygen_markers.hmm"
EVALUE_THRESHOLD = 1e-5  # report a hit only if the per-domain e-value is at least this strict


def download_markers() -> Path:
    """Fetch each Pfam HMM from InterPro and concatenate into one file.

    Idempotent: skips families already present and reuses MARKER_LIB if it
    contains all 10 names.
    """
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    if MARKER_LIB.exists():
        text = MARKER_LIB.read_text()
        if all(name in text for name in OXYGEN_MARKERS.values()):
            return MARKER_LIB

    parts: list[str] = []
    for pfam_id, friendly in OXYGEN_MARKERS.items():
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
            # Rewrite NAME to the friendly tag so hits report a usable column key.
            lines = hmm_text.splitlines()
            for i, line in enumerate(lines):
                if line.startswith("NAME "):
                    lines[i] = f"NAME  {friendly}"
                    break
            cached.write_text("\n".join(lines) + "\n")
        parts.append(cached.read_text().rstrip() + "\n")

    MARKER_LIB.write_text("\n".join(parts))
    print(f"  wrote {MARKER_LIB} ({len(OXYGEN_MARKERS)} HMMs)")
    return MARKER_LIB


def _load_hmms(lib_path: Path) -> list[pyhmmer.plan7.HMM]:
    with pyhmmer.plan7.HMMFile(str(lib_path)) as fh:
        return list(fh)


def scan_proteins(
    proteins: list[str],
    hmms: list[pyhmmer.plan7.HMM],
    alphabet: pyhmmer.easel.Alphabet,
) -> dict[str, dict[str, float]]:
    """Run hmmsearch and return {marker_name: {n_hits, top_bitscore, top_evalue}}."""
    seqs: list[pyhmmer.easel.DigitalSequence] = []
    for i, prot in enumerate(proteins):
        if not prot:
            continue
        ts = pyhmmer.easel.TextSequence(name=f"p{i}".encode(), sequence=prot)
        seqs.append(ts.digitize(alphabet))

    summary: dict[str, dict[str, float]] = {
        friendly: {"n_hits": 0.0, "top_bitscore": 0.0, "top_evalue": 1.0}
        for friendly in OXYGEN_MARKERS.values()
    }
    if not seqs:
        return summary

    for top_hits in pyhmmer.hmmer.hmmsearch(hmms, seqs, E=EVALUE_THRESHOLD):
        name = top_hits.query.name
        marker = name.decode() if isinstance(name, bytes) else name
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
        summary[marker] = {
            "n_hits": float(n),
            "top_bitscore": best_score,
            "top_evalue": best_evalue,
        }
    return summary


def _process_one(args: tuple[int, str, str]) -> dict[str, Any] | None:
    bacdive_id, accession, lib_path = args
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
    summary = scan_proteins(proteins, hmms, alphabet)

    row: dict[str, Any] = {"bacdive_id": bacdive_id, "genome_accession": accession}
    for marker, stats in summary.items():
        row[f"hmm_{marker}_n"] = stats["n_hits"]
        row[f"hmm_{marker}_score"] = stats["top_bitscore"]
        row[f"hmm_{marker}_present"] = float(stats["n_hits"] > 0)
    return row


def _existing_ids(jsonl_path: Path) -> set[int]:
    if not jsonl_path.exists():
        return set()
    seen: set[int] = set()
    with open(jsonl_path) as fh:
        for line in fh:
            try:
                seen.add(int(json.loads(line)["bacdive_id"]))
            except Exception:
                continue
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100,
                        help="Max strains to scan (default 100 for the smoke test).")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    print("Step 1: ensuring marker library is present")
    lib_path = download_markers()
    n_hmms = len(_load_hmms(lib_path))
    print(f"  loaded {n_hmms} HMMs from {lib_path}")
    if n_hmms != len(OXYGEN_MARKERS):
        raise SystemExit(f"  expected {len(OXYGEN_MARKERS)} HMMs, got {n_hmms}")

    print("\nStep 2: selecting strains with both genome + oxygen label")
    pheno = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet")
    has_genome = pheno["genome_accession"].notna()
    has_oxygen = pheno["oxygen_requirement"].notna()
    ready = pheno.loc[has_genome & has_oxygen].head(args.limit).copy()
    print(f"  selected {len(ready)} strains")
    print(f"  oxygen distribution: {ready['oxygen_requirement'].value_counts().to_dict()}")

    out_jsonl = config.DATA / "hmm_features_oxygen.jsonl"
    out_parquet = config.DATA / "hmm_features_oxygen.parquet"
    done = _existing_ids(out_jsonl)
    pending = [
        (int(b), str(a), str(lib_path))
        for b, a in zip(ready["bacdive_id"], ready["genome_accession"], strict=True)
        if int(b) not in done
    ]
    print(f"  {len(done)} cached, {len(pending)} new tasks")

    print(f"\nStep 3: streaming fetch + predict + scan ({args.workers} workers)")
    t0 = time.time()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "a") as log, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, t): t for t in pending}
        with tqdm(total=len(pending), unit="strain") as bar:
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

    print("\nStep 4: materializing parquet + sanity-check crosstab")
    rows = []
    with open(out_jsonl) as fh:
        for line in fh:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    print(f"  wrote {out_parquet} ({len(df)} rows × {len(df.columns)} cols)")

    merged = df.merge(
        pheno[["bacdive_id", "oxygen_requirement"]],
        on="bacdive_id",
        how="inner",
    )
    print()
    aerobic_cols = [c for c in df.columns if c.endswith("_aerobic_present")]
    anaerobic_cols = [c for c in df.columns if c.endswith("_anaerobic_present")]
    if aerobic_cols and anaerobic_cols:
        merged["aerobic_marker_count"] = merged[aerobic_cols].sum(axis=1)
        merged["anaerobic_marker_count"] = merged[anaerobic_cols].sum(axis=1)
        print("Mean aerobic marker count by oxygen_requirement:")
        print(merged.groupby("oxygen_requirement")["aerobic_marker_count"].mean().round(2))
        print()
        print("Mean anaerobic marker count by oxygen_requirement:")
        print(merged.groupby("oxygen_requirement")["anaerobic_marker_count"].mean().round(2))


if __name__ == "__main__":
    main()
