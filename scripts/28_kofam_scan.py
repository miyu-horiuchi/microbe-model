"""KOfam scan — assign KEGG Orthologue (KO) hits to every genome.

Same architecture as scripts/24_unified_hmm_scan.py, but the HMM library is
KOfam (~25,000 KOs) instead of our curated Pfam set. Output is the SET OF KOs
present in each genome (not a per-marker count, since each KO is itself one
specific gene we either find or don't).

Per-KO bitscore thresholds come from KEGG's ko_list — using the threshold
prevents false positives from distant Pfam-style domain matches.

Usage:
    # one-time: fetch + extract KOfam (~5 min, 1.6 GB on disk)
    python scripts/28_kofam_scan.py --fetch-only

    # smoke test
    python scripts/28_kofam_scan.py --limit 10 --workers 4

    # full corpus
    python scripts/28_kofam_scan.py --workers 8

Output: data/kofam_hits.parquet — one row per genome:
    genome_accession, ko_K00001, ko_K00002, ..., ko_K25000   (binary 0/1)

WARNING: full corpus scan against 25K HMMs is slow on local CPU. Plan ~2-4 min
per genome × 22K genomes / 8 workers ≈ ~3-4 days single machine, or run on
Modal CPU containers (much cheaper than GPU; pennies per hour each).
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tarfile
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
from microbe_model.pipeline import _fetch_fasta_bytes

KOFAM_DIR = config.DATA / "kofam"
KO_LIST_URL = "https://www.genome.jp/ftp/db/kofam/ko_list.gz"
PROFILES_URL = "https://www.genome.jp/ftp/db/kofam/profiles.tar.gz"
KOFAM_HMM = KOFAM_DIR / "kofam.hmm"            # concatenation of ALL 27K HMMs
KOFAM_RELEVANT_HMM = KOFAM_DIR / "kofam_relevant.hmm"  # subset matching kegg/relevant_kos.txt
KO_THRESHOLDS = KOFAM_DIR / "ko_thresholds.tsv" # parsed thresholds
RELEVANT_KOS_PATH = config.DATA / "kegg" / "relevant_kos.txt"
DEFAULT_EVALUE = 1e-5  # used when KO has no recommended bitscore threshold


def build_relevant_library() -> Path:
    """Concatenate only the HMMs whose KO is in data/kegg/relevant_kos.txt.

    Reduces scan cost by ~9× (3K vs 27K HMMs). Built once on first invocation.
    """
    if KOFAM_RELEVANT_HMM.exists():
        return KOFAM_RELEVANT_HMM
    if not RELEVANT_KOS_PATH.exists():
        raise SystemExit(
            f"Missing {RELEVANT_KOS_PATH}. Run scripts/27_fetch_kegg_modules.py "
            "first so we know which KOs to scan."
        )
    relevant = set(RELEVANT_KOS_PATH.read_text().strip().splitlines())
    profiles_dir = KOFAM_DIR / "profiles"
    if not profiles_dir.exists():
        raise SystemExit(f"Missing {profiles_dir}. Run --fetch-only first.")

    print(f"Building relevant-KO library ({len(relevant):,} KOs)...")
    found = 0
    with open(KOFAM_RELEVANT_HMM, "wb") as out:
        for ko in sorted(relevant):
            f = profiles_dir / f"{ko}.hmm"
            if f.exists():
                out.write(f.read_bytes())
                found += 1
    print(f"  wrote {KOFAM_RELEVANT_HMM} ({found:,} HMMs)")
    return KOFAM_RELEVANT_HMM


def fetch_kofam() -> None:
    """Download and extract KOfam profiles + thresholds."""
    KOFAM_DIR.mkdir(parents=True, exist_ok=True)

    if KOFAM_RELEVANT_HMM.exists() and KO_THRESHOLDS.exists():
        return

    ko_list_gz = KOFAM_DIR / "ko_list.gz"
    if not KO_THRESHOLDS.exists():
        if not ko_list_gz.exists():
            print(f"Downloading {KO_LIST_URL} (~900 KB)...")
            r = requests.get(KO_LIST_URL, stream=True, timeout=120)
            r.raise_for_status()
            with open(ko_list_gz, "wb") as fh:
                shutil.copyfileobj(r.raw, fh)
        with gzip.open(ko_list_gz, "rt") as fh, open(KO_THRESHOLDS, "w") as out:
            out.write("ko\tthreshold\tscore_type\tprofile_type\tf_measure\tnseq\tnseq_used\talen\tmlen\teff_nseq\tre_pos\tdefinition\n")
            next(fh)  # skip header
            for line in fh:
                out.write(line)
        print(f"  parsed → {KO_THRESHOLDS}")

    profiles_tgz = KOFAM_DIR / "profiles.tar.gz"
    profiles_dir = KOFAM_DIR / "profiles"
    if not profiles_dir.exists():
        if not profiles_tgz.exists():
            print(f"Downloading {PROFILES_URL} (~1.55 GB) — go grab coffee...")
            r = requests.get(PROFILES_URL, stream=True, timeout=600)
            r.raise_for_status()
            with open(profiles_tgz, "wb") as fh:
                shutil.copyfileobj(r.raw, fh, length=1024 * 1024)
        print("Extracting profiles tarball...")
        with tarfile.open(profiles_tgz, "r:gz") as tf:
            tf.extractall(KOFAM_DIR)

    if not KOFAM_HMM.exists():
        print("Concatenating individual .hmm files into one library...")
        hmm_files = sorted(profiles_dir.glob("K*.hmm"))
        with open(KOFAM_HMM, "wb") as out:
            for h in hmm_files:
                out.write(h.read_bytes())
        print(f"  wrote {KOFAM_HMM} ({len(hmm_files)} HMMs)")


def load_ko_thresholds() -> dict[str, float]:
    """Return {KO: bitscore_threshold}. Default to 0 if missing/non-numeric."""
    if not KO_THRESHOLDS.exists():
        return {}
    df = pd.read_csv(KO_THRESHOLDS, sep="\t")
    out: dict[str, float] = {}
    for ko, thr in zip(df["ko"], df["threshold"], strict=True):
        try:
            out[str(ko)] = float(thr)
        except (TypeError, ValueError):
            out[str(ko)] = 0.0
    return out


def _load_hmms(lib_path: Path) -> list[pyhmmer.plan7.HMM]:
    with pyhmmer.plan7.HMMFile(str(lib_path)) as fh:
        return list(fh)


def scan_proteins(
    proteins: list[str],
    hmms: list[pyhmmer.plan7.HMM],
    alphabet: pyhmmer.easel.Alphabet,
    thresholds: dict[str, float],
) -> set[str]:
    seqs: list[pyhmmer.easel.DigitalSequence] = []
    for i, prot in enumerate(proteins):
        if not prot:
            continue
        ts = pyhmmer.easel.TextSequence(name=f"p{i}".encode(), sequence=prot)
        seqs.append(ts.digitize(alphabet))
    found: set[str] = set()
    if not seqs:
        return found
    for top_hits in pyhmmer.hmmer.hmmsearch(hmms, seqs, E=DEFAULT_EVALUE, cpus=1):
        raw_name = top_hits.query.name
        ko = raw_name.decode() if isinstance(raw_name, bytes) else raw_name
        thr = thresholds.get(ko, 0.0)
        for hit in top_hits:
            if hit.score >= thr and hit.evalue <= DEFAULT_EVALUE:
                found.add(ko)
                break
    return found


def _process_one(args: tuple[str, str, str]) -> dict[str, Any] | None:
    accession, lib_path, thresholds_path = args
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
    thresholds = load_ko_thresholds()
    ko_hits = scan_proteins(proteins, hmms, alphabet, thresholds)
    return {"genome_accession": accession, "ko_hits": sorted(ko_hits)}


def _existing_accessions(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    seen: set[str] = set()
    with open(jsonl_path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:
                continue
            acc = row.get("genome_accession") or row.get("accession")
            if acc:
                seen.add(str(acc))
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-only", action="store_true",
                        help="Just download + extract KOfam; don't scan.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    fetch_kofam()
    if args.fetch_only:
        build_relevant_library()
        return

    lib_path = build_relevant_library()
    n_hmms = sum(1 for _ in _load_hmms(lib_path))
    print(f"\nLoaded {n_hmms:,} KOfam HMMs from {lib_path}")

    feats = pd.read_parquet(config.DATA / "features.parquet")
    unique_accs = feats["genome_accession"].dropna().astype(str).unique().tolist()
    if args.limit:
        unique_accs = unique_accs[: args.limit]
    print(f"{len(unique_accs):,} unique genome accessions to scan")

    out_jsonl = config.DATA / "kofam_hits.jsonl"
    done = _existing_accessions(out_jsonl)
    pending = [(acc, str(lib_path), str(KO_THRESHOLDS)) for acc in unique_accs if acc not in done]
    print(f"{len(done):,} cached, {len(pending):,} new tasks")

    t0 = time.time()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "a") as log, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, t): t for t in pending}
        with tqdm(total=len(pending), unit="genome") as bar:
            n_ok = 0
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                except Exception:
                    r = None
                bar.update(1)
                if r is None:
                    continue
                log.write(json.dumps(r) + "\n")
                log.flush()
                n_ok += 1
                bar.set_postfix(ok=n_ok)
    print(f"Scan finished in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
