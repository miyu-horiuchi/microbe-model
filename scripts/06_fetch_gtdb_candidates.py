"""Fetch GTDB representative genomes that don't appear in BacDive.

The point: BacDive only contains organisms that have been cultured at least once.
GTDB representative genomes that lack a BacDive entry are the "novel cultivation
candidates" we want to predict on — these are the closest thing we have to
genuinely uncultured organisms while still being concrete genomes we can fetch.

Inputs:
  - data/bacdive_phenotypes.parquet (must exist)
  - https://data.gtdb.ecogenomic.org/releases/latest/bac120_metadata.tsv.gz

Output:
  - data/gtdb_candidates.parquet: subset of GTDB representatives not in BacDive

Usage:
    uv run python scripts/06_fetch_gtdb_candidates.py --max 5000
"""
from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from microbe_model import config

GTDB_BAC120_URL = "https://data.gtdb.ecogenomic.org/releases/latest/bac120_metadata.tsv.gz"


def _normalize_accession(acc: str) -> str:
    """Strip prefixes like 'GB_' / 'RS_' that GTDB prepends, return bare accession."""
    if not isinstance(acc, str):
        return ""
    m = re.match(r"^(?:GB_|RS_)?(.+)$", acc)
    return m.group(1) if m else acc


def _strip_version(acc: str) -> str:
    return re.sub(r"\.\d+$", "", acc) if acc else acc


def download_gtdb_metadata() -> Path:
    """Cache the GTDB metadata TSV locally; ~275MB compressed."""
    out = config.DATA / "bac120_metadata.tsv.gz"
    if out.exists():
        return out
    print(f"Downloading {GTDB_BAC120_URL} → {out} (~275 MB, takes a few min)")
    with requests.get(GTDB_BAC120_URL, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(out, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc="GTDB"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
                bar.update(len(chunk))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=5000,
                        help="Cap number of candidates returned (default 5000).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pheno_path = config.DATA / "bacdive_phenotypes.parquet"
    if not pheno_path.exists():
        raise SystemExit(f"Missing {pheno_path}. Run scripts/01_fetch_bacdive.py first.")
    pheno = pd.read_parquet(pheno_path)
    bacdive_acc = set(pheno["genome_accession"].dropna().astype(str).map(_strip_version))
    print(f"BacDive has {len(bacdive_acc):,} unique unversioned accessions")

    metadata_path = download_gtdb_metadata()
    print("Reading GTDB metadata (this takes ~30-60s)...")
    with gzip.open(metadata_path, "rt") as fh:
        meta = pd.read_csv(
            fh, sep="\t", low_memory=False,
            usecols=lambda c: c in {
                "accession", "gtdb_representative", "gtdb_taxonomy",
                "ncbi_genbank_assembly_accession", "ncbi_organism_name",
                "checkm_completeness", "ncbi_assembly_level",
            },
        )
    print(f"Loaded {len(meta):,} GTDB rows")

    representatives = meta[meta["gtdb_representative"] == "t"].copy()
    print(f"{len(representatives):,} are GTDB representatives")

    # Build the bare-accession candidate set (use ncbi_genbank_assembly_accession;
    # GTDB's `accession` column has GB_/RS_ prefixes — strip them too as a fallback.)
    def _accession(row: pd.Series) -> str:
        acc = row.get("ncbi_genbank_assembly_accession")
        if isinstance(acc, str) and acc.startswith("GCA_"):
            return _strip_version(acc)
        return _strip_version(_normalize_accession(row.get("accession", "")))

    representatives["_acc_clean"] = representatives.apply(_accession, axis=1)
    representatives = representatives[representatives["_acc_clean"] != ""]
    not_in_bacdive = representatives[~representatives["_acc_clean"].isin(bacdive_acc)].copy()
    print(f"{len(not_in_bacdive):,} representatives are not in BacDive")

    # Light quality filter: prefer reasonably complete genomes
    not_in_bacdive["checkm_completeness"] = pd.to_numeric(
        not_in_bacdive["checkm_completeness"], errors="coerce"
    )
    high_quality = not_in_bacdive[not_in_bacdive["checkm_completeness"] >= 90].copy()
    print(f"{len(high_quality):,} are >=90% complete (CheckM)")

    if len(high_quality) < args.max:
        sample = high_quality
    else:
        sample = high_quality.sample(n=args.max, random_state=args.seed)

    sample = sample.rename(columns={
        "_acc_clean": "genome_accession",
        "ncbi_genbank_assembly_accession": "ncbi_assembly_accession_versioned",
    })
    sample = sample[[
        "accession", "genome_accession", "ncbi_assembly_accession_versioned",
        "gtdb_taxonomy", "ncbi_organism_name", "checkm_completeness",
    ]]
    out = config.DATA / "gtdb_candidates.parquet"
    sample.to_parquet(out, index=False)
    print(f"\nWrote {len(sample):,} candidates to {out}")


if __name__ == "__main__":
    main()
