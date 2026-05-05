"""Resolve BacDive species names → NCBI representative genome accessions.

Targets the phenotype-labeled strains that lack a `genome_accession` in BacDive.
Many of those species DO have a sequenced genome — BacDive just doesn't link to it.
We query NCBI Datasets v2 for one RefSeq assembly per unique species name and write
the {species: accession} map so the next pipeline step can pull the FASTAs.

Output: data/species_to_genome.parquet  (species, ncbi_accession, status)

Resumable: re-runs skip species already present in the output.
"""
from __future__ import annotations

import time
from urllib.parse import quote

import pandas as pd
import requests
from tqdm import tqdm

from microbe_model import config

API_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/taxon/{taxon}/dataset_report"
RATE_LIMIT_S = 0.11 if config.NCBI_API_KEY else 0.36
OUT_PATH = config.DATA / "species_to_genome.parquet"


def fetch_one(species: str, session: requests.Session) -> tuple[str | None, str]:
    """Return (accession, status) for a species. status ∈ {hit, miss, error}."""
    headers: dict[str, str] = {"Accept": "application/json"}
    if config.NCBI_API_KEY:
        headers["api-key"] = config.NCBI_API_KEY
    params = {"filters.assembly_source": "RefSeq", "page_size": 1}

    for attempt in range(3):
        try:
            time.sleep(RATE_LIMIT_S)
            resp = session.get(
                API_URL.format(taxon=quote(species)),
                headers=headers,
                params=params,
                timeout=30,
            )
            if resp.status_code == 404:
                return None, "miss"
            if resp.status_code in (429, 502, 503):
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            reports = resp.json().get("reports", [])
            if reports:
                acc = reports[0].get("accession")
                return (acc, "hit") if acc else (None, "miss")
            return None, "miss"
        except requests.RequestException:
            if attempt == 2:
                return None, "error"
            time.sleep(2 ** attempt)
    return None, "error"


def main() -> None:
    df = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet")
    has_label = df[
        ["optimal_temperature_c", "optimal_ph", "oxygen_requirement", "salt_tolerance_pct"]
    ].notna().any(axis=1)
    no_genome = df["genome_accession"].isna()
    valid_species = df["species"].notna() & df["species"].str.contains(" ", na=False)
    gap_species = sorted(df[has_label & no_genome & valid_species]["species"].unique())
    print(f"unique species to resolve: {len(gap_species):,}")

    # Resume from prior partial run
    done: dict[str, tuple[str | None, str]] = {}
    if OUT_PATH.exists():
        prev = pd.read_parquet(OUT_PATH)
        for _, row in prev.iterrows():
            done[row["species"]] = (row["ncbi_accession"], row["status"])
        print(f"resuming — {len(done):,} already cached")

    todo = [s for s in gap_species if s not in done]
    print(f"to fetch: {len(todo):,}")

    session = requests.Session()
    rows: list[dict] = [
        {"species": sp, "ncbi_accession": acc, "status": st}
        for sp, (acc, st) in done.items()
    ]
    n_hits = sum(1 for _, st in done.values() if st == "hit")

    try:
        for sp in tqdm(todo, desc="resolving", unit="species"):
            acc, status = fetch_one(sp, session)
            rows.append({"species": sp, "ncbi_accession": acc, "status": status})
            if status == "hit":
                n_hits += 1
            # Periodic checkpoint every 200 species so an interrupt doesn't lose progress
            if len(rows) % 200 == 0:
                pd.DataFrame(rows).to_parquet(OUT_PATH, index=False)
    finally:
        pd.DataFrame(rows).to_parquet(OUT_PATH, index=False)

    out = pd.DataFrame(rows)
    print(f"\nwrote {len(out):,} rows to {OUT_PATH}")
    print(f"  hit:   {(out['status'] == 'hit').sum():,} ({100 * (out['status'] == 'hit').mean():.0f}%)")
    print(f"  miss:  {(out['status'] == 'miss').sum():,}")
    print(f"  error: {(out['status'] == 'error').sum():,}")


if __name__ == "__main__":
    main()
