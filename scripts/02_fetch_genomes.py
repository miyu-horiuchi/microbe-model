"""Download genome FASTAs for every BacDive strain that has an accession.

Skips strains already cached on disk. Run after 01_fetch_bacdive.py.
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from microbe_model import config
from microbe_model.data.ncbi import GenomeNotFound, fetch_genome


def main() -> None:
    table = config.DATA / "bacdive_phenotypes.parquet"
    if not table.exists():
        raise SystemExit(f"Missing {table}. Run scripts/01_fetch_bacdive.py first.")

    df = pd.read_parquet(table)
    accessions = df["genome_accession"].dropna().unique().tolist()
    print(f"{len(accessions)} unique genome accessions to fetch.")

    failed: list[tuple[str, str]] = []
    for acc in tqdm(accessions, desc="NCBI", unit="genome"):
        try:
            fetch_genome(acc)
        except GenomeNotFound:
            failed.append((acc, "not_found"))
        except Exception as exc:  # noqa: BLE001 — log and continue, don't kill the batch
            failed.append((acc, type(exc).__name__))

    print(f"\nDownloaded: {len(accessions) - len(failed)} / {len(accessions)}")
    if failed:
        log = config.DATA / "genome_fetch_failures.tsv"
        pd.DataFrame(failed, columns=["accession", "error"]).to_csv(log, sep="\t", index=False)
        print(f"Failures logged to {log}")


if __name__ == "__main__":
    main()
