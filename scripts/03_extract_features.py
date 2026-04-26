"""Extract tabular genome features for every cached genome.

Reads the BacDive phenotype table + the cached FASTAs in data/genomes/, runs Pyrodigal +
amino-acid-composition feature extraction, and writes data/features.parquet (one row per strain).
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from microbe_model import config
from microbe_model.data.ncbi import genome_path
from microbe_model.features.genome import extract_features


def main() -> None:
    pheno_path = config.DATA / "bacdive_phenotypes.parquet"
    if not pheno_path.exists():
        raise SystemExit(f"Missing {pheno_path}. Run 01 then 02 first.")
    pheno = pd.read_parquet(pheno_path)

    rows = []
    for _, row in tqdm(pheno.iterrows(), total=len(pheno), desc="features"):
        accession = row["genome_accession"]
        if not accession:
            continue
        path = genome_path(accession)
        if not path.exists():
            continue
        try:
            feats = extract_features(path)
        except Exception as exc:  # noqa: BLE001 — bad FASTA shouldn't kill the run
            print(f"  skip {accession}: {type(exc).__name__}: {exc}")
            continue
        feats["bacdive_id"] = row["bacdive_id"]
        feats["genome_accession"] = accession
        rows.append(feats)

    feats_df = pd.DataFrame(rows)
    out = config.DATA / "features.parquet"
    feats_df.to_parquet(out, index=False)
    print(f"\nWrote {len(feats_df)} rows to {out}")


if __name__ == "__main__":
    main()
