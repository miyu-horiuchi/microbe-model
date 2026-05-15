"""Build data/per_marker_embeddings.parquet from data/per_marker_embeddings.jsonl.

Streams the (large) jsonl, keeps the last record per genome_accession (Modal jobs may
have written multiple times during retries), and casts the embedding dims to float32
to keep the parquet under ~500 MB.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from microbe_model import config


def main() -> None:
    src = config.DATA / "per_marker_embeddings.jsonl"
    if not src.exists():
        raise SystemExit(f"Missing {src}")

    t0 = time.time()
    by_genome: dict[str, dict] = {}
    n_lines = 0
    with open(src) as fh:
        for line in fh:
            n_lines += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ga = r.get("genome_accession") or r.get("accession")
            if not ga:
                continue
            by_genome[ga] = r
            if n_lines % 5000 == 0:
                print(f"  read {n_lines:,} lines, {len(by_genome):,} unique genomes")

    print(f"Parsed {n_lines:,} lines → {len(by_genome):,} unique genomes ({time.time()-t0:.1f}s)")

    rows = list(by_genome.values())
    df = pd.DataFrame(rows)
    if "bacdive_id" in df.columns:
        df["bacdive_id"] = pd.to_numeric(df["bacdive_id"], errors="coerce").astype("Int64")

    float_cols = [c for c in df.columns if c.startswith("pme_") and c != "pme_marker_proteins_total"]
    df[float_cols] = df[float_cols].astype(np.float32)
    if "pme_marker_proteins_total" in df.columns:
        df["pme_marker_proteins_total"] = pd.to_numeric(
            df["pme_marker_proteins_total"], errors="coerce"
        ).astype("Int32")

    # Reorder: ids first, then features
    id_cols = [c for c in ("bacdive_id", "genome_accession") if c in df.columns]
    other_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + other_cols]

    out = config.DATA / "per_marker_embeddings.parquet"
    df.to_parquet(out, index=False)
    sz_mb = out.stat().st_size / 1e6
    print(f"Wrote {len(df):,} rows × {df.shape[1]} cols → {out} ({sz_mb:.1f} MB, {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
