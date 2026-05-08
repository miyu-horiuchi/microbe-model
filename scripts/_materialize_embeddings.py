"""Build data/embeddings.parquet from data/embeddings.jsonl.

Use after Modal/local extraction finishes (or partway through, to checkpoint).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from microbe_model import config


def main() -> None:
    src = config.DATA / "embeddings.jsonl"
    if not src.exists():
        raise SystemExit(f"Missing {src}")
    rows: list[dict] = []
    with open(src) as fh:
        for line in fh:
            r = json.loads(line)
            d = {"bacdive_id": r["bacdive_id"], "genome_accession": r["genome_accession"]}
            d.update({f"emb_{i}": float(v) for i, v in enumerate(r["embedding"])})
            rows.append(d)
    df = pd.DataFrame(rows)
    out = config.DATA / "embeddings.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df):,} embeddings × {df.shape[1]} cols to {out}")


if __name__ == "__main__":
    main()
