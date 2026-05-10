"""Compute per-genome KEGG module completeness from KOfam hits.

Reads:
  data/kegg/modules.json     — module definitions (from script 27)
  data/kofam_hits.jsonl      — per-genome KO sets (from script 28)

Writes:
  data/kegg_modules.parquet  — one row per genome, columns are kegg_<module_id>

Each cell is a 0.0-1.0 fractional completeness score (KEGG-style: average over
the AND chain, max across OR alternatives). XGBoost can read these directly.

Quick to run (~seconds for 22K genomes); rerun any time the module set changes.
"""
from __future__ import annotations

import json

import pandas as pd
from tqdm import tqdm

from microbe_model import config
from microbe_model.features.kegg_modules import module_completeness


def main() -> None:
    modules_path = config.DATA / "kegg" / "modules.json"
    if not modules_path.exists():
        raise SystemExit(f"Missing {modules_path}. Run scripts/27_fetch_kegg_modules.py first.")
    hits_path = config.DATA / "kofam_hits.jsonl"
    if not hits_path.exists():
        raise SystemExit(f"Missing {hits_path}. Run scripts/28_kofam_scan.py first.")

    with open(modules_path) as fh:
        modules = json.load(fh)
    print(f"Loaded {len(modules)} KEGG modules")

    # Pre-parse each rule once so we don't re-parse per genome
    from microbe_model.features.kegg_modules import parse_definition, evaluate
    parsed: list[tuple[str, object]] = []
    for m in modules:
        try:
            ast = parse_definition(m["definition"])
            parsed.append((m["id"], ast))
        except Exception as exc:
            print(f"  ⚠ couldn't parse {m['id']}: {exc}")
    print(f"Parsed {len(parsed)} module rules")

    rows: list[dict] = []
    n = 0
    with open(hits_path) as fh:
        for line in tqdm(fh, desc="genomes"):
            r = json.loads(line)
            ko_set = set(r["ko_hits"])
            row: dict = {"genome_accession": r["genome_accession"]}
            for mod_id, ast in parsed:
                row[f"kegg_{mod_id}"] = evaluate(ast, ko_set, fractional=True)
            rows.append(row)
            n += 1

    df = pd.DataFrame(rows)
    out = config.DATA / "kegg_modules.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote {out}: {len(df):,} rows × {df.shape[1]} cols")

    if len(df) > 0:
        print("\nMean completeness per module (top 10 most-present):")
        means = df.iloc[:, 1:].mean().sort_values(ascending=False).head(10)
        for col, val in means.items():
            mod_id = col[len("kegg_"):]
            name = next((m["name"] for m in modules if m["id"] == mod_id), "")
            print(f"  {val:.2f}  {mod_id}  {name[:55]}")


if __name__ == "__main__":
    main()
