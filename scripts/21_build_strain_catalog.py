"""Build a unified per-strain catalog with label provenance.

Every BacDive strain in data/bacdive_phenotypes.parquet gets a row. For each of the
4 phenotype targets, two columns are emitted:
  - <target>          numeric/categorical value (or NaN if unknown)
  - <target>_source   one of: 'bacdive' | 'mediadive_weak' | 'unknown'

`bacdive` means the value came from BacDive's curated optimum / oxygen tolerance
(high-quality). `mediadive_weak` means we derived it from the median pH/NaCl% of
the DSMZ media the strain has been recorded as growing on (lower-quality fallback,
only filled in when BacDive has no value). `unknown` means we have nothing.

Saves to data/strain_catalog.parquet.
"""
from __future__ import annotations

import pandas as pd

from microbe_model import config


def main() -> None:
    pheno = pd.read_parquet(config.DATA / "bacdive_phenotypes.parquet").copy()
    pheno["bacdive_id"] = pheno["bacdive_id"].astype(int)

    # Pull MediaDive-derived signals (per-strain median across grown media)
    md_path = config.DATA / "mediadive_features.parquet"
    md = pd.read_parquet(md_path) if md_path.exists() else pd.DataFrame()
    if len(md):
        md["bacdive_id"] = md["bacdive_id"].astype(int)

    # Build per-target value + source
    out_cols: dict[str, pd.Series] = {}
    for col in ("bacdive_id", "species", "genus", "family", "ncbi_taxon_id",
                "genome_accession", "genome_source"):
        if col in pheno.columns:
            out_cols[col] = pheno[col]

    # Direct BacDive labels (canonical). For each, fall back to MediaDive when missing.
    targets = {
        "optimal_temperature_c": None,        # no MediaDive proxy for temperature
        "optimal_ph": "md_ph_median" if len(md) else None,
        "oxygen_requirement": None,           # no media-based proxy
        "salt_tolerance_pct": "md_nacl_pct_median" if len(md) else None,
    }

    md_indexed = md.set_index("bacdive_id") if len(md) else None
    for target, md_col in targets.items():
        bacdive_vals = pheno.set_index("bacdive_id")[target] if target in pheno.columns else None
        # Reindex to match pheno order
        ordered_bacdive = (
            bacdive_vals.reindex(pheno["bacdive_id"]).values if bacdive_vals is not None else None
        )
        # MediaDive-derived fallback (numeric only; salt% capped at 30 already)
        weak_vals = (
            md_indexed[md_col].reindex(pheno["bacdive_id"]).values
            if md_col and md_indexed is not None and md_col in md_indexed.columns
            else None
        )

        values = []
        sources = []
        for i in range(len(pheno)):
            v = ordered_bacdive[i] if ordered_bacdive is not None else None
            # pandas-aware NaN check
            v_is_na = (v is None) or (isinstance(v, float) and pd.isna(v)) or (
                isinstance(v, str) and not v
            )
            if not v_is_na:
                values.append(v)
                sources.append("bacdive")
                continue
            wv = weak_vals[i] if weak_vals is not None else None
            wv_is_na = wv is None or (isinstance(wv, float) and pd.isna(wv))
            if not wv_is_na:
                values.append(wv)
                sources.append("mediadive_weak")
                continue
            values.append(None)
            sources.append("unknown")

        out_cols[target] = pd.Series(values, dtype="object")
        out_cols[f"{target}_source"] = pd.Series(sources, dtype="string")

    catalog = pd.DataFrame(out_cols)
    out = config.DATA / "strain_catalog.parquet"
    catalog.to_parquet(out, index=False)
    print(f"wrote {len(catalog):,} strains to {out}\n")

    # Summary per target
    for target in targets:
        src_col = f"{target}_source"
        counts = catalog[src_col].value_counts().to_dict()
        n_known = counts.get("bacdive", 0) + counts.get("mediadive_weak", 0)
        print(f"{target}")
        print(f"  bacdive (curated):   {counts.get('bacdive', 0):>7,}")
        print(f"  mediadive_weak:      {counts.get('mediadive_weak', 0):>7,}")
        print(f"  unknown:             {counts.get('unknown', 0):>7,}")
        print(f"  ─ any-known:         {n_known:>7,} ({100*n_known/len(catalog):.0f}%)")
        print()


if __name__ == "__main__":
    main()
