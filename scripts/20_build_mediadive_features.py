"""Build per-strain MediaDive features from strain_media + media_recipes + raw JSON.

For each BacDive strain, compute the median pH and NaCl% across all DSMZ media that
strain has been recorded as growing on. These are NOT labels — they're additional
features the model can use to predict the actual phenotype optima. Saves to
data/mediadive_features.parquet (joined into the training table by scripts/03).

Per-strain features written:
  - md_n_media:        count of media the strain grows on
  - md_ph_median:      median midpoint(min_pH, max_pH) across those media
  - md_ph_range:       max - min of medium pH across those media
  - md_nacl_pct_median:median NaCl % w/v across those media
  - md_nacl_pct_max:   max NaCl % w/v (highest tolerated)

Sanity check: where a BacDive optimum_pH or salt_tolerance_pct exists, we expect
moderate (not perfect) correlation with the corresponding MediaDive feature.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from microbe_model import config

NACL_CAP_PCT = 30.0  # clip recipes with absurd NaCl values (parse artifacts)


def build_medium_ph_map() -> dict[str, float]:
    """Return {medium_id: midpoint pH} from raw MediaDive cache."""
    out: dict[str, float] = {}
    for path in Path(config.DATA / "mediadive").glob("*.json"):
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        m = d.get("medium")
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        min_ph = m.get("min_pH")
        max_ph = m.get("max_pH")
        if mid is None or min_ph is None or max_ph is None:
            continue
        try:
            out[str(mid)] = (float(min_ph) + float(max_ph)) / 2
        except (ValueError, TypeError):
            continue
    return out


def build_medium_nacl_map() -> dict[str, float]:
    """Return {medium_id: NaCl % w/v} summed from recipe compounds (clipped)."""
    mr = pd.read_parquet(config.DATA / "media_recipes.parquet")
    nacl = mr[mr["compound"].str.contains(r"sodium chlor|^nacl$", case=False, na=False, regex=True)]
    pct = (nacl.groupby("medium_id")["g_l"].sum() / 10).clip(upper=NACL_CAP_PCT)
    return pct.astype(float).to_dict()


def main() -> None:
    sm = pd.read_parquet(config.DATA / "strain_media.parquet")
    sm = sm[sm["growth"].str.lower() == "yes"].copy()
    sm["medium_id"] = sm["medium_id"].astype(str)

    ph_map = build_medium_ph_map()
    nacl_map = build_medium_nacl_map()
    print(f"medium pH map: {len(ph_map):,} media")
    print(f"medium NaCl map: {len(nacl_map):,} media")

    sm["m_ph"] = sm["medium_id"].map(ph_map)
    # Strains may grow on media not in the recipe table — treat absent as 0% NaCl
    sm["m_nacl"] = sm["medium_id"].map(nacl_map).fillna(0.0)

    # Aggregate per-strain
    grouped = sm.groupby("bacdive_id")
    feat = pd.DataFrame({
        "md_n_media": grouped.size(),
        "md_ph_median": grouped["m_ph"].median(),
        "md_ph_range": grouped["m_ph"].max() - grouped["m_ph"].min(),
        "md_nacl_pct_median": grouped["m_nacl"].median(),
        "md_nacl_pct_max": grouped["m_nacl"].max(),
    }).reset_index()

    out = config.DATA / "mediadive_features.parquet"
    feat.to_parquet(out, index=False)
    print(f"\nwrote {len(feat):,} strains to {out}")
    print(feat.describe().round(2).to_string())


if __name__ == "__main__":
    main()
