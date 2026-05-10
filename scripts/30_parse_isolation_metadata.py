"""Extract richer environment-of-origin features from raw BacDive JSON.

Reads:
  data/bacdive/*.json  — one file per BacDive ID

Writes:
  data/isolation_metadata.parquet — one row per bacdive_id with cols:
    - iso_country, iso_continent (categorical → caller can one-hot)
    - iso_lat, iso_lon (float, NaN if missing)
    - iso_collection_year (int from sampling/isolation date, NaN if missing)
    - iso_host_species (string, NaN if missing)
    - iso_sample_text (free-text description, for downstream NLP if needed)
    - iso_continent_<X> binary one-hots (8 continents)
    - iso_country_<X> top-30 country one-hots
    - iso_host_kingdom (animal / plant / human / fungal / NaN — coarse map)

Wires into baseline by genome's bacdive_id (one strain per row, not per genome).
"""
from __future__ import annotations

import glob
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from microbe_model import config

DATE_RE = re.compile(r"\b(19|20)\d{2}\b")
HOST_KINGDOM_KEYWORDS = {
    "human": "human",
    "homo sapiens": "human",
    "patient": "human",
    "infant": "human",
    "mouse": "animal", "rat": "animal", "cow": "animal", "bovine": "animal",
    "pig": "animal", "swine": "animal", "chicken": "animal", "fish": "animal",
    "honey bee": "animal", "insect": "animal", "termite": "animal",
    "bird": "animal", "tick": "animal",
    "plant": "plant", "rice": "plant", "wheat": "plant", "soybean": "plant",
    "tomato": "plant", "leaf": "plant", "root": "plant", "rhizosphere": "plant",
    "fungus": "fungal", "yeast": "fungal", "mushroom": "fungal",
}


def coerce_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_year(s) -> int | None:
    if not s or not isinstance(s, str):
        return None
    m = DATE_RE.search(s)
    if not m:
        return None
    y = int(m.group(0))
    return y if 1850 <= y <= 2100 else None


def host_kingdom(host_str) -> str | None:
    if not host_str or not isinstance(host_str, str):
        return None
    s = host_str.lower()
    for k, v in HOST_KINGDOM_KEYWORDS.items():
        if k in s:
            return v
    return "other"


def extract_one(path: Path) -> dict | None:
    try:
        with open(path) as fh:
            d = json.load(fh)
    except Exception:
        return None

    bid_str = path.stem  # filename is e.g. "12345.json"
    try:
        bid = int(bid_str)
    except ValueError:
        return None

    iso_section = d.get("Isolation, sampling and environmental information", {})
    if not isinstance(iso_section, dict):
        return {"bacdive_id": bid}

    iso = iso_section.get("isolation", {})
    if isinstance(iso, list):
        iso = iso[0] if iso else {}
    if not isinstance(iso, dict):
        iso = {}

    sample_type = iso.get("sample type")
    sample_text = sample_type if isinstance(sample_type, str) else None

    year = parse_year(iso.get("sampling date")) or parse_year(iso.get("isolation date"))
    host_species = iso.get("host species") if isinstance(iso.get("host species"), str) else None

    return {
        "bacdive_id": bid,
        "iso_country": iso.get("country") if isinstance(iso.get("country"), str) else None,
        "iso_continent": iso.get("continent") if isinstance(iso.get("continent"), str) else None,
        "iso_lat": coerce_float(iso.get("latitude")),
        "iso_lon": coerce_float(iso.get("longitude")),
        "iso_collection_year": year,
        "iso_host_species": host_species,
        "iso_host_kingdom": host_kingdom(host_species) or host_kingdom(sample_text),
        "iso_sample_text": sample_text,
        "iso_geographic_location": iso.get("geographic location") if isinstance(iso.get("geographic location"), str) else None,
    }


def add_categorical_onehots(df: pd.DataFrame, top_n_countries: int = 30) -> pd.DataFrame:
    # continent one-hots (small, fixed set)
    continents = [c for c in df["iso_continent"].dropna().unique()]
    for c in continents:
        slug = re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")
        if slug:
            df[f"iso_continent_{slug}"] = (df["iso_continent"] == c).astype(int)

    # top-N country one-hots (long tail; cap to keep feature count manageable)
    top_countries = df["iso_country"].value_counts().head(top_n_countries).index.tolist()
    for c in top_countries:
        slug = re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")
        if slug:
            df[f"iso_country_{slug}"] = (df["iso_country"] == c).astype(int)

    # host-kingdom one-hots (very small fixed set)
    for k in ("human", "animal", "plant", "fungal", "other"):
        df[f"iso_host_kingdom_{k}"] = (df["iso_host_kingdom"] == k).astype(int)

    return df


def main() -> None:
    bacdive_dir = config.DATA / "bacdive"
    if not bacdive_dir.exists():
        raise SystemExit(f"Missing {bacdive_dir}")

    paths = list(bacdive_dir.glob("*.json"))
    print(f"Parsing {len(paths):,} BacDive JSON files...")

    rows: list[dict] = []
    for p in tqdm(paths, unit="file"):
        r = extract_one(p)
        if r:
            rows.append(r)
    df = pd.DataFrame(rows)
    print(f"Parsed {len(df):,} rows")

    # Coverage report on the high-value fields
    for col in ["iso_country", "iso_continent", "iso_lat", "iso_lon",
                "iso_collection_year", "iso_host_species", "iso_host_kingdom"]:
        nn = df[col].notna().sum() if col in df.columns else 0
        print(f"  {col:30s} {nn:>6,} populated  ({100*nn/len(df):.1f}%)")

    df = add_categorical_onehots(df)

    out = config.DATA / "isolation_metadata.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote {out}: {len(df):,} rows × {df.shape[1]} cols")

    # Sample
    print("\nMost common countries (sanity check):")
    print(df["iso_country"].value_counts().head(10))


if __name__ == "__main__":
    main()
