"""Fetch full recipes for every medium referenced in data/strain_media.parquet.

Sequential, polite — MediaDive is a small public API. ~25 min for ~1,500 media at
0.3s/call. Outputs:
  - data/media_metadata.parquet — one row per medium (name, pH range, source, etc.)
  - data/media_recipes.parquet  — one row per (medium_id, compound_id) recipe entry
"""
from __future__ import annotations

import json

import pandas as pd
from tqdm import tqdm

from microbe_model import config
from microbe_model.data.mediadive import MediaDiveClient, normalize_recipe


def main() -> None:
    links_path = config.DATA / "strain_media.parquet"
    if not links_path.exists():
        raise SystemExit(f"Missing {links_path}. Run scripts/08_extract_strain_media.py first.")
    links = pd.read_parquet(links_path)

    medium_ids = sorted(links["medium_id"].dropna().unique().tolist())
    print(f"Fetching {len(medium_ids):,} unique medium recipes from MediaDive")

    cache_dir = config.DATA / "mediadive"
    cache_dir.mkdir(parents=True, exist_ok=True)

    client = MediaDiveClient()
    metadata_rows = []
    recipe_rows = []
    failed: list[str] = []

    for mid in tqdm(medium_ids, desc="MediaDive", unit="medium"):
        cache_path = cache_dir / f"{mid}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
        else:
            payload = client.fetch_medium(mid)
            if payload is None:
                failed.append(mid)
                continue
            cache_path.write_text(json.dumps(payload))

        medium = payload.get("medium") or {}
        metadata_rows.append({
            "medium_id": str(mid),
            "name": medium.get("name"),
            "complex_medium": medium.get("complex_medium"),
            "min_pH": medium.get("min_pH"),
            "max_pH": medium.get("max_pH"),
            "source": medium.get("source"),
            "link": medium.get("link"),
            "n_solutions": len(payload.get("solutions") or []),
        })
        recipe_rows.extend(normalize_recipe(payload))

    md = pd.DataFrame(metadata_rows)
    rc = pd.DataFrame(recipe_rows)
    md.to_parquet(config.DATA / "media_metadata.parquet", index=False)
    rc.to_parquet(config.DATA / "media_recipes.parquet", index=False)

    print(f"\nWrote {len(md):,} media to media_metadata.parquet")
    print(f"Wrote {len(rc):,} compound-rows to media_recipes.parquet")
    print(f"Failed to fetch: {len(failed)}")
    if rc.empty:
        return

    print(f"\nUnique compounds: {rc['compound'].nunique():,}")
    print("Top 15 most-used compounds across all recipes:")
    top = (rc.groupby("compound")
            .agg(n_media=("medium_id", "nunique"), median_g_l=("g_l", "median"))
            .sort_values("n_media", ascending=False)
            .head(15))
    print(top.to_string())


if __name__ == "__main__":
    main()
