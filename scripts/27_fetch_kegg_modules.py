"""Fetch KEGG module definitions via REST API.

Each module has a DEFINITION line that is a boolean rule over KEGG Orthologue
(KO) IDs:

    DEFINITION  K00001 (K01810,K06859) K00850 ...

Rule grammar:
  - whitespace      → AND  (sequential pathway steps)
  - comma in parens → OR   (alternative isozymes)
  - parens          → grouping

We parse this rule into an AST so it can be evaluated against a per-genome KO
set to produce a 0.0-1.0 completeness score (see 28_compute_module_completeness).

Output: data/kegg/modules.json
    [
      {"id": "M00001", "name": "Glycolysis ...", "definition": "K00844 ...",
       "ko_list": ["K00844", ...], "category": "Carbohydrate metabolism"},
      ...
    ]

KEGG REST has a soft rate limit; we sleep 100 ms between calls to stay polite.
~470 modules → ~1 minute total.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from microbe_model import config

LIST_URL = "https://rest.kegg.jp/list/module"
GET_URL = "https://rest.kegg.jp/get/{module_id}"
SLEEP_S = 0.10


def fetch_module_list() -> list[tuple[str, str]]:
    """Return [(M00001, 'Glycolysis ...'), ...]."""
    resp = requests.get(LIST_URL, timeout=30)
    resp.raise_for_status()
    out: list[tuple[str, str]] = []
    for line in resp.text.splitlines():
        if "\t" not in line:
            continue
        mod_id, name = line.split("\t", 1)
        if mod_id.startswith("M") and mod_id[1:].isdigit():
            out.append((mod_id, name))
    return out


def fetch_module_detail(module_id: str) -> dict | None:
    resp = requests.get(GET_URL.format(module_id=module_id), timeout=30)
    if resp.status_code != 200:
        return None
    text = resp.text

    # Extract DEFINITION line (may span multiple lines if very long)
    definition = ""
    name = ""
    category = ""
    in_definition = False
    in_class = False
    for line in text.splitlines():
        if line.startswith("NAME"):
            name = line[len("NAME"):].strip()
            in_definition = False
            in_class = False
        elif line.startswith("DEFINITION"):
            definition = line[len("DEFINITION"):].strip()
            in_definition = True
            in_class = False
        elif line.startswith("CLASS"):
            category = line[len("CLASS"):].strip()
            in_class = True
            in_definition = False
        elif line.startswith(" ") and (in_definition or in_class):
            cont = line.strip()
            if in_definition:
                definition += " " + cont
            else:
                category += " " + cont
        else:
            in_definition = False
            in_class = False

    if not definition:
        return None

    # Extract every KO (Knnnnn) appearing in the definition
    ko_list = sorted(set(re.findall(r"K\d{5}", definition)))

    return {
        "id": module_id,
        "name": name,
        "definition": definition.strip(),
        "ko_list": ko_list,
        "category": category.strip(),
    }


def main() -> None:
    out_dir = config.DATA / "kegg"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "modules.json"

    print("Fetching KEGG module list...")
    modules = fetch_module_list()
    print(f"  {len(modules)} modules listed")

    print("\nFetching definitions (with 100 ms politeness sleep)...")
    records: list[dict] = []
    for i, (mod_id, _) in enumerate(modules, 1):
        time.sleep(SLEEP_S)
        rec = fetch_module_detail(mod_id)
        if rec is None:
            print(f"  ⚠ {mod_id}: failed to fetch detail")
            continue
        records.append(rec)
        if i % 50 == 0:
            print(f"  {i}/{len(modules)}  latest: {mod_id} {rec['name'][:50]}")

    print(f"\nFetched {len(records)} modules")
    with open(out_path, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"Wrote {out_path}")

    n_kos = len({ko for r in records for ko in r["ko_list"]})
    print(f"Distinct KOs across all modules: {n_kos:,}")
    print()
    print("Sample modules (first 5):")
    for r in records[:5]:
        print(f"  {r['id']}: {r['name'][:60]}  ({len(r['ko_list'])} KOs)")


if __name__ == "__main__":
    main()
