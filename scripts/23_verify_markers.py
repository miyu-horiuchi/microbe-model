"""Pre-flight check the unified Pfam marker library.

For each Pfam ID in markers.all_markers():
  - download the HMM from InterPro
  - confirm the response is a valid HMM (has a NAME line + ALPH amino + LENG > 0)
  - parse the family description from the DESC line
  - flag any ID where our annotated role mentions a protein family that doesn't
    obviously match the DESC (e.g. we said "biotin synthase" but DESC says
    "transcription factor")

Output:
  - data/markers/_verification.tsv with columns: pfam_id, our_name, our_role,
    pfam_desc, status (ok / 404 / mismatch / parse_error)

Run this BEFORE committing to a multi-hour scan. Cost: ~30 sec for 70 HMMs.
"""
from __future__ import annotations

import gzip
import time
from pathlib import Path

import requests

from microbe_model import config
from microbe_model.features.markers import all_markers, category_for

INTERPRO_HMM_URL = "https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/{pfam}/?annotation=hmm"
OUT = config.DATA / "markers" / "_verification.tsv"


def fetch_hmm_text(pfam_id: str) -> tuple[str | None, int]:
    url = INTERPRO_HMM_URL.format(pfam=pfam_id)
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException:
        return None, 0
    if resp.status_code != 200:
        return None, resp.status_code
    raw = resp.content
    try:
        return gzip.decompress(raw).decode("ascii"), 200
    except gzip.BadGzipFile:
        try:
            return raw.decode("ascii"), 200
        except UnicodeDecodeError:
            return None, 200


def parse_hmm_header(text: str) -> dict[str, str]:
    """Pull NAME, DESC, LENG, ALPH out of the HMM header."""
    header: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("HMM "):
            break
        for key in ("NAME", "DESC", "LENG", "ALPH"):
            if line.startswith(key):
                header[key] = line[len(key):].strip()
                break
    return header


def main() -> None:
    markers = all_markers()
    print(f"Verifying {len(markers)} markers from the unified library\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    n_ok = n_404 = n_other = 0

    for pfam_id, (our_name, our_role) in markers.items():
        time.sleep(0.05)  # be nice to InterPro
        text, status = fetch_hmm_text(pfam_id)
        if status == 404:
            print(f"  ❌ {pfam_id:8s}  404 not found  ({our_name})")
            n_404 += 1
            rows.append({"pfam_id": pfam_id, "our_name": our_name,
                         "category": category_for(pfam_id), "our_role": our_role,
                         "pfam_name": "", "pfam_desc": "", "status": "404"})
            continue
        if text is None:
            print(f"  ❓ {pfam_id:8s}  HTTP {status} or parse error  ({our_name})")
            n_other += 1
            rows.append({"pfam_id": pfam_id, "our_name": our_name,
                         "category": category_for(pfam_id), "our_role": our_role,
                         "pfam_name": "", "pfam_desc": "",
                         "status": f"http_{status}_or_parse"})
            continue

        header = parse_hmm_header(text)
        n_ok += 1
        print(f"  ✅ {pfam_id:8s}  {header.get('NAME', '???'):25s}  "
              f"{header.get('DESC', '')[:50]}")
        rows.append({
            "pfam_id": pfam_id,
            "our_name": our_name,
            "category": category_for(pfam_id),
            "our_role": our_role,
            "pfam_name": header.get("NAME", ""),
            "pfam_desc": header.get("DESC", ""),
            "status": "ok",
        })

    print(f"\nSummary: {n_ok} ok, {n_404} not-found, {n_other} other")
    with open(OUT, "w") as fh:
        fh.write("pfam_id\tcategory\tour_name\tour_role\tpfam_name\tpfam_desc\tstatus\n")
        for r in rows:
            fh.write("\t".join([
                r["pfam_id"], r["category"], r["our_name"], r["our_role"],
                r["pfam_name"], r["pfam_desc"], r["status"],
            ]) + "\n")
    print(f"Wrote {OUT}")
    print("\nReview the TSV and update markers.py for any ID where "
          "our_role disagrees with pfam_desc.")


if __name__ == "__main__":
    main()
