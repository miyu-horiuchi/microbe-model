"""Extract HMM-gated marker protein sequences locally.

This is the local CPU fallback for ``scripts/36_extract_marker_sequences.py`` when
Modal is unavailable. It emits the same JSONL schema expected by
``scripts/39_predict_hybrid.py``:

    {
      "bacdive_id": 1000000000,
      "genome_accession": "GCA_...",
      "by_category": {"oxygen": ["M..."], ...},
      "category_counts": {"oxygen": 3, ...}
    }

Example for the 5,000 uncultured UI genomes:

    PYTHONPATH=src uv run --python 3.11 python scripts/40_extract_marker_sequences_local.py \
      --input-path data/gtdb_candidates.parquet \
      --id-col "" \
      --accession-col genome_accession \
      --fetch-accession-col ncbi_assembly_accession_versioned \
      --out-path data/uncultured_marker_sequences.jsonl \
      --workers 6
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

DATASETS_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{acc}/download"
VERSION_FALLBACKS = (".1", ".2", ".3", ".4")
EMPTY_ZIP_BYTES = 2_000
EVALUE_THRESHOLD = 1e-5
MAX_PROTEIN_LEN = 1022

MARKER_TO_CATEGORY: dict[str, str] = {
    "Hsp70_DnaK": "temperature",
    "Hsp90": "temperature",
    "Cpn60_GroEL": "temperature",
    "Hsp20": "temperature",
    "CSD_cold_shock": "temperature",
    "TGS_thermosome": "temperature",
    "ATP_synth_alphabeta": "ph",
    "ATP_synth_alphabeta_C": "ph",
    "ATP_synth_F0_B": "ph",
    "NhaA_Na_H_exch": "ph",
    "NhaB_Na_H_exch": "ph",
    "Pyridoxal_decarbox": "ph",
    "MotA_TolQ_ExbB": "ph",
    "V_ATPase_subH_N": "ph",
    "COX1_aerobic": "oxygen",
    "COX2_TM_aerobic": "oxygen",
    "COX2_periplasm_aero": "oxygen",
    "Cyt_CBB3_microaero": "oxygen",
    "Rieske_2Fe2S": "oxygen",
    "Catalase": "oxygen",
    "SOD_FeMn": "oxygen",
    "SOD_CuZn": "oxygen",
    "FeFe_hyd_anaerobic": "oxygen",
    "NiFe_hyd_anaerobic": "oxygen",
    "FAD_binding_FrdA": "oxygen",
    "Fer4_FeS_4Fe4S": "oxygen",
    "KdpD_osmosensor": "salt",
    "TrkH_K_channel": "salt",
    "BCCT_compatible": "salt",
    "BPD_transp_1": "salt",
    "EctC_ectoine_synth": "salt",
    "Bact_rhodopsin": "salt",
    "TP_methylase_B12": "vitamin",
    "Peripla_BP_2": "vitamin",
    "THF_DHG_CYH_folate": "vitamin",
    "FolB_folate": "vitamin",
    "PdxJ_pyridoxine": "vitamin",
    "DHBP_riboflavin": "vitamin",
    "NifH_nitrogenase": "nitrogen",
    "NifDK_nitrogenase": "nitrogen",
    "NIR_SIR_ferredoxin": "nitrogen",
    "RuBisCO_large_form1": "carbon",
    "RuBisCO_small_form1": "carbon",
    "Alpha_amylase": "carbon",
    "Cellulase_GH5": "carbon",
    "CBM_cellulose": "carbon",
    "Molybdopterin_OR": "special",
    "UvrD_helicase_C": "special",
}
CATEGORIES = ["temperature", "ph", "oxygen", "salt", "vitamin", "nitrogen", "carbon", "special"]
_HMM_CACHE: list[Any] | None = None
_ALPHABET: Any | None = None


def _has_version(accession: str) -> bool:
    return "." in accession and accession.rsplit(".", 1)[-1].isdigit()


def _candidate_accessions(accession: str) -> list[str]:
    if _has_version(accession):
        return [accession]
    return [accession + v for v in VERSION_FALLBACKS]


def _parse_fasta(raw: bytes) -> list[tuple[str, str]]:
    contigs: list[tuple[str, str]] = []
    name = None
    chunks: list[str] = []
    for line in raw.decode("utf-8", errors="ignore").splitlines():
        if line.startswith(">"):
            if name is not None:
                contigs.append((name, "".join(chunks).upper()))
            name = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line.strip())
    if name is not None:
        contigs.append((name, "".join(chunks).upper()))
    return contigs


def _fetch_fasta_bytes(accession: str) -> list[tuple[str, str]] | None:
    import requests

    headers = {"Accept": "application/zip"}
    ncbi_key = os.environ.get("NCBI_API_KEY")
    if ncbi_key:
        headers["api-key"] = ncbi_key

    for cand in _candidate_accessions(accession):
        for attempt in range(3):
            try:
                time.sleep(0.1 if ncbi_key else 0.34)
                resp = requests.get(
                    DATASETS_URL.format(acc=cand),
                    params={"include_annotation_type": "GENOME_FASTA"},
                    headers=headers,
                    timeout=120,
                )
                if resp.status_code == 404:
                    break
                if resp.status_code in (429, 502, 503):
                    time.sleep(2**attempt)
                    continue
                resp.raise_for_status()
            except requests.RequestException:
                if attempt == 2:
                    break
                time.sleep(2**attempt)
                continue

            if len(resp.content) < EMPTY_ZIP_BYTES:
                break
            try:
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    fasta_names = [name for name in zf.namelist() if name.endswith(".fna")]
                    if not fasta_names:
                        break
                    with zf.open(fasta_names[0]) as src:
                        return _parse_fasta(src.read())
            except zipfile.BadZipFile:
                break
    return None


def _predict_proteins(contigs: list[tuple[str, str]]) -> list[str]:
    import pyrodigal

    if not contigs:
        return []
    total = sum(len(seq) for _, seq in contigs)
    meta = total < 100_000
    finder = pyrodigal.GeneFinder(meta=meta)
    if not meta:
        finder.train(*[seq.encode() for _, seq in contigs])
    proteins: list[str] = []
    for _, seq in contigs:
        for gene in finder.find_genes(seq.encode()):
            proteins.append(gene.translate().rstrip("*"))
    return proteins


def _scan_for_markers(proteins: list[str], hmm_path: Path) -> dict[str, list[int]]:
    import pyhmmer
    import pyhmmer.easel
    import pyhmmer.plan7

    global _ALPHABET, _HMM_CACHE
    result: dict[str, list[int]] = {name: [] for name in MARKER_TO_CATEGORY}
    if not proteins:
        return result

    if _ALPHABET is None:
        _ALPHABET = pyhmmer.easel.Alphabet.amino()
    if _HMM_CACHE is None:
        with pyhmmer.plan7.HMMFile(str(hmm_path)) as hmm_file:
            _HMM_CACHE = list(hmm_file)

    seqs = []
    for idx, prot in enumerate(proteins):
        if prot:
            seqs.append(
                pyhmmer.easel.TextSequence(name=f"p{idx}".encode(), sequence=prot).digitize(_ALPHABET)
            )
    if not seqs:
        return result

    for top_hits in pyhmmer.hmmer.hmmsearch(_HMM_CACHE, seqs, E=EVALUE_THRESHOLD):
        raw = top_hits.query.name
        marker = raw.decode() if isinstance(raw, bytes) else raw
        if marker not in result:
            continue
        for hit in top_hits:
            if hit.evalue > EVALUE_THRESHOLD:
                continue
            hit_name = hit.name.decode() if isinstance(hit.name, bytes) else hit.name
            if hit_name.startswith("p"):
                try:
                    result[marker].append(int(hit_name[1:]))
                except ValueError:
                    pass
    return result


def _extract_one(task: tuple[int, str, str, int, str]) -> dict[str, Any] | None:
    record_id, genome_accession, fetch_accession, max_per_cat, hmm_path_str = task
    contigs = _fetch_fasta_bytes(fetch_accession)
    if not contigs:
        return None
    proteins = _predict_proteins(contigs)
    if not proteins:
        return None
    marker_to_idx = _scan_for_markers(proteins, Path(hmm_path_str))

    by_category: dict[str, list[str]] = {cat: [] for cat in CATEGORIES}
    for cat in CATEGORIES:
        idxs: set[int] = set()
        for marker, protein_ids in marker_to_idx.items():
            if MARKER_TO_CATEGORY.get(marker) == cat:
                idxs.update(protein_ids)
        ranked = sorted(idxs, key=lambda i: len(proteins[i]))
        kept = ranked[:max_per_cat]
        by_category[cat] = [proteins[i][:MAX_PROTEIN_LEN] for i in kept]

    return {
        "bacdive_id": int(record_id),
        "genome_accession": genome_accession,
        "by_category": by_category,
        "category_counts": {cat: len(by_category[cat]) for cat in CATEGORIES},
    }


def _load_done(path: Path) -> tuple[set[int], set[str]]:
    done_ids: set[int] = set()
    done_accessions: set[str] = set()
    if not path.exists():
        return done_ids, done_accessions
    with open(path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
                done_ids.add(int(row["bacdive_id"]))
                done_accessions.add(str(row["genome_accession"]))
            except Exception:
                continue
    return done_ids, done_accessions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=Path("data/gtdb_candidates.parquet"))
    parser.add_argument("--out-path", type=Path, default=Path("data/uncultured_marker_sequences.jsonl"))
    parser.add_argument("--id-col", default="")
    parser.add_argument("--accession-col", default="genome_accession")
    parser.add_argument("--fetch-accession-col", default="ncbi_assembly_accession_versioned")
    parser.add_argument("--hmm-path", type=Path, default=Path("data/markers/unified/unified_markers.hmm"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-per-cat", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_parquet(args.input_path)
    if args.accession_col not in source.columns:
        raise SystemExit(f"Missing accession column: {args.accession_col}")
    if args.fetch_accession_col and args.fetch_accession_col not in source.columns:
        raise SystemExit(f"Missing fetch accession column: {args.fetch_accession_col}")

    ready = source[source[args.accession_col].notna()].copy().reset_index(drop=True)
    if args.id_col and args.id_col in ready.columns:
        ready["_record_id"] = ready[args.id_col].astype(int)
    else:
        ready["_record_id"] = ready.index + 1_000_000_000
    ready["_genome_accession"] = ready[args.accession_col].astype(str)
    if args.fetch_accession_col:
        ready["_fetch_accession"] = ready[args.fetch_accession_col].fillna(ready[args.accession_col]).astype(str)
    else:
        ready["_fetch_accession"] = ready["_genome_accession"]

    done_ids, done_accessions = _load_done(args.out_path)
    pending = ready[
        ~ready["_record_id"].isin(done_ids)
        & ~ready["_genome_accession"].isin(done_accessions)
    ]
    if args.limit:
        pending = pending.head(args.limit)
    tasks = [
        (
            int(row["_record_id"]),
            str(row["_genome_accession"]),
            str(row["_fetch_accession"]),
            args.max_per_cat,
            str(args.hmm_path),
        )
        for row in pending[["_record_id", "_genome_accession", "_fetch_accession"]].to_dict("records")
    ]

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Marker-sequence local extract: {len(tasks):,} pending ({len(done_accessions):,} cached)")
    print(f"input_path={args.input_path}")
    print(f"out_path={args.out_path}")
    print(f"workers={args.workers} max_per_cat={args.max_per_cat}")
    if not tasks:
        return

    n_ok = 0
    n_fail = 0
    with open(args.out_path, "a") as log, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_extract_one, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception:
                result = None
            if result is None:
                n_fail += 1
            else:
                log.write(json.dumps(result) + "\n")
                log.flush()
                n_ok += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(f"  {completed:,}/{len(tasks):,} complete  ok={n_ok:,} fail={n_fail:,}", flush=True)
    print(f"Finished. {n_ok:,} succeeded, {n_fail:,} failed.")


if __name__ == "__main__":
    main()
