"""Per-marker ESM-2 embedding — local Mac MPS version.

Local port of scripts/modal_per_marker_embed.py. Same logic, no Modal:
  fetch FASTA → pyrodigal → pyhmmer (50 markers) → ESM-2 on hit proteins only
  → group by 8 categories → 8 × embed_dim features per genome.

Output: data/per_marker_embeddings.jsonl (one row per genome, append-only,
resumable on bacdive_id).

Usage:
    uv run --extra embeddings python scripts/29_per_marker_embed_local.py \\
        --model facebook/esm2_t30_150M_UR50D --batch-size 16 --max 10

    # Full corpus
    uv run --extra embeddings python scripts/29_per_marker_embed_local.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyhmmer
import pyhmmer.easel
import pyhmmer.plan7
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from microbe_model import config
from microbe_model.features.genome import predict_genes
from microbe_model.pipeline import _fetch_fasta_bytes

MARKER_TO_CATEGORY: dict[str, str] = {
    "Hsp70_DnaK": "temperature", "Hsp90": "temperature", "Cpn60_GroEL": "temperature",
    "Hsp20": "temperature", "CSD_cold_shock": "temperature", "TGS_thermosome": "temperature",
    "ATP_synth_alphabeta": "ph", "ATP_synth_alphabeta_C": "ph", "ATP_synth_F0_B": "ph",
    "NhaA_Na_H_exch": "ph", "NhaB_Na_H_exch": "ph", "Pyridoxal_decarbox": "ph",
    "MotA_TolQ_ExbB": "ph", "V_ATPase_subH_N": "ph",
    "COX1_aerobic": "oxygen", "COX2_TM_aerobic": "oxygen", "COX2_periplasm_aero": "oxygen",
    "Cyt_CBB3_microaero": "oxygen", "Rieske_2Fe2S": "oxygen", "Catalase": "oxygen",
    "SOD_FeMn": "oxygen", "SOD_CuZn": "oxygen", "FeFe_hyd_anaerobic": "oxygen",
    "NiFe_hyd_anaerobic": "oxygen", "FAD_binding_FrdA": "oxygen", "Fer4_FeS_4Fe4S": "oxygen",
    "KdpD_osmosensor": "salt", "TrkH_K_channel": "salt", "BCCT_compatible": "salt",
    "BPD_transp_1": "salt", "EctC_ectoine_synth": "salt", "Bact_rhodopsin": "salt",
    "TP_methylase_B12": "vitamin", "Peripla_BP_2": "vitamin", "THF_DHG_CYH_folate": "vitamin",
    "FolB_folate": "vitamin", "PdxJ_pyridoxine": "vitamin", "DHBP_riboflavin": "vitamin",
    "NifH_nitrogenase": "nitrogen", "NifDK_nitrogenase": "nitrogen",
    "NIR_SIR_ferredoxin": "nitrogen",
    "RuBisCO_large_form1": "carbon", "RuBisCO_small_form1": "carbon",
    "Alpha_amylase": "carbon", "Cellulase_GH5": "carbon", "CBM_cellulose": "carbon",
    "Molybdopterin_OR": "special", "UvrD_helicase_C": "special",
}
CATEGORIES = ["temperature", "ph", "oxygen", "salt", "vitamin", "nitrogen", "carbon", "special"]
EVALUE_THRESHOLD = 1e-5
MARKERS_HMM = config.DATA / "markers" / "unified" / "unified_markers.hmm"


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_done_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    ids: set[int] = set()
    with open(path) as fh:
        for line in fh:
            try:
                ids.add(int(json.loads(line)["bacdive_id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return ids


def _scan_markers(
    proteins: list[str],
    hmms: list[pyhmmer.plan7.HMM],
    alphabet: pyhmmer.easel.Alphabet,
) -> dict[str, list[int]]:
    seqs: list[pyhmmer.easel.DigitalSequence] = []
    for i, prot in enumerate(proteins):
        if not prot:
            continue
        ts = pyhmmer.easel.TextSequence(name=f"p{i}".encode(), sequence=prot)
        seqs.append(ts.digitize(alphabet))
    result: dict[str, list[int]] = {name: [] for name in MARKER_TO_CATEGORY}
    if not seqs:
        return result
    for top_hits in pyhmmer.hmmer.hmmsearch(hmms, seqs, E=EVALUE_THRESHOLD):
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


def _embed_proteins(
    proteins: list[str], tokenizer, model, device, batch_size: int, embed_dim: int,
) -> np.ndarray:
    if not proteins:
        return np.zeros((0, embed_dim), dtype=np.float32)
    out: list = []
    for i in range(0, len(proteins), batch_size):
        batch = proteins[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            outs = model(**enc)
        last_hidden = outs.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=0,
                        help="This worker's shard (0-indexed). With --num-shards M, "
                             "process bacdive_ids where id %% M == shard_id.")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total shard count for multi-VM parallel runs.")
    parser.add_argument("--out-name", default=None,
                        help="Override output filename. Defaults to "
                             "per_marker_embeddings.<shard_id>.jsonl when sharded.")
    args = parser.parse_args()

    if not MARKERS_HMM.exists():
        raise SystemExit(f"Missing {MARKERS_HMM}. Build it first.")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit(f"shard-id must be in [0, num-shards)")

    pheno_path = config.DATA / "bacdive_phenotypes.parquet"
    pheno = pd.read_parquet(pheno_path)
    has_genome = pheno["genome_accession"].notna()
    label_cols = list(config.PHENOTYPE_TARGETS.keys())
    has_label = pheno[label_cols].notna().any(axis=1)
    ready = pheno[has_genome & has_label].copy()
    ready["bacdive_id"] = ready["bacdive_id"].astype(int)

    if args.num_shards > 1:
        ready = ready[ready["bacdive_id"] % args.num_shards == args.shard_id]
        out_name = args.out_name or f"per_marker_embeddings.{args.shard_id}.jsonl"
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(ready):,} genomes assigned")
    else:
        out_name = args.out_name or "per_marker_embeddings.jsonl"

    out_path = config.DATA / out_name
    done_ids = _load_done_ids(out_path)
    pending = ready[~ready["bacdive_id"].isin(done_ids)]
    if args.max:
        pending = pending.head(args.max)
    print(f"Embedding {len(pending):,} genomes (skipping {len(done_ids):,} cached)")

    device = _pick_device()
    print(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.train(False)
    embed_dim = model.config.hidden_size
    print(f"  device={device}, embed_dim={embed_dim}, batch_size={args.batch_size}")

    alphabet = pyhmmer.easel.Alphabet.amino()
    with pyhmmer.plan7.HMMFile(str(MARKERS_HMM)) as fh:
        hmms = list(fh)
    print(f"  loaded {len(hmms)} marker HMMs")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_ok = n_fail = 0
    with open(out_path, "a") as log:
        for _, row in tqdm(pending.iterrows(), total=len(pending),
                           desc="embed", unit="genome"):
            bid = int(row["bacdive_id"])
            acc = str(row["genome_accession"])
            try:
                contigs = _fetch_fasta_bytes(acc)
                if not contigs:
                    n_fail += 1
                    continue
                proteins, _, _ = predict_genes(contigs)
                if not proteins:
                    n_fail += 1
                    continue
                marker_idx = _scan_markers(proteins, hmms, alphabet)
                hit_indices = sorted({i for ids in marker_idx.values() for i in ids})
                payload: dict[str, Any] = {
                    "bacdive_id": bid,
                    "genome_accession": acc,
                    "pme_marker_proteins_total": len(hit_indices),
                }
                if hit_indices:
                    hit_proteins = [proteins[i] for i in hit_indices]
                    hit_matrix = _embed_proteins(
                        hit_proteins, tokenizer, model, device, args.batch_size, embed_dim,
                    )
                    gi_to_ri = {gi: ri for ri, gi in enumerate(hit_indices)}
                    for cat in CATEGORIES:
                        idxs: set[int] = set()
                        for marker, gis in marker_idx.items():
                            if MARKER_TO_CATEGORY.get(marker) == cat:
                                idxs.update(gis)
                        payload[f"pme_{cat}_n"] = len(idxs)
                        if idxs:
                            rows = [gi_to_ri[gi] for gi in idxs if gi in gi_to_ri]
                            if rows:
                                cat_mean = hit_matrix[rows].mean(axis=0).astype(np.float32)
                                for d, v in enumerate(cat_mean):
                                    payload[f"pme_{cat}_{d}"] = float(v)
                                continue
                        for d in range(embed_dim):
                            payload[f"pme_{cat}_{d}"] = 0.0
                else:
                    for cat in CATEGORIES:
                        payload[f"pme_{cat}_n"] = 0
                        for d in range(embed_dim):
                            payload[f"pme_{cat}_{d}"] = 0.0
            except Exception as exc:
                print(f"  skip {acc}: {type(exc).__name__}: {exc}")
                n_fail += 1
                continue
            log.write(json.dumps(payload) + "\n")
            log.flush()
            n_ok += 1

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed/60:.1f} min. {n_ok:,} succeeded, {n_fail:,} failed.")


if __name__ == "__main__":
    main()
