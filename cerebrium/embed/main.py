"""Per-marker ESM-2 t30 embedding service — runs on a Cerebrium L4 GPU container.

embed_genome(bacdive_id, accession) → {"ok": bool, "row": {pme_<cat>_<dim>: float, ...}}
or {"ok": False, "reason": ...}.

The unified-marker HMM library is baked into the image. Each replica loads
ESM-2 + HMMs once at startup, then serves multiple genomes from the warm
container.
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from typing import Any

import numpy as np
import pyhmmer
import pyhmmer.easel
import pyhmmer.plan7
import pyrodigal
import requests
import torch
from transformers import AutoModel, AutoTokenizer

DATASETS_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{acc}/download"
VERSION_FALLBACKS = (".1", ".2", ".3", ".4")
EMPTY_ZIP_BYTES = 2_000
EVALUE_THRESHOLD = 1e-5

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

_model_name = os.environ.get("ESM2_MODEL", "facebook/esm2_t30_150M_UR50D")
_batch_size = int(os.environ.get("ESM2_BATCH_SIZE", "16"))
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_dtype = torch.float16 if _device.type == "cuda" else torch.float32

print(f"[boot] loading {_model_name} on {_device} ({_dtype})", flush=True)
_tokenizer = AutoTokenizer.from_pretrained(_model_name)
_model = AutoModel.from_pretrained(_model_name, dtype=_dtype)
_model.to(_device)
_model.train(False)
_embed_dim = _model.config.hidden_size

_alphabet = pyhmmer.easel.Alphabet.amino()
with pyhmmer.plan7.HMMFile("/cortex/app/markers.hmm") as _fh:
    _hmms = list(_fh)
_ncbi_key = os.environ.get("NCBI_API_KEY")
print(f"[boot] loaded {len(_hmms)} marker HMMs, embed_dim={_embed_dim}, "
      f"ncbi_key={'yes' if _ncbi_key else 'no'}", flush=True)


def _has_version(acc: str) -> bool:
    return "." in acc and acc.rsplit(".", 1)[-1].isdigit()


def _candidates(acc: str) -> list[str]:
    return [acc] if _has_version(acc) else [acc + v for v in VERSION_FALLBACKS]


def _fetch_fasta(acc: str) -> list[tuple[str, str]] | None:
    rate = 0.1 if _ncbi_key else 0.34
    headers = {"Accept": "application/zip"}
    if _ncbi_key:
        headers["api-key"] = _ncbi_key
    params = {"include_annotation_type": "GENOME_FASTA"}
    for cand in _candidates(acc):
        zip_bytes: bytes | None = None
        for attempt in range(3):
            try:
                time.sleep(rate)
                resp = requests.get(
                    DATASETS_URL.format(acc=cand), params=params,
                    headers=headers, timeout=120,
                )
                if resp.status_code == 404:
                    break
                if resp.status_code in (429, 502, 503):
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
            except requests.RequestException:
                if attempt == 2:
                    break
                time.sleep(2 ** attempt)
                continue
            if len(resp.content) < EMPTY_ZIP_BYTES:
                break
            zip_bytes = resp.content
            break
        if zip_bytes is None:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                fna = [n for n in zf.namelist() if n.endswith(".fna")]
                if not fna:
                    continue
                with zf.open(fna[0]) as src:
                    raw = src.read()
        except zipfile.BadZipFile:
            continue
        return _parse_fasta(raw)
    return None


def _parse_fasta(raw: bytes) -> list[tuple[str, str]]:
    contigs: list[tuple[str, str]] = []
    cur: str | None = None
    chunks: list[str] = []
    for line in raw.splitlines():
        if not line:
            continue
        if line.startswith(b">"):
            if cur is not None:
                contigs.append((cur, "".join(chunks).upper()))
            cur = line[1:].decode("ascii", errors="replace").split()[0]
            chunks = []
        else:
            chunks.append(line.decode("ascii", errors="replace"))
    if cur is not None:
        contigs.append((cur, "".join(chunks).upper()))
    return contigs


def _predict_proteins(contigs: list[tuple[str, str]]) -> list[str]:
    encoded = [(n, s.encode("ascii")) for n, s in contigs]
    total_nt = sum(len(s) for _, s in encoded)
    if total_nt >= 20_000:
        finder = pyrodigal.GeneFinder(meta=False)
        try:
            finder.train(b"TTAATTAATTAA".join(s for _, s in encoded))
        except Exception:
            finder = pyrodigal.GeneFinder(meta=True)
    else:
        finder = pyrodigal.GeneFinder(meta=True)
    proteins: list[str] = []
    for _, s in encoded:
        for gene in finder.find_genes(s):
            proteins.append(gene.translate().rstrip("*"))
    return proteins


def _embed_proteins(proteins: list[str]) -> np.ndarray:
    if not proteins:
        return np.zeros((0, _embed_dim), dtype=np.float32)
    out: list = []
    for i in range(0, len(proteins), _batch_size):
        batch = proteins[i : i + _batch_size]
        enc = _tokenizer(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=1024)
        enc = {k: v.to(_device) for k, v in enc.items()}
        with torch.inference_mode():
            outs = _model(**enc)
        last_hidden = outs.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def _scan_markers(proteins: list[str]) -> dict[str, list[int]]:
    seqs = []
    for i, prot in enumerate(proteins):
        if not prot:
            continue
        ts = pyhmmer.easel.TextSequence(name=f"p{i}".encode(), sequence=prot)
        seqs.append(ts.digitize(_alphabet))
    result: dict[str, list[int]] = {name: [] for name in MARKER_TO_CATEGORY}
    if not seqs:
        return result
    for top_hits in pyhmmer.hmmer.hmmsearch(_hmms, seqs, E=EVALUE_THRESHOLD):
        raw = top_hits.query.name
        marker = raw.decode() if isinstance(raw, bytes) else raw
        if marker not in result:
            continue
        for hit in top_hits:
            if hit.evalue > EVALUE_THRESHOLD:
                continue
            name = hit.name.decode() if isinstance(hit.name, bytes) else hit.name
            if name.startswith("p"):
                try:
                    result[marker].append(int(name[1:]))
                except ValueError:
                    pass
    return result


def embed_genome(bacdive_id: int, accession: str) -> dict[str, Any]:
    try:
        contigs = _fetch_fasta(accession)
        if not contigs:
            return {"ok": False, "reason": "fetch_empty", "bacdive_id": bacdive_id, "accession": accession}
        proteins = _predict_proteins(contigs)
        if not proteins:
            return {"ok": False, "reason": "no_proteins", "bacdive_id": bacdive_id, "accession": accession}
        marker_idx = _scan_markers(proteins)
        hit_indices = sorted({i for ids in marker_idx.values() for i in ids})
        row: dict[str, Any] = {
            "bacdive_id": int(bacdive_id),
            "genome_accession": accession,
            "pme_marker_proteins_total": len(hit_indices),
        }
        if not hit_indices:
            for cat in CATEGORIES:
                row[f"pme_{cat}_n"] = 0
                for d in range(_embed_dim):
                    row[f"pme_{cat}_{d}"] = 0.0
            return {"ok": True, "row": row}
        hit_proteins = [proteins[i] for i in hit_indices]
        hit_matrix = _embed_proteins(hit_proteins)
        gi_to_ri = {gi: ri for ri, gi in enumerate(hit_indices)}
        for cat in CATEGORIES:
            idxs: set[int] = set()
            for marker, gis in marker_idx.items():
                if MARKER_TO_CATEGORY.get(marker) == cat:
                    idxs.update(gis)
            row[f"pme_{cat}_n"] = len(idxs)
            if idxs:
                rows = [gi_to_ri[gi] for gi in idxs if gi in gi_to_ri]
                if rows:
                    cat_mean = hit_matrix[rows].mean(axis=0).astype(np.float32)
                    for d, v in enumerate(cat_mean):
                        row[f"pme_{cat}_{d}"] = float(v)
                    continue
            for d in range(_embed_dim):
                row[f"pme_{cat}_{d}"] = 0.0
        return {"ok": True, "row": row}
    except Exception as exc:
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}",
                "bacdive_id": bacdive_id, "accession": accession}
