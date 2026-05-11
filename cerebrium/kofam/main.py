"""KOfam scan service — runs on a Cerebrium CPU container.

scan_genome(accession) → {"ok": bool, "ko_hits": [...]} or {"ok": False, "reason": ...}.

The relevant-KO HMM library (~734 MB) and per-KO bitscore thresholds are baked
into the image via `include` in cerebrium.toml, so each replica loads them
once at startup.
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from typing import Any

import pyhmmer
import pyhmmer.easel
import pyhmmer.plan7
import pyrodigal
import requests

DATASETS_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{acc}/download"
VERSION_FALLBACKS = (".1", ".2", ".3", ".4")
EMPTY_ZIP_BYTES = 2_000
DEFAULT_EVALUE = 1e-5

HMM_PATH = "/cortex/app/kofam_relevant.hmm"
THRESHOLDS_PATH = "/cortex/app/ko_thresholds.tsv"

_alphabet = pyhmmer.easel.Alphabet.amino()
with pyhmmer.plan7.HMMFile(HMM_PATH) as _fh:
    _hmms = list(_fh)
_thresholds: dict[str, float] = {}
with open(THRESHOLDS_PATH) as _fh:
    next(_fh)
    for _line in _fh:
        _parts = _line.rstrip("\n").split("\t")
        if len(_parts) < 2:
            continue
        try:
            _thresholds[_parts[0]] = float(_parts[1])
        except (TypeError, ValueError):
            _thresholds[_parts[0]] = 0.0
_ncbi_key = os.environ.get("NCBI_API_KEY")
print(f"[boot] loaded {len(_hmms):,} HMMs, {len(_thresholds):,} thresholds, "
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


def _scan(proteins: list[str]) -> set[str]:
    seqs = []
    for i, prot in enumerate(proteins):
        if not prot:
            continue
        ts = pyhmmer.easel.TextSequence(name=f"p{i}".encode(), sequence=prot)
        seqs.append(ts.digitize(_alphabet))
    found: set[str] = set()
    if not seqs:
        return found
    for top_hits in pyhmmer.hmmer.hmmsearch(_hmms, seqs, E=DEFAULT_EVALUE):
        raw = top_hits.query.name
        ko = raw.decode() if isinstance(raw, bytes) else raw
        thr = _thresholds.get(ko, 0.0)
        for hit in top_hits:
            if hit.score >= thr and hit.evalue <= DEFAULT_EVALUE:
                found.add(ko)
                break
    return found


def scan_genome(accession: str) -> dict[str, Any]:
    try:
        contigs = _fetch_fasta(accession)
        if not contigs:
            return {"ok": False, "reason": "fetch_empty", "accession": accession}
        proteins = _predict_proteins(contigs)
        if not proteins:
            return {"ok": False, "reason": "no_proteins", "accession": accession}
        ko_hits = _scan(proteins)
        return {"ok": True, "accession": accession, "ko_hits": sorted(ko_hits)}
    except Exception as exc:
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}", "accession": accession}
