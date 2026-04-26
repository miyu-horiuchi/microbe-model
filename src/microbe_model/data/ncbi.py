"""NCBI genome fetcher.

Uses the NCBI Datasets v2 REST API to download a single nucleotide FASTA per accession.
This API doesn't require auth, but providing NCBI_API_KEY raises the rate limit from
3 req/s to 10 req/s.
"""
from __future__ import annotations

import gzip
import io
import time
import zipfile
from pathlib import Path

import requests

from microbe_model import config

DATASETS_BASE = "https://api.ncbi.nlm.nih.gov/datasets/v2"
RATE_LIMIT_S = 0.1 if config.NCBI_API_KEY else 0.34


class GenomeNotFound(RuntimeError):
    pass


def genome_path(accession: str) -> Path:
    return config.GENOME_DIR / f"{accession}.fna.gz"


def fetch_genome(accession: str, *, force: bool = False) -> Path:
    """Download a genome FASTA for the given assembly accession (e.g. GCF_000005845.2).

    The Datasets API returns a zip; we extract the FASTA, gzip it, and write to disk.
    Idempotent — returns immediately if the file is already cached.
    """
    out = genome_path(accession)
    if out.exists() and not force:
        return out

    url = f"{DATASETS_BASE}/genome/accession/{accession}/download"
    params = {"include_annotation_type": "GENOME_FASTA"}
    headers = {"Accept": "application/zip"}
    if config.NCBI_API_KEY:
        headers["api-key"] = config.NCBI_API_KEY

    time.sleep(RATE_LIMIT_S)
    resp = requests.get(url, params=params, headers=headers, timeout=120, stream=True)
    if resp.status_code == 404:
        raise GenomeNotFound(accession)
    resp.raise_for_status()

    buf = io.BytesIO(resp.content)
    with zipfile.ZipFile(buf) as zf:
        fasta_names = [n for n in zf.namelist() if n.endswith(".fna")]
        if not fasta_names:
            raise GenomeNotFound(f"{accession} (no .fna in archive)")
        with zf.open(fasta_names[0]) as src, gzip.open(out, "wb") as dst:
            for chunk in iter(lambda: src.read(1 << 16), b""):
                dst.write(chunk)
    return out
