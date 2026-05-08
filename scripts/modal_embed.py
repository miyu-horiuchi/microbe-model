"""Modal app — extract ESM-2 embeddings for the full BacDive corpus.

Each Modal container loads ESM-2 once on its GPU, then processes a stream of
(bacdive_id, accession) tasks. The local entrypoint dispatches all training-ready
strains via Modal's parallel .map(), and streams results back to local
data/embeddings.jsonl as they complete (resumable: re-running skips finished IDs).

Usage:
    # one-time:
    modal setup                                   # OAuth Modal account
    modal secret create ncbi-key NCBI_API_KEY=...   # paste your NCBI key
    # run:
    modal run scripts/modal_embed.py
    # or with custom flags:
    modal run scripts/modal_embed.py --gpu A10G --sample-n 50 --workers 16

Cost (as of 2026, A10G at ~$1/hr):
    22K genomes × ~1 sec/genome on A10G ÷ 16 parallel containers ≈ 25 min wall time
    ≈ $7–10 total
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

# --- Modal image ------------------------------------------------------------

# Pin Python and bundle the deps that genome → proteins → ESM-2 needs.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.2",
        "transformers>=4.40",
        "accelerate>=0.30",
        "pyrodigal>=3.5",
        "biopython>=1.83",
        "requests>=2.32",
        "numpy>=1.26",
    ])
)

app = modal.App("microbe-esm2", image=image)

DEFAULT_MODEL = "facebook/esm2_t30_150M_UR50D"
DATASETS_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{acc}/download"
VERSION_FALLBACKS = (".1", ".2", ".3", ".4")
EMPTY_ZIP_BYTES = 2_000


# --- Self-contained helpers (run inside the container) ----------------------

def _has_version(accession: str) -> bool:
    if "." not in accession:
        return False
    return accession.rsplit(".", 1)[-1].isdigit()


def _candidate_accessions(accession: str) -> list[str]:
    if _has_version(accession):
        return [accession]
    return [accession + v for v in VERSION_FALLBACKS]


def _fetch_fasta_bytes(accession: str, ncbi_key: str | None) -> list[tuple[str, str]] | None:
    import io
    import time
    import zipfile

    import requests

    rate = 0.1 if ncbi_key else 0.34
    headers: dict[str, str] = {"Accept": "application/zip"}
    if ncbi_key:
        headers["api-key"] = ncbi_key
    params = {"include_annotation_type": "GENOME_FASTA"}

    for candidate in _candidate_accessions(accession):
        zip_bytes: bytes | None = None
        for attempt in range(3):
            try:
                time.sleep(rate)
                resp = requests.get(
                    DATASETS_URL.format(acc=candidate),
                    params=params, headers=headers, timeout=120,
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
                fasta_names = [n for n in zf.namelist() if n.endswith(".fna")]
                if not fasta_names:
                    continue
                with zf.open(fasta_names[0]) as src:
                    raw = src.read()
        except zipfile.BadZipFile:
            continue
        return _parse_fasta(raw)
    return None


def _parse_fasta(raw: bytes) -> list[tuple[str, str]]:
    contigs: list[tuple[str, str]] = []
    current_id: str | None = None
    chunks: list[str] = []
    for line in raw.splitlines():
        if not line:
            continue
        if line.startswith(b">"):
            if current_id is not None:
                contigs.append((current_id, "".join(chunks).upper()))
            current_id = line[1:].decode("ascii", errors="replace").split()[0]
            chunks = []
        else:
            chunks.append(line.decode("ascii", errors="replace"))
    if current_id is not None:
        contigs.append((current_id, "".join(chunks).upper()))
    return contigs


def _predict_proteins(contigs: list[tuple[str, str]]) -> list[str]:
    import pyrodigal

    encoded = [(name, seq.encode("ascii")) for name, seq in contigs]
    total_nt = sum(len(s) for _, s in encoded)
    if total_nt >= 20_000:
        finder = pyrodigal.GeneFinder(meta=False)
        train_seq = b"TTAATTAATTAA".join(seq for _, seq in encoded)
        try:
            finder.train(train_seq)
        except Exception:
            finder = pyrodigal.GeneFinder(meta=True)
    else:
        finder = pyrodigal.GeneFinder(meta=True)

    proteins: list[str] = []
    for _, seq in encoded:
        for gene in finder.find_genes(seq):
            proteins.append(gene.translate().rstrip("*"))
    return proteins


# --- Modal class: loads ESM-2 once per container, batches embeddings --------

@app.cls(
    gpu="A10G",
    timeout=3600 * 4,
    secrets=[modal.Secret.from_name("ncbi-key", required_keys=["NCBI_API_KEY"])],
    max_containers=16,
    scaledown_window=60,
)
class Embedder:
    @modal.enter()
    def setup(self):
        import os

        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.np = np
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        # Read configurable knobs from env (set by the local entrypoint)
        self.model_name = os.environ.get("ESM2_MODEL", DEFAULT_MODEL)
        self.sample_n = int(os.environ.get("ESM2_SAMPLE_N", "50"))
        self.batch_size = int(os.environ.get("ESM2_BATCH_SIZE", "16"))
        print(f"[setup] loading {self.model_name} on {self.device}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, dtype=self.dtype)
        self.model.to(self.device)
        self.model.train(False)
        self.embed_dim = self.model.config.hidden_size
        self.ncbi_key = os.environ.get("NCBI_API_KEY")
        self.rng = np.random.default_rng(0)
        print(f"[setup] embed_dim={self.embed_dim}, "
              f"sample_n={self.sample_n}, batch={self.batch_size}, ready", flush=True)

    def _embed_proteins(self, proteins: list[str]):
        import torch

        if not proteins:
            return self.np.zeros((0, self.embed_dim), dtype=self.np.float32)
        out: list = []
        for i in range(0, len(proteins), self.batch_size):
            batch = proteins[i : i + self.batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=1024,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                outputs = self.model(**enc)
            last_hidden = outputs.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            out.append(pooled.float().cpu().numpy())
        return self.np.concatenate(out, axis=0)

    @modal.method()
    def embed_genome(self, bacdive_id: int, accession: str) -> dict | None:
        try:
            contigs = _fetch_fasta_bytes(accession, self.ncbi_key)
            if not contigs:
                return None
            proteins = _predict_proteins(contigs)
            if not proteins:
                return None
            if self.sample_n is not None and self.sample_n < len(proteins):
                idx = self.rng.choice(len(proteins), size=self.sample_n, replace=False)
                proteins = [proteins[i] for i in idx]
            matrix = self._embed_proteins(proteins)
            vec = matrix.mean(axis=0).astype(self.np.float32)
            return {
                "bacdive_id": int(bacdive_id),
                "genome_accession": accession,
                "embed_dim": int(len(vec)),
                "embedding": vec.tolist(),
            }
        except Exception as exc:
            print(f"  skip {accession}: {type(exc).__name__}: {exc}", flush=True)
            return None


# --- Local entrypoint -------------------------------------------------------

@app.local_entrypoint()
def main(
    model: str = DEFAULT_MODEL,
    sample_n: int = 50,
    batch_size: int = 16,
    gpu: str = "A10G",
    out_path: str = "data/embeddings.jsonl",
    limit: int = 0,
):
    """Dispatch all training-ready genomes to Modal and stream results to disk."""
    import pandas as pd

    pheno = pd.read_parquet("data/bacdive_phenotypes.parquet")
    has_genome = pheno["genome_accession"].notna()
    label_cols = ["optimal_temperature_c", "optimal_ph", "oxygen_requirement", "salt_tolerance_pct"]
    has_label = pheno[label_cols].notna().any(axis=1)
    ready = pheno[has_genome & has_label].copy()
    ready["bacdive_id"] = ready["bacdive_id"].astype(int)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    if out.exists():
        with open(out) as fh:
            for line in fh:
                try:
                    done.add(int(json.loads(line)["bacdive_id"]))
                except Exception:
                    continue

    pending = ready[~ready["bacdive_id"].isin(done)]
    if limit:
        pending = pending.head(limit)
    tasks = list(zip(pending["bacdive_id"], pending["genome_accession"].astype(str), strict=True))
    print(f"Embedding {len(tasks):,} genomes (skipping {len(done):,} cached)")
    print(f"Model: {model}  sample_n={sample_n}  batch={batch_size}  gpu={gpu}")

    if not tasks:
        print("Nothing to do.")
        return

    config_secret = modal.Secret.from_dict({
        "ESM2_MODEL": model,
        "ESM2_SAMPLE_N": str(sample_n),
        "ESM2_BATCH_SIZE": str(batch_size),
    })
    embedder = Embedder.with_options(
        gpu=gpu,
        secrets=[
            modal.Secret.from_name("ncbi-key", required_keys=["NCBI_API_KEY"]),
            config_secret,
        ],
    )()

    n_ok = 0
    n_fail = 0
    with open(out, "a") as log:
        for result in embedder.embed_genome.starmap(tasks, return_exceptions=True):
            if isinstance(result, Exception):
                n_fail += 1
                continue
            if result is None:
                n_fail += 1
                continue
            log.write(json.dumps(result) + "\n")
            log.flush()
            n_ok += 1
            if n_ok % 100 == 0:
                print(f"  {n_ok:,} ok / {n_fail:,} fail")

    print(f"\nFinished. {n_ok:,} succeeded, {n_fail:,} failed.")
    print(f"Streamed to {out}")
    print("Run scripts/_materialize_embeddings.py (or the snippet at the bottom of "
          "scripts/11_extract_embeddings.py) to build the parquet from this JSONL.")
