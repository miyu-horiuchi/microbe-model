"""KOfam scan on Modal CPU — fast parallel HMMER scan against KEGG-relevant KOs.

Same architecture as scripts/28_kofam_scan.py but each Modal container has its
own NCBI fetch IP (bypasses the rate limit that bottlenecks local parallelism)
and its own CPU resources. The 700 MB KOfam relevant library is bundled into
the Modal image so containers don't redownload.

Usage:
    modal setup  # one-time
    modal run scripts/modal_kofam.py --limit 10        # smoke test
    modal run scripts/modal_kofam.py                   # full corpus

Cost: ~$2-4 for 22K genomes at 4-CPU containers, 16-way parallel.
Resumes from data/kofam_hits.jsonl — already-processed accessions are skipped.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "pyrodigal>=3.5",
        "pyhmmer>=0.12",
        "biopython>=1.83",
        "requests>=2.32",
    ])
    .add_local_file("data/kofam/kofam_relevant.hmm", "/root/kofam.hmm", copy=True)
    .add_local_file("data/kofam/ko_thresholds.tsv", "/root/ko_thresholds.tsv", copy=True)
)

app = modal.App("microbe-kofam", image=image)

DATASETS_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{acc}/download"
VERSION_FALLBACKS = (".1", ".2", ".3", ".4")
EMPTY_ZIP_BYTES = 2_000
DEFAULT_EVALUE = 1e-5


def _has_version(accession: str) -> bool:
    if "." not in accession:
        return False
    return accession.rsplit(".", 1)[-1].isdigit()


def _candidate_accessions(accession: str) -> list[str]:
    if _has_version(accession):
        return [accession]
    return [accession + v for v in VERSION_FALLBACKS]


def _fetch_fasta_bytes(accession: str, ncbi_key: str | None) -> list[tuple[str, str]] | None:
    import io, time, zipfile
    import requests

    rate = 0.1 if ncbi_key else 0.34
    headers = {"Accept": "application/zip"}
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
        try:
            finder.train(b"TTAATTAATTAA".join(seq for _, seq in encoded))
        except Exception:
            finder = pyrodigal.GeneFinder(meta=True)
    else:
        finder = pyrodigal.GeneFinder(meta=True)
    proteins: list[str] = []
    for _, seq in encoded:
        for gene in finder.find_genes(seq):
            proteins.append(gene.translate().rstrip("*"))
    return proteins


@app.cls(
    cpu=4,                               # 4 vCPU per container; pyhmmer is multi-threaded
    memory=4096,                         # 4 GB RAM (fits 700 MB HMM + working set)
    timeout=3600 * 4,
    secrets=[modal.Secret.from_name("ncbi-key", required_keys=["NCBI_API_KEY"])],
    max_containers=16,
    scaledown_window=120,
)
class KofamScanner:
    @modal.enter()
    def setup(self):
        import os

        import pyhmmer
        import pyhmmer.easel
        import pyhmmer.plan7

        self.pyhmmer = pyhmmer
        self.alphabet = pyhmmer.easel.Alphabet.amino()
        with pyhmmer.plan7.HMMFile("/root/kofam.hmm") as fh:
            self.hmms = list(fh)

        # Load per-KO bitscore thresholds
        thresholds: dict[str, float] = {}
        with open("/root/ko_thresholds.tsv") as fh:
            next(fh)  # header
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                ko = parts[0]
                try:
                    thresholds[ko] = float(parts[1])
                except (TypeError, ValueError):
                    thresholds[ko] = 0.0
        self.thresholds = thresholds
        self.ncbi_key = os.environ.get("NCBI_API_KEY")
        print(f"[setup] loaded {len(self.hmms):,} KOfam HMMs, "
              f"{len(self.thresholds):,} thresholds", flush=True)

    def _scan(self, proteins: list[str]) -> set[str]:
        seqs = []
        for i, prot in enumerate(proteins):
            if not prot:
                continue
            ts = self.pyhmmer.easel.TextSequence(name=f"p{i}".encode(), sequence=prot)
            seqs.append(ts.digitize(self.alphabet))
        found: set[str] = set()
        if not seqs:
            return found
        for top_hits in self.pyhmmer.hmmer.hmmsearch(self.hmms, seqs, E=DEFAULT_EVALUE):
            raw = top_hits.query.name
            ko = raw.decode() if isinstance(raw, bytes) else raw
            thr = self.thresholds.get(ko, 0.0)
            for hit in top_hits:
                if hit.score >= thr and hit.evalue <= DEFAULT_EVALUE:
                    found.add(ko)
                    break
        return found

    @modal.method()
    def scan_genome(self, accession: str) -> dict | None:
        try:
            contigs = _fetch_fasta_bytes(accession, self.ncbi_key)
            if not contigs:
                return None
            proteins = _predict_proteins(contigs)
            if not proteins:
                return None
            ko_hits = self._scan(proteins)
            return {"genome_accession": accession, "ko_hits": sorted(ko_hits)}
        except Exception as exc:
            print(f"  skip {accession}: {type(exc).__name__}: {exc}", flush=True)
            return None


@app.local_entrypoint()
def main(out_path: str = "data/kofam_hits.jsonl", limit: int = 0):
    import pandas as pd

    feats = pd.read_parquet("data/features.parquet")
    unique_accs = feats["genome_accession"].dropna().astype(str).unique().tolist()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        with open(out) as fh:
            for line in fh:
                try:
                    done.add(str(json.loads(line)["genome_accession"]))
                except Exception:
                    continue

    pending = [a for a in unique_accs if a not in done]
    if limit:
        pending = pending[:limit]
    print(f"KOfam scan on Modal: {len(pending):,} pending ({len(done):,} cached)")
    if not pending:
        return

    scanner = KofamScanner()

    n_ok = n_fail = 0
    with open(out, "a") as log:
        for result in scanner.scan_genome.map(pending, return_exceptions=True):
            if isinstance(result, Exception) or result is None:
                n_fail += 1
                continue
            log.write(json.dumps(result) + "\n")
            log.flush()
            n_ok += 1
            if n_ok % 200 == 0:
                print(f"  {n_ok:,} ok / {n_fail:,} fail")
    print(f"\nFinished. {n_ok:,} succeeded, {n_fail:,} failed.")
