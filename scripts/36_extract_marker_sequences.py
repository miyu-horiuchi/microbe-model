"""Extract HMM-gated protein sequences per genome for LoRA fine-tuning.

This is a sibling to scripts/modal_per_marker_embed.py — same fetch+pyrodigal+pyhmmer
pipeline — but instead of mean-pooling ESM-2 embeddings, it emits the raw protein
sequences themselves, grouped by phenotype category. Those sequences become the input
to scripts/37_train_lora.py for end-to-end LoRA fine-tuning.

Per-genome output (one JSONL line):
    {
      "bacdive_id": 482,
      "genome_accession": "GCF_000005845.2",
      "by_category": {
          "oxygen": ["MLDF...", "MFKK...", ...],
          "temperature": ["MAKH...", ...],
          ...
      },
      "category_counts": {"oxygen": 12, "temperature": 8, ...}
    }

CPU-only (skips ESM-2). With 16 concurrent Modal containers each with a unique IP
(bypassing NCBI's 3 req/s per-IP limit), ~22K genomes should finish in ~30-60 minutes
of wall time for ~$2-5 of Modal compute.

Usage:
    modal run scripts/36_extract_marker_sequences.py --limit 50
    modal run scripts/36_extract_marker_sequences.py --max-per-cat 16
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
        "requests>=2.32",
    ])
    .add_local_file("data/markers/unified/unified_markers.hmm", "/root/markers.hmm")
)

app = modal.App("microbe-extract-marker-seqs", image=image)

DATASETS_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{acc}/download"
VERSION_FALLBACKS = (".1", ".2", ".3", ".4")
EMPTY_ZIP_BYTES = 2_000

MARKER_TO_CATEGORY: dict[str, str] = {
    # temperature
    "Hsp70_DnaK": "temperature", "Hsp90": "temperature", "Cpn60_GroEL": "temperature",
    "Hsp20": "temperature", "CSD_cold_shock": "temperature", "TGS_thermosome": "temperature",
    # pH
    "ATP_synth_alphabeta": "ph", "ATP_synth_alphabeta_C": "ph", "ATP_synth_F0_B": "ph",
    "NhaA_Na_H_exch": "ph", "NhaB_Na_H_exch": "ph", "Pyridoxal_decarbox": "ph",
    "MotA_TolQ_ExbB": "ph", "V_ATPase_subH_N": "ph",
    # oxygen
    "COX1_aerobic": "oxygen", "COX2_TM_aerobic": "oxygen", "COX2_periplasm_aero": "oxygen",
    "Cyt_CBB3_microaero": "oxygen", "Rieske_2Fe2S": "oxygen", "Catalase": "oxygen",
    "SOD_FeMn": "oxygen", "SOD_CuZn": "oxygen", "FeFe_hyd_anaerobic": "oxygen",
    "NiFe_hyd_anaerobic": "oxygen", "FAD_binding_FrdA": "oxygen", "Fer4_FeS_4Fe4S": "oxygen",
    # salt
    "KdpD_osmosensor": "salt", "TrkH_K_channel": "salt", "BCCT_compatible": "salt",
    "BPD_transp_1": "salt", "EctC_ectoine_synth": "salt", "Bact_rhodopsin": "salt",
    # vitamin
    "TP_methylase_B12": "vitamin", "Peripla_BP_2": "vitamin", "THF_DHG_CYH_folate": "vitamin",
    "FolB_folate": "vitamin", "PdxJ_pyridoxine": "vitamin", "DHBP_riboflavin": "vitamin",
    # nitrogen
    "NifH_nitrogenase": "nitrogen", "NifDK_nitrogenase": "nitrogen",
    "NIR_SIR_ferredoxin": "nitrogen",
    # carbon
    "RuBisCO_large_form1": "carbon", "RuBisCO_small_form1": "carbon",
    "Alpha_amylase": "carbon", "Cellulase_GH5": "carbon", "CBM_cellulose": "carbon",
    # special
    "Molybdopterin_OR": "special", "UvrD_helicase_C": "special",
}
CATEGORIES = ["temperature", "ph", "oxygen", "salt", "vitamin", "nitrogen", "carbon", "special"]
EVALUE_THRESHOLD = 1e-5
MAX_PROTEIN_LEN = 1022  # ESM-2 context window minus special tokens


def _has_version(accession: str) -> bool:
    return "." in accession and accession.rsplit(".", 1)[-1].isdigit()


def _candidate_accessions(accession: str) -> list[str]:
    if _has_version(accession):
        return [accession]
    return [accession + v for v in VERSION_FALLBACKS]


def _fetch_fasta_bytes(accession: str, ncbi_key: str | None) -> list[tuple[str, str]] | None:
    import io
    import zipfile

    import requests

    headers = {"api-key": ncbi_key} if ncbi_key else {}
    for cand in _candidate_accessions(accession):
        url = DATASETS_URL.format(acc=cand)
        try:
            resp = requests.get(
                url,
                params={"include_annotation_type": "GENOME_FASTA"},
                headers=headers,
                timeout=120,
            )
        except requests.RequestException:
            continue
        if resp.status_code != 200 or len(resp.content) < EMPTY_ZIP_BYTES:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
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
    name = None
    chunks: list[str] = []
    for line in raw.decode("utf-8", errors="ignore").splitlines():
        if line.startswith(">"):
            if name is not None:
                contigs.append((name, "".join(chunks)))
            name = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line.strip())
    if name is not None:
        contigs.append((name, "".join(chunks)))
    return contigs


def _predict_proteins(contigs: list[tuple[str, str]]) -> list[str]:
    import pyrodigal

    if not contigs:
        return []
    total = sum(len(s) for _, s in contigs)
    meta = total < 100_000
    orf = pyrodigal.GeneFinder(meta=meta)
    if not meta:
        orf.train(*[s.encode() for _, s in contigs])
    proteins: list[str] = []
    for _, seq in contigs:
        for gene in orf.find_genes(seq.encode()):
            proteins.append(gene.translate().rstrip("*"))
    return proteins


@app.cls(
    cpu=2.0,
    memory=2048,
    timeout=3600 * 4,
    secrets=[modal.Secret.from_name("ncbi-key", required_keys=["NCBI_API_KEY"])],
    max_containers=16,
    scaledown_window=60,
)
class MarkerSeqExtractor:
    @modal.enter()
    def setup(self):
        import os

        import pyhmmer
        import pyhmmer.easel
        import pyhmmer.plan7

        self.pyhmmer = pyhmmer
        self.alphabet = pyhmmer.easel.Alphabet.amino()

        with pyhmmer.plan7.HMMFile("/root/markers.hmm") as fh:
            self.hmms = list(fh)
        print(f"[setup] loaded {len(self.hmms)} marker HMMs", flush=True)
        self.ncbi_key = os.environ.get("NCBI_API_KEY")
        self.max_per_cat = int(os.environ.get("MAX_PER_CATEGORY", "16"))

    def _scan_for_markers(self, proteins: list[str]) -> dict[str, list[int]]:
        seqs = []
        for i, prot in enumerate(proteins):
            if not prot:
                continue
            ts = self.pyhmmer.easel.TextSequence(name=f"p{i}".encode(), sequence=prot)
            seqs.append(ts.digitize(self.alphabet))
        result: dict[str, list[int]] = {name: [] for name in MARKER_TO_CATEGORY}
        if not seqs:
            return result
        for top_hits in self.pyhmmer.hmmer.hmmsearch(self.hmms, seqs, E=EVALUE_THRESHOLD):
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

    @modal.method()
    def extract_genome(self, bacdive_id: int, accession: str) -> dict | None:
        try:
            contigs = _fetch_fasta_bytes(accession, self.ncbi_key)
            if not contigs:
                return None
            proteins = _predict_proteins(contigs)
            if not proteins:
                return None

            marker_to_idx = self._scan_for_markers(proteins)

            by_category: dict[str, list[str]] = {c: [] for c in CATEGORIES}
            for cat in CATEGORIES:
                # Gather unique protein indices for this category
                idxs: set[int] = set()
                for marker, gis in marker_to_idx.items():
                    if MARKER_TO_CATEGORY.get(marker) == cat:
                        idxs.update(gis)
                # Take top-K shortest proteins (preference for unique/specific hits)
                ranked = sorted(idxs, key=lambda i: len(proteins[i]))
                kept = ranked[: self.max_per_cat]
                by_category[cat] = [proteins[i][:MAX_PROTEIN_LEN] for i in kept]

            return {
                "bacdive_id": int(bacdive_id),
                "genome_accession": accession,
                "by_category": by_category,
                "category_counts": {c: len(by_category[c]) for c in CATEGORIES},
            }
        except Exception as exc:
            print(f"  skip {accession}: {type(exc).__name__}: {exc}", flush=True)
            return None


@app.local_entrypoint()
def main(
    out_path: str = "data/marker_sequences.jsonl",
    limit: int = 0,
    max_per_cat: int = 16,
):
    """Dispatch genomes to Modal containers; stream sequences to local JSONL."""
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
    print(f"Marker-sequence extract: {len(tasks):,} genomes pending ({len(done):,} cached)")
    print(f"max_per_cat={max_per_cat}")
    if not tasks:
        return

    config_secret = modal.Secret.from_dict({"MAX_PER_CATEGORY": str(max_per_cat)})
    extractor = MarkerSeqExtractor.with_options(
        secrets=[
            modal.Secret.from_name("ncbi-key", required_keys=["NCBI_API_KEY"]),
            config_secret,
        ],
    )()

    n_ok = n_fail = 0
    with open(out, "a") as log:
        for result in extractor.extract_genome.starmap(tasks, return_exceptions=True):
            if isinstance(result, Exception) or result is None:
                n_fail += 1
                continue
            log.write(json.dumps(result) + "\n")
            log.flush()
            n_ok += 1
            if n_ok % 100 == 0:
                print(f"  {n_ok:,} ok / {n_fail:,} fail")
    print(f"\nFinished. {n_ok:,} succeeded, {n_fail:,} failed.")
    print(f"Streamed to {out}")
