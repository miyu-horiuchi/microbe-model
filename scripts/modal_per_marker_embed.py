"""Per-marker ESM-2 pooling — embed only HMMER-hit proteins, group by marker category.

Architecture:
  fetch FASTA → pyrodigal → pyhmmer scan → keep proteins matching marker library
                ↓
                ESM-2 forward pass on hit proteins only
                ↓
                group by marker category (8 categories), mean within each
                ↓
                concat → 8 × embed_dim features per genome

Output: data/per_marker_embeddings.parquet
  cols: genome_accession, pme_temperature_0..D-1, pme_ph_0..D-1, ...,
        pme_special_0..D-1   (8 × D total embedding cols)
        plus pme_<category>_n   (count of hit proteins in each category)
        plus pme_marker_proteins_total (total hit count across all markers)

Bundles unified_markers.hmm into the Modal image so each container can scan
locally without re-downloading.

Usage:
    modal setup  # one time
    modal run scripts/modal_per_marker_embed.py --limit 10        # smoke test
    modal run scripts/modal_per_marker_embed.py                   # full corpus
    modal run scripts/modal_per_marker_embed.py --model facebook/esm2_t30_150M_UR50D

Cost: t6 ≈ $1-2 for 22K, t30 ≈ $5-15.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

# Modal image: same deps as modal_embed.py + pyhmmer for scanning.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.2",
        "transformers>=4.40",
        "accelerate>=0.30",
        "pyrodigal>=3.5",
        "pyhmmer>=0.12",
        "biopython>=1.83",
        "requests>=2.32",
        "numpy>=1.26",
    ])
    # Bundle the unified marker library into the image so containers don't
    # have to download it. The local file must exist at this relative path.
    .add_local_file(
        "data/markers/unified/unified_markers.hmm",
        "/root/markers.hmm",
    )
)

app = modal.App("microbe-per-marker-embed", image=image)

DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"
DATASETS_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{acc}/download"
VERSION_FALLBACKS = (".1", ".2", ".3", ".4")
EMPTY_ZIP_BYTES = 2_000

# Friendly name → category map. Built from src/microbe_model/features/markers.py.
# Hard-coded here so the Modal container doesn't need the package import.
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
    gpu="A10G",
    timeout=3600 * 4,
    secrets=[modal.Secret.from_name("ncbi-key", required_keys=["NCBI_API_KEY"])],
    max_containers=8,
    scaledown_window=120,
)
class PerMarkerEmbedder:
    @modal.enter()
    def setup(self):
        import os

        import numpy as np
        import pyhmmer
        import pyhmmer.easel
        import pyhmmer.plan7
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.np = np
        self.pyhmmer = pyhmmer
        self.alphabet = pyhmmer.easel.Alphabet.amino()
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.model_name = os.environ.get("ESM2_MODEL", DEFAULT_MODEL)
        self.batch_size = int(os.environ.get("ESM2_BATCH_SIZE", "16"))
        print(f"[setup] loading {self.model_name} on {self.device}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, dtype=self.dtype)
        self.model.to(self.device)
        self.model.train(False)
        self.embed_dim = self.model.config.hidden_size

        with pyhmmer.plan7.HMMFile("/root/markers.hmm") as fh:
            self.hmms = list(fh)
        print(f"[setup] loaded {len(self.hmms)} marker HMMs, embed_dim={self.embed_dim}",
              flush=True)
        self.ncbi_key = os.environ.get("NCBI_API_KEY")

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

    def _scan_for_markers(self, proteins: list[str]) -> dict[str, list[int]]:
        """Return {marker_friendly_name: [protein indices that match]}."""
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
                # name is "p<i>" — recover the index
                if hit_name.startswith("p"):
                    try:
                        result[marker].append(int(hit_name[1:]))
                    except ValueError:
                        pass
        return result

    @modal.method()
    def embed_genome(self, bacdive_id: int, accession: str) -> dict | None:
        try:
            contigs = _fetch_fasta_bytes(accession, self.ncbi_key)
            if not contigs:
                return None
            proteins = _predict_proteins(contigs)
            if not proteins:
                return None

            marker_to_protein_idx = self._scan_for_markers(proteins)

            # Collect unique protein indices that hit any marker.
            hit_indices = sorted({i for ids in marker_to_protein_idx.values() for i in ids})
            if not hit_indices:
                # Genome has no marker hits. Return zeros so the row still merges.
                row = {"bacdive_id": int(bacdive_id), "genome_accession": accession,
                       "pme_marker_proteins_total": 0}
                for cat in CATEGORIES:
                    row[f"pme_{cat}_n"] = 0
                    for d in range(self.embed_dim):
                        row[f"pme_{cat}_{d}"] = 0.0
                return row

            hit_proteins = [proteins[i] for i in hit_indices]
            hit_matrix = self._embed_proteins(hit_proteins)  # (n_hit, D)
            idx_to_row = {gi: ri for ri, gi in enumerate(hit_indices)}

            # For each category, gather protein rows belonging to its markers.
            row = {"bacdive_id": int(bacdive_id), "genome_accession": accession,
                   "pme_marker_proteins_total": len(hit_indices)}
            for cat in CATEGORIES:
                idxs: set[int] = set()
                for marker, gis in marker_to_protein_idx.items():
                    if MARKER_TO_CATEGORY.get(marker) == cat:
                        idxs.update(gis)
                row[f"pme_{cat}_n"] = len(idxs)
                if idxs:
                    cat_rows = [idx_to_row[gi] for gi in idxs if gi in idx_to_row]
                    if cat_rows:
                        cat_matrix = hit_matrix[cat_rows]
                        cat_mean = cat_matrix.mean(axis=0).astype(self.np.float32)
                        for d, v in enumerate(cat_mean):
                            row[f"pme_{cat}_{d}"] = float(v)
                        continue
                # No hits for this category: fill with zeros
                for d in range(self.embed_dim):
                    row[f"pme_{cat}_{d}"] = 0.0
            return row
        except Exception as exc:
            print(f"  skip {accession}: {type(exc).__name__}: {exc}", flush=True)
            return None


@app.local_entrypoint()
def main(
    model: str = DEFAULT_MODEL,
    batch_size: int = 16,
    gpu: str = "A10G",
    out_path: str = "data/per_marker_embeddings.jsonl",
    limit: int = 0,
):
    """Dispatch genomes to Modal containers; stream results to local JSONL."""
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
    print(f"Per-marker embed: {len(tasks):,} genomes pending ({len(done):,} cached)")
    print(f"Model: {model}  batch={batch_size}  gpu={gpu}")
    if not tasks:
        return

    config_secret = modal.Secret.from_dict({
        "ESM2_MODEL": model,
        "ESM2_BATCH_SIZE": str(batch_size),
    })
    embedder = PerMarkerEmbedder.with_options(
        gpu=gpu,
        secrets=[
            modal.Secret.from_name("ncbi-key", required_keys=["NCBI_API_KEY"]),
            config_secret,
        ],
    )()

    n_ok = n_fail = 0
    with open(out, "a") as log:
        for result in embedder.embed_genome.starmap(tasks, return_exceptions=True):
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
