from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def load_script_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "42_prepare_external_benchmarks.py"
    spec = importlib.util.spec_from_file_location("external_benchmark_prep", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_fasta_path_accepts_common_suffixes(tmp_path: Path) -> None:
    mod = load_script_module()
    fasta = tmp_path / "GCF_000001.fna.gz"
    fasta.write_text(">contig\nATGC\n")

    assert mod.local_fasta_path(tmp_path, "GCF_000001") == fasta
    assert mod.local_fasta_path(tmp_path, "GCF_000002") is None


def test_fasta_coverage_counts_unique_accessions(tmp_path: Path) -> None:
    mod = load_script_module()
    (tmp_path / "GCF_000001.fna").write_text(">contig\nATGC\n")
    manifest = pd.DataFrame(
        {
            "genome_accession": ["GCF_000001", "GCF_000001", "GCF_000002"],
        }
    )

    coverage = mod.fasta_coverage(manifest, tmp_path)

    assert coverage["unique_accessions"] == 2
    assert coverage["present_fastas"] == 1
    assert coverage["missing_fastas"] == 1
    assert coverage["coverage_pct"] == 50.0
