from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def load_script_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "43_run_genomespot_benchmark.py"
    spec = importlib.util.spec_from_file_location("genomespot_benchmark", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_one_reuses_existing_prediction_file(monkeypatch, tmp_path: Path) -> None:
    mod = load_script_module()
    accession = "GCA_000001"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    pred_path = output_dir / f"{accession}.predictions.tsv"
    pred_path.write_text(
        "\tvalue\terror\n"
        "temperature_optimum\t42.5\t1.5\n"
        "ph_optimum\t7.25\t0.2\n"
        "salinity_optimum\t1.75\t0.3\n"
        "oxygen\ttolerant\t0.9\n"
    )

    monkeypatch.setattr(
        mod,
        "ensure_inputs",
        lambda row, fasta_dir: (tmp_path / "genome.fna.gz", tmp_path / "proteins.faa.gz", None),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("GenomeSPOT command should not run when predictions already exist")

    monkeypatch.setattr(mod.subprocess, "run", fail_if_called)

    row = pd.Series(
        {
            "bacdive_id": 1,
            "genome_accession": accession,
            "fold": 0,
            "optimal_temperature_c": 40.0,
            "optimal_ph": 7.0,
            "salt_tolerance_pct": 2.0,
            "oxygen_requirement": "aerobe",
        }
    )

    result = mod.run_one(
        row,
        genome_spot_dir=tmp_path / "GenomeSPOT",
        fasta_dir=tmp_path / "fastas",
        output_dir=output_dir,
    )

    assert result["status"] == "ok"
    assert result["genomespot_temperature_c"] == 42.5
    assert result["genomespot_ph"] == 7.25
    assert result["genomespot_salt_pct"] == 1.75
    assert result["genomespot_oxygen"] == "tolerant"
