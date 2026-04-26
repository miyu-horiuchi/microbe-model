"""Smoke test the feature extractor on a tiny synthetic genome."""
from __future__ import annotations

import gzip
from pathlib import Path

from microbe_model.features.genome import extract_features


def _write_fake_genome(path: Path) -> None:
    """Write a tiny FASTA with two contigs of synthetic GC-balanced sequence."""
    contigs = [
        (">contig_1\n" + ("ATGCGTACGTAGCTAGCTAGCATGCGTACG" * 200) + "\n"),
        (">contig_2\n" + ("CGTACGATCGATCGTACGTAGCTACGATGC" * 200) + "\n"),
    ]
    with gzip.open(path, "wt") as fh:
        fh.write("".join(contigs))


def test_extract_features_runs(tmp_path: Path) -> None:
    fasta = tmp_path / "fake.fna.gz"
    _write_fake_genome(fasta)

    feats = extract_features(fasta)

    assert feats["genome_size_nt"] > 0
    assert 0 <= feats["gc_content"] <= 1
    assert feats["n_contigs"] == 2
    assert feats["n_predicted_cds"] >= 0  # synthetic seq may have no real ORFs

    # Amino acid fractions should sum to ~1 if any proteins were found, else 0.
    aa_total = sum(v for k, v in feats.items() if k.startswith("aa_frac_"))
    assert aa_total == 0.0 or abs(aa_total - 1.0) < 1e-6


def test_isoelectric_point_in_range() -> None:
    from microbe_model.features.genome import _isoelectric_point

    assert 0 <= _isoelectric_point("AAAAA") <= 14
    assert 0 <= _isoelectric_point("DDDDD") <= 14
    assert 0 <= _isoelectric_point("KKKKK") <= 14
    # Acidic protein should have lower pI than basic
    assert _isoelectric_point("DDDDD") < _isoelectric_point("KKKKK")
