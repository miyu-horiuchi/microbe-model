"""Tests for tetranucleotide + codon-frequency features."""
from __future__ import annotations

from microbe_model.features.composition import codon_freqs, tetranucleotide_freqs


def test_tetranucleotide_freqs_sum_to_one() -> None:
    contigs = [("c1", "ACGT" * 100)]  # 400 nt → 397 4-mers
    out = tetranucleotide_freqs(contigs)
    assert len(out) == 256
    total = sum(out.values())
    assert abs(total - 1.0) < 1e-6


def test_tetranucleotide_freqs_handles_n() -> None:
    contigs = [("c1", "ACGNACGTACGT")]
    out = tetranucleotide_freqs(contigs)
    # All 4-mers containing N should be skipped; valid ones (ACGT, CGTA, GTAC, TACG) counted
    nonzero = {k: v for k, v in out.items() if v > 0}
    assert all(("N" not in k.removeprefix("tetra_")) for k in nonzero)
    assert nonzero  # we should have some non-N kmers


def test_tetranucleotide_freqs_empty() -> None:
    out = tetranucleotide_freqs([])
    assert len(out) == 256
    assert all(v == 0.0 for v in out.values())


def test_codon_freqs_sum_to_one() -> None:
    cds_list = ["ATG" * 30 + "TAA"]  # 30 ATG codons, 1 stop
    out = codon_freqs(cds_list)
    assert len(out) == 64
    total = sum(out.values())
    assert abs(total - 1.0) < 1e-6
    # ATG should be 30/31 of the codons
    assert abs(out["codon_ATG"] - 30 / 31) < 1e-6


def test_codon_freqs_skips_non_acgt() -> None:
    cds_list = ["ATGNNNATG"]  # codon NNN should be skipped
    out = codon_freqs(cds_list)
    assert out["codon_ATG"] == 1.0  # only the two ATG codons counted, both same
    assert sum(out.values()) == 1.0
