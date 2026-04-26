"""Test the streaming pipeline helpers (excluding network calls)."""
from __future__ import annotations

from microbe_model.pipeline import (
    _candidate_accessions,
    _has_version,
    _parse_fasta_bytes,
)


def test_has_version() -> None:
    assert _has_version("GCF_000005845.2") is True
    assert _has_version("GCA_000511385.1") is True
    assert _has_version("GCA_000511385") is False
    assert _has_version("GCA_000511385.X") is False  # non-numeric suffix
    assert _has_version("plain_text") is False


def test_candidate_accessions_versioned_passes_through() -> None:
    assert _candidate_accessions("GCF_000005845.2") == ["GCF_000005845.2"]


def test_candidate_accessions_unversioned_expands() -> None:
    candidates = _candidate_accessions("GCA_000511385")
    assert candidates == [
        "GCA_000511385.1",
        "GCA_000511385.2",
        "GCA_000511385.3",
        "GCA_000511385.4",
    ]


def test_parse_fasta_bytes_minimal() -> None:
    raw = b">contig_1\nACGTACGT\nGCTA\n>contig_2 description here\nNNNN\n"
    contigs = _parse_fasta_bytes(raw)
    assert contigs == [("contig_1", "ACGTACGTGCTA"), ("contig_2", "NNNN")]


def test_parse_fasta_bytes_uppercases() -> None:
    raw = b">x\nacgt\n"
    assert _parse_fasta_bytes(raw) == [("x", "ACGT")]


def test_parse_fasta_bytes_empty() -> None:
    assert _parse_fasta_bytes(b"") == []
