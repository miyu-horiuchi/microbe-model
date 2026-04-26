"""Test BacDive phenotype extraction against a fixture of the real v2 schema."""
from __future__ import annotations

from microbe_model.data.bacdive import _derive_optimum, extract_phenotypes

# Trimmed-down version of a real /v2/fetch/24493 response (Phaeobacter gallaeciensis BS 107).
SAMPLE_RECORD = {
    "General": {
        "BacDive-ID": 24493,
        "NCBI tax id": [
            {"NCBI tax id": 1423144, "Matching level": "strain"},
            {"NCBI tax id": 60890, "Matching level": "species"},
        ],
    },
    "Name and taxonomic classification": {
        "LPSN": {
            "domain": "Bacteria",
            "phylum": "Pseudomonadota",
            "class": "Alphaproteobacteria",
            "order": "Rhodobacterales",
            "family": "Roseobacteraceae",
            "genus": "Phaeobacter",
            "species": "Phaeobacter gallaeciensis",
        },
        "genus": "Phaeobacter",
        "species": "Phaeobacter gallaeciensis",
    },
    "Culture and growth conditions": {
        "culture temp": [
            {"growth": "positive", "type": "growth", "temperature": "25"},
            {"growth": "positive", "type": "growth", "temperature": "22"},
            {"growth": "positive", "type": "growth", "temperature": "5-30"},
            {"growth": "negative", "type": "growth", "temperature": "37"},
        ],
    },
    "Physiology and metabolism": {
        "oxygen tolerance": [{"oxygen tolerance": "obligate aerobe"}],
    },
    "Sequence information": {
        "Genome sequences": [
            {"INSDC accession": "GCA_000511385", "assembly level": "complete"},
            {"INSDC accession": "GCA_000819625", "assembly level": "contig"},
        ],
    },
}


def test_extract_phenotypes_real_schema() -> None:
    out = extract_phenotypes(SAMPLE_RECORD)
    assert out["bacdive_id"] == 24493
    assert out["species"] == "Phaeobacter gallaeciensis"
    assert out["genus"] == "Phaeobacter"
    assert out["family"] == "Roseobacteraceae"
    assert out["ncbi_taxon_id"] == 1423144
    assert out["genome_accession"] == "GCA_000511385"  # first listed
    assert out["oxygen_requirement"] == "obligate aerobe"

    # Three positive-growth temps: 25, 22, midpoint(5-30)=17.5 → median = 22
    assert out["optimal_temperature_c"] == 22.0


def test_derive_optimum_prefers_explicit_optimum() -> None:
    entries = [
        {"type": "growth", "growth": "positive", "temperature": "30"},
        {"type": "optimum", "temperature": "37"},
        {"type": "growth", "growth": "positive", "temperature": "25"},
    ]
    assert _derive_optimum(entries, "temperature") == 37.0


def test_derive_optimum_falls_back_to_growth_median() -> None:
    entries = [
        {"type": "growth", "growth": "positive", "temperature": "20"},
        {"type": "growth", "growth": "positive", "temperature": "30"},
        {"type": "growth", "growth": "negative", "temperature": "45"},  # ignored
    ]
    assert _derive_optimum(entries, "temperature") == 25.0


def test_extract_phenotypes_handles_missing_fields() -> None:
    out = extract_phenotypes({})
    assert out["bacdive_id"] is None
    assert out["genome_accession"] is None
    assert out["optimal_temperature_c"] is None
