"""Project paths and shared constants."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

BACDIVE_DIR = DATA / "bacdive"
GENOME_DIR = DATA / "genomes"
FEATURE_DIR = DATA / "features"

for _d in (DATA, ARTIFACTS, BACDIVE_DIR, GENOME_DIR, FEATURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

NCBI_API_KEY = os.environ.get("NCBI_API_KEY")

PHENOTYPE_TARGETS = {
    "optimal_temperature_c": "regression",
    "optimal_ph": "regression",
    "oxygen_requirement": "classification",
    "salt_tolerance_pct": "regression",
}
