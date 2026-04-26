"""BacDive REST API client.

BacDive (https://bacdive.dsmz.de/) is the largest curated database of bacterial phenotypes.
Free registration is required; credentials are read from BACDIVE_USER / BACDIVE_PASSWORD.

This client does the minimum needed for v0:
  - log in and obtain an OAuth token
  - paginate through the strain catalog
  - fetch full records by BacDive ID
  - extract the phenotype targets we predict (T_opt, pH_opt, oxygen, salt)
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests

from microbe_model import config

BASE_URL = "https://api.bacdive.dsmz.de"
TOKEN_URL = "https://sso.dsmz.de/auth/realms/dsmz/protocol/openid-connect/token"


class BacDiveAuthError(RuntimeError):
    pass


class BacDiveClient:
    def __init__(self, user: str | None = None, password: str | None = None) -> None:
        self.user = user or config.BACDIVE_USER
        self.password = password or config.BACDIVE_PASSWORD
        if not self.user or not self.password:
            raise BacDiveAuthError(
                "Set BACDIVE_USER and BACDIVE_PASSWORD in .env (register at bacdive.dsmz.de)."
            )
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._session = requests.Session()

    def _refresh_token(self) -> None:
        resp = self._session.post(
            TOKEN_URL,
            data={
                "grant_type": "password",
                "client_id": "api.bacdive.public",
                "username": self.user,
                "password": self.password,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise BacDiveAuthError(f"BacDive auth failed: {resp.status_code} {resp.text}")
        body = resp.json()
        self._token = body["access_token"]
        self._token_expires_at = time.time() + body.get("expires_in", 300) - 30

    def _headers(self) -> dict[str, str]:
        if self._token is None or time.time() >= self._token_expires_at:
            self._refresh_token()
        return {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        url = f"{BASE_URL}{path}"
        for attempt in range(3):
            resp = self._session.get(url, headers=self._headers(), params=params, timeout=60)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return {}

    def iter_strain_ids(self, limit: int | None = None) -> Iterator[int]:
        """Page through the BacDive catalog and yield strain IDs."""
        page_url: str | None = "/fetch/"
        seen = 0
        while page_url:
            body = self._get(page_url)
            for record in body.get("results", []):
                yield int(record["id"])
                seen += 1
                if limit is not None and seen >= limit:
                    return
            next_url = body.get("next")
            if not next_url:
                return
            page_url = next_url.replace(BASE_URL, "")

    def fetch_record(self, bacdive_id: int) -> dict[str, Any]:
        body = self._get(f"/fetch/{bacdive_id}")
        results = body.get("results") or {}
        if isinstance(results, list):
            return results[0] if results else {}
        if isinstance(results, dict) and str(bacdive_id) in results:
            return results[str(bacdive_id)]
        return results


def extract_phenotypes(record: dict[str, Any]) -> dict[str, Any]:
    """Pull the v0 prediction targets out of a BacDive record.

    BacDive's record schema is deeply nested and field names vary across record versions.
    We tolerate missing fields — anything we can't find becomes None and is dropped at training time.
    """
    out: dict[str, Any] = {
        "bacdive_id": record.get("General", {}).get("BacDive-ID"),
        "species": record.get("Name and taxonomic classification", {}).get("species"),
        "ncbi_taxon_id": record.get("General", {}).get("NCBI tax id"),
        "optimal_temperature_c": None,
        "optimal_ph": None,
        "oxygen_requirement": None,
        "salt_tolerance_pct": None,
        "genome_accession": None,
    }

    culture = record.get("Culture and growth conditions", {})
    temps = _as_list(culture.get("culture temp"))
    for t in temps:
        if isinstance(t, dict) and t.get("type", "").lower() in {"optimum", "optimal"}:
            out["optimal_temperature_c"] = _to_float(t.get("temperature"))
            break

    phs = _as_list(culture.get("culture pH"))
    for p in phs:
        if isinstance(p, dict) and p.get("type", "").lower() in {"optimum", "optimal"}:
            out["optimal_ph"] = _to_float(p.get("pH"))
            break

    physio = record.get("Physiology and metabolism", {})
    oxygen = _as_list(physio.get("oxygen tolerance"))
    if oxygen and isinstance(oxygen[0], dict):
        out["oxygen_requirement"] = oxygen[0].get("oxygen tolerance")

    salt = _as_list(physio.get("halophily"))
    for s in salt:
        if isinstance(s, dict) and "concentration" in s:
            out["salt_tolerance_pct"] = _to_float(s.get("concentration"))
            break

    seq = record.get("Sequence information", {})
    genomes = _as_list(seq.get("genome sequence"))
    for g in genomes:
        if isinstance(g, dict) and g.get("accession"):
            out["genome_accession"] = g["accession"]
            break

    return out


def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(str(x).split()[0])
    except (ValueError, AttributeError):
        return None


def cache_path(bacdive_id: int) -> Path:
    return config.BACDIVE_DIR / f"{bacdive_id}.json"


def fetch_with_cache(client: BacDiveClient, bacdive_id: int) -> dict[str, Any]:
    path = cache_path(bacdive_id)
    if path.exists():
        return json.loads(path.read_text())
    record = client.fetch_record(bacdive_id)
    path.write_text(json.dumps(record))
    return record
