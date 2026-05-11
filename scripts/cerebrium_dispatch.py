"""Dispatch KOfam scans and/or ESM-2 embeddings to Cerebrium.

Each Cerebrium replica is a stateless HTTP endpoint that handles one genome at
a time (`replica_concurrency = 1`). We fan out across `max_replicas` parallel
in-flight requests; results stream to JSONL as they arrive.

Usage:
    # Smoke test: 5 KOfam scans
    uv run python scripts/cerebrium_dispatch.py kofam --limit 5

    # Smoke test: 5 embeddings
    uv run python scripts/cerebrium_dispatch.py embed --limit 5

    # Full corpus (defaults: concurrency = max_replicas of the deployed app)
    uv run python scripts/cerebrium_dispatch.py kofam
    uv run python scripts/cerebrium_dispatch.py embed
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yaml

PROJECT_ID = "p-11b3c6c6"
REGION_HOST = "https://api.aws.us-east-1.cerebrium.ai"

APP_CONFIG = {
    "kofam": {
        "function": "scan_genome",
        "concurrency": 10,
        "out_path": Path("data/kofam_hits.jsonl"),
        "id_field": "genome_accession",
        "request_timeout": 600,
        "ok_keys": ("ko_hits",),
    },
    "embed": {
        "function": "embed_genome",
        "concurrency": 3,
        "out_path": Path("data/per_marker_embeddings.jsonl"),
        "id_field": "bacdive_id",
        "request_timeout": 600,
        "ok_keys": ("row",),
    },
}


def _read_access_token() -> str:
    env_token = os.environ.get("CEREBRIUM_API_KEY") or os.environ.get("CEREBRIUM_INFERENCE_KEY")
    if env_token:
        return env_token
    sys.exit(
        "Set CEREBRIUM_API_KEY to a JWT from the dashboard's API Keys section. "
        "The CLI's accesstoken doesn't work for inference endpoints."
    )


def _load_pending_kofam(limit: int) -> list[dict[str, Any]]:
    feats = pd.read_parquet("data/features.parquet")
    accs = feats["genome_accession"].dropna().astype(str).unique().tolist()
    done: set[str] = set()
    out_path = APP_CONFIG["kofam"]["out_path"]
    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    done.add(str(json.loads(line)["genome_accession"]))
                except Exception:
                    continue
    pending = [a for a in accs if a not in done]
    if limit:
        pending = pending[:limit]
    return [{"accession": a} for a in pending]


def _load_pending_embed(limit: int) -> list[dict[str, Any]]:
    import microbe_model.config as cfg
    pheno = pd.read_parquet("data/bacdive_phenotypes.parquet")
    has_genome = pheno["genome_accession"].notna()
    label_cols = list(cfg.PHENOTYPE_TARGETS.keys())
    has_label = pheno[label_cols].notna().any(axis=1)
    ready = pheno[has_genome & has_label].copy()
    ready["bacdive_id"] = ready["bacdive_id"].astype(int)

    done: set[int] = set()
    out_path = APP_CONFIG["embed"]["out_path"]
    if out_path.exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    done.add(int(json.loads(line)["bacdive_id"]))
                except Exception:
                    continue
    pending = ready[~ready["bacdive_id"].isin(done)]
    if limit:
        pending = pending.head(limit)
    return [
        {"bacdive_id": int(row["bacdive_id"]), "accession": str(row["genome_accession"])}
        for _, row in pending.iterrows()
    ]


async def _call_once(
    client: httpx.AsyncClient, app: str, payload: dict[str, Any], token: str,
    timeout: float,
) -> dict[str, Any]:
    url = f"{REGION_HOST}/v4/{PROJECT_ID}/{app}/{APP_CONFIG[app]['function']}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = await client.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "result" in data:
        return data["result"]
    return data


async def _worker(
    app: str, queue: asyncio.Queue, log_fh, results: dict[str, int],
    token: str, timeout: float, sem: asyncio.Semaphore,
):
    async with httpx.AsyncClient() as client:
        while True:
            payload = await queue.get()
            if payload is None:
                queue.task_done()
                return
            async with sem:
                start = time.time()
                for attempt in range(3):
                    try:
                        out = await _call_once(client, app, payload, token, timeout)
                        elapsed = time.time() - start
                        if isinstance(out, dict) and out.get("ok"):
                            log_fh.write(json.dumps(out.get("row") if app == "embed" else out) + "\n")
                            log_fh.flush()
                            results["ok"] += 1
                            results["elapsed_sum"] += elapsed
                        else:
                            results["fail"] += 1
                            reason = out.get("reason", "?") if isinstance(out, dict) else "non-dict"
                            print(f"  fail {payload}: {reason}", flush=True)
                        break
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code in (429, 502, 503, 504):
                            await asyncio.sleep(2 ** attempt)
                            continue
                        results["fail"] += 1
                        print(f"  http {exc.response.status_code} {payload}: {exc.response.text[:200]}",
                              flush=True)
                        break
                    except (httpx.TimeoutException, httpx.TransportError) as exc:
                        if attempt < 2:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        results["fail"] += 1
                        print(f"  timeout {payload}: {exc}", flush=True)
            queue.task_done()


async def _run(app: str, jobs: list[dict[str, Any]], concurrency: int):
    cfg = APP_CONFIG[app]
    token = _read_access_token()
    out_path: Path = cfg["out_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue = asyncio.Queue()
    for j in jobs:
        await queue.put(j)
    for _ in range(concurrency):
        await queue.put(None)

    results = {"ok": 0, "fail": 0, "elapsed_sum": 0.0}
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    with open(out_path, "a") as log_fh:
        workers = [
            asyncio.create_task(_worker(
                app, queue, log_fh, results, token, cfg["request_timeout"], sem,
            ))
            for _ in range(concurrency)
        ]
        last_report = t0
        while any(not w.done() for w in workers):
            await asyncio.sleep(15)
            now = time.time()
            done = results["ok"] + results["fail"]
            if done == 0:
                continue
            rate = done / (now - t0)
            remaining = len(jobs) - done
            eta = remaining / rate if rate > 0 else float("inf")
            if now - last_report >= 30:
                print(
                    f"  [{int(now - t0)}s] ok={results['ok']:,} fail={results['fail']:,} "
                    f"rate={rate:.2f}/s eta={int(eta/60)}min", flush=True,
                )
                last_report = now
        await asyncio.gather(*workers)
    elapsed = time.time() - t0
    avg_per_ok = results["elapsed_sum"] / max(results["ok"], 1)
    print(f"\nDone in {elapsed/60:.1f} min. ok={results['ok']:,} fail={results['fail']:,} "
          f"avg/ok={avg_per_ok:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("app", choices=list(APP_CONFIG.keys()))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=0,
                        help="Override default (default: app's max_replicas)")
    args = parser.parse_args()

    if args.app == "kofam":
        jobs = _load_pending_kofam(args.limit)
    else:
        jobs = _load_pending_embed(args.limit)
    if not jobs:
        print("Nothing to do.")
        return
    concurrency = args.concurrency or APP_CONFIG[args.app]["concurrency"]
    print(f"Dispatching {len(jobs):,} jobs to Cerebrium app '{args.app}' "
          f"at concurrency={concurrency}.")
    print(f"  Output: {APP_CONFIG[args.app]['out_path']}")
    asyncio.run(_run(args.app, jobs, concurrency))


if __name__ == "__main__":
    main()
