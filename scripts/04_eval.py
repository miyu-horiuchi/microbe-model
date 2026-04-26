"""Render the v0 eval report from the trained-results JSON + training table."""
from __future__ import annotations

from microbe_model import config
from microbe_model.eval import render_report


def main() -> None:
    results_path = config.ARTIFACTS / "baseline_results.json"
    dataset_path = config.DATA / "training_table.parquet"
    out_path = config.ARTIFACTS / "eval_report.md"

    if not results_path.exists():
        raise SystemExit(f"Missing {results_path}. Run scripts/03_train_baseline.py first.")
    if not dataset_path.exists():
        raise SystemExit(f"Missing {dataset_path}. Run scripts/03_train_baseline.py first.")

    render_report(results_path, dataset_path, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
