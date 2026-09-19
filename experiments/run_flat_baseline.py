"""Milestone 2 checkpoint: flat (everything-HOT, never tiered) baseline run."""

from __future__ import annotations

from pathlib import Path

import yaml

from kvtier.metrics import print_summary, save_timeseries_csv
from kvtier.policy import FlatPolicy
from kvtier.simulator import run_session
from kvtier.tiers import load_tier_config
from kvtier.trace import generate_session

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    session = generate_session(**cfg["trace"])
    tiers = load_tier_config(CONFIG_PATH)
    metrics = run_session(session, FlatPolicy(), tiers)

    print_summary(metrics)
    out_path = RESULTS_DIR / "flat_baseline_timeseries.csv"
    save_timeseries_csv(metrics, out_path)
    print(f"\nTime series written to {out_path}")


if __name__ == "__main__":
    main()
