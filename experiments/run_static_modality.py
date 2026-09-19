"""Milestone 2 checkpoint: static modality-only tiering baseline run.

VISUAL entries are placed in WARM and TEXT entries in HOT at creation, and
neither ever moves afterward — no runtime importance monitoring, no
reconfiguration cost.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from kvtier.metrics import print_summary, save_timeseries_csv
from kvtier.policy import StaticModalityPolicy
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
    metrics = run_session(session, StaticModalityPolicy(), tiers)

    print_summary(metrics)
    out_path = RESULTS_DIR / "static_modality_timeseries.csv"
    save_timeseries_csv(metrics, out_path)
    print(f"\nTime series written to {out_path}")


if __name__ == "__main__":
    main()
