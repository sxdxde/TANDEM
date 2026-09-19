"""Milestone 3 checkpoint: dynamic hybrid policy with reconfiguration cost.

Entries start at their modality-prior tier (VISUAL->WARM, TEXT->HOT) and
are promoted/demoted live based on their importance EMA, with hysteresis
and an explicit reconfiguration cost charged on every tier change. Prints
total cost with the reconfiguration overhead broken out as its own line
item, plus the promotion/demotion counts.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from kvtier.metrics import print_summary, save_timeseries_csv
from kvtier.policy import DynamicHybridPolicy
from kvtier.reconfig import load_reconfig_config
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
    reconfig_manager = load_reconfig_config(CONFIG_PATH)
    policy = DynamicHybridPolicy(**cfg["policy"]["dynamic_hybrid"])

    metrics = run_session(
        session,
        policy,
        tiers,
        reconfig_manager=reconfig_manager,
        ema_decay=cfg["trace"]["ema_decay"],
    )

    print_summary(metrics)
    out_path = RESULTS_DIR / "dynamic_hybrid_timeseries.csv"
    save_timeseries_csv(metrics, out_path)
    print(f"\nTime series written to {out_path}")


if __name__ == "__main__":
    main()
