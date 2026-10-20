"""Milestone 4b: sweep DynamicHybridPolicy's warmup / min-dwell damping.

Averages over several trace seeds. Selection rule (fixed before looking at
results): lowest mean energy among configs whose mean quality loss is at most
``QUALITY_BUDGET`` x static_modality's, i.e. dynamic must keep delivering the
quality gain that justifies it. Writes results/damping_sweep.csv.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
import yaml

from kvtier.policy import DynamicHybridPolicy, StaticModalityPolicy
from kvtier.reconfig import load_reconfig_config
from kvtier.simulator import run_session
from kvtier.tiers import load_tier_config
from kvtier.trace import generate_session

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

SEEDS = [0, 1, 2, 3, 4]
WARMUP = [0, 2, 4, 8, 12, 16]
DWELL = [0, 3, 6, 10, 20, 40]
QUALITY_BUDGET = 0.8


def main() -> None:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    tiers = load_tier_config(CONFIG_PATH)
    reconfig = load_reconfig_config(CONFIG_PATH)
    base = {k: v for k, v in cfg["policy"]["dynamic_hybrid"].items()
            if k not in ("warmup_steps", "min_dwell_steps", "damp_promotions")}

    def run(seed, policy):
        trace_cfg = {**cfg["trace"], "seed": seed}
        return run_session(generate_session(**trace_cfg), policy, tiers,
                           reconfig_manager=reconfig, ema_decay=cfg["trace"]["ema_decay"])

    static = {s: run(s, StaticModalityPolicy()) for s in SEEDS}
    rows = []
    for dp, w, d in itertools.product([True, False], WARMUP, DWELL):
        rs = {s: run(s, DynamicHybridPolicy(**base, warmup_steps=w, min_dwell_steps=d, damp_promotions=dp)) for s in SEEDS}
        rows.append(dict(
            damp_promotions=dp, warmup=w, dwell=d,
            energy_vs_static_pct=100 * sum(rs[s].total_energy_pj / static[s].total_energy_pj - 1 for s in SEEDS) / len(SEEDS),
            latency_vs_static_pct=100 * sum(rs[s].total_latency_ns / static[s].total_latency_ns - 1 for s in SEEDS) / len(SEEDS),
            quality_vs_static_pct=100 * sum(rs[s].mean_quality_loss / static[s].mean_quality_loss - 1 for s in SEEDS) / len(SEEDS),
            reconfig_energy_share_pct=100 * sum(rs[s].total_reconfig_energy_pj / rs[s].total_energy_pj for s in SEEDS) / len(SEEDS),
            moves=sum(rs[s].total_promotions + rs[s].total_demotions for s in SEEDS) / len(SEEDS),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "damping_sweep.csv", index=False)

    print(f"Undamped (warmup 0, dwell 0):\n{df[(df.warmup == 0) & (df.dwell == 0)].head(1).to_string(index=False)}\n")
    ok = df[df.quality_vs_static_pct <= -100 * (1 - QUALITY_BUDGET)]
    print(f"{len(ok)}/{len(df)} configs keep >= {100 * (1 - QUALITY_BUDGET):.0f}% quality gain. Best 8 by energy:")
    print(ok.sort_values("energy_vs_static_pct").head(8).to_string(index=False))


if __name__ == "__main__":
    main()
