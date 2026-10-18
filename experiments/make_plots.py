"""Milestone 4: policy comparison bar chart and tier-over-time timeline.

Runs flat, static_modality and dynamic_hybrid on the default config and
writes results/comparison_bars.png, results/tier_timeline.png and
results/milestone4_comparison.csv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from kvtier.metrics import combine_summaries
from kvtier.policy import DynamicHybridPolicy, FlatPolicy, StaticModalityPolicy
from kvtier.reconfig import load_reconfig_config
from kvtier.simulator import run_session
from kvtier.tiers import load_tier_config
from kvtier.trace import generate_session

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    tiers = load_tier_config(CONFIG_PATH)
    reconfig = load_reconfig_config(CONFIG_PATH)

    runs = []
    for policy in (FlatPolicy(), StaticModalityPolicy(),
                   DynamicHybridPolicy(**cfg["policy"]["dynamic_hybrid"])):
        session = generate_session(**cfg["trace"])
        runs.append(run_session(session, policy, tiers, reconfig_manager=reconfig,
                                ema_decay=cfg["trace"]["ema_decay"]))

    summary = combine_summaries(runs)
    summary.to_csv(RESULTS_DIR / "milestone4_comparison.csv", index=False)
    print(summary[["policy", "total_latency_ns", "total_energy_pj", "mean_quality_loss",
                   "warm_attention_mass_pct", "n_promotions", "n_demotions"]].to_string(index=False))

    # Bar chart: energy (log), latency, quality loss.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    for ax, col, title, log in (
        (axes[0], "total_energy_pj", "Total energy (pJ, log)", True),
        (axes[1], "total_latency_ns", "Total latency (ns)", False),
        (axes[2], "mean_quality_loss", "Mean quality loss", False),
    ):
        ax.bar(summary.policy, summary[col], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        if log:
            ax.set_yscale("log")
    fig.suptitle("Policy comparison, default config (reconfig cost 10x)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "comparison_bars.png", dpi=150)

    # Timeline: HOT vs WARM share of attention-served accesses + tier moves.
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for m, c in zip(runs, colors):
        ts = m.to_timeseries_df()
        axes[0].plot(ts.step, ts.warm_attention_mass, label=m.policy_name, color=c)
    axes[0].set_ylabel("attention mass served from WARM")
    axes[0].legend()
    dyn = runs[2].to_timeseries_df()
    axes[1].bar(dyn.step, dyn.n_promotions, label="promotions", color="#55a868")
    axes[1].bar(dyn.step, -dyn.n_demotions, label="demotions", color="#c44e52")
    axes[1].set_ylabel("dynamic_hybrid tier moves")
    axes[1].set_xlabel("decode step")
    axes[1].legend()
    fig.suptitle("Tier usage over time")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "tier_timeline.png", dpi=150)


if __name__ == "__main__":
    main()
