"""Milestone 4: sensitivity sweep of dynamic_hybrid vs. static_modality.

Sweeps promote/demote thresholds and the reconfiguration cost multiplier,
recording dynamic_hybrid's energy/latency relative to the static_modality
baseline. Writes results/sensitivity_sweep.csv and sensitivity plots.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from kvtier.policy import DynamicHybridPolicy, FlatPolicy, StaticModalityPolicy
from kvtier.reconfig import ReconfigurationManager
from kvtier.simulator import run_session
from kvtier.kv_entry import Tier
from kvtier.tiers import load_tier_config
from kvtier.trace import generate_session

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

PROMOTE = [0.004, 0.006, 0.008, 0.010, 0.014, 0.020]
DEMOTE = [0.0005, 0.001, 0.002, 0.003, 0.004]
COST_MULT = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]


def _run(cfg, tiers, policy, mult):
    session = generate_session(**cfg["trace"])
    mgr = ReconfigurationManager(hot=tiers[Tier.HOT], cost_multiplier=mult)
    return run_session(session, policy, tiers, reconfig_manager=mgr,
                       ema_decay=cfg["trace"]["ema_decay"])


class AllWarmPolicy:
    """Reference: everything in WARM. Shows what the sweep's 'wins' amount to."""

    name = "all_warm"

    def initial_tier(self, entry):
        return Tier.WARM

    def update(self, entry):
        return entry.tier


def _pareto_mask(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """True for rows not dominated (<= on all cols, < on one) by any other row."""
    vals = df[cols].to_numpy()
    keep = []
    for i, v in enumerate(vals):
        dominated = ((vals <= v).all(axis=1) & (vals < v).any(axis=1)).any()
        keep.append(not dominated)
    return pd.Series(keep, index=df.index)


def main() -> None:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    tiers = load_tier_config(CONFIG_PATH)

    static = _run(cfg, tiers, StaticModalityPolicy(), 10.0)
    s_e, s_l = static.total_energy_pj, static.total_latency_ns
    s_q = static.mean_quality_loss

    refs = []
    for name, pol in (("flat", FlatPolicy()), ("static_modality", StaticModalityPolicy()),
                      ("all_warm", AllWarmPolicy())):
        r = _run(cfg, tiers, pol, 10.0)
        refs.append(dict(policy=name, energy_pj=r.total_energy_pj,
                         latency_ns=r.total_latency_ns,
                         mean_quality_loss=r.mean_quality_loss,
                         warm_attention_mass_pct=100 * r.mean_warm_attention_mass))
    refs = pd.DataFrame(refs)
    refs.to_csv(RESULTS_DIR / "sensitivity_references.csv", index=False)
    print("References (reconfig-free):")
    print(refs.to_string(index=False), "\n")

    rows = []
    for p, d, m in itertools.product(PROMOTE, DEMOTE, COST_MULT):
        if d >= p:
            continue
        r = _run(cfg, tiers, DynamicHybridPolicy(p, d), m)
        rows.append(dict(
            promote=p, demote=d, cost_mult=m,
            energy_vs_static_pct=100 * (r.total_energy_pj / s_e - 1),
            latency_vs_static_pct=100 * (r.total_latency_ns / s_l - 1),
            quality_loss=r.mean_quality_loss,
            quality_vs_static_pct=100 * (r.mean_quality_loss / s_q - 1),
            warm_attention_mass_pct=100 * r.mean_warm_attention_mass,
            n_promotions=r.total_promotions, n_demotions=r.total_demotions,
            reconfig_energy_pj=r.total_reconfig_energy_pj,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "sensitivity_sweep.csv", index=False)

    # "Wins" now require beating static on energy WITHOUT worse quality.
    win = df[(df.energy_vs_static_pct < 0) & (df.quality_vs_static_pct <= 0)]
    print(f"Configs beating static on energy AND quality: {len(win)} / {len(df)}")
    print(f"Configs beating static on energy only:        {(df.energy_vs_static_pct < 0).sum()} / {len(df)}")
    if len(win):
        print(win.sort_values("energy_vs_static_pct").head(8).to_string(index=False))
    for m, g in df.groupby("cost_mult"):
        w = g[(g.energy_vs_static_pct < 0) & (g.quality_vs_static_pct <= 0)]
        best = f"best {w.energy_vs_static_pct.min():+.1f}%" if len(w) else "none"
        print(f"cost_mult={m:>5}: {len(w):>2} dominating configs, {best}")

    # Plot 1: energy vs quality loss, one panel per reconfig cost.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, m in zip(axes, (1.0, 10.0, 20.0)):
        g = df[df.cost_mult == m]
        sc = ax.scatter(g.quality_loss, g.energy_vs_static_pct / 100 + 1, c=g.n_promotions + g.n_demotions,
                        cmap="viridis", s=18)
        for _, r in refs.iterrows():
            ax.scatter(r.mean_quality_loss, r.energy_pj / s_e, marker="*", s=140, edgecolor="k",
                       label=r.policy, zorder=5)
        ax.set_title(f"reconfig cost {m:g}x")
        ax.set_xlabel("mean quality loss")
        ax.set_yscale("log")
    axes[0].set_ylabel("energy / static_modality energy")
    axes[0].legend(fontsize=8)
    fig.colorbar(sc, ax=axes[-1], label="# tier moves")
    fig.suptitle("Energy-quality trade-off (each dot = one threshold pair)")
    fig.savefig(RESULTS_DIR / "sensitivity_pareto.png", dpi=150, bbox_inches="tight")

    # Plot 2: best dominating energy delta vs reconfig cost.
    fig, ax = plt.subplots(figsize=(6, 4))
    xs, ys = [], []
    for m, g in df.groupby("cost_mult"):
        w = g[(g.energy_vs_static_pct < 0) & (g.quality_vs_static_pct <= 0)]
        if len(w):
            xs.append(m); ys.append(w.energy_vs_static_pct.min())
    ax.plot(xs, ys, "o-")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("reconfig cost multiplier (x HOT access)")
    ax.set_ylabel("best energy vs static (%), quality no worse")
    ax.set_title("Dynamic advantage vs reconfiguration cost")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sensitivity_reconfig_cost.png", dpi=150)

    # Plot 3: threshold heatmaps at the default cost multiplier.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in ((axes[0], "energy_vs_static_pct", "Energy vs static (%)"),
                           (axes[1], "quality_vs_static_pct", "Quality loss vs static (%)")):
        h = df[df.cost_mult == 10.0].pivot(index="demote", columns="promote", values=col)
        im = ax.imshow(h.values, aspect="auto", origin="lower", cmap="RdBu_r")
        ax.set_xticks(range(len(h.columns)), [f"{c:g}" for c in h.columns])
        ax.set_yticks(range(len(h.index)), [f"{c:g}" for c in h.index])
        ax.set_xlabel("promote threshold"); ax.set_ylabel("demote threshold")
        ax.set_title(f"{title}, cost 10x")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sensitivity_thresholds.png", dpi=150)


if __name__ == "__main__":
    main()
