"""Milestone 5: run flat / static / dynamic policies on real VLM attention traces.

Protocol (fixed before looking at results):
  1. Thresholds are expressed as multiples of uniform attention (1/n_prefill)
     because the synthetic-trace thresholds don't transfer to ~900-entry traces.
  2. Train on TRAIN traces: pick the (promote, demote) pair with the lowest mean
     energy whose mean quality loss is <= static_modality's (i.e. no worse).
  3. Evaluate that pair on the held-out TEST trace.
Also reports the synthetic-default thresholds on real traces, for contrast.
Writes results/real_trace_results.csv and results/real_trace_sweep.csv.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
import yaml

from kvtier.policy import DynamicHybridPolicy, FlatPolicy, StaticModalityPolicy
from kvtier.real_trace import load_trace
from kvtier.reconfig import load_reconfig_config
from kvtier.simulator import run_session
from kvtier.tiers import load_tier_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
TRAIN = ["chart", "scene"]
TEST = ["noise_label"]
PROMOTE_X = [0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 4.0]   # x uniform attention
DEMOTE_X = [0.02, 0.05, 0.1, 0.2, 0.4]


def main() -> None:
    cfg = yaml.safe_load(open(CONFIG_PATH))
    tiers = load_tier_config(CONFIG_PATH)
    reconfig = load_reconfig_config(CONFIG_PATH)
    damp = {k: v for k, v in cfg["policy"]["dynamic_hybrid"].items()
            if k in ("warmup_steps", "min_dwell_steps")}
    sessions = {n: lambda n=n: load_trace(ROOT / "traces" / f"{n}.npz") for n in TRAIN + TEST}

    def run(name, policy):
        s = sessions[name]()
        return run_session(s, policy, tiers, reconfig_manager=reconfig,
                           ema_decay=cfg["trace"]["ema_decay"])

    uniform = {n: 1.0 / (len(sessions[n]().entries) - len(sessions[n]().steps)) for n in sessions}
    static = {n: run(n, StaticModalityPolicy()) for n in sessions}
    flat = {n: run(n, FlatPolicy()) for n in sessions}

    rows = []
    for px, dx in itertools.product(PROMOTE_X, DEMOTE_X):
        if dx >= px:
            continue
        for n in sessions:
            u = uniform[n]
            r = run(n, DynamicHybridPolicy(px * u, dx * u, **damp))
            rows.append(dict(trace=n, promote_x=px, demote_x=dx,
                             energy_vs_static_pct=100 * (r.total_energy_pj / static[n].total_energy_pj - 1),
                             latency_vs_static_pct=100 * (r.total_latency_ns / static[n].total_latency_ns - 1),
                             quality=r.mean_quality_loss,
                             quality_vs_static_pct=100 * (r.mean_quality_loss / static[n].mean_quality_loss - 1),
                             warm_mass_pct=100 * r.mean_warm_attention_mass,
                             moves=r.total_promotions + r.total_demotions))
    sweep = pd.DataFrame(rows)
    sweep.to_csv(ROOT / "results" / "real_trace_sweep.csv", index=False)

    tr = sweep[sweep.trace.isin(TRAIN)].groupby(["promote_x", "demote_x"]).agg(
        e=("energy_vs_static_pct", "mean"), q=("quality_vs_static_pct", "mean"),
        moves=("moves", "mean")).reset_index()
    ok = tr[tr.q <= 0]
    print(f"Train ({TRAIN}): {len(ok)}/{len(tr)} threshold pairs have quality no worse than static")
    chosen = ok.sort_values("e").iloc[0] if len(ok) else None
    if chosen is None:
        print("No pair matches static quality; falling back to the lowest-quality-loss pair.")
        chosen = tr.sort_values("q").iloc[0]
    print("Chosen on train:\n", chosen.to_string())

    out = []
    for n in sessions:
        u = uniform[n]
        pols = {"flat": flat[n], "static_modality": static[n]}
        pols["dynamic (chosen)"] = run(n, DynamicHybridPolicy(chosen.promote_x * u, chosen.demote_x * u, **damp))
        pols["dynamic (synthetic defaults)"] = run(
            n, DynamicHybridPolicy(**cfg["policy"]["dynamic_hybrid"]))
        for k, r in pols.items():
            out.append(dict(trace=n, split="train" if n in TRAIN else "TEST", policy=k,
                            energy_pj=r.total_energy_pj, latency_ns=r.total_latency_ns,
                            quality_loss=r.mean_quality_loss,
                            warm_mass_pct=100 * r.mean_warm_attention_mass,
                            promos=r.total_promotions, demos=r.total_demotions,
                            energy_vs_static_pct=100 * (r.total_energy_pj / static[n].total_energy_pj - 1),
                            quality_vs_static_pct=(100 * (r.mean_quality_loss / static[n].mean_quality_loss - 1)
                                                   if static[n].mean_quality_loss else float("nan"))))
    res = pd.DataFrame(out)
    res.to_csv(ROOT / "results" / "real_trace_results.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
