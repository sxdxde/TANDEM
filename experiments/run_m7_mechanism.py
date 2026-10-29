"""Milestone 7: where does the dynamic policy's saving come from? (instrumented, not inferred)

Usage: python experiments/run_m7_mechanism.py <tag> <promote_x> <demote_x>
On a stratified subset of traces, run static_modality, full dynamic, demote-only (promote threshold
unreachable) and promote-only (demote threshold ~0), all at the given thresholds (dense reads, 10x
cost), and count moves by entry class: prompt text, visual, generated. Writes results/m7_mechanism_<tag>.csv.
"""

from __future__ import annotations

import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from kvtier.kv_entry import Modality, Tier
from kvtier.policy import DynamicHybridPolicy, StaticModalityPolicy
from kvtier.real_trace import load_layer_trace
from kvtier.reconfig import ReconfigurationManager
from kvtier.simulator import run_session
from kvtier.tiers import load_tier_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
SUBSET = ["natural_00", "natural_05", "natural_10", "natural_15", "natural_20",
          "chart_00", "chart_03", "chart_06", "doc_00", "doc_02", "doc_05"]
VARIANTS = ["static", "full", "demote_only", "promote_only"]


def _job(args):
    tag, trace, variant, px, dx = args
    cfg = yaml.safe_load(open(CONFIG_PATH))
    tiers = load_tier_config(CONFIG_PATH)
    rm = ReconfigurationManager(hot=tiers[Tier.HOT], cost_multiplier=10.0)
    s = load_layer_trace(ROOT / "traces" / tag / f"{trace}_layers.npz")
    n_prefill = len(s.entries) - sum(len(st.new_positions) for st in s.steps)
    u = 1.0 / n_prefill
    kw = dict(warmup_steps=cfg["policy"]["dynamic_hybrid"]["warmup_steps"])
    if variant == "static":
        pol = StaticModalityPolicy()
    elif variant == "full":
        pol = DynamicHybridPolicy(px * u, dx * u, **kw)
    elif variant == "demote_only":
        pol = DynamicHybridPolicy(1e9, dx * u, **kw)
    else:
        pol = DynamicHybridPolicy(px * u, 1e-12, **kw)
    r = run_session(s, pol, tiers, reconfig_manager=rm, ema_decay=cfg["trace"]["ema_decay"])
    cls = Counter()
    for i, e in enumerate(s.entries):
        c = "generated" if i >= n_prefill else ("visual" if e.modality == Modality.VISUAL else "prompt_text")
        cls[c] += e.tier_changes
    return dict(tag=tag, trace=trace, variant=variant, energy=r.total_energy_pj, quality=r.mean_quality_loss,
                warm_mass_pct=100 * r.mean_warm_attention_mass, promos=r.total_promotions, demos=r.total_demotions,
                moves_prompt_text=cls["prompt_text"], moves_visual=cls["visual"], moves_generated=cls["generated"],
                reconfig_share_pct=100 * r.total_reconfig_energy_pj / r.total_energy_pj)


def main() -> None:
    tag, px, dx = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    traces = [t for t in SUBSET if (ROOT / "traces" / tag / f"{t}_layers.npz").exists()]
    jobs = [(tag, t, v, px, dx) for t in traces for v in VARIANTS]
    with ProcessPoolExecutor(max_workers=7) as ex:
        df = pd.DataFrame(list(ex.map(_job, jobs, chunksize=1)))
    st = df[df.variant == "static"].set_index("trace")
    df["e_vs_static_pct"] = 100 * (df.energy.values / st.energy[df.trace].values - 1)
    df["q_vs_static_pct"] = 100 * (df.quality.values / st.quality[df.trace].values - 1)
    df.to_csv(ROOT / "results" / f"m7_mechanism_{tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.groupby("variant")[["e_vs_static_pct", "q_vs_static_pct", "promos", "demos", "moves_prompt_text",
                                  "moves_visual", "moves_generated", "reconfig_share_pct"]].mean().round(2)
          .loc[VARIANTS].to_string())


if __name__ == "__main__":
    main()
