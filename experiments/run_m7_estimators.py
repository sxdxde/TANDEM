"""Milestone 8a: does the per-layer dynamic win survive a non-oracle importance signal?

Usage: python experiments/run_m7_estimators.py <tag>
Replays the dense per-layer dynamic policy on every trace with each estimator in kvtier.estimators
(dense reads, 10 threshold configs). Static / oracle numbers are reused from results/m7_<tag>_sims.csv.
Writes results/m8_<tag>_estimators_sims.csv (resumable via a .partial.csv, like run_m7.py).
"""

from __future__ import annotations

import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from kvtier import estimators
from kvtier.kv_entry import Tier
from kvtier.policy import DynamicHybridPolicy
from kvtier.real_trace import load_layer_trace
from kvtier.reconfig import ReconfigurationManager
from kvtier.simulator import run_session
from kvtier.tiers import load_tier_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
VARIANTS = ["noise0.25", "noise0.5", "noise1", "sub2", "sub4", "sub8", "delay1", "delay4",
            "token_shared", "shared_sub4", "noise0.5_sub4"]
GRID = [(p, d) for p in (1.0, 2.0, 4.0, 8.0) for d in (0.1, 0.4) if d < p] + [(4.0, 0.8), (16.0, 0.4)]


def _job(args):
    tag, trace, variant, px, dx = args
    cfg = yaml.safe_load(open(CONFIG_PATH))
    tiers = load_tier_config(CONFIG_PATH)
    reconfig = ReconfigurationManager(hot=tiers[Tier.HOT], cost_multiplier=1.0)
    session = load_layer_trace(ROOT / "traces" / tag / f"{trace}_layers.npz")
    n_prefill = len(session.entries) - sum(len(s.new_positions) for s in session.steps)
    n_layers = len(session.steps[0].new_positions)
    u = 1.0 / n_prefill
    pol = DynamicHybridPolicy(px * u, dx * u, warmup_steps=cfg["policy"]["dynamic_hybrid"]["warmup_steps"])
    est = estimators.make(variant, n_layers, seed=zlib.crc32(f"{trace}/{variant}".encode()))
    r = run_session(session, pol, tiers, reconfig_manager=reconfig, ema_decay=cfg["trace"]["ema_decay"],
                    estimator=est)
    return dict(trace=trace, variant=variant, promote_x=px, demote_x=dx,
                access_energy_pj=r.total_energy_pj - r.total_reconfig_energy_pj,
                access_latency_ns=r.total_latency_ns - r.total_reconfig_latency_ns,
                reconfig_energy_1x=r.total_reconfig_energy_pj, reconfig_latency_1x=r.total_reconfig_latency_ns,
                quality=r.mean_quality_loss, promos=r.total_promotions, demos=r.total_demotions)


def main() -> None:
    tag = sys.argv[1]
    traces = sorted(p.name[: -len("_layers.npz")] for p in (ROOT / "traces" / tag).glob("*_layers.npz"))
    jobs = [(tag, t, v, p, d) for t in traces for v in VARIANTS for p, d in GRID]
    out = ROOT / "results" / f"m8_{tag}_estimators_sims.csv"
    part = out.with_suffix(".partial.csv")
    done = pd.read_csv(part) if part.exists() else pd.DataFrame()
    have = set(zip(done.trace, done.variant, done.promote_x, done.demote_x)) if len(done) else set()
    todo = [j for j in jobs if (j[1], j[2], j[3], j[4]) not in have]
    print(f"{tag}: {len(jobs)} simulations, {len(have)} done, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=7) as ex, open(part, "a") as f:
        for i, row in enumerate(ex.map(_job, todo, chunksize=1)):
            pd.DataFrame([row]).to_csv(f, header=f.tell() == 0, index=False)
            f.flush()
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
    pd.read_csv(part).drop_duplicates(["trace", "variant", "promote_x", "demote_x"]).to_csv(out, index=False)
    part.unlink()
    print("wrote", out)


if __name__ == "__main__":
    main()
