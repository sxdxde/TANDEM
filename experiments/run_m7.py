"""Milestone 7: simulate the whole image set (per-layer entries) and store raw per-run metrics.

Usage: python experiments/run_m7.py <tag> [--extras]
Writes results/m7_<tag>_sims.csv, one row per (trace, policy, access mode). Reconfiguration is
simulated at a 1x multiplier and stored separately (reconfig_*_1x): policy decisions depend only on
the importance EMA, never on cost, so energy/latency at multiplier m is exactly
access + m * reconfig_1x (checked in tests/test_m7_linearity.py). Selection / CV / plots live in
analyze_m7.py so they can be iterated without re-simulating.

Grids: full threshold grid for dense and sparse top-10%. With --extras: reduced grid for top-k
fractions 5/20/50% and for min-dwell variants (0.3 of the cost, extra analyses).
"""

from __future__ import annotations

import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from kvtier.policy import DynamicHybridPolicy, FlatPolicy, StaticModalityPolicy
from kvtier.real_trace import load_layer_trace
from kvtier.reconfig import ReconfigurationManager
from kvtier.simulator import SparseAccess, run_session
from kvtier.tiers import load_tier_config
from kvtier.kv_entry import Tier

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
# 16/32 and 0.8/1.6 were added after the first 256M pass: every CV fold picked 8/0.4, the old grid corner.
PROMOTE_X = [0.2, 0.4, 0.7, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
DEMOTE_X = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
FULL = [(p, d) for p, d in itertools.product(PROMOTE_X, DEMOTE_X) if d < p]
REDUCED = [(p, d) for p, d in itertools.product([1.0, 2.0, 4.0, 8.0], [0.1, 0.4]) if d < p]
DWELLS = [5, 20, 40]
K_EXTRA = [0.05, 0.2, 0.5]


def _job(args):
    tag, trace, spec, k = args
    cfg = yaml.safe_load(open(CONFIG_PATH))
    tiers = load_tier_config(CONFIG_PATH)
    reconfig = ReconfigurationManager(hot=tiers[Tier.HOT], cost_multiplier=1.0)
    session = load_layer_trace(ROOT / "traces" / tag / f"{trace}_layers.npz")
    n_prefill = len(session.entries) - sum(len(s.new_positions) for s in session.steps)
    if spec[0] == "flat":
        pol = FlatPolicy()
    elif spec[0] == "static":
        pol = StaticModalityPolicy()
    else:
        _, px, dx, dwell = spec
        u = 1.0 / n_prefill
        pol = DynamicHybridPolicy(px * u, dx * u, warmup_steps=cfg["policy"]["dynamic_hybrid"]["warmup_steps"],
                                  min_dwell_steps=dwell)
    sparse = SparseAccess(top_k_fraction=k) if k else None
    r = run_session(session, pol, tiers, reconfig_manager=reconfig, ema_decay=cfg["trace"]["ema_decay"],
                    sparse=sparse)
    return dict(tag=tag, trace=trace, policy=spec[0], promote_x=spec[1] if len(spec) > 1 else None,
                demote_x=spec[2] if len(spec) > 2 else None, dwell=spec[3] if len(spec) > 3 else None,
                k=k or 0.0,
                access_energy_pj=r.total_energy_pj - r.total_reconfig_energy_pj,
                access_latency_ns=r.total_latency_ns - r.total_reconfig_latency_ns,
                reconfig_energy_1x=r.total_reconfig_energy_pj, reconfig_latency_1x=r.total_reconfig_latency_ns,
                quality=r.mean_quality_loss, warm_mass_pct=100 * r.mean_warm_attention_mass,
                skipped_pct=100 * r.mean_skipped_attention_mass,
                promos=r.total_promotions, demos=r.total_demotions)


def build_jobs(tag: str, traces: list[str], extras: bool) -> list:
    jobs = []
    for tr in traces:
        for k in ([0.0, 0.1] + (K_EXTRA if extras else [])):
            grid = FULL if k in (0.0, 0.1) else REDUCED
            jobs += [(tag, tr, ("flat",), k), (tag, tr, ("static",), k)]
            jobs += [(tag, tr, ("dyn", p, d, 0), k) for p, d in grid]
        if extras:
            jobs += [(tag, tr, ("dyn", p, d, w), 0.0) for p, d in REDUCED for w in DWELLS]
    return jobs


def _key(r) -> tuple:
    f = lambda x: None if pd.isna(x) else float(x)  # noqa: E731
    return (r["trace"], r["policy"], f(r["promote_x"]), f(r["demote_x"]), f(r["dwell"]), float(r["k"]))


def _job_key(j) -> tuple:
    _, trace, spec, k = j
    g = lambda i: float(spec[i]) if len(spec) > i else None  # noqa: E731
    return (trace, spec[0], g(1), g(2), g(3), float(k or 0.0))


def main() -> None:
    """Resumable: finished runs are appended to <out>.partial.csv as they complete, and a restart
    skips them. The final CSV is written (and the partial removed) only when every run is done."""
    tag = sys.argv[1]
    extras = "--extras" in sys.argv
    traces = sorted(p.name[: -len("_layers.npz")] for p in (ROOT / "traces" / tag).glob("*_layers.npz"))
    jobs = build_jobs(tag, traces, extras)
    out = ROOT / "results" / f"m7_{tag}_sims.csv"
    part = out.with_suffix(".partial.csv")
    done = pd.read_csv(part) if part.exists() else pd.DataFrame()
    have = {_key(r) for _, r in done.iterrows()}
    todo = [j for j in jobs if _job_key(j) not in have]
    print(f"{tag}: {len(traces)} traces, {len(jobs)} simulations, {len(have)} already done, "
          f"{len(todo)} to run", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=7) as ex, open(part, "a") as f:
            for i, row in enumerate(ex.map(_job, todo, chunksize=1)):
                pd.DataFrame([row]).to_csv(f, header=f.tell() == 0 and not len(done), index=False)
                f.flush()
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(todo)}", flush=True)
    final = pd.read_csv(part).drop_duplicates(subset=["trace", "policy", "promote_x", "demote_x", "dwell", "k"])
    final.to_csv(out, index=False)
    part.unlink()
    print("wrote", out, len(final), "rows")


if __name__ == "__main__":
    main()
