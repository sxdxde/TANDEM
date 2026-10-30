"""Milestone 9: computation placement under explicit resources, and placement across access regimes.

Usage: python experiments/run_m9.py [e8] [e9] [e9b]      (default: all three)

e8  compute-placement scheduler x CIM capacity   (dense reads; 42 traces)
      policies flat / static / dynamic (8x/0.4x, the 256M leave-one-out choice) x scheduler tier / host /
      adaptive x cim_units {2,4,8,16,32,64}. The "host" scheduler never uses the CIM units, so it is run once.
e9  access regimes: reusable (dense), streaming, random 3/10/30%       (42 traces)
      flat / static / dynamic / dynamic+reuse gate, at a bottom-up reconfiguration cost (scale 1) and a
      10x-stressed one (scale 10). The gate's break-even read count is derived from each cost model.
e9b reconfiguration-cost sweep in the explicit model (dense, 14 traces: every third)
e9c POST-HOC (added after e9 showed the gate leaves a residual loss under random access): the reuse gate with a
    minimum age before the read-frequency estimate is trusted, min_age in {4, 8, 16} ("dynr_a<age>"). Not part of the
    pre-specified protocol and not held out; reported as a diagnosis, not as a headline result.

Writes results/m9_system.csv (resumable through results/m9_system.partial.csv).
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from kvtier.kv_entry import Tier
from kvtier.policy import DynamicHybridPolicy, FlatPolicy, ReuseAwareDynamicPolicy, StaticModalityPolicy
from kvtier.real_trace import load_layer_trace
from kvtier.system import AccessRegime, SystemConfig, breakeven_accesses, run_system_session
from kvtier.tiers import load_tier_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
PX, DX = 8.0, 0.4
HORIZON = 32.0
CIM_SWEEP = [2, 4, 8, 16, 32, 64]
REGIMES = [("dense", 1.0), ("stream", 0.0), ("random", 0.03), ("random", 0.1), ("random", 0.3)]
KEYS = ["exp", "trace", "regime", "frac", "policy", "scheduler", "cim_units", "scale"]


def jobs_for(exps: list[str], traces: list[str]) -> list[dict]:
    out = []
    for i, tr in enumerate(traces):
        base = dict(trace=tr, seed=i, regime="dense", frac=1.0, scheduler="tier", cim_units=16, scale=1.0)
        if "e8" in exps:
            out.append({**base, "exp": "e8", "policy": "flat"})
            for pol in ("static", "dyn"):
                out.append({**base, "exp": "e8", "policy": pol, "scheduler": "host"})
                for sched in ("tier", "adaptive"):
                    out += [{**base, "exp": "e8", "policy": pol, "scheduler": sched, "cim_units": c} for c in CIM_SWEEP]
        if "e9" in exps:
            for regime, frac in REGIMES:
                rb = {**base, "exp": "e9", "regime": regime, "frac": frac}
                out += [{**rb, "policy": "flat"}, {**rb, "policy": "static"}]
                for scale in (1.0, 10.0):
                    out += [{**rb, "policy": p, "scale": scale} for p in ("dyn", "dynr")]
        if "e9c" in exps:
            for regime, frac in REGIMES:
                if regime == "stream":
                    continue
                rb = {**base, "exp": "e9c", "regime": regime, "frac": frac}
                ages = (4, 8, 16) if regime == "random" else (8, 16)
                out += [{**rb, "policy": f"dynr_a{a}", "scale": sc} for a in ages for sc in (1.0, 10.0)]
        if "e9b" in exps and i % 3 == 0:
            out += [{**base, "exp": "e9b", "policy": "dyn", "scale": s} for s in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)]
    return out


def key(j: dict) -> tuple:
    return tuple(j[k] if k not in ("frac", "scale") else float(j[k]) for k in KEYS)


def run_job(j: dict) -> dict:
    cfg_yaml = yaml.safe_load(open(CONFIG_PATH))
    tiers = load_tier_config(CONFIG_PATH)
    session = load_layer_trace(ROOT / "traces" / "s256" / f"{j['trace']}_layers.npz")
    n_prefill = len(session.entries) - sum(len(s.new_positions) for s in session.steps)
    u = 1.0 / n_prefill
    warm = cfg_yaml["policy"]["dynamic_hybrid"]["warmup_steps"]
    cfg = replace(SystemConfig(), scheduler=j["scheduler"], cim_units=j["cim_units"], reconfig_scale=j["scale"])
    if j["policy"] == "flat":
        pol = FlatPolicy()
    elif j["policy"] == "static":
        pol = StaticModalityPolicy()
    elif j["policy"] == "dyn":
        pol = DynamicHybridPolicy(PX * u, DX * u, warmup_steps=warm)
    else:
        n_min = breakeven_accesses(cfg, tiers, session.entries[0])
        age = int(j["policy"].split("_a")[1]) if "_a" in j["policy"] else 0
        pol = ReuseAwareDynamicPolicy(PX * u, DX * u, warmup_steps=warm, min_expected_accesses=n_min,
                                      horizon_steps=HORIZON, min_age=age)
    regime = AccessRegime(j["regime"], random_fraction=j["frac"] if j["regime"] == "random" else 0.1, seed=j["seed"])
    r = run_system_session(session, pol, tiers, cfg, regime, ema_decay=cfg_yaml["trace"]["ema_decay"])
    return {**{k: j[k] for k in KEYS}, "energy_pj": r.energy_pj, "energy_access_pj": r.energy_access_pj,
            "energy_reconfig_pj": r.energy_reconfig_pj, "latency_ns": r.latency_ns,
            "latency_reconfig_ns": r.latency_reconfig_ns, "bus_bytes": r.bus_bytes,
            "bus_bytes_reconfig": r.bus_bytes_reconfig, "quality": r.quality, "warm_mass": r.warm_mass,
            "skipped_mass": r.skipped_mass, "moves": r.promotions + r.demotions, "promos": r.promotions,
            "reads": r.reads, "packets": r.packets, "warm_packets": r.warm_packets,
            "spilled_packets": r.spilled_packets, "util_bus": r.util_bus, "util_host": r.util_host,
            "util_cim": r.util_cim, "wait_bus_ns": r.wait_ns_bus, "wait_cim_ns": r.wait_ns_cim}


def main() -> None:
    exps = [a for a in sys.argv[1:] if a in ("e8", "e9", "e9b", "e9c")] or ["e8", "e9", "e9b"]
    traces = sorted(p.name[: -len("_layers.npz")] for p in (ROOT / "traces" / "s256").glob("*_layers.npz"))
    jobs = jobs_for(exps, traces)
    out = ROOT / "results" / "m9_system.csv"
    part = out.with_suffix(".partial.csv")
    existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
    partial = pd.read_csv(part) if part.exists() else pd.DataFrame()
    done = pd.concat([existing, partial], ignore_index=True)
    have = {tuple(float(r[k]) if k in ("frac", "scale") else r[k] for k in KEYS) for _, r in done.iterrows()}
    todo = [j for j in jobs if key(j) not in have]
    print(f"{len(jobs)} simulations, {len(have)} already done, {len(todo)} to run", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=7) as ex, open(part, "a") as f:
            for i, row in enumerate(ex.map(run_job, todo, chunksize=1)):
                pd.DataFrame([row]).to_csv(f, header=f.tell() == 0, index=False)
                f.flush()
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(todo)}", flush=True)
    new = pd.read_csv(part) if part.exists() else pd.DataFrame()
    final = pd.concat([existing, new], ignore_index=True).drop_duplicates(subset=KEYS)
    final.to_csv(out, index=False)
    if part.exists():
        part.unlink()
    print("wrote", out, len(final), "rows")


if __name__ == "__main__":
    main()
