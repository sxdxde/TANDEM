"""Milestone 6: per-layer entries x sparse-attention cost model.

Four modes: {layer-averaged, per-layer entries} x {dense, sparse top-10% access}.
Same protocol as run_real_trace.py (fixed in advance): thresholds are
multiples of uniform attention; pick the pair on TRAIN traces with the lowest
mean energy whose mean quality loss is <= static_modality's, then evaluate on
the held-out TEST trace. Writes results/m6_sweep.csv and results/m6_results.csv.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from kvtier.policy import DynamicHybridPolicy, FlatPolicy, StaticModalityPolicy
from kvtier.real_trace import load_layer_trace, load_trace
from kvtier.reconfig import load_reconfig_config
from kvtier.simulator import SparseAccess, run_session
from kvtier.tiers import load_tier_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
TRAIN, TEST = ["chart", "scene"], ["noise_label"]
PROMOTE_X = [0.2, 0.4, 0.7, 1.0, 2.0, 4.0, 8.0]
DEMOTE_X = [0.02, 0.05, 0.1, 0.2, 0.4]
SPARSE = SparseAccess(top_k_fraction=0.1)
MODES = {
    "avg_dense": ("avg", None),
    "avg_sparse10": ("avg", SPARSE),
    "layer_dense": ("layer", None),
    "layer_sparse10": ("layer", SPARSE),
}


def _load(kind, name):
    return load_trace(ROOT / "traces" / f"{name}.npz") if kind == "avg" \
        else load_layer_trace(ROOT / "traces" / f"{name}_layers.npz")


def _job(args):
    mode, trace, policy_spec = args
    kind, sparse = MODES[mode]
    cfg = yaml.safe_load(open(CONFIG_PATH))
    tiers = load_tier_config(CONFIG_PATH)
    reconfig = load_reconfig_config(CONFIG_PATH)
    session = _load(kind, trace)
    n_prefill = len(session.entries) - sum(len(s.new_positions) for s in session.steps)
    if policy_spec[0] == "flat":
        pol = FlatPolicy()
    elif policy_spec[0] == "static":
        pol = StaticModalityPolicy()
    else:
        _, px, dx = policy_spec
        u = 1.0 / n_prefill  # uniform attention per entry (entries already include layers)
        damp = {k: v for k, v in cfg["policy"]["dynamic_hybrid"].items()
                if k in ("warmup_steps", "min_dwell_steps")}
        pol = DynamicHybridPolicy(px * u, dx * u, **damp)
    r = run_session(session, pol, tiers, reconfig_manager=reconfig,
                    ema_decay=cfg["trace"]["ema_decay"], sparse=sparse)
    return dict(mode=mode, trace=trace, policy=policy_spec[0],
                promote_x=policy_spec[1] if len(policy_spec) > 1 else None,
                demote_x=policy_spec[2] if len(policy_spec) > 2 else None,
                energy_pj=r.total_energy_pj, latency_ns=r.total_latency_ns,
                quality=r.mean_quality_loss, warm_mass_pct=100 * r.mean_warm_attention_mass,
                skipped_pct=100 * r.mean_skipped_attention_mass,
                promos=r.total_promotions, demos=r.total_demotions)


def main() -> None:
    traces = TRAIN + TEST
    jobs = []
    for mode in MODES:
        for tr in traces:
            jobs += [(mode, tr, ("flat",)), (mode, tr, ("static",))]
            jobs += [(mode, tr, ("dyn", px, dx))
                     for px, dx in itertools.product(PROMOTE_X, DEMOTE_X) if dx < px]
    # slowest (layered) first for better pool utilisation
    jobs.sort(key=lambda j: MODES[j[0]][0] != "layer")
    print(f"{len(jobs)} simulations", flush=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(_job, jobs, chunksize=4))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "m6_sweep.csv", index=False)

    out = []
    for mode in MODES:
        d = df[df["mode"] == mode]
        st = d[d.policy == "static"].set_index("trace")
        fl = d[d.policy == "flat"].set_index("trace")
        dyn = d[d.policy == "dyn"].copy()
        dyn["e_vs_s"] = dyn.apply(lambda r: 100 * (r.energy_pj / st.energy_pj[r.trace] - 1), axis=1)
        dyn["q_vs_s"] = dyn.apply(lambda r: 100 * (r.quality / st.quality[r.trace] - 1), axis=1)
        tr = dyn[dyn.trace.isin(TRAIN)].groupby(["promote_x", "demote_x"]).agg(
            e=("e_vs_s", "mean"), q=("q_vs_s", "mean")).reset_index()
        ok = tr[tr.q <= 0]
        chosen = (ok.sort_values("e").iloc[0] if len(ok) else tr.sort_values("q").iloc[0])
        sel = dyn[(dyn.promote_x == chosen.promote_x) & (dyn.demote_x == chosen.demote_x)]
        print(f"\n=== {mode}: {len(ok)}/{len(tr)} train pairs have quality <= static; "
              f"chosen promote {chosen.promote_x}x demote {chosen.demote_x}x", flush=True)
        for trace in traces:
            s, f = st.loc[trace], fl.loc[trace]
            dd = sel[sel.trace == trace].iloc[0]
            out.append(dict(mode=mode, trace=trace, split="TEST" if trace in TEST else "train",
                            flat_energy=f.energy_pj, static_energy=s.energy_pj, dyn_energy=dd.energy_pj,
                            static_quality=s.quality, dyn_quality=dd.quality,
                            skipped_pct=s.skipped_pct, static_warm_mass_pct=s.warm_mass_pct,
                            dyn_warm_mass_pct=dd.warm_mass_pct,
                            flat_over_static_energy=f.energy_pj / s.energy_pj,
                            dyn_energy_vs_static_pct=dd.e_vs_s, dyn_quality_vs_static_pct=dd.q_vs_s,
                            moves=dd.promos + dd.demos, chosen_promote_x=chosen.promote_x,
                            chosen_demote_x=chosen.demote_x))
    res = pd.DataFrame(out)
    res.to_csv(ROOT / "results" / "m6_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[["mode", "trace", "split", "flat_over_static_energy", "skipped_pct",
               "static_quality", "dyn_quality", "dyn_energy_vs_static_pct",
               "dyn_quality_vs_static_pct", "moves"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
