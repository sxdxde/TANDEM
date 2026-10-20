"""Layer/role-aware static placement vs static_modality vs dynamic (per-layer traces).

Leave-one-trace-out CV over {chart, scene, noise_label}. For each held-out
trace: build the (modality x layer) profile from the other two traces, pick
theta (threshold, x uniform attention) = lowest mean energy among thetas whose
mean quality on those two traces is <= static_modality's (fallback: lowest
quality), then evaluate on the held-out trace. An in-sample ``ceiling``
(profile from the held-out trace itself, same theta) bounds what the profile
could give. Dynamic's per-fold choice reuses results/m6_sweep.csv (same rule).
Both dense and sparse top-10% access. Groupings: (modality, layer) and (role, layer) with role in sink/prompt-text/generated/visual.
Writes results/layer_aware_results.csv and layer_aware_theta_curve.csv.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from kvtier.policy import LayerModalityPolicy
from kvtier.profile import GROUPINGS, average_profiles, group_importance, hot_groups, n_prefill_tokens
from kvtier.real_trace import load_layer_trace
from kvtier.reconfig import load_reconfig_config
from kvtier.simulator import SparseAccess, run_session
from kvtier.tiers import load_tier_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
TRACES = ["chart", "scene", "noise_label"]
THETAS = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 4.0]
GROUPING_NAMES = ["modality_layer", "role_layer"]
SPARSE = {"layer_dense": None, "layer_sparse10": SparseAccess(top_k_fraction=0.1)}


def _sess(name):
    return load_layer_trace(ROOT / "traces" / f"{name}_layers.npz")


def _job(args):
    grouping, mode, profile_traces, theta, eval_trace = args
    gfn = GROUPINGS[grouping]
    tiers = load_tier_config(CONFIG_PATH)
    reconfig = load_reconfig_config(CONFIG_PATH)
    profile = average_profiles([group_importance(_sess(t), gfn) for t in profile_traces])
    hot = hot_groups(profile, theta)
    sess = _sess(eval_trace)
    r = run_session(sess, LayerModalityPolicy(hot, gfn, n_prefill_tokens(sess)), tiers, reconfig_manager=reconfig,
                    ema_decay=0.9, sparse=SPARSE[mode])
    return dict(grouping=grouping, mode=mode, profile=",".join(profile_traces), theta=theta, trace=eval_trace,
                energy_pj=r.total_energy_pj, quality=r.mean_quality_loss,
                warm_mass_pct=100 * r.mean_warm_attention_mass, n_hot_groups=len(hot))


def _pct(df, sm, col):
    return df.apply(lambda r: 100 * (r[col] / sm[col][(r["mode"], r.trace)] - 1), axis=1)


def main() -> None:
    sweep = pd.read_csv(ROOT / "results" / "m6_sweep.csv")
    pool = ProcessPoolExecutor(max_workers=6)
    all_rows, all_curves = [], []

    # Baselines (grouping-independent): static / flat / dynamic(CV) per held-out trace.
    base = []
    for m in SPARSE:
        s = sweep[sweep["mode"] == m]
        st = s[s.policy == "static"].set_index("trace")
        dyn = s[s.policy == "dyn"].copy()
        dyn["e"] = dyn.apply(lambda r: 100 * (r.energy_pj / st.energy_pj[r.trace] - 1), axis=1)
        dyn["q"] = dyn.apply(lambda r: 100 * (r.quality / st.quality[r.trace] - 1), axis=1)
        for ho in TRACES:
            g = dyn[dyn.trace != ho].groupby(["promote_x", "demote_x"]).agg(
                e=("e", "mean"), q=("q", "mean")).reset_index()
            ok = g[g.q <= 0]
            c = (ok.sort_values("e") if len(ok) else g.sort_values("q")).iloc[0]
            r = dyn[(dyn.trace == ho) & (dyn.promote_x == c.promote_x) & (dyn.demote_x == c.demote_x)].iloc[0]
            fl = s[(s.policy == "flat") & (s.trace == ho)].iloc[0]
            for name, rr, mv in (("dynamic (CV)", r, r.promos + r.demos),
                                 ("static_modality", st.loc[ho], 0), ("flat", fl, 0)):
                base.append(dict(grouping="-", mode=m, trace=ho, policy=name, energy_pj=rr.energy_pj,
                                 quality=rr.quality, warm_mass_pct=rr.warm_mass_pct, moves=mv))

    for grouping in GROUPING_NAMES:
        p1 = [(grouping, m, tuple(t for t in TRACES if t != ho), th, ev)
              for m in SPARSE for ho in TRACES for th in THETAS for ev in TRACES if ev != ho]
        p1_rows = pd.DataFrame(list(pool.map(_job, p1, chunksize=2)))
        chosen = {}
        for m in SPARSE:
            st = sweep[(sweep["mode"] == m) & (sweep.policy == "static")].set_index("trace")
            for ho in TRACES:
                d = p1_rows[(p1_rows["mode"] == m) & (p1_rows.profile == ",".join(t for t in TRACES if t != ho))].copy()
                d["e"] = d.apply(lambda r: 100 * (r.energy_pj / st.energy_pj[r.trace] - 1), axis=1)
                d["q"] = d.apply(lambda r: 100 * (r.quality / st.quality[r.trace] - 1), axis=1)
                g = d.groupby("theta").agg(e=("e", "mean"), q=("q", "mean")).reset_index()
                ok = g[g.q <= 0]
                chosen[(m, ho)] = float((ok.sort_values("e") if len(ok) else g.sort_values("q")).iloc[0].theta)
        p2 = []
        for (m, ho), th in chosen.items():
            p2 += [(grouping, m, tuple(t for t in TRACES if t != ho), th, ho),
                   (grouping, m, (ho,), th, ho)]
        for a, r in zip(p2, pool.map(_job, p2)):
            kind = f"{grouping} (CV)" if len(a[2]) == 2 else f"{grouping} (ceiling, in-sample)"
            all_rows.append({**r, "policy": kind, "moves": 0})
        p3 = [(grouping, m, tuple(t for t in TRACES if t != ho), th, ho)
              for m in SPARSE for ho in TRACES for th in THETAS]
        all_curves.append(pd.DataFrame(list(pool.map(_job, p3, chunksize=2))))
        print(f"{grouping}: chosen theta {chosen}", flush=True)

    allr = pd.concat([pd.DataFrame(base), pd.DataFrame(all_rows)], ignore_index=True)
    sm = allr[allr.policy == "static_modality"].set_index(["mode", "trace"])
    allr["energy_vs_static_pct"] = _pct(allr, sm, "energy_pj")
    allr["quality_vs_static_pct"] = _pct(allr, sm, "quality")
    allr.to_csv(ROOT / "results" / "layer_aware_results.csv", index=False)
    curve = pd.concat(all_curves, ignore_index=True)
    curve["energy_vs_static_pct"] = _pct(curve, sm, "energy_pj")
    curve["quality_vs_static_pct"] = _pct(curve, sm, "quality")
    curve.to_csv(ROOT / "results" / "layer_aware_theta_curve.csv", index=False)

    pd.set_option("display.width", 250)
    print("\nTheta curve, mean over held-out folds:")
    print(curve.groupby(["grouping", "mode", "theta"])[["n_hot_groups", "energy_vs_static_pct", "quality_vs_static_pct"]]
          .mean().round(2).to_string())
    print("\nMean over folds (held-out):")
    print(allr[allr.policy != "flat"].groupby(["mode", "policy"])[["energy_vs_static_pct", "quality_vs_static_pct", "moves"]]
          .mean().round(2).to_string())


if __name__ == "__main__":
    main()
