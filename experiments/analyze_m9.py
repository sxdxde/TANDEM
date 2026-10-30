"""Milestone 9 analysis: compute placement under contention (E8) and access regimes (E9, E9b).

Everything is a per-trace paired ratio against a named baseline, then averaged over traces, with
paired-bootstrap 95% CIs (same convention as analyze_m7.py). Usage: python experiments/analyze_m9.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_m7 import RES, boot_ci

K = ["trace"]


def pct(a: pd.Series, b: pd.Series) -> pd.Series:
    return 100.0 * (a.values - b.values) / b.values


def ci_row(x: np.ndarray) -> tuple[float, float, float]:
    lo, hi = boot_ci(x)
    return float(np.mean(x)), float(lo), float(hi)


def e8_table(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d.exp == "e8"].copy()
    flat = d[d.policy == "flat"].set_index("trace")
    host = d[d.scheduler == "host"]
    parts = [d[d.scheduler != "host"]] + [host.assign(cim_units=c) for c in sorted(d.cim_units.unique())]
    d = pd.concat([p for p in parts if len(p)])
    d = d[d.policy != "flat"].copy()
    st = d[d.policy == "static"].set_index(["trace", "scheduler", "cim_units"])
    rows = []
    for (pol, sched, cim), g in d.groupby(["policy", "scheduler", "cim_units"]):
        g = g.set_index("trace")
        f = flat.loc[g.index]
        s = st.loc[[(t, sched, cim) for t in g.index]]
        r = dict(policy=pol, scheduler=sched, cim_units=cim,
                 energy_vs_flat=float(np.mean(g.energy_pj.values / f.energy_pj.values)),
                 latency_vs_flat=float(np.mean(g.latency_ns.values / f.latency_ns.values)),
                 bus_vs_flat=float(np.mean(g.bus_bytes.values / f.bus_bytes.values)),
                 latency_us=float(g.latency_ns.mean() / 1e3), energy_uj=float(g.energy_pj.mean() / 1e6),
                 spill=float(g.spilled_packets.sum() / max(g.warm_packets.sum(), 1)),
                 util_cim=float(g.util_cim.mean()), util_bus=float(g.util_bus.mean()),
                 wait_cim_us=float(g.wait_cim_ns.mean() / 1e3), moves=float(g.moves.mean()))
        r["latency_vs_flat_sd"] = float(np.std(g.latency_ns.values / f.latency_ns.values))
        r["energy_vs_flat_sd"] = float(np.std(g.energy_pj.values / f.energy_pj.values))
        if pol == "dyn":
            for col, nm in (("energy_pj", "e"), ("latency_ns", "l"), ("bus_bytes", "b"), ("quality", "q")):
                r[f"{nm}_vs_static"] = float(np.mean(pct(g[col], s[col])))
            r["e_vs_static_lo"], r["e_vs_static_hi"] = (float(v) for v in boot_ci(pct(g.energy_pj, s.energy_pj)))
            r["l_vs_static_lo"], r["l_vs_static_hi"] = (float(v) for v in boot_ci(pct(g.latency_ns, s.latency_ns)))
        rows.append(r)
    out = pd.DataFrame(rows)
    out["flat_latency_us"] = float(flat.latency_ns.mean() / 1e3)
    out["flat_energy_uj"] = float(flat.energy_pj.mean() / 1e6)
    return out


def e9_table(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d.exp.isin(["e9", "e9c"])].copy()
    rows = []
    for (reg, frac), g in d.groupby(["regime", "frac"]):
        key = ["trace"]
        st = g[g.policy == "static"].set_index("trace")
        fl = g[g.policy == "flat"].set_index("trace")
        g = g[g.exp.isin(["e9", "e9c"])]
        rows.append(dict(regime=reg, frac=frac, scale=np.nan, policy="flat", moves=0.0, reads=float(fl.reads.mean()),
                         skipped=float(fl.skipped_mass.mean()),
                         e_ratio=float(np.mean(fl.energy_pj.values / st.energy_pj.values)),
                         **dict(zip(["e", "e_lo", "e_hi"], ci_row(pct(fl.energy_pj, st.energy_pj)))),
                         **dict(zip(["q", "q_lo", "q_hi"], ci_row(pct(fl.quality + 1e-12, st.quality + 1e-12)))),
                         b=float(np.mean(pct(fl.bus_bytes, st.bus_bytes))), l=float(np.mean(pct(fl.latency_ns, st.latency_ns))),
                         win=np.nan))
        for (pol, scale), h in g[g.policy.str.startswith("dyn")].groupby(["policy", "scale"]):
            h = h.set_index("trace").loc[st.index]
            e = pct(h.energy_pj, st.energy_pj)
            qd = (h.quality.values - st.quality.values) / np.maximum(st.quality.values, 1e-12) * 100.0
            rows.append(dict(regime=reg, frac=frac, scale=scale, policy=pol, moves=float(h.moves.mean()),
                             reads=float(h.reads.mean()), skipped=float(h.skipped_mass.mean()), e_ratio=np.nan,
                             **dict(zip(["e", "e_lo", "e_hi"], ci_row(e))),
                             **dict(zip(["q", "q_lo", "q_hi"], ci_row(qd))),
                             b=float(np.mean(pct(h.bus_bytes, st.bus_bytes))), l=float(np.mean(pct(h.latency_ns, st.latency_ns))),
                             win=float(np.mean((e < 0) & (qd <= 2.0)))))
    return pd.DataFrame(rows)


def e9b_table(d: pd.DataFrame) -> pd.DataFrame:
    e9 = d[(d.exp == "e9") & (d.regime == "dense")]
    sub = d[d.exp == "e9b"]
    dyn = pd.concat([sub, e9[e9.policy == "dyn"]]).drop_duplicates(subset=["trace", "scale"])
    dyn = dyn[dyn.trace.isin(sub.trace.unique())]
    st = e9[e9.policy == "static"].set_index("trace")
    rows = []
    for scale, h in dyn.groupby("scale"):
        h = h.set_index("trace")
        s = st.loc[h.index]
        e = pct(h.energy_pj, s.energy_pj)
        q = (h.quality.values - s.quality.values) / s.quality.values * 100.0
        rows.append(dict(scale=scale, n=len(h), **dict(zip(["e", "e_lo", "e_hi"], ci_row(e))), q=float(np.mean(q)),
                         l=float(np.mean(pct(h.latency_ns, s.latency_ns))), moves=float(h.moves.mean()),
                         reconfig_share=float(np.mean(h.energy_reconfig_pj / h.energy_pj))))
    return pd.DataFrame(rows).sort_values("scale")


def main() -> None:
    d = pd.read_csv(RES / "m9_system.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    e8, e9, e9b = e8_table(d), e9_table(d), e9b_table(d)
    e8.to_csv(RES / "m9_e8_summary.csv", index=False)
    e9.to_csv(RES / "m9_e9_summary.csv", index=False)
    e9b.to_csv(RES / "m9_e9b_summary.csv", index=False)
    print("== E8: compute placement (dense, per-trace ratios vs flat, mean of 42) ==")
    print(e8.round(3).to_string(index=False))
    print("\n== E9: regimes (dyn/dynr vs static; flat row = headroom) ==")
    print(e9.round(2).to_string(index=False))
    print("\n== E9b: reconfiguration scale, dense, 14 traces ==")
    print(e9b.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
