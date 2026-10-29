"""Milestone 8a analysis: leave-one-trace-out CV per importance estimator (dense reads, 10x reconfig cost).

Same protocol as analyze_m7.py (thresholds chosen on the other 41 traces = lowest energy with mean quality
loss <= static's), applied to results/m8_s256_estimators_sims.csv; the oracle row and the static baseline
come from results/m7_s256_sims.csv restricted to the same 10-config grid.
Usage: python experiments/analyze_m8.py
"""

from __future__ import annotations

import pandas as pd

from analyze_m7 import RES, boot_ci, loo, summarize
from run_m7_estimators import GRID, VARIANTS

ORDER = ["oracle"] + VARIANTS


def _frame(sims: pd.DataFrame, variant: str, static: pd.DataFrame) -> pd.DataFrame:
    d = sims[sims.variant == variant].copy()
    d["policy"], d["dwell"], d["k"] = "dyn", 0, 0.0
    return pd.concat([static, d.drop(columns="variant")], ignore_index=True)


def estimator_table() -> pd.DataFrame:
    base = pd.read_csv(RES / "m7_s256_sims.csv")
    base["dwell"] = base.dwell.fillna(0).astype(int)
    static = base[(base.k == 0.0) & (base.policy == "static")]
    sims = pd.read_csv(RES / "m8_s256_estimators_sims.csv")
    orc = base[(base.k == 0.0) & (base.policy == "dyn") & (base.dwell == 0)].copy()
    orc = orc[[(p, d) in set(GRID) for p, d in zip(orc.promote_x, orc.demote_x)]]
    orc["variant"] = "oracle"
    sims = pd.concat([sims, orc[sims.columns.intersection(orc.columns)]], ignore_index=True)
    rows = []
    for v in ORDER:
        if v not in set(sims.variant):
            continue
        df = _frame(sims, v, static)
        r = loo(df, 0.0, 10, grid=GRID)
        s = summarize(r)
        fixed = loo_fixed(df, "8/0.4")
        rows.append(dict(variant=v, e=s["e_mean"], e_lo=s["e_lo"], e_hi=s["e_hi"], q=s["q_mean"], q_lo=s["q_lo"],
                         q_hi=s["q_hi"], moves=s["moves"], win_rate=s["win_rate"],
                         top_cfg=r.cfg.value_counts().index[0], top_cfg_n=int(r.cfg.value_counts().iloc[0]),
                         fixed_e=fixed[0], fixed_q=fixed[1]))
    return pd.DataFrame(rows)


def loo_fixed(df: pd.DataFrame, cfg: str) -> tuple[float, float]:
    from analyze_m7 import tables
    T, _ = tables(df, 0.0, 10, 0, GRID)
    return float(T["e"][cfg].mean()), float(T["q"][cfg].mean())


def main() -> None:
    t = estimator_table()
    t.to_csv(RES / "m8_s256_estimators_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
