"""Milestone 7 analysis: leave-one-trace-out CV, reconfig-cost sweep, top-k sweep, cross-model transfer.

Reads results/m7_<tag>_sims.csv (see run_m7.py) and applies the protocol fixed in NOTES.md:
thresholds chosen on the other traces = lowest mean energy among configs whose mean quality loss is
<= static_modality's (fallback: lowest quality), evaluated on the held-out trace. Energy / latency at
reconfig multiplier m = access + m * reconfig_1x. Percentages are vs static_modality, per trace.
Usage: python experiments/analyze_m7.py [--plots]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
COSTS = [1, 2, 5, 10, 20, 50, 100]
REDUCED = [(p, d) for p in (1.0, 2.0, 4.0, 8.0) for d in (0.1, 0.4) if d < p]
RNG = np.random.default_rng(0)


def kind_of(name: str) -> str:
    return name.split("_")[0]


def load(tag: str) -> pd.DataFrame:
    df = pd.read_csv(RES / f"m7_{tag}_sims.csv")
    df["kind"] = df.trace.map(kind_of)
    df["dwell"] = df.dwell.fillna(0).astype(int)
    return df


def add_totals(df: pd.DataFrame, mult: float) -> pd.DataFrame:
    df = df.copy()
    df["energy"] = df.access_energy_pj + mult * df.reconfig_energy_1x
    df["latency"] = df.access_latency_ns + mult * df.reconfig_latency_1x
    return df


def tables(df, k, mult, dwell=0, grid=None):
    """Per-trace % vs static for every dynamic config: matrices E, Q, L, M (trace x config)."""
    d = add_totals(df[df.k == k], mult)
    st = d[d.policy == "static"].set_index("trace")
    dyn = d[(d.policy == "dyn") & (d.dwell == dwell)].copy()
    if grid is not None:
        dyn = dyn[[(p, q) in set(grid) for p, q in zip(dyn.promote_x, dyn.demote_x)]]
    dyn["cfg"] = [f"{p:g}/{d:g}" for p, d in zip(dyn.promote_x, dyn.demote_x)]  # "promote_x/demote_x"
    dyn["e"] = 100 * (dyn.energy.values / st.energy[dyn.trace].values - 1)
    dyn["q"] = 100 * (dyn.quality.values / st.quality[dyn.trace].values - 1)
    dyn["l"] = 100 * (dyn.latency.values / st.latency[dyn.trace].values - 1)
    dyn["m"] = dyn.promos + dyn.demos
    return {c: dyn.pivot(index="trace", columns="cfg", values=c) for c in "eqlm"}, st


def select(E, Q, train_idx):
    e, q = E.loc[train_idx].mean(), Q.loc[train_idx].mean()
    ok = q[q <= 0]
    return e[ok.index].idxmin() if len(ok) else q.idxmin()


def loo(df, k, mult, dwell=0, grid=None, fit_traces=None):
    """Leave-one-out CV. If fit_traces is given, thresholds come from those traces only (cross-model)."""
    T, st = tables(df, k, mult, dwell, grid)
    E, Q, L, M = T["e"], T["q"], T["l"], T["m"]
    rows = []
    for h in E.index:
        train = [t for t in E.index if t != h]
        cfg = select(E, Q, train)
        rows.append(dict(trace=h, kind=kind_of(h), cfg=cfg, e=E.loc[h, cfg], q=Q.loc[h, cfg], l=L.loc[h, cfg],
                         moves=M.loc[h, cfg]))
    return pd.DataFrame(rows)


def transfer(df_fit, df_eval, k, mult, grid=None):
    """Select on all of df_fit's traces, evaluate on every trace of df_eval, no refitting."""
    Tf, _ = tables(df_fit, k, mult, 0, grid)
    cfg = select(Tf["e"], Tf["q"], list(Tf["e"].index))
    Te, _ = tables(df_eval, k, mult, 0, grid)
    return pd.DataFrame(dict(trace=Te["e"].index, kind=[kind_of(t) for t in Te["e"].index], cfg=[cfg] * len(Te["e"]),
                             e=Te["e"][cfg].values, q=Te["q"][cfg].values, l=Te["l"][cfg].values,
                             moves=Te["m"][cfg].values))


def boot_ci(x, n=4000):
    x = np.asarray(x, float)
    m = RNG.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def summarize(r: pd.DataFrame) -> dict:
    elo, ehi = boot_ci(r.e)
    qlo, qhi = boot_ci(r.q)
    return dict(n=len(r), e_mean=r.e.mean(), e_lo=elo, e_hi=ehi, q_mean=r.q.mean(), q_lo=qlo, q_hi=qhi,
                l_mean=r.l.mean(), moves=r.moves.mean(),
                win_rate=float(((r.e < 0) & (r.q <= 0)).mean()), e_neg_rate=float((r.e < 0).mean()))


def verdict(r: pd.DataFrame) -> dict:
    """The criteria fixed in NOTES.md (item 2)."""
    s = summarize(r)
    per_kind = r.groupby("kind").e.mean()
    return dict(energy_ge_10=s["e_mean"] <= -10, quality_le_2=s["q_mean"] <= 2, ci_excludes_0=s["e_hi"] < 0,
                every_kind_negative=bool((per_kind < 0).all()),
                robust=bool(s["e_mean"] <= -10 and s["q_mean"] <= 2 and s["e_hi"] < 0 and (per_kind < 0).all()))


def fmt(d: dict) -> str:
    return (f"energy {d['e_mean']:+.1f}% [{d['e_lo']:+.1f},{d['e_hi']:+.1f}]  quality {d['q_mean']:+.1f}% "
            f"[{d['q_lo']:+.1f},{d['q_hi']:+.1f}]  latency {d['l_mean']:+.1f}%  moves {d['moves']:.0f}  "
            f"win(e<0&q<=0) {100 * d['win_rate']:.0f}%  e<0 {100 * d['e_neg_rate']:.0f}%")


def main() -> None:
    plots = "--plots" in sys.argv
    tags = [t for t in ("s256", "s500", "s2200") if (RES / f"m7_{t}_sims.csv").exists()]
    data = {t: load(t) for t in tags}
    out = {}
    for tag, df in data.items():
        n = df.trace.nunique()
        print(f"\n################ {tag}: {n} traces "
              f"({df.drop_duplicates('trace').kind.value_counts().to_dict()})")
        for k in sorted(df.k.unique()):
            s = df[(df.k == k) & (df.policy == "static")]
            f = df[(df.k == k) & (df.policy == "flat")]
            ratio = (f.set_index("trace").access_energy_pj / s.set_index("trace").access_energy_pj)
            print(f"  k={k}: flat/static energy {ratio.mean():.2f}x (min {ratio.min():.2f}, max {ratio.max():.2f}); "
                  f"static quality {s.quality.mean():.5f}; skipped mass {s.skipped_pct.mean():.2f}%")
        # --- primary
        for k in (0.0, 0.1):
            r = loo(df, k, 10)
            out[(tag, "primary", k)] = r
            print(f"\n[{tag}] PRIMARY LOO-CV, k={k or 'dense'}, 10x:  {fmt(summarize(r))}")
            print("   verdict:", verdict(r))
            for kd, g in r.groupby("kind"):
                print(f"     {kd:8s} n={len(g):2d}  {fmt(summarize(g))}")
            print("   chosen configs:", r.cfg.value_counts().to_dict())
        # --- cost sweep (dense and sparse10)
        crows = []
        for k in (0.0, 0.1):
            for m in COSTS:
                r = loo(df, k, m)
                crows.append(dict(tag=tag, k=k, mult=m, **summarize(r), robust=verdict(r)["robust"]))
        cs = pd.DataFrame(crows)
        cs.to_csv(RES / f"m7_{tag}_cost_sweep.csv", index=False)
        print(f"\n[{tag}] RECONFIG-COST SWEEP (thresholds reselected per cost):")
        print(cs[["k", "mult", "e_mean", "e_lo", "e_hi", "q_mean", "l_mean", "moves", "win_rate", "robust"]]
              .round(2).to_string(index=False))
        # --- top-k (reduced grid, only if extras present)
        if (df.k > 0.1).any():
            krows = []
            for k in (0.0, 0.5, 0.2, 0.1, 0.05):
                r = loo(df, k, 10, grid=REDUCED)
                sd = df[(df.k == k) & (df.policy == "static")].set_index("trace")
                fl = df[(df.k == k) & (df.policy == "flat")].set_index("trace")
                krows.append(dict(tag=tag, k=k, flat_over_static=(fl.access_energy_pj / sd.access_energy_pj).mean(),
                                  skipped_pct=sd.skipped_pct.mean(), **summarize(r)))
            ks = pd.DataFrame(krows)
            ks.to_csv(RES / f"m7_{tag}_topk.csv", index=False)
            print(f"\n[{tag}] TOP-K SWEEP (reduced grid, 10x):")
            print(ks[["k", "flat_over_static", "skipped_pct", "e_mean", "e_lo", "e_hi", "q_mean", "moves", "win_rate"]]
                  .round(2).to_string(index=False))
        # --- dwell
        if (df.dwell > 0).any():
            drows = []
            for w in [0, 5, 20, 40]:
                r = loo(df, 0.0, 10, dwell=w, grid=REDUCED)
                drows.append(dict(tag=tag, dwell=w, **summarize(r)))
            ds = pd.DataFrame(drows)
            ds.to_csv(RES / f"m7_{tag}_dwell.csv", index=False)
            print(f"\n[{tag}] MIN-DWELL VARIANTS (dense, reduced grid, 10x):")
            print(ds[["dwell", "e_mean", "e_lo", "e_hi", "q_mean", "moves", "win_rate"]].round(2).to_string(index=False))
    # --- cross-model: thresholds fitted on all 256M traces, applied unchanged to larger models
    for tag in tags:
        if tag == "s256" or "s256" not in data:
            continue
        for k in (0.0, 0.1):
            r = transfer(data["s256"], data[tag], k, 10)
            out[(tag, "transfer", k)] = r
            print(f"\n[transfer 256M->{tag}] k={k or 'dense'}, thresholds {r.cfg.iloc[0]} chosen on 256M: {fmt(summarize(r))}")
            print("   verdict:", verdict(r))
    for key, r in out.items():
        r.to_csv(RES / f"m7_{key[0]}_{key[1]}_k{key[2]}.csv", index=False)
    if plots:
        make_plots(data)


def make_plots(data) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"natural": "#0072B2", "chart": "#D55E00", "doc": "#009E73"}
    for tag, df in data.items():
        # per-trace scatter, primary dense
        r = loo(df, 0.0, 10)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for kd, g in r.groupby("kind"):
            ax[0].scatter(g.e, g.q, label=kd, c=colors[kd], s=36, alpha=0.85)
        ax[0].axhline(0, c="grey", lw=0.8); ax[0].axvline(0, c="grey", lw=0.8)
        ax[0].set_xlabel("held-out energy vs static_modality (%)"); ax[0].set_ylabel("held-out quality loss vs static (%)")
        ax[0].set_title(f"{tag}: per-image LOO-CV, dense, 10x cost"); ax[0].legend(frameon=False)
        cs = pd.read_csv(RES / f"m7_{tag}_cost_sweep.csv")
        for k, ls in ((0.0, "-"), (0.1, "--")):
            c = cs[cs.k == k]
            lab = "dense" if k == 0 else "sparse top-10%"
            ax[1].plot(c.mult, c.e_mean, ls, marker="o", c="#0072B2", label=f"energy, {lab}")
            ax[1].fill_between(c.mult, c.e_lo, c.e_hi, color="#0072B2", alpha=0.12)
        ax[1].axhline(0, c="grey", lw=0.8); ax[1].set_xscale("log")
        ax[1].set_xlabel("reconfiguration cost (x one HOT access)"); ax[1].set_ylabel("held-out energy vs static (%)")
        ax[1].set_title("break-even reconfiguration cost"); ax[1].legend(frameon=False)
        fig.tight_layout(); fig.savefig(RES / f"m7_{tag}_summary.png", dpi=140); plt.close(fig)
        kp = RES / f"m7_{tag}_topk.csv"
        if kp.exists():
            ks = pd.read_csv(kp)
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
            x = [1 if k == 0 else k for k in ks.k]
            ax[0].plot(x, ks.flat_over_static, marker="o", c="#D55E00"); ax[0].set_xscale("log")
            ax[0].set_xlabel("fraction of entries read per step (1 = dense)"); ax[0].set_ylabel("flat / static energy")
            ax[0].set_title("value of tiering shrinks with sparse reads")
            ax[1].errorbar(x, ks.e_mean, yerr=[ks.e_mean - ks.e_lo, ks.e_hi - ks.e_mean], marker="o", c="#0072B2", capsize=3)
            ax[1].axhline(0, c="grey", lw=0.8); ax[1].set_xscale("log")
            ax[1].set_xlabel("fraction read per step"); ax[1].set_ylabel("dynamic energy vs static (%)")
            ax[1].set_title("held-out dynamic energy saving")
            fig.tight_layout(); fig.savefig(RES / f"m7_{tag}_topk.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
