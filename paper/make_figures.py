"""Builds every data figure of the paper from results/*.csv into paper/figs/*.pdf.
Usage (repo root): .venv/bin/python paper/make_figures.py
Figures that depend on M8 outputs are skipped until their CSVs exist."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, OUT = ROOT / "results", ROOT / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

C = dict(flat="#8a8a8a", static="#3b6fb6", dyn="#d9622b", warm="#2a9d8f", ink="#222222", grid="#dddddd",
         m256="#3b6fb6", m500="#d9622b", good="#2a9d8f", bad="#c0392b")
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
    "axes.linewidth": 0.7, "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    "font.family": "DejaVu Sans",
})


def panel(ax, letter):
    ax.text(-0.02, 1.07, letter, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- synthetic study (M1-M4)
def fig_synth():
    m4 = pd.read_csv(R / "milestone4_comparison.csv").set_index("policy")
    ref = pd.read_csv(R / "sensitivity_references.csv").set_index("policy")
    names = ["flat", "static_modality", "dynamic_hybrid", "all_warm"]
    lab = ["Flat", "Static\nmodality", "Dynamic\nhybrid", "All-WARM"]
    en = [m4.total_energy_pj["flat"], m4.total_energy_pj["static_modality"], m4.total_energy_pj["dynamic_hybrid"],
          ref.energy_pj["all_warm"]]
    q = [0, m4.mean_quality_loss["static_modality"], m4.mean_quality_loss["dynamic_hybrid"], 0.05]
    cols = [C["flat"], C["static"], C["dyn"], "#b9b9b9"]
    dy = pd.read_csv(R / "dynamic_hybrid_timeseries.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    ax[0].bar(range(4), np.array(en) / 1e6, color=cols, width=0.7)
    ax[0].set_yscale("log"); ax[0].set_xticks(range(4)); ax[0].set_xticklabels(lab, fontsize=6.5)
    ax[0].set_yticks([1, 10, 100]); ax[0].set_yticklabels(["1", "10", "100"]); ax[0].set_ylim(1, 130)
    ax[0].set_ylabel("Total energy (µJ, log)")
    for i, v in enumerate(en):
        ax[0].text(i, v / 1e6 * 1.12, f"{v / 1e6:.1f}", ha="center", fontsize=7)
    panel(ax[0], "a")
    ax[1].bar(range(4), q, color=cols, width=0.7)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(lab, fontsize=6.5); ax[1].set_ylabel("Mean quality loss")
    for i, v in enumerate(q):
        ax[1].text(i, v + 0.0012, f"{v:.3f}", ha="center", fontsize=7)
    panel(ax[1], "b")
    ax[2].axhline(100 * m4.warm_attention_mass_pct["static_modality"] / 100, color=C["static"], lw=1.6, ls="--",
                  label="Static modality (mean)")
    ax[2].plot(dy.step, 100 * dy["warm_attention_mass"], color=C["dyn"], lw=1.6, label="Dynamic hybrid")
    ax[2].set_xlabel("Decode step"); ax[2].set_ylabel("Attention served from WARM (%)")
    ax[2].legend(frameon=False, loc="upper right"); ax[2].set_ylim(0, 100)
    panel(ax[2], "c")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_synth")


# ---------------------------------------------------------------- real attention anatomy (M5)
def fig_anatomy():
    traces = ["chart", "scene", "noise+label"]
    vis, sink, gen = np.array([6, 7, 5]), np.array([35, 34, 34]), np.array([36, 37, 37])
    other = 100 - vis - sink - gen
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    left = np.zeros(3)
    for v, name, col in ((sink, "Attention sink (pos. 0)", "#7b6fb0"), (gen, "Generated tokens", C["warm"]),
                         (other, "Other prompt text", "#b9b9b9"), (vis, "Visual tokens", C["dyn"])):
        ax[0].barh(traces, v, left=left, color=col, label=name, height=0.62)
        for i, (l, w) in enumerate(zip(left, v)):
            if w >= 5:
                ax[0].text(l + w / 2, i, f"{w}", ha="center", va="center", color="white", fontsize=7.5)
        left += v
    ax[0].set_xlim(0, 100); ax[0].set_xlabel("Share of attention mass (%)"); ax[0].invert_yaxis()
    ax[0].legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    panel(ax[0], "a")
    ax[1].bar([0, 1], [68, 6], color=[C["flat"], C["dyn"]], width=0.55)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["Synthetic trace\n(assumed)", "Real SmolVLM\n(measured)"])
    ax[1].set_ylabel("Attention mass on visual tokens (%)")
    for i, v in enumerate([68, 6]):
        ax[1].text(i, v + 2, f"{v}%", ha="center", fontsize=8)
    ax[1].set_ylim(0, 80)
    panel(ax[1], "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_anatomy")


# ---------------------------------------------------------------- ladder: each stage, held-out energy/quality
def fig_ladder():
    s256 = pd.read_csv(R / "m7_s256_primary_k0.0.csv"); s500 = pd.read_csv(R / "m7_s500_primary_k0.0.csv")
    m5 = pd.read_csv(R / "m6_results.csv")
    def hold(mode):
        d = m5[(m5["mode"] == mode) & (m5.split == "TEST")]
        return d.dyn_energy_vs_static_pct.mean(), d.dyn_quality_vs_static_pct.mean()
    ci = lambda x: (x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)))
    stages = [("M5  layer-avg\n3 imgs, 256M", *hold("avg_dense"), None, None),
              ("M6  per-layer\n3 imgs, 256M", *hold("layer_dense"), None, None),
              ("M7  per-layer\n42 imgs, 256M", s256.e.mean(), s256.q.mean(), ci(s256.e)[1], ci(s256.q)[1]),
              ("M7  per-layer\n42 imgs, 500M", s500.e.mean(), s500.q.mean(), ci(s500.e)[1], ci(s500.q)[1])]
    x = np.arange(len(stages))
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.4))
    for a, idx, ttl, ylab in ((ax[0], 1, "Energy vs static modality", "Energy change (%)"),
                              (ax[1], 2, "Quality loss vs static modality", "Quality-loss change (%)")):
        vals = [s[idx] for s in stages]
        err = [s[idx + 2] if s[idx + 2] is not None else 0 for s in stages]
        cols = [C["good"] if v <= 0 else C["bad"] for v in vals]
        a.bar(x, vals, yerr=err, color=cols, width=0.62, error_kw=dict(lw=0.8, capsize=2))
        a.axhline(0, color=C["ink"], lw=0.8)
        a.set_xticks(x); a.set_xticklabels([s[0] for s in stages], fontsize=6.8); a.set_ylabel(ylab)
        a.set_title(ttl, loc="left")
        for i, v in enumerate(vals):
            e = err[i]
            a.text(i, v + (e + 0.8 if v >= 0 else -(e + 0.8)), f"{v:+.1f}",
                   ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax[0].set_ylim(-31, 6); ax[1].set_ylim(-9, 18)
    panel(ax[0], "a"); panel(ax[1], "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_ladder")


# ---------------------------------------------------------------- M6b layer-aware static baseline
def fig_m6b():
    t = pd.read_csv(R / "layer_aware_theta_curve.csv")
    g = t.groupby(["grouping", "mode", "theta"])[["energy_vs_static_pct", "quality_vs_static_pct"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(4.1, 2.9))
    for mode, mk, ls in (("layer_dense", "o", "-"),):
        d = g[(g.grouping == "role_layer") & (g["mode"] == mode) & (g.theta >= 0.4)].sort_values("theta")
        ax.plot(d.energy_vs_static_pct, d.quality_vs_static_pct, ls, marker=mk, color=C["static"], ms=4, lw=1.4,
                label="Static (role × layer) profile, sweep of θ")
        for _, r in d.iterrows():
            ax.annotate(f"θ={r.theta:g}", (r.energy_vs_static_pct, r.quality_vs_static_pct), fontsize=6.5,
                        xytext=(4, -9), textcoords="offset points", color=C["static"])
    ax.plot([0], [0], "s", color=C["flat"], ms=6, label="Static modality (reference)")
    ax.plot([-19.8], [-3.0], "*", color=C["dyn"], ms=13, label="Dynamic per-entry policy")
    ax.axhline(0, color=C["ink"], lw=0.6); ax.axvline(0, color=C["ink"], lw=0.6)
    ax.set_xlabel("Energy vs static modality (%)"); ax.set_ylabel("Quality loss vs static modality (%)")
    ax.legend(frameon=False, loc="upper right", fontsize=6.8)
    ax.set_ylim(-9, 46)
    ax.annotate("lower-left is better", xy=(-27.5, -7.5), fontsize=7, color="#555555", style="italic")
    save(fig, "fig_m6b")


# ---------------------------------------------------------------- M7: reconfiguration cost + sparse reads
def fig_cost():
    a, b = pd.read_csv(R / "m7_s256_cost_sweep.csv"), pd.read_csv(R / "m7_s500_cost_sweep.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.5))
    for d, col, lab in ((a, C["m256"], "SmolVLM-256M"), (b, C["m500"], "SmolVLM-500M")):
        for k, ls, kl in ((0.0, "-", "dense reads"), (0.1, "--", "top-10% reads")):
            s = d[d.k == k]
            ax[0].plot(s.mult, s.e_mean, ls, marker="o", ms=3.2, color=col, lw=1.4, label=f"{lab}, {kl}")
            ax[0].fill_between(s.mult, s.e_lo, s.e_hi, color=col, alpha=0.15, lw=0)
    ax[0].axhline(0, color=C["ink"], lw=0.8); ax[0].set_xscale("log")
    ax[0].set_xlabel("Reconfiguration cost (× one HOT access)"); ax[0].set_ylabel("Energy vs static modality (%)")
    ax[0].axvline(10, color="#999999", lw=0.8, ls=":"); ax[0].text(10.8, -38, "default 10×", fontsize=6.8, color="#666666")
    ax[0].set_ylim(-42, 75); ax[0].legend(frameon=False, fontsize=6.3, loc="upper left")
    ax[0].set_xticks([1, 2, 5, 10, 20, 50, 100]); ax[0].set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    panel(ax[0], "a")
    tk = pd.read_csv(R / "m7_s256_topk.csv").sort_values("k")
    ax[1].bar(range(len(tk)), tk.e_mean, yerr=[tk.e_mean - tk.e_lo, tk.e_hi - tk.e_mean],
              color=[C["good"] if v < 0 else C["bad"] for v in tk.e_mean], width=0.6,
              error_kw=dict(lw=0.8, capsize=2))
    ax[1].set_xticks(range(len(tk)))
    ax[1].set_xticklabels([("dense" if k == 0 else f"top-{int(round(k * 100))}%") + f"\n({r:.1f}×)"
                           for k, r in zip(tk.k, tk.flat_over_static)])
    ax[1].axhline(0, color=C["ink"], lw=0.8)
    for i, v in enumerate(tk.e_mean):
        ax[1].text(i, v + (1.5 if v >= 0 else -1.5), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    ax[1].set_ylabel("Energy vs static modality (%)"); ax[1].set_xlabel("Read pattern (flat/static energy ratio)")
    ax[1].set_ylim(-30, 22)
    panel(ax[1], "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_cost_sparse")


# ---------------------------------------------------------------- M7: mechanism + per-kind
def fig_mech():
    m = pd.read_csv(R / "m7_mechanism_s256.csv")
    g = m[m.variant != "static"].groupby("variant")[["e_vs_static_pct", "q_vs_static_pct"]].mean()
    order = [("demote_only", "Demote only"), ("promote_only", "Promote only"), ("full", "Full policy")]
    p = pd.read_csv(R / "m7_s256_primary_k0.0.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.3), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    for a, col, ttl in ((ax[0], "e_vs_static_pct", "Energy vs static (%)"), (ax[1], "q_vs_static_pct", "Quality loss vs static (%)")):
        vals = [g[col][k] for k, _ in order]
        a.bar(range(3), vals, color=[C["good"] if v < 0 else C["bad"] for v in vals], width=0.6)
        a.axhline(0, color=C["ink"], lw=0.8); a.set_xticks(range(3)); a.set_xticklabels([l.replace(" ", "\n", 1) for _, l in order])
        a.set_title(ttl, loc="left")
        for i, v in enumerate(vals):
            a.text(i, v + (0.8 if v >= 0 else -0.8), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
        a.set_ylim(min(vals) * 1.35 if min(vals) < 0 else -3, max(vals) * 1.4 + 2)
    panel(ax[0], "a"); panel(ax[1], "b")
    kinds = ["natural", "chart", "doc"]
    for i, k in enumerate(kinds):
        d = p[p.kind == k]
        ax[2].scatter(d.e, d.q, s=14, alpha=0.8, label=f"{k} (n={len(d)})", color=[C["dyn"], C["static"], C["warm"]][i], lw=0)
    ax[2].axhline(0, color=C["ink"], lw=0.7); ax[2].axvline(0, color=C["ink"], lw=0.7)
    ax[2].set_xlabel("Energy vs static (%)"); ax[2].set_ylabel("Quality loss vs static (%)")
    ax[2].legend(frameon=False, fontsize=6.8, loc="upper right")
    panel(ax[2], "c")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_mech")


# ---------------------------------------------------------------- M8a: non-oracle importance estimators
def fig_estimators():
    f = R / "m8_s256_estimators_sims.csv"
    if not f.exists():
        return print("skip fig_estimators (no data)")
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    from analyze_m8 import estimator_table
    t = estimator_table()
    t.to_csv(R / "m8_s256_estimators_summary.csv", index=False)
    labels = {"oracle": "Oracle", "noise0.25": "Noise σ=.25", "noise0.5": "Noise σ=.5", "noise1": "Noise σ=1",
              "sub2": "Every 2nd step", "sub4": "Every 4th step", "sub8": "Every 8th step", "delay1": "Delay 1",
              "delay4": "Delay 4", "token_shared": "Token-shared", "shared_sub4": "Shared + 4th step",
              "noise0.5_sub4": "Noise .5 + 4th"}
    t = t.set_index("variant").loc[[v for v in labels if v in set(t.variant)]]
    y = np.arange(len(t))[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    for a, col, lo, hi, ttl in ((ax[0], "e", "e_lo", "e_hi", "Energy vs static modality (%)"),
                                (ax[1], "q", "q_lo", "q_hi", "Quality loss vs static modality (%)")):
        cols = [C["ink"] if v == "oracle" else (C["good"] if t[col][v] < 0 else C["bad"]) for v in t.index]
        a.barh(y, t[col], color=cols, height=0.62, xerr=[t[col] - t[lo], t[hi] - t[col]], error_kw=dict(lw=0.8, capsize=1.5))
        a.axvline(0, color=C["ink"], lw=0.8); a.set_title(ttl, loc="left")
        for yy, v, l, h in zip(y, t[col], t[lo], t[hi]):
            a.text(h + 0.5 if v >= 0 else l - 0.5, yy, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)
    ax[0].set_yticks(y); ax[0].set_yticklabels([labels[v] for v in t.index])
    ax[0].set_xlim(-31, 5); ax[1].set_xlim(-13, 6)
    panel(ax[0], "a"); panel(ax[1], "b")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_estimators")


# ---------------------------------------------------------------- M8b: real-model output check
def fig_real_output():
    f = R / "m8_real_output.csv"
    if not f.exists():
        return print("skip fig_real_output (no data)")
    d = pd.read_csv(f)
    order = [("static", "Static", C["static"]), ("dynamic", "Dynamic", C["dyn"]),
             ("random_matched", "Random\n(same #)", "#9a9a9a"), ("all_int4", "All\nINT4", "#555555")]
    pv = d.pivot(index="image", columns="policy", values="kl")
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.4), gridspec_kw=dict(width_ratios=[1.1, 0.9, 1.05]))
    means = [pv[k].mean() for k, _, _ in order]
    se = [pv[k].std(ddof=1) / np.sqrt(len(pv)) * 1.96 for k, _, _ in order]
    ax[0].bar(range(4), means, yerr=se, color=[c for _, _, c in order], width=0.62, error_kw=dict(lw=0.8, capsize=2))
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels([l for _, l, _ in order], fontsize=6.6)
    ax[0].set_ylabel("KL(full-precision ‖ policy), nats")
    for i, v in enumerate(means):
        ax[0].text(i, v + max(se) * 1.1 + 0.0003, f"{v:.4f}", ha="center", fontsize=6.8)
    panel(ax[0], "a")
    # paired: dynamic vs static per image
    ax[1].scatter(pv.static, pv.dynamic, s=14, color=C["dyn"], lw=0, alpha=0.85)
    lim = max(pv.static.max(), pv.dynamic.max()) * 1.08
    ax[1].plot([0, lim], [0, lim], color=C["ink"], lw=0.8); ax[1].set_xlim(0, lim); ax[1].set_ylim(0, lim)
    ax[1].set_xlabel("KL, static modality"); ax[1].set_ylabel("KL, dynamic")
    ax[1].text(lim * 0.04, lim * 0.9, f"dynamic lower on\n{(pv.dynamic < pv.static).sum()}/{len(pv)} images", fontsize=7)
    panel(ax[1], "b")
    # proxy vs real
    w = d[d.policy != "dynamic"].copy()
    kcol = dict(static=C["static"], random_matched="#9a9a9a", all_int4="#555555")
    for pol in ["static", "random_matched", "all_int4"]:
        s = d[d.policy == pol]
        ax[2].scatter(s.proxy, s.kl, s=11, color=kcol[pol], lw=0, alpha=0.8)
    s = d[d.policy == "dynamic"]; ax[2].scatter(s.proxy, s.kl, s=11, color=C["dyn"], lw=0, alpha=0.9)
    rho = d.proxy.rank().corr(d.kl.rank())
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xticks([0.004, 0.01, 0.02, 0.04]); ax[2].set_xticklabels(["0.004", "0.01", "0.02", "0.04"])
    ax[2].set_yticks([0.001, 0.01]); ax[2].set_yticklabels(["0.001", "0.01"])
    ax[2].minorticks_off()
    ax[2].set_xlabel("Simulator quality proxy"); ax[2].set_ylabel("Measured KL (nats)")
    ax[2].text(0.04, 0.93, f"Spearman ρ = {rho:.2f}\n(all {len(d)} image×policy points)", transform=ax[2].transAxes, fontsize=7, va="top")
    panel(ax[2], "c")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_real_output")


if __name__ == "__main__":
    for fn in (fig_synth, fig_anatomy, fig_ladder, fig_m6b, fig_cost, fig_mech, fig_estimators, fig_real_output):
        fn()
