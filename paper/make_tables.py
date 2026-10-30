"""Generates the appendix tables (LaTeX) of the paper from results/m9_*_summary.csv.
Usage (repo root): .venv/bin/python paper/make_tables.py"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, P = ROOT / "results", ROOT / "paper"


def e8() -> None:
    t = pd.read_csv(R / "m9_e8_summary.csv")
    rows = []
    for pol, name in (("static", "Static modality"), ("dyn", "Dynamic ($8\\times/0.4\\times$)")):
        rows.append(f"\\multicolumn{{11}}{{l}}{{\\textit{{{name}}}}}\\\\")
        for c in sorted(t.cim_units.unique()):
            cells = []
            for col in ("latency_vs_flat", "energy_vs_flat", "bus_vs_flat"):
                for s in ("tier", "host", "adaptive"):
                    v = t[(t.policy == pol) & (t.scheduler == s) & (t.cim_units == c)][col].iloc[0]
                    cells.append(f"{v:.3f}" if v < 10 else f"{v:.1f}")
            sp = t[(t.policy == pol) & (t.scheduler == "adaptive") & (t.cim_units == c)].spill.iloc[0]
            rows.append(f"{c} & " + " & ".join(cells) + f" & {100 * sp:.0f}\\%\\\\")
        rows.append("\\midrule")
    body = "\n".join(rows[:-1])
    (P / "app_e8.tex").write_text(
        "\\begin{table}[h]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3.2pt}\n"
        "\\caption{E8 in full: per-trace ratios to flat, averaged over 42 traces, by CIM capacity. T = compute in memory (tier), "
        "H = host only, A = adaptive spill. Last column: share of \\warm{} packets the adaptive scheduler spilled. "
        "Flat: mean makespan %.0f $\\mu$s, mean energy %.0f $\\mu$J.}\n\\label{tab:e8full}\n"
        "\\begin{tabular}{c ccc ccc ccc c}\n\\toprule\n & \\multicolumn{3}{c}{Makespan / flat} & \\multicolumn{3}{c}{Energy / flat} "
        "& \\multicolumn{3}{c}{Bus bytes / flat} & \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
        "CIM units & T & H & A & T & H & A & T & H & A & Spill (A)\\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        % (t.flat_latency_us.iloc[0], t.flat_energy_uj.iloc[0], body))


def e9() -> None:
    t = pd.read_csv(R / "m9_e9_summary.csv")
    order = [("dense", 1.0, "Reusable (dense)"), ("random", 0.3, "Random 30\\%"), ("random", 0.1, "Random 10\\%"),
             ("random", 0.03, "Random 3\\%"), ("stream", 0.0, "Streaming")]
    names = {"dyn": "Dynamic", "dynr": "+ reuse gate", "dynr_a4": "+ gate, min age 4$^\\dagger$",
             "dynr_a8": "+ gate, min age 8$^\\dagger$", "dynr_a16": "+ gate, min age 16$^\\dagger$"}
    rows = []
    for reg, fr, label in order:
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{label}}}}}\\\\")
        g = t[(t.regime == reg) & (t.frac == fr)]
        fl = g[g.policy == "flat"].iloc[0]
        rows.append(f"\\quad Flat (headroom) & -- & {fl.e_ratio:.2f}$\\times$ static & -- & -- & -- & -- & --\\\\")
        for pol in names:
            for sc in (1.0, 10.0):
                r = g[(g.policy == pol) & (g.scale == sc)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append(f"\\quad {names[pol]} & {sc:g}$\\times$ & {r.e:+.1f} [{r.e_lo:+.1f}, {r.e_hi:+.1f}] & "
                            f"{r.q:+.1f} [{r.q_lo:+.1f}, {r.q_hi:+.1f}] & {r.moves:.0f} & {r.b:+.1f} & {r.l:+.1f} & {r.win:.2f}\\\\")
        rows.append("\\midrule")
    (P / "app_e9.tex").write_text(
        "\\begin{table}[h]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{E9 in full: change against static modality (\\%, mean over 42 traces, paired-bootstrap 95\\% CI in brackets). "
        "Scale = migration cost relative to the bottom-up estimate. Moves per session; bus bytes and makespan as \\% change; "
        "Win = fraction of traces with lower energy and quality loss at most 2\\% worse. $^\\dagger$ Post-hoc variants (see text); "
        "streaming rows for gated variants were not run (the dynamic policy makes no moves there).}\n\\label{tab:e9full}\n"
        "\\begin{tabular}{llcccccc}\n\\toprule\nPolicy & Scale & Energy & Quality loss & Moves & Bus bytes & Makespan & Win\\\\\n\\midrule\n"
        + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


if __name__ == "__main__":
    e8()
    e9()
    print("wrote app_e8.tex, app_e9.tex")
