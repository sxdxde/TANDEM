"""Builds report/index.html from template.html by embedding CSV data inline.
Reads only results/*.csv; writes only inside report/."""
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def rows(name):
    with open(R / name) as f:
        return list(csv.DictReader(f))


def fl(x):
    return float(x)


m4 = {r["policy"]: r for r in rows("milestone4_comparison.csv")}
refs = {r["policy"]: r for r in rows("sensitivity_references.csv")}
policies = []
for name, r in (("flat", m4["flat"]), ("static_modality", m4["static_modality"]),
                ("dynamic_hybrid", m4["dynamic_hybrid"])):
    policies.append(dict(policy=name, energy=fl(r["total_energy_pj"]), latency=fl(r["total_latency_ns"]),
                         quality=fl(r["mean_quality_loss"]), warm=fl(r["warm_attention_mass_pct"]),
                         promos=int(r["n_promotions"]), demos=int(r["n_demotions"])))
a = refs["all_warm"]
policies.append(dict(policy="all_warm", energy=fl(a["energy_pj"]), latency=fl(a["latency_ns"]),
                     quality=fl(a["mean_quality_loss"]), warm=fl(a["warm_attention_mass_pct"]),
                     promos=0, demos=0))

sweep = [[fl(r["promote"]), fl(r["demote"]), fl(r["cost_mult"]), round(fl(r["energy_vs_static_pct"]), 3),
          round(fl(r["latency_vs_static_pct"]), 3), round(fl(r["quality_loss"]), 5),
          round(fl(r["quality_vs_static_pct"]), 3), int(r["n_promotions"]) + int(r["n_demotions"]),
          round(fl(r["warm_attention_mass_pct"]), 2)] for r in rows("sensitivity_sweep.csv")]

damping = [dict(dp=r["damp_promotions"] == "True", warmup=int(r["warmup"]), dwell=int(r["dwell"]),
                e=round(fl(r["energy_vs_static_pct"]), 3), l=round(fl(r["latency_vs_static_pct"]), 3),
                q=round(fl(r["quality_vs_static_pct"]), 3), share=round(fl(r["reconfig_energy_share_pct"]), 2),
                moves=round(fl(r["moves"]), 1)) for r in rows("damping_sweep.csv")]

ts = [dict(step=int(r["step"]), warm=fl(r["warm_attention_mass"]), promo=int(r["n_promotions"]),
           demo=int(r["n_demotions"])) for r in rows("dynamic_hybrid_timeseries.csv")]

real = [dict(trace=r["trace"], split=r["split"], policy=r["policy"], energy=fl(r["energy_pj"]),
             latency=fl(r["latency_ns"]), quality=fl(r["quality_loss"]), warm=fl(r["warm_mass_pct"]),
             promos=int(r["promos"]), demos=int(r["demos"]), e=fl(r["energy_vs_static_pct"]),
             q=fl(r["quality_vs_static_pct"])) for r in rows("real_trace_results.csv")]

data = dict(policies=policies, sweep=sweep, damping=damping, ts=ts, real=real)
html = (Path(__file__).parent / "template.html").read_text()
html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
(Path(__file__).parent / "index.html").write_text(html)
print("wrote index.html", len(html), "bytes")
