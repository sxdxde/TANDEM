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

m6 = [dict(mode=r["mode"], trace=r["trace"], split=r["split"], ratio=fl(r["flat_over_static_energy"]),
           skipped=fl(r["skipped_pct"]), e=fl(r["dyn_energy_vs_static_pct"]), q=fl(r["dyn_quality_vs_static_pct"]),
           moves=int(r["moves"]), px=fl(r["chosen_promote_x"]), dx=fl(r["chosen_demote_x"]),
           warm_s=fl(r["static_warm_mass_pct"]), warm_d=fl(r["dyn_warm_mass_pct"]))
      for r in rows("m6_results.csv")]

# measured INT4 error: per-layer mean (g32), and scheme summary
from collections import defaultdict
import statistics as st
_i = rows("int4_error.csv")
_layer = defaultdict(list)
_sum = defaultdict(list)
for r in _i:
    v = (fl(r["err_visual"]) + fl(r["err_text"])) / 2
    if r["scheme"] == "tok_g32":
        _layer[(r["tensor"], int(r["layer"]))].append(v)
    _sum[(r["tensor"], r["scheme"], "visual")].append(fl(r["err_visual"]))
    _sum[(r["tensor"], r["scheme"], "text")].append(fl(r["err_text"]))
nl = 1 + max(int(r["layer"]) for r in _i)
int4 = dict(
    K=[round(st.mean(_layer[("K", l)]), 4) for l in range(nl)],
    V=[round(st.mean(_layer[("V", l)]), 4) for l in range(nl)],
    summary=[dict(t=k[0], scheme=k[1], who=k[2], err=round(st.mean(v), 4), n=len(v)) for k, v in sorted(_sum.items())],
)

# M6b: layer-aware static baseline (means over the three leave-one-trace-out folds)
_c = rows("layer_aware_theta_curve.csv")
_agg = defaultdict(list)
for r in _c:
    _agg[(r["grouping"], r["mode"], fl(r["theta"]))].append(
        (fl(r["energy_vs_static_pct"]), fl(r["quality_vs_static_pct"]), fl(r["n_hot_groups"])))
curve = [dict(g=k[0], mode=k[1], theta=k[2], e=round(st.mean(x[0] for x in v), 2),
              q=round(st.mean(x[1] for x in v), 2), hot=round(st.mean(x[2] for x in v), 1))
         for k, v in sorted(_agg.items())]
_la = rows("layer_aware_results.csv")
_dyn = defaultdict(list)
_pol = defaultdict(list)
for r in _la:
    if r["policy"] == "dynamic (CV)":
        _dyn[r["mode"]].append((fl(r["energy_vs_static_pct"]), fl(r["quality_vs_static_pct"]), int(r["moves"])))
    elif r["policy"].startswith(("modality_layer", "role_layer")):
        _pol[(r["grouping"], r["mode"], r["policy"])].append(
            (fl(r["energy_vs_static_pct"]), fl(r["quality_vs_static_pct"]), r["theta"]))
dyn = {m: dict(e=round(st.mean(x[0] for x in v), 2), q=round(st.mean(x[1] for x in v), 2),
               moves=round(st.mean(x[2] for x in v)), folds=[dict(e=round(x[0], 2), q=round(x[1], 2), moves=x[2]) for x in v])
       for m, v in _dyn.items()}
la_sel = [dict(g=k[0], mode=k[1], policy=k[2], e=round(st.mean(x[0] for x in v), 2),
               q=round(st.mean(x[1] for x in v), 2), theta=v[0][2]) for k, v in sorted(_pol.items())]

data = dict(policies=policies, sweep=sweep, damping=damping, ts=ts, real=real, m6=m6, int4=int4,
            la=dict(curve=curve, dyn=dyn, sel=la_sel))
html = (Path(__file__).parent / "template.html").read_text()
html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
(Path(__file__).parent / "index.html").write_text(html)
print("wrote index.html", len(html), "bytes")
