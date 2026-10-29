# kvtier — Tiered Memory-Centric Architecture for Modality-Aware KV-Cache Placement

Vision-language models expand each image into hundreds or thousands of visual
tokens, causing the LLM decoder's KV cache to balloon and dominate memory
traffic during decode. A large, active body of work (TopV, AirCache,
VL-Cache, ZipVL, HybridKV, TokenCarve) shows that visual-token KV entries
carry systematically less attention importance than text-token entries on
average, and exploits this by pruning, quantizing, or merging tokens. All of
that work operates at the *algorithm* layer: it assumes a flat,
undifferentiated KV cache in ordinary GPU memory and only decides what to
discard. This project asks the *architecture* question instead: should
differently-important KV entries physically live in different memory tiers
with different compute attached, and what does it cost, in hardware terms,
to move an entry between tiers when its importance changes?

We build a two-tier memory-centric architecture — **HOT** (conventional,
high-fidelity) and **WARM** (compute-in-memory, reduced-precision) — with a
runtime importance monitor and a dynamic promotion/demotion policy that
explicitly charges a reconfiguration cost for tier changes, and compare it
against a flat (untiered) baseline and a static modality-only tiering
baseline.

## Related work

**Algorithm-layer visual-KV compression** (the gap this project fills at the
hardware layer): TopV (Yang et al., 2025), AirCache (Huang et al., 2025),
VL-Cache (Tu et al., 2024), ZipVL (He et al., 2024), HybridKV (Zeng et al.,
2026), TokenCarve (Tan et al., 2025). All show visual tokens are
on-average less important to attention and compressible with minimal
accuracy loss (VL-Cache: 10% retention, no accuracy drop).

**Hardware KV-cache accelerators** (single-tier, non-modality-aware —
adjacent but distinct): LOKI, Sangam, MATA, HiKV.

**Our contribution / gap**: no prior work treats modality-and-importance-aware
KV-cache placement as a *tiered hardware architecture problem* with an
explicit reconfiguration-cost model for promotion/demotion. This project
does.

## Status

**Milestone 1** (synthetic trace + tier cost model, no simulator yet) is
implemented: `kv_entry.py`, `trace.py`, `tiers.py`, and a validation script
that plots average importance by modality over decode, confirming the trace
generator reproduces the literature's reported pattern (visual tokens'
average importance declining while a small salient minority stays high)
before anything is built on top of it.

**Milestone 2** (SimPy simulator + flat and static-modality baselines) is
implemented: `simulator.py` walks a trace session step by step inside a
SimPy process, charging each active entry's current tier for that step's
attention lookup; `policy.py` adds the two static baselines (`FlatPolicy`,
`StaticModalityPolicy`); `metrics.py` aggregates per-step energy, latency,
and per-tier memory footprint into a comparison table. On the default
config (20 text + 500 visual prefill tokens, 64 decode steps),
`static_modality` cuts total energy by ~88% versus `flat` at the cost of
~34% higher total latency — WARM's much lower per-byte energy wins on
energy, but its higher per-access base latency and the sheer number of
visual-token accesses each step outweigh that once most of the trace's
active entries live there. See
`results/milestone2_baseline_comparison.csv`.

**Milestone 3** (dynamic hybrid policy + reconfiguration cost) is
implemented: `monitor.py` tracks each entry's importance as a *live* EMA
during simulation (reset to 0 at the start, updated once per step from
only that step's attention weights — deliberately kept separate from
`trace.py`'s full-session EMA, which would otherwise leak future decode
steps into a live policy's decisions); `reconfig.py` charges a
configurable multiple of a HOT access's cost on every tier change;
`policy.py` adds `DynamicHybridPolicy`, which starts entries at their
modality-prior tier and promotes/demotes based on live importance with
hysteresis (`promote_threshold=0.006`, `demote_threshold=0.002`,
`configs/default.yaml`).

On the default config, `dynamic_hybrid` currently **loses** to
`static_modality`: +67% more total energy and +2% more total latency,
from 82 promotions + 64 demotions (146 reconfig events at 10x a HOT
access each). The cause is visible in the per-step time series: visual
entries created late in prefill get an artificially high attention weight
in the first few decode steps purely from position-proximity (the trace's
recency bias), which pushes their live EMA above `promote_threshold`
almost immediately; as that recency term decays over the next ~20 steps,
the non-salient ones fall back below `demote_threshold` and get demoted
again. Nothing prevents an entry from round-tripping like this, and each
round trip costs 10x a HOT access. This is reported as a finding, not
patched around — it's exactly the trade-off Milestone 4's sensitivity
sweep (over thresholds and reconfiguration cost) is meant to
characterize: whether and where a dynamic hybrid's advantage over static
tiering actually survives reconfiguration overhead. See
`results/milestone3_comparison.csv`.

**Milestone 4** (comparison, visualization, sensitivity) is implemented. A
quality penalty was added (`quant_error` per tier, attention-weighted) because
an energy/latency-only comparison was degenerate: an all-WARM policy beat
everything. With quality included, `dynamic_hybrid` trades roughly +55% energy for
-49% quality loss vs `static_modality` at the default config; see `NOTES.md`
for the full log of observations and caveats (the 0.05 INT4 error is an
assumption). The dynamic policy now defaults to `warmup_steps: 2` (see the M4b
section of `NOTES.md`). Run `experiments/make_plots.py` and
`experiments/run_sensitivity.py`; outputs are in `results/`.

**Milestone 5** (real attention traces from SmolVLM-256M) is implemented
(`experiments/extract_vlm_trace.py`, `experiments/run_real_trace.py`, traces in
`traces/`). Headline: the M4 trade-off does **not** replicate on real traces.
Visual tokens receive only ~6% of attention (vs 68% of mass served from WARM
in the synthetic trace), so `static_modality` is already near the quality
floor and the dynamic policy's benefit is marginal and failed a held-out
check. See `NOTES.md` for the full analysis and caveats (3 traces, 1 model).

**Milestone 6** (per-layer entries, sparse-attention cost model, measured INT4
error) is implemented (`experiments/run_m6.py`). With one entry per
(token, layer) and dense reads, the dynamic policy saves ~20% energy vs static
at roughly neutral quality; with sparse reads that benefit disappears. Measured
INT4 error is ~0.09, not the assumed 0.05 (ratios unaffected). Details and
caveats in `NOTES.md`. A cross-validated layer/role-aware *static* baseline
(`experiments/run_layer_aware.py`) does not match the dynamic policy at equal
energy (about +16..+26% worse quality), so its advantage on real dense-read
traces comes from per-entry information, not a layer prior.

**Milestone 7** (scale-up: 42 images, SmolVLM-256M and 500M, leave-one-trace-out, bootstrap CIs) establishes the
dense-read per-layer dynamic win under a pass rule fixed in advance: -22.3% (256M) / -23.4% (500M) energy vs static
modality at neutral quality, break-even reconfiguration cost about 50x, thresholds transfer across sizes, and no win
under sparse (top-10%) reads. **Milestone 8** checks it against a non-oracle importance signal (robust to noise,
staleness and sampling, but not to one score per token) and against the real model's output with INT4 actually
applied (dynamic placement changes the next-token distribution 21.8% less than static, 9.4x less than random
placement of the same size). The write-up is in `paper/` (`main.tex`, `main.pdf`).

## Repository structure

```
kvtier-memcentric/
  README.md, NOTES.md (lab notes, chronological), pyproject.toml
  configs/default.yaml      # tier costs, trace generation, policy + reconfig parameters
  src/kvtier/
    kv_entry.py             # TokenKV dataclass, Modality enum, Tier enum
    trace.py                # synthetic prefill+decode trace with attention weights
    tiers.py                # HOT/WARM per-access latency + energy + quant-error model
    policy.py               # Flat, StaticModality, DynamicHybrid, LayerModality policies
    monitor.py              # live per-entry importance EMA (separate from trace.py's)
    reconfig.py             # promotion/demotion cost model
    simulator.py            # SimPy environment driving a trace through a policy (+ SparseAccess)
    metrics.py              # energy/latency/quality/footprint aggregation
    real_trace.py           # (M5/M6) real attention -> TraceSession, layer-averaged or per-layer
    profile.py              # (M6b) group importance profiles for layer/role-aware static policies
  experiments/
    validate_trace.py, run_flat_baseline.py, run_static_modality.py, run_dynamic_hybrid.py  # M1-M3
    run_sensitivity.py, run_damping_sweep.py, make_plots.py                                 # M4/M4b
    extract_vlm_trace.py, run_real_trace.py                                                 # M5
    extract_vlm_layer_trace.py, run_m6.py, measure_int4_error.py                            # M6
    run_layer_aware.py                                                                      # M6b
    make_dataset.py, extract_many.py, run_m7.py, analyze_m7.py, run_m7_mechanism.py         # M7
    run_m7_estimators.py, analyze_m8.py, validate_real_output.py, run_m7_part2.sh            # M8
  tests/                    # pytest (61+)
  data/                     # M7 image set (manifest.csv; COCO photos are third-party, re-downloadable)
  traces/                   # real attention traces (*.npz; M7 under traces/<model-tag>/)
  results/                  # CSVs and plots
  report/                   # HTML report source (build.py + template.html + parts/), M1-M6b
  paper/                    # research article (LaTeX), figure script, figures
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the Milestone 1 checkpoint

```bash
python experiments/validate_trace.py
```

This writes `results/importance_by_modality.png` and prints final average
importance per group (TEXT, all VISUAL, salient VISUAL subset, non-salient
VISUAL).

## Running the Milestone 2 checkpoint

```bash
python experiments/run_flat_baseline.py
python experiments/run_static_modality.py
```

Each prints a summary (total latency, total energy, per-tier breakdown,
final memory footprint) and writes its per-step time series to
`results/*_timeseries.csv`. A combined comparison table is at
`results/milestone2_baseline_comparison.csv`.

## Running the Milestone 3 checkpoint

```bash
python experiments/run_dynamic_hybrid.py
```

Prints the same summary as the baselines, plus reconfiguration overhead
(latency, energy) broken out as its own line and the promotion/demotion
counts. A combined comparison against both M2 baselines is at
`results/milestone3_comparison.csv` — see the Status section above for
what it currently shows.

## Tests

```bash
pytest
```

## Cost-model assumptions (Milestone 1)

The HOT/WARM access-cost ratios in `configs/default.yaml` /
`src/kvtier/tiers.py` are explicit, named constants rather than derived at
runtime:

- **Energy**: WARM's per-byte access energy is ~10x lower than HOT's, an
  order-of-magnitude figure consistent with commonly cited compute-in-memory
  / near-memory energy savings over conventional DRAM/HBM access + separate
  compute.
- **Latency**: WARM is ~1.5x slower per access than HOT, reflecting the
  common CIM tradeoff where analog/near-memory compute adds latency overhead
  even as it saves energy and data movement.
- **Precision**: HOT stores entries at FP16 (2 bytes/element), WARM at INT4
  (0.5 bytes/element), consistent with KV-cache quantization results in the
  literature above (e.g. HybridKV, ZipVL).

## Cost-model assumptions (Milestone 3)

`configs/default.yaml`'s `policy.dynamic_hybrid` and `reconfig` blocks,
also explicit named constants:

- **Promotion/demotion thresholds**: `promote_threshold=0.006`,
  `demote_threshold=0.002` on the live importance EMA. Chosen relative to
  Milestone 1's validation plot, where salient visual tokens converge to
  ~0.0088 and average TEXT to ~0.005-0.006 — i.e. promotion is meant to
  trigger around text-level importance or higher. The hysteresis band
  between them turned out not to be wide enough to suppress
  recency-driven promotion churn on the default trace; see the Status
  section above.
- **Reconfiguration cost multiplier**: `10.0` — a tier move (read +
  requantize + write) costs 10x a single HOT access, in both latency and
  energy, applied identically regardless of direction (`reconfig.py`).

## Milestone plan

1. **Synthetic trace + tier cost model** (no simulator) — done.
2. **SimPy simulator + flat and static-modality baselines** — done.
3. **Dynamic hybrid policy + reconfiguration cost**, with promotion/demotion
   counts and reconfiguration overhead broken out as their own metric — done.
4. **Comparison, visualization, sensitivity** — done: bar charts of total
   energy/latency across the three policies, a tier-over-time timeline plot,
   and a sensitivity sweep over thresholds and reconfiguration penalty
   showing how the dynamic hybrid's advantage shrinks as reconfiguration
   cost grows.
5. **Real VLM attention traces** (stretch) — done, see NOTES.md — replace the synthetic trace with
   real attention weights hooked from a small open VLM, plus an oracle
   comparison against ground-truth top-k attended tokens.
# KV-Cache-Tier
