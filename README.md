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

Milestones 3–5 (dynamic hybrid policy + reconfiguration cost,
comparison/sensitivity plots, real-VLM attention traces) are not yet
implemented — see the milestone plan below.

## Repository structure

```
kvtier-memcentric/
  README.md
  pyproject.toml
  configs/
    default.yaml          # tier costs, trace generation parameters
  src/kvtier/
    kv_entry.py             # TokenKV dataclass, Modality enum, Tier enum
    trace.py                # synthetic prefill+decode trace with attention weights
    tiers.py                # HOT/WARM per-access latency + energy cost model
    policy.py               # FlatPolicy, StaticModalityPolicy; (M3) DynamicHybridPolicy
    simulator.py            # SimPy environment driving a trace through a policy
    metrics.py              # energy/latency/footprint aggregation + comparison tables
    monitor.py              # (M3) per-entry importance tracker
    reconfig.py             # (M3) promotion/demotion cost model
    viz.py                  # (M4) timeline + comparison plots
  experiments/
    validate_trace.py       # Milestone 1 checkpoint
    run_flat_baseline.py    # Milestone 2 checkpoint
    run_static_modality.py  # Milestone 2 checkpoint
    run_dynamic_hybrid.py   # (M3)
    compare_all.py          # (M4)
  tests/
  results/
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

## Milestone plan

1. **Synthetic trace + tier cost model** (no simulator) — done.
2. **SimPy simulator + flat and static-modality baselines** — done.
3. **Dynamic hybrid policy + reconfiguration cost**, with promotion/demotion
   counts and reconfiguration overhead broken out as their own metric.
4. **Comparison, visualization, sensitivity** — bar charts of total
   energy/latency across the three policies, a tier-over-time timeline plot,
   and a sensitivity sweep over thresholds and reconfiguration penalty
   showing how the dynamic hybrid's advantage shrinks as reconfiguration
   cost grows.
5. **(Stretch) Real VLM attention traces** — replace the synthetic trace with
   real attention weights hooked from a small open VLM, plus an oracle
   comparison against ground-truth top-k attended tokens.
# KV-Cache-Tier
