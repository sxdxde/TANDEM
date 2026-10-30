# TANDEM — joint data and computation placement for VLM KV caches

**Course project, Computer Architecture — Sudarshan Sudhakar, IIITDM Kancheepuram**

> **TANDEM** = **T**iered, **A**daptive, **N**ear-memory **D**ata-and-compute placement for **E**fficient **M**ultimodal KV caches.
> The system makes two runtime decisions *in tandem*: where each KV entry **lives** (data placement) and where each read is **computed** (computation placement).

The paper is in [`paper/main.pdf`](paper/main.pdf) (21 pages, key points and contributions on page 1). The lab notebook with every decision, deviation and failed hypothesis is [`NOTES.md`](NOTES.md).

---

## 1. What this project is

The assigned topic is *Memory-Centric Computing Architecture: data-local execution and intelligent computation placement* — an architecture that decides, at run time, where computation should happen from the **location, movement, reuse and characteristics of data**, evaluated in a simulation in which memory, compute resources, data movement, placement and runtime decisions explicitly interact, including the **feasibility and overhead** of dynamic placement for **reusable, streaming and randomly accessed** data.

We instantiate this on a concrete, data-intensive workload: the **key–value (KV) cache of a vision-language model (VLM)**. A VLM expands an image into hundreds of visual tokens; every generated token re-reads their KV entries in every layer, so the cache dominates memory traffic during decode.

The memory has two tiers:

| Tier | Storage | Compute | Properties (stylised ratios, see §6) |
|---|---|---|---|
| **HOT** | FP16 DRAM/HBM | host | lossless reference |
| **WARM** | INT4 compute-in-memory (CIM) arrays | in memory, or spilled to host | 10× lower energy per byte, 1.5× slower per access, lossy (INT4) |

A runtime watches live importance and reuse and makes two decisions:

1. **Data placement** — which tier each (token, layer) entry occupies. Changing it is a *migration* that costs time, energy and bus traffic.
2. **Computation placement** — whether a read of WARM data runs in memory or is pulled to the host (a scheduler that sees queue state).

### How the task statement maps onto the work

| Requirement | Where it is addressed |
|---|---|
| Decide placement from location, movement, reuse, data characteristics | Location = tier; movement = bytes on a shared bus (reads, spills, partial sums, migrations); reuse = observed read frequency; characteristics = attention importance and INT4 error tolerance |
| Compute near the data when beneficial, conventional compute otherwise | CIM units vs host lanes, adaptive spill scheduler (`src/kvtier/system.py`, experiment E8) |
| Memory, compute, movement, placement and runtime decisions interacting | SimPy model with finite bus channels, host lanes, CIM units, work packets, per-step barrier, migration traffic sharing the bus |
| Feasibility and overhead of dynamic placement | Bottom-up migration cost, break-even sweep, imperfect importance estimators, contention (E5, E6, E8, E9b) |
| Reusable / streaming / random data | Three access regimes on real attention (E9) |

---

## 2. Headline results

All numbers come from the CSVs in `results/` (per-simulation rows and summaries). Intervals are paired-bootstrap 95% CIs in the paper.

| # | Result |
|---|---|
| E1 | On a *synthetic* trace, dynamic placement trades energy for quality against static modality tiering. |
| E2 | **Negative:** that trade-off does **not** replicate on real attention (visual tokens get only ~6% of attention). |
| E3–E4 | Per-(token, layer) entries with dense reads recover a win; a cross-validated layer/role-aware *static* baseline cannot match it, so it is per-entry information, **not** a layer prior (a falsification test). |
| E5 | 42 images, SmolVLM-256M and 500M, leave-one-trace-out: dynamic placement lowers energy **−22.3% / −23.4%** vs static modality at neutral quality (dense reads, 10× migration cost). Thresholds transfer between sizes; break-even ≈ 50× migration cost; **no win under sparse (top-10%) reads**. |
| E6 | Robust to noisy, stale or sampled importance (−18.6 to −21.8%), **not** to one score per token (−7.6%). |
| E7 | With INT4 applied inside the real model, dynamic placement changes the next-token distribution **21.8% less** than static tiering and **9.4× less** than random placement of the same size. |
| E8 | Under contention, tiering moves **11.8× fewer bus bytes** at **0.64× the makespan** of a flat memory (the earlier serial-model "+34% latency" was an artefact). Always computing in memory is up to **5× slower** than flat with 2 CIM units; an **adaptive spill scheduler beats both** fixed choices. Bottom-up migration cost ≈ **1.05 HOT accesses** (vs the assumed 10×). |
| E9 | Reusable data: **−27.4%** energy. Streamed data: policy is inert (no entry is read twice, so nothing repays a move). Random access: up to **+82% energy** at 10× migration cost unless migrations are priced against observed reuse; the reuse gate leaves **+1 to +2%** and **misses our pre-specified criterion**. Break-even ≈ 45× the bottom-up cost. |

**What did not work (reported in the paper, not hidden):** the synthetic-trace win (E2); the layer-prior explanation (E4); one-score-per-token importance (E6); two pre-specified E9 criteria — H9c and the second half of H9d (scorecard in `NOTES.md` and paper Table 8). A minimum-age variant of the reuse gate was added *after* seeing those failures and is labelled post hoc — it is a diagnosis, not a headline result.

---

## 3. Repository layout

```
.
├── README.md                this file
├── NOTES.md                 lab notebook: chronological, with pre-specified hypotheses and scorecards
├── pyproject.toml           package metadata and dependencies
├── configs/default.yaml     tier costs, trace generation, policy and reconfiguration parameters
├── src/kvtier/              the simulator package
│   ├── kv_entry.py          KV entry (modality, tier, reads, age, reuse_rate)
│   ├── trace.py             synthetic prefill+decode trace with attention weights (E1)
│   ├── real_trace.py        real attention -> trace session (layer-averaged or per-layer)
│   ├── tiers.py             HOT/WARM per-access latency, energy, quantisation-error model
│   ├── policy.py            Flat, StaticModality, DynamicHybrid, LayerModality, ReuseAwareDynamic
│   ├── monitor.py           live per-entry importance EMA (no look-ahead)
│   ├── reconfig.py          lump-sum promotion/demotion cost (access-cost model)
│   ├── simulator.py         access-cost SimPy simulator (E1–E7)
│   ├── system.py            system model: bus, host lanes, CIM units, scheduler, regimes (E8–E9)
│   ├── estimators.py        imperfect importance signals (E6)
│   ├── profile.py           group importance profiles for layer/role-aware static policies (E4)
│   └── metrics.py           energy / latency / quality / footprint aggregation
├── experiments/             one script per experiment (see §5)
├── tests/                   pytest suite, 77 tests
├── data/manifest.csv        the 42-image set (names, kinds, prompts); photos are third-party and re-downloadable
├── traces/                  real attention traces (*.npz) — see the note in §4
├── results/                 all CSVs and plots, including the 3,402-row `m9_system.csv`
├── report/                  HTML report source for the early milestones (M1–M6b)
├── logs/                    run logs (not tracked in git)
└── paper/
    ├── main.tex, sec_*.tex, app_*.tex, fig_*.tex   the paper (LaTeX / TikZ)
    ├── make_figures.py, make_tables.py             regenerate figures and appendix tables from results/
    ├── figs/                                       figure PDFs
    └── main.pdf                                    the built paper
```

---

## 4. Setup

Python ≥ 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # simulator, analysis, tests
pip install -e ".[vlm]"          # only to re-extract attention from the VLM (torch, transformers, pillow)
```

Run the tests (they cover the simulator, policies, estimators, and — importantly — that `system.py` reproduces the original simulator exactly in a degenerate configuration):

```bash
pytest                           # 77 passed
```

**About the traces.** Real attention traces (`traces/s256/`, `traces/s500/`, ~230 MB) are large and are *not* in the git repository (see `.gitignore`); the submission zip includes them. If you cloned from git, either copy them in or regenerate them (needs the `vlm` extra, the SmolVLM weights from Hugging Face, and a GPU/MPS for speed):

```bash
python experiments/make_dataset.py                  # downloads the 42 images listed in data/manifest.csv
python experiments/extract_many.py                  # SmolVLM-256M traces -> traces/s256/
python experiments/extract_many.py s500 HuggingFaceTB/SmolVLM-500M-Instruct mps float32   # 500M -> traces/s500/
```

---

## 5. Reproducing each experiment

Commands are run from the repository root. Simulation scripts are resumable.

| Experiment | What it tests | Command |
|---|---|---|
| **E1** synthetic trace | dynamic vs static vs flat, with reconfiguration cost | `python experiments/validate_trace.py`, `run_flat_baseline.py`, `run_static_modality.py`, `run_dynamic_hybrid.py`, `run_sensitivity.py`, `make_plots.py` |
| **E2** real attention | does E1 replicate? | `python experiments/extract_vlm_trace.py`, `run_real_trace.py` |
| **E3** per-layer entries | sparse reads, measured INT4 error | `python experiments/extract_vlm_layer_trace.py`, `run_m6.py`, `measure_int4_error.py` |
| **E4** falsification | is it just a layer prior? | `python experiments/run_layer_aware.py` |
| **E5** scale-up | 42 images, two model sizes, leave-one-out CV | `python experiments/run_m7.py s256` · `run_m7.py s500` · `analyze_m7.py` · `run_m7_mechanism.py` |
| **E6** non-oracle importance | noisy / stale / sampled / per-token signals | `python experiments/run_m7_estimators.py` · `analyze_m8.py` |
| **E7** real-output check | INT4 applied in the model, KL of next-token distribution | `python experiments/validate_real_output.py 42 mps` |
| **E8–E9** system model | compute placement under contention; reuse regimes | `python experiments/run_m9.py e8 e9 e9b e9c` · `analyze_m9.py` (3,402 simulations, ≈25 min on 7 cores) |
| **Paper** | figures, tables, PDF | `python paper/make_figures.py && python paper/make_tables.py && cd paper && pdflatex main.tex && pdflatex main.tex` |

`experiments/run_m7_part2.sh` and `run_overnight.sh` chain the long runs end to end.

E8–E9 only need `traces/s256/` and the config; they do not need the `vlm` extra. To re-check the E8–E9 numbers without re-simulating, `python experiments/analyze_m9.py` regenerates `results/m9_e8_summary.csv`, `m9_e9_summary.csv` and `m9_e9b_summary.csv` from `results/m9_system.csv`.

---

## 6. Modelling choices and honest limitations

**Cost model.** All tier costs come from one stylised table (`configs/default.yaml`, paper Table "tiers"): WARM has ~10× lower energy per byte, ~1.5× higher per-access latency, and 0.5 vs 2 bytes per element. They are ratios in the range of commonly cited CIM/near-memory figures, **not a specific device**, so absolute energies are not hardware claims. What the study supports is *ratios between policies under one shared cost model* and how those ratios move when the assumptions move.

**Two simulators, on purpose.**
- *Access-cost model* (`simulator.py`, E1–E7): the tier implies the compute site, latencies are summed serially, a migration is a lump-sum multiple of a HOT access.
- *System model* (`system.py`, E8–E9): finite bus channels, host lanes and CIM units; 64-entry work packets; a per-step barrier; migration traffic shares the bus and uses a host lane; schedulers `tier` / `host` / `adaptive`. With `SystemConfig.legacy()` it reduces to the first simulator to 1e-9 relative error (unit test), so everything new is an addition, not a re-parameterisation.

**Quality.** Quality loss is an attention-weighted INT4 error proxy; E7 checks it against the real model's next-token distribution.

**Evaluation discipline.** Leave-one-trace-out cross-validation, paired-bootstrap CIs, and pass criteria written in `NOTES.md` before the corresponding runs (self-imposed, not an external registration). E8–E9 hypotheses H8a–H9e were written after the runs were launched and before any result was inspected. Fixed thresholds (8× / 0.4× of uniform attention) were chosen in E5 and never re-tuned for E8–E9.

**Limitations.**
- No real hardware or CIM data; the migration cost is derived bottom-up from the *same* stylised table, so it cannot independently corroborate the other assumptions.
- Narrow workload: one model family (SmolVLM-256M and 500M), 42 images, short sessions.
- The streaming and random regimes are **overlays on real attention traces**, not natively streamed or randomly accessed workloads.
- Dynamic placement is **not regret-free** under random access, even when migration is cheap.
- The reference list in the paper was written from memory and has not been checked against the sources.

---

## 7. Development history

| Milestone | Content | Experiment |
|---|---|---|
| M1 | Synthetic trace and tier cost model | E1 |
| M2 | SimPy simulator, flat and static-modality baselines | E1 |
| M3–M4 | Dynamic hybrid policy with reconfiguration cost; quality penalty; sensitivity sweeps | E1 |
| M5 | Real attention traces from SmolVLM-256M | E2 |
| M6 / M6b | Per-(token, layer) entries, sparse reads, measured INT4 error; layer-aware static baseline | E3, E4 |
| M7 | Scale-up to 42 images, 256M + 500M, leave-one-trace-out, bootstrap CIs | E5 |
| M8 | Non-oracle importance; real-output validation; first paper | E6, E7 |
| M9 | System model with compute resources, data movement and scheduler; three access regimes; reframed as TANDEM | E8, E9 |

`git log` has one commit per milestone; `NOTES.md` records every observation and deviation in order.
