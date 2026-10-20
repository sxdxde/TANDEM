# Lab notes

Chronological log of what was done and observed. Newest entries at the bottom.
Numbers are from `configs/default.yaml` (seed 0, 20 text + 500 visual tokens,
64 decode steps) unless stated.

## Milestones 1-3 (recap, from commit history)
- M1 synthetic trace + tier cost model; M2 SimPy simulator + flat / static_modality
  baselines; M3 dynamic hybrid policy + reconfiguration cost.
- M3 result: dynamic_hybrid *lost* to static_modality (+67% energy, +2% latency)
  from recency-driven promotion churn (82 promotions, 64 demotions).

## M4, step 1: sensitivity sweep, energy + latency only
Script: `experiments/run_sensitivity.py` (6 promote x 5 demote x 6 reconfig-cost
multipliers = 174 valid configs).

Observed:
- 87/174 configs beat static_modality on energy, but the best had 0-1 promotions:
  they only demote TEXT tokens to WARM.
- A trivial **all-WARM** policy gets -78.6% energy / +2.6% latency vs static,
  beating every dynamic config.
- Cause: in the cost model WARM is ~10x cheaper in energy and only ~1.5x slower,
  with unlimited capacity, and nothing penalises INT4. HOT has no reason to exist,
  so the energy/latency-only comparison is degenerate.
- Among configs that really churn (>=20 promotions) at 10x cost, only 3/17 beat static.

Decision (user): add a quality penalty for reading from the INT4 tier (option 1);
HOT capacity limit (option 2) left as a possible later extension.

## M4, step 2: quality penalty added
Changes:
- `TierResource.quant_error` (HOT 0.0, WARM 0.05; in `tiers.py` + `default.yaml`).
  **0.05 is an assumed order-of-magnitude INT4 relative error, not measured.**
  Should be swept / replaced with a measured value (M5 or literature).
- Simulator accrues per-step `quality_loss = sum(attention_weight x tier.quant_error)`
  and `warm_attention_mass` (fraction of attention served from WARM).
  Session-level: `mean_quality_loss`, `mean_warm_attention_mass`.
- Quality is a *reported* axis, not folded into a scalar cost; policies do not see
  it directly (they still act on the importance EMA + thresholds).
- Tests: 2 new (zero loss for flat; loss == warm_mass x warm_error). 43 pass.

Observed (reconfig-free references):

| policy | energy (pJ) | latency (ns) | quality loss | % attention from WARM |
|---|---|---|---|---|
| flat | 72.6M | 797.7k | 0.0000 | 0% |
| static_modality | 8.55M | 1066.5k | 0.0340 | 68% |
| all_warm | 1.83M | 1094.7k | 0.0500 | 100% |

- all_warm is no longer a free win: it has the worst quality. The degeneracy is gone.
- 68% of attention mass hits WARM under static_modality because the salient
  visual tokens (the ones that get attention) are placed in WARM by the modality prior.
- 24/174 configs beat static on energy AND quality, but only by ~0.27% on quality
  (the energy win, up to -28%, comes from demoting low-attention TEXT tokens to WARM).
  That is the modality prior being slightly wrong about text in this synthetic
  trace, not evidence for dynamic promotion. Treat with caution: the synthetic
  trace gives text little attention by construction.
- The real dynamic-hybrid story is on the other side of the trade-off. Promoting
  salient visual tokens cuts quality loss by up to 56% vs static. Example
  (promote 0.006 / demote 0.002): quality -44.6% at energy +35.8% (1x reconfig cost),
  +67.3% (10x), +102.4% (20x); latency -0.8% / +2.0% / +5.1%.
  So the M3 "loss" is really a trade: ~+67% energy buys ~-45% quality loss.
  Compare flat: -100% quality loss for +748% energy. Dynamic gets a much better
  quality-per-energy rate than interpolating between static and flat
  (see `results/sensitivity_pareto.png`, log y-axis).
- Quality-improvement is independent of reconfig cost (same -44.6% at every cost);
  reconfig cost only moves the energy side. At 20x the +102% energy for -45% quality
  is a much worse deal.

## M4, step 3: comparison bars + tier timeline
Script: `experiments/make_plots.py` -> `results/comparison_bars.png`,
`tier_timeline.png`, `milestone4_comparison.csv` (default config, 10x reconfig).

| policy | energy (pJ) | latency (ns) | quality loss | % attn from WARM | promos / demos |
|---|---|---|---|---|---|
| flat | 72.6M | 797.7k | 0.0000 | 0% | 0 / 0 |
| static_modality | 8.55M | 1066.5k | 0.0340 | 68% | 0 / 0 |
| dynamic_hybrid | 14.31M | 1087.9k | 0.0188 | 38% | 82 / 64 |

Timeline observations (dynamic_hybrid):
- **Step 0 startup artifact:** 21 demotions at step 0. TEXT tokens start in HOT with
  importance EMA = 0, which is below the demote threshold, so all 20 prefill text
  tokens (+1) are demoted immediately, each paying 10x reconfig. This is wasted
  reconfig cost caused by the EMA's cold start, not a real importance signal.
  A warmup period / EMA initialisation fix would remove it.
- **Steps 4-11 promotion burst** (~50 promotions): salient visual tokens crossing the
  promote threshold as their EMA warms up. This is the useful work; it drops the WARM
  attention share from ~97% to ~36% by step 12 (static stays ~60-70%).
- **Steady state:** low-rate promote/demote churn (0-3 moves/step), more demotions
  late in the run (steps 50-63). Mostly recency-boundary entries, consistent with the
  M3 churn explanation.
- Dynamic's WARM attention share ends ~25% vs ~57% for static; gap is stable after step 12.

## M4b: damping the dynamic policy (cold start + churn)
Goal: remove the wasted reconfig cost seen in the step-0 burst and the boundary churn.
Changes (all default to the old behaviour, so M3 results are reproducible):
- `DynamicHybridPolicy(warmup_steps, min_dwell_steps, damp_promotions)`.
  Warmup = entry can't move until its EMA has N updates; dwell = can't move for N steps
  after creation/last move. `TokenKV` gained `observed_steps`, `steps_since_move`,
  `raw_importance`; `ImportanceMonitor(bias_correct=...)`; `run_session(bias_correct_ema=...)`.
- Script `experiments/run_damping_sweep.py` (6 warmup x 6 dwell x both damp_promotions,
  mean of 5 trace seeds) -> `results/damping_sweep.csv`.

Results (default thresholds 0.006/0.002, 10x reconfig, mean over 5 seeds, vs static):

| variant | energy | quality loss | reconfig share of energy | tier moves |
|---|---|---|---|---|
| undamped (M3/M4) | +64.8% | -46.1% | 19.1% | 131 |
| warmup 2 (chosen) | +55.8% | -49.3% | 13.0% | 85 |
| dwell 20 | +49.5% | -37.9% | 13.0% | 81 |

Observations:
- **Warmup 2 strictly dominates undamped** on energy and quality, and has the best
  quality-gain-per-energy ratio in the grid. Chosen as the new default
  (`warmup_steps: 2`, `min_dwell_steps: 0`).
- **Deviation from my pre-set rule.** I had fixed "lowest energy subject to >=20% quality
  gain" before the run. That rule drifts to the edge of the quality budget (it picked
  dwell 20, which gives away 11 points of quality gain just to do less). I switched to
  gain-per-energy after seeing this, because warmup 2 dominates undamped outright. This was
  a post-hoc choice and is recorded as such.
- **Dwell 20 batches every move into one burst at step 19**: steady-state churn vanishes
  but the first 20 steps serve 70-97% of attention from WARM. Slow reaction.
- **Warmup ending the step-0 burst: no, it only moves it to step 1.** 20 TEXT demotions still
  happen at step 1 because at decay 0.9 the EMA needs ~20 steps to converge, so text tokens
  look unimportant at 2 updates. These are mostly *correct* demotions in this trace (text
  gets low attention) arriving early, not pure noise. Remaining cost is real reconfig energy.
- Warmup is redundant once dwell >= warmup (dwell counts from creation).
- `damp_promotions=False` (damp demotions only) gave identical results at dwell 0 and no
  benefit elsewhere; kept as an option, default True.
- **Bias-corrected EMA tried (Adam-style): neutral, not adopted.** Median over a
  threshold x warmup grid: energy +78.8% / quality -61.1% / efficiency 0.76, vs raw EMA
  +55.9% / -49.3% / 0.86. It shifts the operating point (more promotions) rather than
  improving the frontier, and the thresholds were calibrated on the raw EMA. Left in as
  `bias_correct=False` by default. Would need recalibrated thresholds to judge fairly.
- Remaining energy overhead (+55%) is mostly the inherent cost of reading promoted
  salient tokens from HOT (reconfig is only 13% of dynamic energy). So the policy is
  now close to "pays energy to buy quality" with little pure waste in this trace.
- Re-ran with the new default (seed 0): dynamic_hybrid = 13.29M pJ, 1073.1k ns, quality
  0.0174 (-48.8% vs static), 52 promotions / 33 demotions. At 1x / 10x / 20x reconfig cost
  the energy overhead is +37.0% / +55.4% / +75.8% with identical -48.8% quality.
  Sensitivity sweep: 48/174 configs beat static on both energy and quality
  (best -22.7% energy at 10x, again via TEXT demotion).
- Note: best efficiency in the threshold sweep is promote 0.008 / demote 0.001
  (+35% energy for -38% quality), i.e. thresholds still matter more than damping.
  Not adopted as default: efficiency is a heuristic, and the right operating point depends on
  how the user weighs energy against quality.

## M5: real VLM attention traces (SmolVLM-256M-Instruct)
Setup: `experiments/extract_vlm_trace.py` (optional deps: `pip install -e .[vlm]`; torch 2.14,
transformers 5.20, CPU float32, eager attention) runs the model on 3 locally generated
images (chart / scene / noise+label), greedy-decodes 64 tokens, and records each decode
step's attention row averaged over all 30 layers x 9 heads. `src/kvtier/real_trace.py`
(torch-free, tested) converts them into a `TraceSession`; saved in `traces/*.npz`.
Alignment: last prompt token = decode step 0's query, so prefill = first L-1 positions.
The model answered correctly on all three images (bar chart / triangle+square / "Hello"),
so the traces are from a model that is really using the image.

Trace statistics (layer/head-averaged; ~876 prefill entries, 832 of them visual = 95%):

| quantity | chart | scene | noise_label |
|---|---|---|---|
| attention mass on visual tokens | 6% | 7% | 5% |
| attention sink (position 0) | 35% | 34% | 34% |
| mass on generated tokens | 36% | 37% | 37% |
| top 8% of visual tokens' share of visual mass | 63% | 70% | 61% |
| top-8% set overlap, first vs last step | 74% | 83% | 77% |
| median visual token's EMA vs uniform (1/N) | 0.02x | 0.02x | 0.02x |

Observations:
- **Heavy-tailed, persistent visual salience holds** on real data (the synthetic premise).
- **But visual tokens get far less attention than the synthetic trace assumed**: ~6% of mass
  vs 68% of mass served from WARM under static_modality in the synthetic trace. The synthetic
  trace badly overstated how much attention the WARM tier would serve.
- A huge attention sink (~35% at position 0) and recency (generated tokens ~36%) dominate.
  Position 0 is TEXT, so the modality prior keeps it HOT: correct for free.
- **Synthetic-calibrated thresholds don't transfer**: uniform attention is ~0.0011 and the
  99th-percentile visual EMA is ~0.0008, below the 0.006 promote threshold, so nothing visual
  is ever promoted. With them, dynamic only demotes text: -23..-27% energy, +37..+42%
  quality loss vs static on every real trace. Thresholds had to be re-expressed as
  multiples of uniform attention (1/n_prefill).
- **Pre-registered protocol** (`experiments/run_real_trace.py`): pick thresholds on chart +
  scene (lowest energy with quality loss <= static), test on held-out noise_label.
  Chosen: promote 2.0x / demote 0.4x uniform. Train mean: -6.6% energy, -2.5% quality loss.
  **Held-out: -14.4% energy but +11.9% quality loss (0.0027 -> 0.0031). The no-worse-quality
  criterion FAILED on the held-out trace.** Train per-trace was also mixed (chart +5.1% quality
  loss, scene -10.0%).
- Absolute stakes are small: static_modality's quality loss is only ~0.003 (floor 0, WARM-only
  0.05). Static modality placement already captures nearly everything on real traces:
  flat -> static is -88..-90% energy for ~0.003 quality loss; dynamic moves energy by only
  about +-10% around that.
- Bottom line: **on real SmolVLM traces the synthetic M4 result does not replicate**. The
  "+55% energy buys -49% quality loss" trade existed because the synthetic trace put heavy
  attention on WARM-resident salient visual tokens. With real attention, the dynamic policy's
  benefit is at best marginal and not robust on a held-out trace (n=3 traces, 1 model).

Per-layer check (chart, 16 steps; scratch analysis, not saved as a script): the
layer-averaged 6% hides large variation. Visual attention mass per layer ranges from ~3% to
39% (layer 0: 39%, up to 70% for the strongest head; layers 21-24 ~ 20%). The sink is ~0% in
layers 0-1 and 11, 40-78% in layers 12-30. So the single-entry-per-token abstraction (one
averaged weight per token) is a real modelling limitation: real KV caches are per-layer/head,
and per-layer placement could have more to gain than this averaged view shows.

## M6: per-layer entries, sparse-attention cost model, measured INT4 error
Code: `DecodeStep.extra_new_positions` (multi-entry steps), `SparseAccess` in
`simulator.py`, `trace_from_layer_attention` / `load_layer_trace` in `real_trace.py`,
`experiments/extract_vlm_layer_trace.py` (head-averaged per-layer attention, `traces/*_layers.npz`),
`experiments/run_m6.py`, `experiments/measure_int4_error.py`. 55 tests pass.

**Sparse access model.** Each step reads only the top-10% of active entries by live importance
EMA plus entries created in the last step; step 0 is dense; skipped attention mass is reported
and costs no energy. Assumption: the importance estimate sees true weights at zero cost (an
upper bound on a real approximate scorer). Skipped mass is placement-independent (tested).
**Per-layer entries.** One entry per (token, layer); each layer's attention renormalised and
divided by n_layers so mass per step is 1. Entry size matches single-entry tokens, so absolute
energies are ~30x the layer-averaged ones: compare ratios only. Because quality is linear in
attention weights, *static* policies give identical quality in both views (0.0031 on chart);
per-layer entries only matter for what the dynamic policy can do.

**Measured INT4 error** (`results/int4_error.csv`, SmolVLM prefill KV, per-token min-max INT4):
relative L2 error K 0.102 / V 0.081 at group size 32; K 0.134 / V 0.094 at group 64; K per-channel
variant not computed (guard skipped it for non-multiple-of-64 token counts). Visual vs text tokens
identical within ~1%; across layers 0.086-0.096. **The assumed 0.05 was ~2x too low** (measured
~0.09 at g32). Not changed in `default.yaml`: quality loss is linear in the error, so every
quality ratio vs static is unchanged and only absolute quality values scale (x~1.8); kept at
0.05 so committed results stay reproducible. Any absolute quality number should be read x~1.8.

Protocol (same as M5, fixed in advance): thresholds in multiples of uniform attention; choose on
chart+scene = lowest energy with mean quality <= static; test on held-out noise_label.
Results (`results/m6_results.csv`, 408 simulations, `results/m6_sweep.csv`):

| mode | flat/static energy | skipped mass | chosen thresholds | held-out energy vs static | held-out quality vs static |
|---|---|---|---|---|---|
| avg, dense (=M5) | 9.3x | 0% | 2.0x / 0.4x | -14.4% | +11.9% |
| avg, sparse10 | 1.6x | 3.2-3.6% | 8.0x / 0.02x | +0.2% (no moves) | 0% |
| per-layer, dense | 9.3x | 0% | 4.0x / 0.4x | -20.0% | +0.8% |
| per-layer, sparse10 | 1.9-2.0x | 2.1-2.3% | 8.0x / 0.2x | +3.7% | -0.2% |

Per-trace per-layer dense: energy -20.6 / -19.4 / -20.0%, quality -2.0 / -10.3 / +0.8%
(chart / scene / noise_label), with 1.2-1.4k tier moves per session.

Observations:
- **Sparse access shrinks the value of tiering a lot but not to zero**: flat/static energy falls
  from ~9.3x to 1.6-2.0x at 10% top-k, with only 2-3.6% of attention mass skipped. Exploratory
  (unsaved ad-hoc run, chart, layer-averaged): 50% top-k 5.4x, 20% 2.5x, 10% 1.6x, 5% 1.4x.
- **Layer-averaged + sparse: the dynamic policy has nothing to do** (chosen thresholds never
  trigger; 0-1 moves). The static modality prior is already as good as it gets.
- **Per-layer + dense gives the first real-trace dynamic win**: ~-20% energy on all three traces,
  quality roughly neutral. The held-out quality is +0.8%, so it technically misses the strict
  "no worse than static" rule by a small margin (stated plainly rather than rounded away).
- **Mechanism (arithmetic, not separately instrumented):** in the static per-layer baseline the
  ~1.3k HOT text entries (44 tokens x 30 layers) cost ~170M of ~380M pJ; the dynamic policy saves
  ~78M, about half of that. WARM-mass share barely moves (6.2 -> 6.1%). So the saving comes from
  demoting text entries in layers where text is unimportant (the per-layer view exposes
  this), with a few promotions of visual entries in layers that attend them; it is NOT mainly
  from rescuing salient visual tokens. This is the same text-demotion channel seen on the
  synthetic trace.
- **Under per-layer + sparse the benefit disappears or reverses** (+3.7..+7.5% energy, quality
  -0.2..-7.8%): with sparse reads, HOT text entries are rarely read, so demoting them saves
  little while ~1k moves still pay reconfiguration cost.
- Overall: the dynamic policy's value on real traces is conditional on dense attention reads and
  per-layer granularity, small (about -20% on top of static's ~-90% vs flat), and needs ~1k
  tier moves per session; with sparse reads it is not worth it at the default 10x reconfig cost.
  Caveats unchanged: 3 images, one 256M model, one held-out trace, selection on 2 traces.

## Open questions / caveats
- Synthetic M4 conclusions did not replicate on real traces (see M5); treat the M4
  quality-vs-energy numbers as a property of the synthetic trace, not of VLMs.
- Real-trace evidence is thin: 1 model (256M), 3 synthetic images, 64 decode steps, n=3
  traces, a single held-out trace. Natural photos/documents and larger models untested.
- WARM quant error 0.05 in the config is below the measured ~0.09 (M6); ratios vs static are
  unaffected, absolute quality values are ~1.8x too small.
- Latency is a serialized total-work figure (see `metrics.py`), not wall-clock.
- Next candidates (per-layer, sparse model and INT4 measurement done in M6): per-head entries;
  natural images / documents and a larger VLM; sweep reconfig cost and top-k fraction on real
  traces; a layer-aware static prior (modality x layer) as a stronger baseline than
  static_modality, since the per-layer dynamic win looks like it comes from a layer effect;
  a real approximate importance estimator for sparse access; oracle top-k comparison (M5 plan).
- No quality-aware policy yet. The policy never sees quality; a promotion rule that
  targets attention-weighted loss directly, or a warmup / recency-damping fix for the
  boundary churn, is the natural next step.
- Churn/cold-start fixes: done in M4b (partially; see above). Quality-aware policy still open.
- Step-3 timeline/table numbers above are for the *undamped* M3 policy; M4b re-ran with warmup 2.

## Artifacts
- `results/sensitivity_sweep.csv`, `results/sensitivity_references.csv`
- `results/milestone4_comparison.csv`, `comparison_bars.png`, `tier_timeline.png`
- `results/m6_results.csv`, `m6_sweep.csv`, `int4_error.csv`, `traces/*_layers.npz`
- `traces/{chart,scene,noise_label}.npz`, `results/real_trace_results.csv`, `results/real_trace_sweep.csv`
- `results/damping_sweep.csv`
- `results/sensitivity_pareto.png`, `sensitivity_reconfig_cost.png`, `sensitivity_thresholds.png`
