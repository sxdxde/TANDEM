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

## M6b: layer-aware static baseline (tests the "it's a layer effect" hypothesis)
Hypothesis (mine, after M6): the per-layer dynamic win is a layer effect, so a static
(modality x layer) profile built offline should capture it with zero tier moves.
Code: `kvtier/profile.py` (group importance per (modality, layer) or (role, layer), role =
sink / prompt text / generated / visual), `LayerModalityPolicy`, `TokenKV.layer/.token`,
`experiments/run_layer_aware.py`. Protocol fixed in advance: leave-one-trace-out CV (profile
and theta from the other two traces, evaluate on the held-out one), same selection rule as
before, plus an in-sample "ceiling". 59 tests pass. Results: `results/layer_aware_results.csv`,
`results/layer_aware_theta_curve.csv`.

Observations (means over the three held-out folds, vs static_modality):
- **(modality, layer) grouping is degenerate**: for every theta >= 0.4 it selects exactly the 30
  TEXT groups = static_modality (0.00% / 0.00%). Text entries are few and include the sink and
  recent tokens, so every text group has mean per-entry importance >= 4x uniform. Only theta
  <= 0.2 changes anything: it promotes one more group (the layer-0 visual entries), giving
  +27.7% energy for -18.6% quality loss (dense), +23.8% / -19.2% (sparse10). The cheap,
  stable end of the profile is therefore "also keep layer-0 visual in HOT".
- **(role, layer) grouping traces a trade-off curve** (dense): theta 0.4: -5.3% energy /
  +5.0% quality loss; 0.7: -17.4% / +16.1%; 1.0: -23.1% / +26.4%; 2.0: -26.0% / +30.5%;
  4.0: -27.5% / +40.9%.
- **Hypothesis falsified.** At matched energy (~-20%) the static role x layer profile has about
  +16..+26% *worse* quality than static_modality, while the dynamic policy gets -19.8% energy at
  -3.0% quality (dense, CV, ~1.2k tier moves). The profile's in-sample ceiling equals its
  cross-validated result (at the selected theta 0.2: +27.2% energy / -18.4% quality in-sample vs +27.7% / -18.6% CV; same selected groups), so this is not
  under-fitting or leakage: the profile transfers across these images, but layer- and
  role-level information is simply too coarse.
- **Correction to the M6 mechanism note.** I wrote that the saving came from "demoting text
  entries in layers where text is unimportant". The energy arithmetic (text entries are ~45% of
  static energy; the saving is about half of that) still stands, but the *layer* part is wrong:
  no text layer group is unimportant on average. The saving comes from per-token variation
  among text entries (which individual prompt tokens matter in this prompt), which only a
  per-entry, online signal sees.
- **Net:** on real traces with dense reads the dynamic policy's advantage is real and
  specific to per-entry information, not a proxy for a layer/role prior. Under sparse reads
  it still does not pay (static role x layer at +23.8..-16.8% energy is a different, costlier
  operating point and dynamic is +5.9% energy / -3.7% quality). Caveats as before: 3 images, one
  small model, theta grid coarse (8 values), static comparators limited to two groupings.

## M7: scale-up and robustness (protocol written BEFORE running; results below it)
Motivation: M5/M6/M6b rest on n=3 images, one 256M model, one held-out trace. Claims to test:
(a) the dense-read per-layer dynamic win (~-20% energy, ~neutral quality vs static_modality),
(b) how it depends on reconfiguration cost, (c) how it depends on sparse top-k fraction,
(d) whether it transfers to a larger model.
Data: 42 images (`experiments/make_dataset.py`): 24 COCO val2017 natural photos, 10 matplotlib
charts, 8 rendered text documents. Models: SmolVLM-256M (primary), SmolVLM-500M (replication).
Traces `traces/<tag>/*_layers.npz` via `experiments/extract_many.py`; simulations via
`experiments/run_m7.py`, analysis via `experiments/analyze_m7.py` (results/m7_*).
Fixed in advance:
1. Primary protocol = leave-one-trace-out CV over all 42 traces: thresholds (promote/demote, x uniform
   attention; grid as M6) are chosen on the other 41 traces = lowest mean energy with mean quality loss
   <= static_modality's (fallback lowest quality), then evaluated on the held-out trace. Dense reads,
   10x reconfiguration cost, warmup 2. Same rule as M5/M6, no new tuning.
2. "Dynamic advantage is robust" requires ALL of: mean held-out energy saving >= 10%; mean held-out
   quality ratio vs static <= +2%; paired-bootstrap 95% CI of the energy saving excludes 0; the sign holds
   in each of the 3 image kinds. Anything weaker is reported as "not established".
3. Reconfiguration cost: multipliers 1,2,5,10,20,50,100; thresholds re-selected per cost. Break-even =
   smallest multiplier at which mean held-out energy saving <= 0. Exact because reconfig cost is additive
   (tests/test_m7_linearity.py).
4. Top-k: dense, 50, 20, 10, 5% on the reduced grid (promote {1,2,4,8}x, demote {0.1,0.4}x).
5. Cross-model: thresholds chosen on all 256M traces, applied unchanged to 500M traces (and the reverse
   is NOT done); plus 500M's own LOO-CV.
6. Cost-reduction check: min-dwell variants {0,5,20,40} on the reduced grid (moves vs energy).
Not done in M7 (stated up front): per-head entries, models other than SmolVLM, real-hardware cost
validation, an actual approximate importance estimator.

### M7 results (256M primary, 500M replication; 42 images each)
Deviation from the protocol above, made after seeing results: the first pass picked the grid corner
(promote 8x / demote 0.4x) in all 42 folds, so the grid was widened (promote up to 32x, demote up to 1.6x)
and the dense/top-10% simulations re-run for the new configs. First-pass CSVs: `results/_m7_first_pass_grid8/`.
The top-k and min-dwell sweeps stay on the reduced 8-config grid by design, so they were not redone.

Pre-registered "robust" test (energy saving >= 10%, quality <= +2%, CI excludes 0, same sign in all 3 image
kinds), dense reads, 10x reconfiguration cost, leave-one-trace-out:
| model | energy vs static | quality vs static | robust? | thresholds chosen |
|---|---|---|---|---|
| 256M (LOO-CV) | -22.3% [-22.8, -21.9] | -2.5% [-4.0, -1.1] | yes | 8/0.4 in 42/42 folds |
| 500M (LOO-CV) | -23.4% [-24.1, -22.6] | -1.8% [-4.6, +0.9] | yes | 4/0.8 in 42/42 folds |
| 256M thresholds -> 500M, unchanged | -23.6% [-24.1, -23.1] | +0.4% [-1.4, +2.1] | yes | 8/0.4 |
Widening the grid did not change the 10x dense result: 8/0.4 was a real optimum, not a search limit.
- By kind (256M): chart -22.8%, doc -20.2%, natural -22.9% energy; energy is negative on every trace.
- Quality is neutral, not improved: per-trace win rate (energy down AND quality no worse) is 76% on 256M,
  55% on 500M (36% for the transferred thresholds); 500M quality CIs include 0.
- Reconfiguration cost (thresholds re-selected per cost): 256M saving is -31.6% at 1x, -26.5% at 5x, -16.1% at 20x,
  -1.7% at 50x, +13.4% at 100x. 500M: -32.3%, -27.7%, -15.1%, -1.9%, +13.2%. Break-even is about 50x on both.
  The 10x operating point has roughly 5x margin to break-even.
- Sparse reads do NOT help: with top-10% reads the dynamic policy costs +5.7% energy on both models (quality
  -0.6%, every trace worse on energy; chosen thresholds 32/0.02 are the new grid corner, so a wider grid could
  move this, but not plausibly to a 10% saving). Reduced-grid top-k sweep (256M): k=50% -20.1%, 20% -6.0%,
  10% +7.5%, 5% +15.0%. The advantage needs dense or near-dense reads.
- Min-dwell (256M, reduced grid) only trades savings for fewer moves: dwell 0/5/20/40 -> -22.3/-19.8/-13.8/-5.9%.
- Mechanism (256M, 8/0.4): the saving comes from demotion. demote-only -24.2% energy but +6.8% quality loss;
  promote-only +1.8% energy, -7.6% quality loss; full policy -22.3% energy, -0.8% quality. Promotion buys back
  the quality that demotion gives up.
Verdict: the dense-read per-layer dynamic win replicates across two model sizes and three image kinds and
survives a >10x reconfiguration cost. It is a dense-read result only.
Not done: SmolVLM-2.2B (planned in the overnight script; dropped as low marginal value), per-head entries,
other model families, real-hardware cost validation. (Non-oracle importance and real-output validation: see M8.)
Reconfiguration cost is a modelled multiplier, not measured.

## M8: non-oracle importance (E6) and real-output validation of the quality proxy (E7)
Motivation: after M7 the two strongest objections were (1) the policy sees the true attention of every
entry for free, (2) "quality" is an attention-weighted proxy, not a measurement of the model's output.
Code: `kvtier/estimators.py` (+ `estimator=` / `tier_log=` hooks in `simulator.py`, `decay=` in `monitor.py`;
`tests/test_estimators.py`), `experiments/run_m7_estimators.py`, `experiments/analyze_m8.py`,
`experiments/validate_real_output.py`, `paper/make_figures.py`. Results: `results/m8_*`.

E6 (256M, 42 traces, dense, 10x, leave-one-out over a 10-config grid; oracle row reproduces M7 exactly):
| signal | energy vs static | quality vs static |
|---|---|---|
| oracle | -22.3% | -2.5% |
| log-normal noise sigma .25 / .5 / 1 | -22.1 / -21.6 / -21.7% | -2.9 / -4.0 / -0.8% |
| sampled every 2nd / 4th / 8th step | -20.4 / -20.6 / -18.6% | -6.0 / -0.5 / -0.6% |
| delayed 1 / 4 steps | -21.9 / -20.4% | -1.7 / -0.9% |
| one score per token (layer mean) | -7.6% [-9.0,-6.1] | -4.3% |
| token-shared + every 4th step | -10.2% | -1.0% |
| noise .5 + every 4th step | -20.2% | -1.5% |
Reading: the saving survives a noisy / stale / sampled per-entry signal (all but the token-shared rows meet the
M7 criterion); collapsing importance to one score per token loses most of it (below the 10% bar). At the oracle's
8/0.4 the token-shared signal gives -18.7% energy but +12.8% quality loss. All estimators are free here: no
energy is charged for computing them, so the oracle's saving is also the upper bound on estimator overhead.
Noise is mean-inflating log-normal (sigma 1 -> 1.65x); thresholds are re-selected per estimator.

E7 (SmolVLM-256M, 42 images, 64 teacher-forced steps; the replay reproduced the saved text on 42/42):
simulator tier schedules are replayed on the real model, WARM entries' cached K and V replaced by INT4 round
trips (g32), KL(full || policy) of the next-token distribution:
| placement | INT4 entries | KL (nats) | top-1 agreement |
|---|---|---|---|
| static modality | 91.8% | 0.00382 | 0.976 |
| dynamic (8/0.4) | 94.9% | 0.00291 | 0.977 |
| random, same INT4 count per step | 94.9% | 0.01946 | 0.930 |
| all INT4 | 100% | 0.02064 | 0.929 |
- Dynamic has 21.8% lower KL than static per image (95% CI 15.3-28.1%), lower on 37/42 images; random placement
  with the same INT4 count is 9.4x worse than dynamic (CI 7.8-11.1x), worse on all 42 images.
- Proxy vs measured KL: Spearman 0.81 over 168 points (Pearson on logs 0.88), but 0.49 within static+dynamic
  only; per-image sign of the dynamic-vs-static difference agrees on 29/42. The proxy predicts -2.5%, the
  measurement is -21.8%: the proxy is ordinally right but understates the advantage.
- Caveats: top-1 agreement does not separate dynamic from static; INT4 effects on this model are small in
  absolute terms (all-INT4 keeps 93% agreement); one model, one INT4 scheme, KL not benchmark accuracy.
Paper: `paper/main.tex` (build: `cd paper && pdflatex main.tex` twice; figures: `.venv/bin/python paper/make_figures.py`).

## Open questions / caveats
- Synthetic M4 conclusions did not replicate on real traces (see M5); treat the M4
  quality-vs-energy numbers as a property of the synthetic trace, not of VLMs.
- Real-trace evidence is thin: 1 model (256M), 3 synthetic images, 64 decode steps, n=3
  traces, a single held-out trace. Natural photos/documents and larger models untested.
- WARM quant error 0.05 in the config is below the measured ~0.09 (M6); ratios vs static are
  unaffected, absolute quality values are ~1.8x too small.
- Latency is a serialized total-work figure (see `metrics.py`), not wall-clock.
- Next candidates (per-layer, sparse model, INT4 measurement, layer-aware static baseline done):
  per-head entries; natural images / documents and a larger VLM (the main external-validity
  gap); sweep reconfig cost and top-k fraction on real traces (does dynamic survive 20x cost?);
  a cheaper dynamic policy (fewer than ~1.2k moves/session); a real approximate importance
  estimator for sparse access; oracle top-k comparison (M5 plan); using the sparse selector's
  top-k set as the tier signal (hypothesis only, untested).
- No quality-aware policy yet. The policy never sees quality; a promotion rule that
  targets attention-weighted loss directly, or a warmup / recency-damping fix for the
  boundary churn, is the natural next step.
- Churn/cold-start fixes: done in M4b (partially; see above). Quality-aware policy still open.
- Step-3 timeline/table numbers above are for the *undamped* M3 policy; M4b re-ran with warmup 2.

## Artifacts
- `results/sensitivity_sweep.csv`, `results/sensitivity_references.csv`
- `results/milestone4_comparison.csv`, `comparison_bars.png`, `tier_timeline.png`
- `results/layer_aware_results.csv`, `layer_aware_theta_curve.csv`
- `results/m6_results.csv`, `m6_sweep.csv`, `int4_error.csv`, `traces/*_layers.npz`
- `traces/{chart,scene,noise_label}.npz`, `results/real_trace_results.csv`, `results/real_trace_sweep.csv`
- `results/damping_sweep.csv`
- `results/sensitivity_pareto.png`, `sensitivity_reconfig_cost.png`, `sensitivity_thresholds.png`
