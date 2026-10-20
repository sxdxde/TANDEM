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

## Open questions / caveats
- The synthetic trace is the only workload; text gets very little attention. Results
  on the text-demotion side likely do not transfer to real VLMs (M5).
- WARM quant error 0.05 is an assumption; the energy/quality trade-off scales with it.
- Latency is a serialized total-work figure (see `metrics.py`), not wall-clock.
- No quality-aware policy yet. The policy never sees quality; a promotion rule that
  targets attention-weighted loss directly, or a warmup / recency-damping fix for the
  boundary churn, is the natural next step.
- Churn/cold-start fixes: done in M4b (partially; see above). Quality-aware policy still open.
- Step-3 timeline/table numbers above are for the *undamped* M3 policy; M4b re-ran with warmup 2.

## Artifacts
- `results/sensitivity_sweep.csv`, `results/sensitivity_references.csv`
- `results/milestone4_comparison.csv`, `comparison_bars.png`, `tier_timeline.png`
- `results/damping_sweep.csv`
- `results/sensitivity_pareto.png`, `sensitivity_reconfig_cost.png`, `sensitivity_thresholds.png`
