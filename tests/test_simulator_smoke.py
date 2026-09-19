import pytest

from kvtier.kv_entry import Tier
from kvtier.metrics import combine_summaries
from kvtier.policy import DynamicHybridPolicy, FlatPolicy, StaticModalityPolicy
from kvtier.simulator import run_session
from kvtier.trace import generate_session


def _small_session(**overrides):
    cfg = dict(
        n_text_tokens=5,
        n_visual_tokens=20,
        n_decode_steps=8,
        salient_visual_fraction=0.1,
        seed=0,
    )
    cfg.update(overrides)
    return generate_session(**cfg)


def test_run_produces_one_step_metric_per_decode_step():
    session = _small_session()
    metrics = run_session(session, FlatPolicy())
    assert len(metrics.steps) == len(session.steps)


def test_flat_policy_never_touches_warm():
    session = _small_session()
    metrics = run_session(session, FlatPolicy())
    assert metrics.total_latency_by_tier()[Tier.WARM] == 0.0
    assert metrics.total_energy_by_tier()[Tier.WARM] == 0.0
    assert metrics.final_footprint_bytes()[Tier.WARM] == 0.0


def test_static_modality_policy_uses_both_tiers():
    session = _small_session()
    metrics = run_session(session, StaticModalityPolicy())
    footprint = metrics.final_footprint_bytes()
    assert footprint[Tier.HOT] > 0.0
    assert footprint[Tier.WARM] > 0.0


def test_static_modality_costs_less_energy_than_flat():
    # Same session (fresh copies, same seed) under each policy: pushing visual
    # entries to WARM should only ever reduce total energy versus all-HOT,
    # since WARM's per-byte energy is strictly lower.
    flat_metrics = run_session(_small_session(), FlatPolicy())
    static_metrics = run_session(_small_session(), StaticModalityPolicy())
    assert static_metrics.total_energy_pj < flat_metrics.total_energy_pj


def test_totals_are_nonnegative():
    session = _small_session()
    metrics = run_session(session, FlatPolicy())
    assert metrics.total_latency_ns > 0.0
    assert metrics.total_energy_pj > 0.0


def test_combine_summaries_has_one_row_per_policy():
    session_a = _small_session()
    session_b = _small_session()
    metrics_a = run_session(session_a, FlatPolicy())
    metrics_b = run_session(session_b, StaticModalityPolicy())
    df = combine_summaries([metrics_a, metrics_b])
    assert list(df["policy"]) == ["flat", "static_modality"]


def test_dynamic_hybrid_reset_prevents_future_leakage():
    # A DynamicHybridPolicy's very first decision (before any live
    # attention has been observed) must not already reflect the full
    # session's importance baked in by trace.py's own EMA update.
    session = _small_session(n_decode_steps=20)
    policy = DynamicHybridPolicy(promote_threshold=0.5, demote_threshold=0.0)
    metrics = run_session(session, policy)
    # threshold of 0.5 is unreachable by a single step's EMA update, so no
    # promotions should occur at all if importance genuinely starts at 0.
    assert metrics.total_promotions == 0


def test_dynamic_hybrid_produces_some_promotions_with_realistic_thresholds():
    session = _small_session(n_decode_steps=40, n_visual_tokens=60, salient_visual_fraction=0.1)
    policy = DynamicHybridPolicy(promote_threshold=0.02, demote_threshold=0.005)
    metrics = run_session(session, policy, ema_decay=0.9)
    assert metrics.total_promotions > 0


def test_dynamic_hybrid_charges_reconfig_cost_as_separate_line_item():
    session = _small_session(n_decode_steps=40, n_visual_tokens=60, salient_visual_fraction=0.1)
    policy = DynamicHybridPolicy(promote_threshold=0.02, demote_threshold=0.005)
    metrics = run_session(session, policy, ema_decay=0.9)
    assert metrics.total_reconfig_latency_ns > 0.0
    assert metrics.total_reconfig_energy_pj > 0.0
    # Reconfig cost must be included in the headline totals, not silently folded away.
    assert metrics.total_latency_ns >= metrics.total_reconfig_latency_ns
    assert metrics.total_energy_pj >= metrics.total_reconfig_energy_pj


def test_dynamic_hybrid_tier_changes_tracked_on_entries():
    session = _small_session(n_decode_steps=40, n_visual_tokens=60, salient_visual_fraction=0.1)
    policy = DynamicHybridPolicy(promote_threshold=0.02, demote_threshold=0.005)
    metrics = run_session(session, policy, ema_decay=0.9)
    total_tier_changes = sum(e.tier_changes for e in session.entries)
    assert total_tier_changes == metrics.total_promotions + metrics.total_demotions
