import pytest

from kvtier.kv_entry import Modality, Tier, TokenKV
from kvtier.policy import DynamicHybridPolicy, FlatPolicy, StaticModalityPolicy


def _entry(modality: Modality, tier: Tier = Tier.HOT, importance: float = 0.0) -> TokenKV:
    return TokenKV(
        position=0, modality=modality, size_bytes=256.0, tier=tier, importance=importance
    )


def test_flat_policy_places_everything_hot():
    policy = FlatPolicy()
    assert policy.initial_tier(_entry(Modality.TEXT)) is Tier.HOT
    assert policy.initial_tier(_entry(Modality.VISUAL)) is Tier.HOT


def test_flat_policy_update_is_a_no_op():
    policy = FlatPolicy()
    entry = _entry(Modality.VISUAL)
    entry.tier = Tier.WARM  # simulate some other code having moved it
    assert policy.update(entry) is Tier.WARM


def test_static_modality_policy_places_visual_warm_text_hot():
    policy = StaticModalityPolicy()
    assert policy.initial_tier(_entry(Modality.TEXT)) is Tier.HOT
    assert policy.initial_tier(_entry(Modality.VISUAL)) is Tier.WARM


def test_static_modality_policy_update_is_a_no_op():
    policy = StaticModalityPolicy()
    entry = _entry(Modality.VISUAL)
    entry.tier = Tier.WARM
    assert policy.update(entry) is Tier.WARM


def test_dynamic_hybrid_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        DynamicHybridPolicy(promote_threshold=0.001, demote_threshold=0.006)


def test_dynamic_hybrid_initial_tier_matches_modality_prior():
    policy = DynamicHybridPolicy(promote_threshold=0.006, demote_threshold=0.002)
    assert policy.initial_tier(_entry(Modality.TEXT)) is Tier.HOT
    assert policy.initial_tier(_entry(Modality.VISUAL)) is Tier.WARM


def test_dynamic_hybrid_promotes_warm_entry_above_threshold():
    policy = DynamicHybridPolicy(promote_threshold=0.006, demote_threshold=0.002)
    entry = _entry(Modality.VISUAL, tier=Tier.WARM, importance=0.01)
    assert policy.update(entry) is Tier.HOT


def test_dynamic_hybrid_demotes_hot_entry_below_threshold():
    policy = DynamicHybridPolicy(promote_threshold=0.006, demote_threshold=0.002)
    entry = _entry(Modality.TEXT, tier=Tier.HOT, importance=0.001)
    assert policy.update(entry) is Tier.WARM


def test_dynamic_hybrid_holds_tier_within_hysteresis_band():
    policy = DynamicHybridPolicy(promote_threshold=0.006, demote_threshold=0.002)
    warm_entry = _entry(Modality.VISUAL, tier=Tier.WARM, importance=0.004)
    hot_entry = _entry(Modality.TEXT, tier=Tier.HOT, importance=0.004)
    assert policy.update(warm_entry) is Tier.WARM
    assert policy.update(hot_entry) is Tier.HOT


def test_dynamic_hybrid_warmup_blocks_early_moves():
    policy = DynamicHybridPolicy(0.006, 0.002, warmup_steps=5)
    entry = _entry(Modality.TEXT, tier=Tier.HOT, importance=0.0)
    entry.observed_steps = 4
    assert policy.update(entry) is Tier.HOT  # cold-start, would otherwise demote
    entry.observed_steps = 5
    assert policy.update(entry) is Tier.WARM


def test_dynamic_hybrid_min_dwell_blocks_round_trips():
    policy = DynamicHybridPolicy(0.006, 0.002, min_dwell_steps=3)
    entry = _entry(Modality.VISUAL, tier=Tier.WARM, importance=0.01)
    entry.steps_since_move = 2
    assert policy.update(entry) is Tier.WARM
    entry.steps_since_move = 3
    assert policy.update(entry) is Tier.HOT


def test_dynamic_hybrid_undamped_promotions_still_demote_damped():
    policy = DynamicHybridPolicy(0.006, 0.002, min_dwell_steps=5, damp_promotions=False)
    warm = _entry(Modality.VISUAL, tier=Tier.WARM, importance=0.01)
    warm.steps_since_move = 0
    hot = _entry(Modality.TEXT, tier=Tier.HOT, importance=0.0)
    hot.steps_since_move = 0
    assert policy.update(warm) is Tier.HOT   # promotion not damped
    assert policy.update(hot) is Tier.HOT    # demotion damped
