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
