from kvtier.kv_entry import Modality, Tier, TokenKV
from kvtier.policy import FlatPolicy, StaticModalityPolicy


def _entry(modality: Modality) -> TokenKV:
    return TokenKV(position=0, modality=modality, size_bytes=256.0, tier=Tier.HOT)


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
