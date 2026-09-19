from kvtier.kv_entry import Modality, Tier, TokenKV


def test_modality_and_tier_values_are_distinct():
    assert Modality.TEXT != Modality.VISUAL
    assert Tier.HOT != Tier.WARM


def test_token_kv_defaults():
    entry = TokenKV(position=0, modality=Modality.TEXT, size_bytes=256.0, tier=Tier.HOT)
    assert entry.importance == 0.0
    assert entry.tier_changes == 0


def test_token_kv_fields_are_mutable():
    entry = TokenKV(position=3, modality=Modality.VISUAL, size_bytes=256.0, tier=Tier.HOT)
    entry.tier = Tier.WARM
    entry.tier_changes += 1
    entry.importance = 0.42
    assert entry.tier is Tier.WARM
    assert entry.tier_changes == 1
    assert entry.importance == 0.42
