from pathlib import Path

from kvtier.kv_entry import Modality, Tier, TokenKV
from kvtier.tiers import (
    TIERS,
    WARM_ENERGY_RATIO,
    WARM_LATENCY_RATIO,
    get_tier,
    load_tier_config,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _entry() -> TokenKV:
    return TokenKV(position=0, modality=Modality.VISUAL, size_bytes=256.0, tier=Tier.HOT)


def test_warm_energy_is_lower_than_hot():
    entry = _entry()
    hot_latency, hot_energy = get_tier(Tier.HOT).access_cost(entry)
    warm_latency, warm_energy = get_tier(Tier.WARM).access_cost(entry)
    assert warm_energy < hot_energy


def test_warm_latency_is_higher_than_hot():
    entry = _entry()
    hot_latency, _ = get_tier(Tier.HOT).access_cost(entry)
    warm_latency, _ = get_tier(Tier.WARM).access_cost(entry)
    assert warm_latency > hot_latency


def test_warm_effective_size_is_quarter_of_hot():
    entry = _entry()
    hot_size = get_tier(Tier.HOT).effective_size_bytes(entry)
    warm_size = get_tier(Tier.WARM).effective_size_bytes(entry)
    assert warm_size == hot_size * 0.25


def test_warm_per_byte_energy_ratio_matches_constant():
    hot = get_tier(Tier.HOT)
    warm = get_tier(Tier.WARM)
    assert warm.energy_per_byte_pj == hot.energy_per_byte_pj * WARM_ENERGY_RATIO
    assert warm.base_energy_pj == hot.base_energy_pj * WARM_ENERGY_RATIO


def test_warm_per_byte_latency_ratio_matches_constant():
    hot = get_tier(Tier.HOT)
    warm = get_tier(Tier.WARM)
    assert warm.latency_per_byte_ns == hot.latency_per_byte_ns * WARM_LATENCY_RATIO
    assert warm.base_latency_ns == hot.base_latency_ns * WARM_LATENCY_RATIO


def test_load_tier_config_matches_builtin_defaults():
    loaded = load_tier_config(CONFIG_PATH)
    for tier in (Tier.HOT, Tier.WARM):
        assert loaded[tier] == TIERS[tier]
