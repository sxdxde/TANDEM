from kvtier.kv_entry import Modality, Tier, TokenKV
from kvtier.reconfig import ReconfigurationManager, default_reconfig_manager
from kvtier.tiers import get_tier


def _entry(tier: Tier = Tier.WARM) -> TokenKV:
    return TokenKV(position=0, modality=Modality.VISUAL, size_bytes=256.0, tier=tier)


def test_apply_no_op_when_tier_unchanged():
    manager = default_reconfig_manager()
    entry = _entry(Tier.WARM)
    latency, energy = manager.apply(entry, Tier.WARM)
    assert (latency, energy) == (0.0, 0.0)
    assert entry.tier_changes == 0


def test_apply_moves_entry_and_records_change():
    manager = default_reconfig_manager()
    entry = _entry(Tier.WARM)
    latency, energy = manager.apply(entry, Tier.HOT)
    assert entry.tier == Tier.HOT
    assert entry.tier_changes == 1
    assert latency > 0.0
    assert energy > 0.0


def test_apply_increments_tier_changes_across_multiple_moves():
    manager = default_reconfig_manager()
    entry = _entry(Tier.WARM)
    manager.apply(entry, Tier.HOT)
    manager.apply(entry, Tier.WARM)
    assert entry.tier_changes == 2


def test_reconfig_cost_is_multiple_of_hot_access_cost():
    hot = get_tier(Tier.HOT)
    manager = ReconfigurationManager(hot=hot, cost_multiplier=10.0)
    entry = _entry()
    hot_latency, hot_energy = hot.access_cost(entry)
    reconfig_latency, reconfig_energy = manager.reconfig_cost(entry)
    assert reconfig_latency == hot_latency * 10.0
    assert reconfig_energy == hot_energy * 10.0


def test_reconfig_cost_same_regardless_of_direction():
    manager = default_reconfig_manager()
    entry_promote = _entry(Tier.WARM)
    entry_demote = _entry(Tier.HOT)
    promote_cost = manager.reconfig_cost(entry_promote)
    demote_cost = manager.reconfig_cost(entry_demote)
    assert promote_cost == demote_cost
