from kvtier.kv_entry import Modality, Tier, TokenKV
from kvtier.monitor import ImportanceMonitor


def _entry(position: int) -> TokenKV:
    return TokenKV(position=position, modality=Modality.TEXT, size_bytes=256.0, tier=Tier.HOT)


def test_reset_zeroes_importance():
    entries = [_entry(0), _entry(1)]
    entries[0].importance = 0.5
    entries[1].importance = 0.9
    ImportanceMonitor(ema_decay=0.9).reset(entries)
    assert entries[0].importance == 0.0
    assert entries[1].importance == 0.0


def test_update_applies_ema_from_this_steps_weights_only():
    entry = _entry(0)
    active = {0: entry}
    monitor = ImportanceMonitor(ema_decay=0.5)

    monitor.update(active, {0: 1.0})
    assert entry.importance == 0.5  # 0.5*0.0 + 0.5*1.0

    monitor.update(active, {0: 1.0})
    assert entry.importance == 0.75  # 0.5*0.5 + 0.5*1.0


def test_update_ignores_entries_not_in_this_steps_weights():
    entry_a = _entry(0)
    entry_b = _entry(1)
    active = {0: entry_a, 1: entry_b}
    monitor = ImportanceMonitor(ema_decay=0.5)

    monitor.update(active, {0: 1.0})  # entry_b receives no attention this step
    assert entry_a.importance == 0.5
    assert entry_b.importance == 0.0


def test_bias_correction_recovers_constant_signal_immediately():
    entry = _entry(0)
    monitor = ImportanceMonitor(ema_decay=0.9, bias_correct=True)
    monitor.reset([entry])
    for _ in range(3):
        monitor.update({0: entry}, {0: 0.01})
    assert entry.importance == pytest.approx(0.01)
    assert entry.raw_importance < 0.01  # raw EMA is still biased toward 0
    assert entry.observed_steps == 3


def test_observed_steps_counts_updates():
    entry = _entry(0)
    monitor = ImportanceMonitor(ema_decay=0.9)
    monitor.reset([entry])
    monitor.update({0: entry}, {0: 0.01})
    monitor.update({0: entry}, {0: 0.01})
    assert entry.observed_steps == 2
