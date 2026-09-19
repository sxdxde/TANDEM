import pytest

from kvtier.kv_entry import Modality
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


def test_entry_count_grows_by_one_per_decode_step():
    session = _small_session()
    assert len(session.entries) == 5 + 20 + 8


def test_salient_visual_positions_are_within_visual_range():
    session = _small_session()
    visual_positions = {e.position for e in session.entries if e.modality is Modality.VISUAL}
    assert session.salient_visual_positions.issubset(visual_positions)
    assert len(session.salient_visual_positions) == 2  # round(0.1 * 20)


def test_attention_weights_sum_to_one_each_step():
    session = _small_session()
    for step in session.steps:
        assert sum(step.attention_weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_importance_stays_in_unit_interval():
    session = _small_session()
    for entry in session.entries:
        assert 0.0 <= entry.importance <= 1.0


def test_generation_is_deterministic_given_seed():
    a = _small_session(seed=42)
    b = _small_session(seed=42)
    a_importances = [e.importance for e in a.entries]
    b_importances = [e.importance for e in b.entries]
    assert a_importances == b_importances
