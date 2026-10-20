import numpy as np
import pytest

from kvtier.kv_entry import Modality
from kvtier.policy import FlatPolicy
from kvtier.real_trace import load_trace, save_trace, trace_from_attention
from kvtier.simulator import run_session


def _inputs(n_prefill=6, n_steps=4, seed=0):
    rng = np.random.default_rng(seed)
    is_visual = np.array([False, False] + [True] * (n_prefill - 2))
    steps = [rng.random(n_prefill + t + 1) for t in range(n_steps)]
    return is_visual, steps


def test_session_shape_matches_simulator_expectations():
    is_visual, steps = _inputs()
    s = trace_from_attention(is_visual, steps)
    assert len(s.entries) == len(is_visual) + len(steps)
    assert [e.modality for e in s.entries[:3]] == [Modality.TEXT, Modality.TEXT, Modality.VISUAL]
    for t, st in enumerate(s.steps):
        assert st.new_entry_position == len(is_visual) + t
        assert len(st.attention_weights) == len(is_visual) + t + 1
        assert sum(st.attention_weights.values()) == pytest.approx(1.0)


def test_wrong_length_rejected():
    is_visual, steps = _inputs()
    steps[1] = steps[1][:-1]
    with pytest.raises(ValueError):
        trace_from_attention(is_visual, steps)


def test_roundtrip_and_simulates(tmp_path):
    is_visual, steps = _inputs()
    save_trace(tmp_path / "t.npz", is_visual, steps)
    s = load_trace(tmp_path / "t.npz")
    assert len(s.steps) == len(steps)
    assert len(run_session(s, FlatPolicy()).steps) == len(steps)
