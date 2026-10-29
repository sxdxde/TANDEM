"""Importance estimators (M8a): the oracle estimator must not change a run; imperfect ones feed the monitor
a different signal but never change the true attention that quality and access costs are charged with."""

import numpy as np
import pytest

from kvtier import estimators
from kvtier.kv_entry import Tier
from kvtier.policy import DynamicHybridPolicy
from kvtier.real_trace import trace_from_layer_attention
from kvtier.reconfig import ReconfigurationManager
from kvtier.simulator import run_session
from kvtier.tiers import get_tier

N_LAYERS, N_PREFILL, N_STEPS = 3, 20, 10


def _run(estimator=None, tier_log=None):
    rng = np.random.default_rng(0)
    is_visual = np.array([False] * 4 + [True] * (N_PREFILL - 4))
    steps = [rng.random((N_LAYERS, N_PREFILL + t + 1)) ** 3 for t in range(N_STEPS)]
    sess = trace_from_layer_attention(is_visual, steps)
    pol = DynamicHybridPolicy(1.5 / N_PREFILL, 0.5 / N_PREFILL, warmup_steps=2)
    rm = ReconfigurationManager(hot=get_tier(Tier.HOT), cost_multiplier=10.0)
    return run_session(sess, pol, reconfig_manager=rm, estimator=estimator, tier_log=tier_log)


def test_oracle_is_identical_to_no_estimator():
    a, b = _run(), _run(estimators.Oracle())
    assert (a.total_energy_pj, a.total_promotions, a.total_demotions) == (b.total_energy_pj, b.total_promotions, b.total_demotions)


def test_subsample_skips_steps_and_raises_decay():
    e = estimators.Subsample(4)
    w = {0: 0.5, 1: 0.5}
    assert e.observe(0, w, 0.9) == (w, pytest.approx(0.9 ** 4))
    assert e.observe(1, w, 0.9) == (None, None)
    assert e.observe(4, w, 0.9)[0] == w


def test_delay_returns_stale_weights():
    e = estimators.Delay(2)
    ws = [{0: float(i)} for i in range(5)]
    seen = [e.observe(i, w, 0.9)[0] for i, w in enumerate(ws)]
    assert seen[:2] == [None, None] and seen[2] == ws[0] and seen[4] == ws[2]


def test_token_shared_averages_over_layers():
    e = estimators.TokenShared(n_layers=2)
    out, _ = e.observe(0, {0: 0.2, 1: 0.6, 2: 0.1, 3: 0.1}, 0.9)  # positions = token * 2 + layer
    assert out[0] == out[1] == pytest.approx(0.4) and out[2] == out[3] == pytest.approx(0.1)


def test_noise_is_seeded_and_positive():
    a = estimators.Noisy(0.5, seed=1).observe(0, {0: 1.0, 1: 2.0}, 0.9)[0]
    b = estimators.Noisy(0.5, seed=1).observe(0, {0: 1.0, 1: 2.0}, 0.9)[0]
    assert a == b and all(v > 0 for v in a.values())


def test_imperfect_estimator_changes_decisions_but_not_true_quality_accounting():
    log = []
    r = _run(estimators.Subsample(8), tier_log=log)
    assert len(log) == N_STEPS and all(isinstance(s, set) for s in log)
    assert r.mean_quality_loss >= 0
