"""Reconfiguration cost is a pure additive multiplier: policy decisions never see cost,
so energy/latency at multiplier m == access + m * (reconfig at 1x). run_m7.py relies on this."""

import numpy as np
import pytest

from kvtier.kv_entry import Tier
from kvtier.policy import DynamicHybridPolicy
from kvtier.real_trace import trace_from_layer_attention
from kvtier.reconfig import ReconfigurationManager
from kvtier.simulator import SparseAccess, run_session
from kvtier.tiers import get_tier


def _session(seed=0, n_prefill=20, n_steps=8, n_layers=3):
    rng = np.random.default_rng(seed)
    is_visual = np.array([False] * 4 + [True] * (n_prefill - 4))
    steps = [rng.random((n_layers, n_prefill + t + 1)) ** 3 for t in range(n_steps)]
    return is_visual, steps


@pytest.mark.parametrize("sparse", [None, SparseAccess(top_k_fraction=0.3)])
def test_cost_sweep_is_linear(sparse):
    def run(mult):
        sess = trace_from_layer_attention(*_session())
        pol = DynamicHybridPolicy(1.5 / 20, 0.5 / 20, warmup_steps=2)
        rm = ReconfigurationManager(hot=get_tier(Tier.HOT), cost_multiplier=mult)
        return run_session(sess, pol, reconfig_manager=rm, sparse=sparse)

    r1, r10 = run(1.0), run(10.0)
    assert r1.total_promotions + r1.total_demotions > 0
    assert (r1.total_promotions, r1.total_demotions) == (r10.total_promotions, r10.total_demotions)
    access = r1.total_energy_pj - r1.total_reconfig_energy_pj
    assert r10.total_energy_pj == pytest.approx(access + 10 * r1.total_reconfig_energy_pj)
    access_l = r1.total_latency_ns - r1.total_reconfig_latency_ns
    assert r10.total_latency_ns == pytest.approx(access_l + 10 * r1.total_reconfig_latency_ns)
    assert r1.mean_quality_loss == pytest.approx(r10.mean_quality_loss)
