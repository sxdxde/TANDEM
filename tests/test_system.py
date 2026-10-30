"""System-level simulator (Milestone 9): reduces to the legacy simulator, and its new mechanisms behave."""

from __future__ import annotations

import numpy as np
import pytest

from kvtier.kv_entry import Tier
from kvtier.policy import DynamicHybridPolicy, FlatPolicy, ReuseAwareDynamicPolicy, StaticModalityPolicy
from kvtier.real_trace import trace_from_layer_attention
from kvtier.reconfig import ReconfigurationManager
from kvtier.simulator import run_session
from kvtier.system import AccessRegime, SystemConfig, breakeven_accesses, run_system_session
from kvtier.tiers import TIERS


def make_session(seed=0, n_tok=40, n_layers=3, steps=12):
    rng = np.random.default_rng(seed)
    is_visual = np.arange(n_tok) < 30
    ws = []
    for t in range(steps):
        w = rng.dirichlet(np.full(n_tok + t + 1, 0.3), size=n_layers)
        ws.append(w)
    return trace_from_layer_attention(is_visual, ws)


def dyn(n_tok=40, n_layers=3):
    u = 1.0 / (n_tok * n_layers)
    return DynamicHybridPolicy(4 * u, 0.4 * u, warmup_steps=2)


@pytest.mark.parametrize("policy_fn", [FlatPolicy, StaticModalityPolicy, dyn])
def test_legacy_config_reproduces_run_session(policy_fn):
    legacy = run_session(make_session(), policy_fn(), reconfig_manager=ReconfigurationManager(hot=TIERS[Tier.HOT]))
    sysr = run_system_session(make_session(), policy_fn(), cfg=SystemConfig.legacy(10.0))
    assert sysr.energy_pj == pytest.approx(legacy.total_energy_pj, rel=1e-9)
    assert sysr.latency_ns == pytest.approx(legacy.total_latency_ns, rel=1e-9)
    assert sysr.promotions + sysr.demotions == legacy.total_promotions + legacy.total_demotions
    assert sysr.quality == pytest.approx(legacy.mean_quality_loss, rel=1e-9)


def test_parallel_resources_never_slower_than_serial():
    s = StaticModalityPolicy
    serial = run_system_session(make_session(), s(), cfg=SystemConfig(serial=True))
    parallel = run_system_session(make_session(), s(), cfg=SystemConfig())
    assert parallel.latency_ns < serial.latency_ns
    assert parallel.energy_pj == pytest.approx(serial.energy_pj)  # energy does not depend on contention


def test_cim_moves_less_data_than_host_compute():
    cim = run_system_session(make_session(), StaticModalityPolicy(), cfg=SystemConfig(scheduler="tier"))
    host = run_system_session(make_session(), StaticModalityPolicy(), cfg=SystemConfig(scheduler="host"))
    assert cim.bus_bytes < host.bus_bytes
    assert cim.energy_pj < host.energy_pj


def test_adaptive_spills_only_when_cim_is_scarce():
    few = run_system_session(make_session(), StaticModalityPolicy(), cfg=SystemConfig(scheduler="adaptive", cim_units=1))
    many = run_system_session(make_session(), StaticModalityPolicy(), cfg=SystemConfig(scheduler="adaptive", cim_units=256))
    tier = run_system_session(make_session(), StaticModalityPolicy(), cfg=SystemConfig(scheduler="tier", cim_units=1))
    assert few.spilled_packets > 0 and many.spilled_packets == 0
    assert few.latency_ns <= tier.latency_ns  # spilling must not make the saturated CIM case slower


def test_regimes_read_different_amounts():
    dense = run_system_session(make_session(), FlatPolicy(), regime=AccessRegime("dense"))
    rand = run_system_session(make_session(), FlatPolicy(), regime=AccessRegime("random", 0.25))
    stream = run_system_session(make_session(), FlatPolicy(), regime=AccessRegime("stream"))
    assert rand.reads < 0.5 * dense.reads and stream.reads < 0.5 * dense.reads
    assert dense.skipped_mass == 0 and rand.skipped_mass > 0 and stream.skipped_mass > 0


def test_explicit_reconfig_is_cheap_relative_to_ten_hot_accesses():
    from kvtier.system import move_cost_in_hot_accesses

    entry = make_session().entries[0]
    m = move_cost_in_hot_accesses(SystemConfig(), TIERS, entry)
    assert 1.0 < m < 2.0
    assert breakeven_accesses(SystemConfig(reconfig_scale=10), TIERS, entry) > breakeven_accesses(SystemConfig(), TIERS, entry)


def test_reuse_gate_is_identity_on_dense_reads_and_blocks_unrecoverable_moves():
    u = 1.0 / (40 * 3)
    gated = lambda m: ReuseAwareDynamicPolicy(4 * u, 0.4 * u, warmup_steps=2, min_expected_accesses=m)  # noqa: E731
    d = run_system_session(make_session(), dyn(), cfg=SystemConfig())
    g = run_system_session(make_session(), gated(2.0), cfg=SystemConfig())
    assert (g.promotions, g.demotions) == (d.promotions, d.demotions)  # reuse rate is 1 under dense reads
    rd = run_system_session(make_session(), dyn(), cfg=SystemConfig(), regime=AccessRegime("random", 0.5))
    rg = run_system_session(make_session(), gated(1000.0), cfg=SystemConfig(), regime=AccessRegime("random", 0.5))
    assert rd.promotions + rd.demotions > 0
    assert rg.promotions + rg.demotions == 0


def test_stream_reads_never_pass_the_warmup_so_no_moves():
    sd = run_system_session(make_session(), dyn(), cfg=SystemConfig(), regime=AccessRegime("stream"))
    assert sd.promotions + sd.demotions == 0
