"""SimPy-driven simulation of a decode session under a placement policy.

Each decode step in the trace performs one attention lookup per active KV
entry — this mirrors ``trace.py``'s ``DecodeStep.attention_weights``, which
covers every position active at that step. This module walks the trace
step by step inside a SimPy process: for each step it charges every active
entry's current tier for that lookup (via ``TierResource.access_cost``),
advances the simulated clock by the step's total latency, then asks the
policy where each entry should live next. For Milestone 2's static
policies that last hook is a no-op; Milestone 3's ``DynamicHybridPolicy``
uses it to promote/demote entries based on live importance.

Prefill is not charged an access cost: the synthetic trace (Milestone 1)
only models attention distributions during decode, so prefill is treated
purely as where entries are created and initially placed.
"""

from __future__ import annotations

import simpy

from kvtier.kv_entry import Tier, TokenKV
from kvtier.metrics import SessionMetrics, StepMetrics
from kvtier.policy import PlacementPolicy
from kvtier.tiers import TierResource, get_tier
from kvtier.trace import TraceSession


def run_session(
    session: TraceSession,
    policy: PlacementPolicy,
    tiers: dict[Tier, TierResource] | None = None,
) -> SessionMetrics:
    """Simulate one trace session under ``policy``, returning aggregated metrics."""
    tiers = tiers if tiers is not None else {t: get_tier(t) for t in Tier}
    env = simpy.Environment()
    metrics = SessionMetrics(policy_name=policy.name)

    n_prefill = len(session.entries) - len(session.steps)
    prefill_entries = session.entries[:n_prefill]
    decode_entries = session.entries[n_prefill:]

    active: dict[int, TokenKV] = {}
    for entry in prefill_entries:
        entry.tier = policy.initial_tier(entry)
        active[entry.position] = entry

    def process():
        for step, new_entry in zip(session.steps, decode_entries):
            assert new_entry.position == step.new_entry_position

            new_entry.tier = policy.initial_tier(new_entry)
            active[new_entry.position] = new_entry

            step_latency = 0.0
            step_energy = 0.0
            tier_latency = {t: 0.0 for t in Tier}
            tier_energy = {t: 0.0 for t in Tier}
            tier_access_counts = {t: 0 for t in Tier}

            for pos in step.attention_weights:
                entry = active[pos]
                latency, energy = tiers[entry.tier].access_cost(entry)
                step_latency += latency
                step_energy += energy
                tier_latency[entry.tier] += latency
                tier_energy[entry.tier] += energy
                tier_access_counts[entry.tier] += 1

            yield env.timeout(step_latency)

            for entry in active.values():
                entry.tier = policy.update(entry)

            tier_footprint = {t: 0.0 for t in Tier}
            for entry in active.values():
                tier_footprint[entry.tier] += tiers[entry.tier].effective_size_bytes(entry)

            metrics.steps.append(
                StepMetrics(
                    step=step.step,
                    total_latency_ns=step_latency,
                    total_energy_pj=step_energy,
                    tier_latency_ns=tier_latency,
                    tier_energy_pj=tier_energy,
                    tier_footprint_bytes=tier_footprint,
                    tier_access_counts=tier_access_counts,
                )
            )

    env.process(process())
    env.run()
    return metrics
