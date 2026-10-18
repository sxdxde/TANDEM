"""SimPy-driven simulation of a decode session under a placement policy.

Each decode step in the trace performs one attention lookup per active KV
entry — this mirrors ``trace.py``'s ``DecodeStep.attention_weights``, which
covers every position active at that step. This module walks the trace
step by step inside a SimPy process. Each step:

1. Charges every active entry's *current* tier for that step's attention
   lookup, and accrues quality loss (attention weight x the tier's
   ``quant_error``) (via ``TierResource.access_cost``) and advances the simulated
   clock by the total access latency.
2. Updates every entry's live importance from this step's attention
   weights (``monitor.ImportanceMonitor`` — see that module for why this
   is a separate live signal from ``trace.py``'s baked-in final importance).
3. Asks the policy where each entry should live now. For Milestone 2's
   static policies this is a no-op; Milestone 3's ``DynamicHybridPolicy``
   may request a tier change, which is charged via ``ReconfigurationManager``
   and advances the clock a second time.

Prefill is not charged an access cost: the synthetic trace (Milestone 1)
only models attention distributions during decode, so prefill is treated
purely as where entries are created and initially placed.
"""

from __future__ import annotations

import simpy

from kvtier.kv_entry import Tier, TokenKV
from kvtier.metrics import SessionMetrics, StepMetrics
from kvtier.monitor import ImportanceMonitor
from kvtier.policy import PlacementPolicy
from kvtier.reconfig import ReconfigurationManager
from kvtier.tiers import TierResource, get_tier
from kvtier.trace import TraceSession

DEFAULT_EMA_DECAY: float = 0.9


def run_session(
    session: TraceSession,
    policy: PlacementPolicy,
    tiers: dict[Tier, TierResource] | None = None,
    reconfig_manager: ReconfigurationManager | None = None,
    ema_decay: float = DEFAULT_EMA_DECAY,
) -> SessionMetrics:
    """Simulate one trace session under ``policy``, returning aggregated metrics."""
    tiers = tiers if tiers is not None else {t: get_tier(t) for t in Tier}
    reconfig_manager = reconfig_manager or ReconfigurationManager(hot=tiers[Tier.HOT])
    monitor = ImportanceMonitor(ema_decay=ema_decay)
    env = simpy.Environment()
    metrics = SessionMetrics(policy_name=policy.name)

    monitor.reset(session.entries)

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
            quality_loss = 0.0
            warm_mass = 0.0

            for pos, weight in step.attention_weights.items():
                entry = active[pos]
                quality_loss += weight * tiers[entry.tier].quant_error
                if entry.tier == Tier.WARM:
                    warm_mass += weight
                latency, energy = tiers[entry.tier].access_cost(entry)
                step_latency += latency
                step_energy += energy
                tier_latency[entry.tier] += latency
                tier_energy[entry.tier] += energy
                tier_access_counts[entry.tier] += 1

            yield env.timeout(step_latency)

            monitor.update(active, step.attention_weights)

            reconfig_latency = 0.0
            reconfig_energy = 0.0
            n_promotions = 0
            n_demotions = 0
            for entry in active.values():
                desired_tier = policy.update(entry)
                if desired_tier == entry.tier:
                    continue
                was_warm = entry.tier == Tier.WARM
                latency, energy = reconfig_manager.apply(entry, desired_tier)
                reconfig_latency += latency
                reconfig_energy += energy
                if was_warm and desired_tier == Tier.HOT:
                    n_promotions += 1
                else:
                    n_demotions += 1

            if reconfig_latency:
                yield env.timeout(reconfig_latency)

            tier_footprint = {t: 0.0 for t in Tier}
            for entry in active.values():
                tier_footprint[entry.tier] += tiers[entry.tier].effective_size_bytes(entry)

            metrics.steps.append(
                StepMetrics(
                    step=step.step,
                    total_latency_ns=step_latency + reconfig_latency,
                    total_energy_pj=step_energy + reconfig_energy,
                    tier_latency_ns=tier_latency,
                    tier_energy_pj=tier_energy,
                    tier_footprint_bytes=tier_footprint,
                    tier_access_counts=tier_access_counts,
                    reconfig_latency_ns=reconfig_latency,
                    reconfig_energy_pj=reconfig_energy,
                    n_promotions=n_promotions,
                    n_demotions=n_demotions,
                    quality_loss=quality_loss,
                    warm_attention_mass=warm_mass,
                )
            )

    env.process(process())
    env.run()
    return metrics
