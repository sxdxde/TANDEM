"""Reconfiguration cost model for promotions/demotions between tiers.

A tier change is not free: the entry has to be read out of its source
tier, requantized/dequantized to the destination tier's precision, and
written into the destination tier. That is qualitatively different work
from a normal attention lookup (which only reads), so we model it as a
fixed multiple of a single HOT access's cost — both latency and energy —
rather than deriving it from the two tiers' per-access costs directly.

``RECONFIG_COST_MULTIPLIER = 10.0``: a reconfiguration costs about as much
as ten ordinary HOT accesses, reflecting that it is a few HOT-access-
equivalents of work (read + requantize + write), not a trivial pointer
update. Applied uniformly regardless of direction (promotion or demotion)
since both require the same read-requantize-write sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from kvtier.kv_entry import Tier, TokenKV
from kvtier.tiers import TierResource, get_tier

RECONFIG_COST_MULTIPLIER: float = 10.0


@dataclass(frozen=True)
class ReconfigurationManager:
    """Charges a latency+energy penalty for every tier change."""

    hot: TierResource
    cost_multiplier: float = RECONFIG_COST_MULTIPLIER

    def reconfig_cost(self, entry: TokenKV) -> tuple[float, float]:
        """Return ``(latency_ns, energy_pj)`` charged for one tier change of ``entry``."""
        latency, energy = self.hot.access_cost(entry)
        return latency * self.cost_multiplier, energy * self.cost_multiplier

    def apply(self, entry: TokenKV, new_tier: Tier) -> tuple[float, float]:
        """Move ``entry`` to ``new_tier``, charging and recording the cost.

        Returns ``(0.0, 0.0)`` without touching the entry if it is already
        in ``new_tier``.
        """
        if new_tier == entry.tier:
            return 0.0, 0.0
        latency, energy = self.reconfig_cost(entry)
        entry.tier = new_tier
        entry.tier_changes += 1
        return latency, energy


def default_reconfig_manager() -> ReconfigurationManager:
    return ReconfigurationManager(hot=get_tier(Tier.HOT))


def load_reconfig_config(path: str | Path) -> ReconfigurationManager:
    """Load the ``reconfig:`` block (see ``configs/default.yaml``), using the
    same file's ``tiers.hot`` block for the underlying HOT access cost."""
    from kvtier.tiers import load_tier_config

    with open(path) as f:
        raw = yaml.safe_load(f)

    tiers = load_tier_config(path)
    cost_multiplier = raw["reconfig"]["cost_multiplier"]
    return ReconfigurationManager(hot=tiers[Tier.HOT], cost_multiplier=cost_multiplier)
