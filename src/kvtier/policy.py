"""Placement policies: decide which tier a KV entry lives in.

Milestone 2 policies are static — they set each entry's tier once, at
creation, and never revisit it. Milestone 3 adds ``DynamicHybridPolicy``,
which uses ``update()`` every decode step to promote/demote entries based
on their live importance (see ``monitor.py``).
"""

from __future__ import annotations

from typing import Protocol

from kvtier.kv_entry import Modality, Tier, TokenKV


class PlacementPolicy(Protocol):
    """Decides tier placement for KV entries."""

    name: str

    def initial_tier(self, entry: TokenKV) -> Tier:
        """Tier assigned to ``entry`` at the moment it is created."""
        ...

    def update(self, entry: TokenKV) -> Tier:
        """Tier ``entry`` should be in right now, given its live state.

        Called once per existing entry at the end of every decode step.
        Static policies simply return the entry's current tier.
        """
        ...


class FlatPolicy:
    """Baseline: every entry lives in HOT for its whole lifetime."""

    name = "flat"

    def initial_tier(self, entry: TokenKV) -> Tier:
        return Tier.HOT

    def update(self, entry: TokenKV) -> Tier:
        return entry.tier


class StaticModalityPolicy:
    """Baseline: VISUAL entries start (and stay) in WARM, TEXT in HOT."""

    name = "static_modality"

    def initial_tier(self, entry: TokenKV) -> Tier:
        return Tier.WARM if entry.modality == Modality.VISUAL else Tier.HOT

    def update(self, entry: TokenKV) -> Tier:
        return entry.tier
