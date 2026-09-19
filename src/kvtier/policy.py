"""Placement policies: decide which tier a KV entry lives in.

Milestone 2's policies are static — they set each entry's tier once, at
creation, and never revisit it. Milestone 3's ``DynamicHybridPolicy`` uses
``update()`` every decode step to promote/demote entries based on their
live importance (see ``monitor.py``), with hysteresis (two distinct
thresholds) so an entry near the boundary doesn't flap tiers every step.
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


class DynamicHybridPolicy:
    """Starts entries at their modality-prior tier, then promotes/demotes
    based on live importance, with hysteresis to avoid flapping.

    - WARM -> HOT once ``entry.importance`` exceeds ``promote_threshold``.
    - HOT -> WARM once ``entry.importance`` falls below ``demote_threshold``.

    ``promote_threshold`` must be strictly greater than ``demote_threshold``
    so there is a dead band between them where an entry holds its current
    tier regardless of which direction it arrived from.
    """

    name = "dynamic_hybrid"

    def __init__(self, promote_threshold: float, demote_threshold: float):
        if not demote_threshold < promote_threshold:
            raise ValueError("demote_threshold must be < promote_threshold (hysteresis band)")
        self.promote_threshold = promote_threshold
        self.demote_threshold = demote_threshold

    def initial_tier(self, entry: TokenKV) -> Tier:
        return Tier.WARM if entry.modality == Modality.VISUAL else Tier.HOT

    def update(self, entry: TokenKV) -> Tier:
        if entry.tier == Tier.WARM and entry.importance > self.promote_threshold:
            return Tier.HOT
        if entry.tier == Tier.HOT and entry.importance < self.demote_threshold:
            return Tier.WARM
        return entry.tier
