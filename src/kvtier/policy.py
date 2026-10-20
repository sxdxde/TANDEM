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


class LayerModalityPolicy:
    """Static layer-aware placement: entries whose group (default (modality,
    layer); see ``profile.GROUPINGS``) is in ``hot_groups`` (built offline) live in HOT, everything
    else in WARM. No tier moves, so no reconfiguration cost."""

    name = "layer_modality"

    def __init__(self, hot_groups: set, group_fn=None, n_prefill_tokens: int = 0):
        from kvtier.profile import modality_layer

        self.hot_groups = set(hot_groups)
        self.group_fn = group_fn or modality_layer
        self.n_prefill_tokens = n_prefill_tokens

    def initial_tier(self, entry: TokenKV) -> Tier:
        in_hot = self.group_fn(entry, self.n_prefill_tokens) in self.hot_groups
        return Tier.HOT if in_hot else Tier.WARM

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

    Two optional damping controls (both default 0 = off, the M3 behaviour):

    - ``warmup_steps``: an entry may not move until its importance EMA has
      been updated this many times. The EMA starts at 0, so without this a
      HOT entry looks unimportant at step 0 and is demoted on cold-start
      alone (observed in M4: 21 wasted demotions at step 0).
    - ``min_dwell_steps``: after any tier change an entry must stay put this
      many steps, which blocks the promote-then-demote round trips caused by
      recency-driven importance spikes at the prefill/decode boundary.

    ``damp_promotions`` (default True) applies both controls to promotions as
    well. Set False to damp demotions only: salient WARM entries are then
    promoted as soon as they cross the threshold, while a HOT entry still
    can't be demoted on cold-start or immediately after a move.
    """

    name = "dynamic_hybrid"

    def __init__(
        self,
        promote_threshold: float,
        demote_threshold: float,
        warmup_steps: int = 0,
        min_dwell_steps: int = 0,
        damp_promotions: bool = True,
    ):
        if not demote_threshold < promote_threshold:
            raise ValueError("demote_threshold must be < promote_threshold (hysteresis band)")
        self.promote_threshold = promote_threshold
        self.demote_threshold = demote_threshold
        self.warmup_steps = warmup_steps
        self.min_dwell_steps = min_dwell_steps
        self.damp_promotions = damp_promotions

    def initial_tier(self, entry: TokenKV) -> Tier:
        return Tier.WARM if entry.modality == Modality.VISUAL else Tier.HOT

    def update(self, entry: TokenKV) -> Tier:
        damped = (
            entry.observed_steps < self.warmup_steps
            or entry.steps_since_move < self.min_dwell_steps
        )
        if entry.tier == Tier.WARM and entry.importance > self.promote_threshold:
            return entry.tier if damped and self.damp_promotions else Tier.HOT
        if entry.tier == Tier.HOT and entry.importance < self.demote_threshold:
            return entry.tier if damped else Tier.WARM
        return entry.tier
