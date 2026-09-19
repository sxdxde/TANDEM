"""Live per-entry importance tracking: an EMA of attention weight received.

``trace.py`` also computes an EMA into ``TokenKV.importance`` during
synthetic generation, but that is a *full-session* summary baked in before
the simulator ever runs (used by Milestone 1's validation plot). Reusing
that value inside a live simulation would leak every future decode step's
attention into an importance a policy queries mid-simulation — a
promotion decision at step 5 would already "know" about step 60.

This module owns the live version instead: the simulator resets every
entry's importance to 0.0 before decode starts, then calls ``update()``
once per step with only that step's attention weights, so a policy only
ever sees what has actually happened so far.
"""

from __future__ import annotations

from dataclasses import dataclass

from kvtier.kv_entry import TokenKV


@dataclass
class ImportanceMonitor:
    """Tracks a live EMA of attention weight per entry."""

    ema_decay: float

    def reset(self, entries: list[TokenKV]) -> None:
        for entry in entries:
            entry.importance = 0.0

    def update(self, active: dict[int, TokenKV], weights: dict[int, float]) -> None:
        """Update every entry that received attention this step."""
        for pos, w in weights.items():
            entry = active[pos]
            entry.importance = self.ema_decay * entry.importance + (1.0 - self.ema_decay) * w
