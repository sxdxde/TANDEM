"""Offline (modality x layer) attention profile for layer-aware static placement.

``group_importance`` measures, on a per-layer trace, the mean attention
weight a single entry of each (modality, layer) group receives, expressed as
a multiple of uniform attention (1 / n_prefill entries). ``hot_groups`` keeps
the groups at or above a threshold multiple. A profile built from *other*
traces and applied to a new one is a static, zero-reconfiguration policy.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Hashable

from kvtier.kv_entry import Modality, TokenKV
from kvtier.trace import TraceSession

Group = Hashable
GroupFn = Callable[[TokenKV, int], Group]


def modality_layer(entry: TokenKV, n_prefill_tokens: int) -> Group:
    return (entry.modality, entry.layer)


def role_layer(entry: TokenKV, n_prefill_tokens: int) -> Group:
    """(role, layer): role is sink (token 0), generated (token >= prompt
    length), prompt text, or visual. Finer than modality, since text tokens
    differ enormously in attention (sink / recent vs mid-prompt)."""
    if entry.token == 0:
        role = "sink"
    elif entry.token >= n_prefill_tokens:
        role = "generated"
    elif entry.modality == Modality.VISUAL:
        role = "visual"
    else:
        role = "text"
    return (role, entry.layer)


GROUPINGS: dict[str, GroupFn] = {"modality_layer": modality_layer, "role_layer": role_layer}


def n_prefill_tokens(session: TraceSession) -> int:
    """Number of prompt tokens in a per-layer trace (first decode step's token index)."""
    return min(s.new_entry_position for s in session.steps) // len(session.steps[0].new_positions)


def group_importance(session: TraceSession, group_fn: GroupFn = modality_layer) -> dict[Group, float]:
    """Mean per-entry attention weight of each (modality, layer) group, in
    multiples of uniform attention, averaged over decode steps."""
    by_pos = {e.position: e for e in session.entries}
    n_tok = n_prefill_tokens(session)
    n_new = sum(len(s.new_positions) for s in session.steps)
    uniform = 1.0 / (len(session.entries) - n_new)
    totals: dict[Group, float] = defaultdict(float)
    for step in session.steps:
        mass: dict[Group, float] = defaultdict(float)
        count: dict[Group, int] = defaultdict(int)
        for pos, w in step.attention_weights.items():
            e = by_pos[pos]
            g = group_fn(e, n_tok)
            mass[g] += w
            count[g] += 1
        for g, m in mass.items():
            totals[g] += m / count[g] / uniform
    return {g: t / len(session.steps) for g, t in totals.items()}


def average_profiles(profiles: list[dict[Group, float]]) -> dict[Group, float]:
    keys = set().union(*profiles)
    return {g: sum(p.get(g, 0.0) for p in profiles) / len(profiles) for g in keys}


def hot_groups(profile: dict[Group, float], theta: float) -> set[Group]:
    """Groups whose mean per-entry importance is at least ``theta`` x uniform."""
    return {g for g, v in profile.items() if v >= theta}
