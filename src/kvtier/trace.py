"""Synthetic prefill + decode trace generation with attention-derived importance.

A session is a prefill burst (TEXT prompt tokens + VISUAL patch tokens) followed
by N decode steps, each of which appends one newly generated TEXT token and
computes a synthetic attention-weight vector over every entry that exists so
far. Two effects are mixed into that vector, matching the heavy-tailed
importance pattern reported for visual tokens in AirCache and VL-Cache:

- **Recency**: entries closer to the current decode position get more weight.
- **Salience**: a small, fixed subset of visual entries (chosen once, at
  prefill) keep receiving disproportionate attention throughout decode,
  regardless of position.

Each entry's ``importance`` is then updated as an EMA of the weights it has
received. This EMA logic is intentionally simple and inline here for
Milestone 1; Milestone 3's ``monitor.py`` takes over owning it once a live
promotion/demotion policy needs to query it mid-simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kvtier.kv_entry import Modality, Tier, TokenKV

#: Abstracted per-token K+V vector width used to size entries. Not tied to a
#: specific model's head_dim x num_layers; kept as a round number so the cost
#: model in tiers.py stays a tractable, simulation-scale abstraction.
DEFAULT_KV_ELEMENTS_PER_TOKEN: int = 128

#: How fast the recency component decays per position of distance from the
#: current decode step. Purely shapes the synthetic trace, not part of the
#: hardware cost story, so it is not exposed as a top-level config knob.
RECENCY_DECAY_PER_POSITION: float = 0.05

#: Salience component given to non-salient visual / all text entries, versus
#: 1.0 for the fixed salient-visual subset. Keeps their attention nonzero
#: without letting them compete with the salient subset.
BASELINE_SALIENCE: float = 0.05


@dataclass
class DecodeStep:
    """One decode step's synthetic attention distribution."""

    step: int
    new_entry_position: int
    attention_weights: dict[int, float]  # position -> weight, sums to 1.0
    # Milestone 6: per-layer traces create one new entry per layer each step.
    # ``new_entry_position`` is the first; any others are listed here (in
    # session.entries order). Empty for the single-entry-per-step traces.
    extra_new_positions: list[int] = field(default_factory=list)

    @property
    def new_positions(self) -> list[int]:
        return [self.new_entry_position, *self.extra_new_positions]


@dataclass
class TraceSession:
    """A full prefill+decode synthetic trace."""

    entries: list[TokenKV]
    steps: list[DecodeStep] = field(default_factory=list)
    salient_visual_positions: set[int] = field(default_factory=set)


def _make_entry(position: int, modality: Modality, kv_elements_per_token: int) -> TokenKV:
    from kvtier.tiers import HOT_BYTES_PER_ELEMENT

    return TokenKV(
        position=position,
        modality=modality,
        size_bytes=kv_elements_per_token * HOT_BYTES_PER_ELEMENT,
        tier=Tier.HOT,
    )


def _attention_weights(
    active_positions: list[int],
    current_position: int,
    salient_visual_positions: set[int],
    recency_bias: float,
    rng: np.random.Generator,
) -> dict[int, float]:
    """Synthesize one decode step's attention distribution over active entries."""
    scores = np.empty(len(active_positions))
    for i, pos in enumerate(active_positions):
        recency = np.exp(-RECENCY_DECAY_PER_POSITION * (current_position - pos))
        salience = 1.0 if pos in salient_visual_positions else BASELINE_SALIENCE
        raw = recency_bias * recency + (1.0 - recency_bias) * salience
        scores[i] = raw

    # Multiplicative log-normal noise, then renormalize to a valid distribution.
    noise = rng.lognormal(mean=0.0, sigma=0.15, size=scores.shape)
    scores = scores * noise
    scores = np.clip(scores, a_min=1e-8, a_max=None)
    weights = scores / scores.sum()

    return {pos: float(w) for pos, w in zip(active_positions, weights)}


def _update_importance_ema(
    entries_by_position: dict[int, TokenKV],
    weights: dict[int, float],
    ema_decay: float,
) -> None:
    for pos, w in weights.items():
        entry = entries_by_position[pos]
        entry.importance = ema_decay * entry.importance + (1.0 - ema_decay) * w


def generate_session(
    n_text_tokens: int = 20,
    n_visual_tokens: int = 500,
    n_decode_steps: int = 64,
    salient_visual_fraction: float = 0.08,
    recency_bias: float = 0.7,
    ema_decay: float = 0.9,
    kv_elements_per_token: int = DEFAULT_KV_ELEMENTS_PER_TOKEN,
    seed: int = 0,
) -> TraceSession:
    """Generate one synthetic prefill+decode session.

    Prefill creates ``n_text_tokens`` TEXT entries followed by
    ``n_visual_tokens`` VISUAL entries. A ``salient_visual_fraction`` slice of
    the visual entries is chosen (once) to keep receiving disproportionate
    attention. Each of the ``n_decode_steps`` decode steps then appends one
    new TEXT entry (the token just generated) and computes an attention
    distribution over every entry that exists at that point, updating each
    entry's importance as an EMA of the weight it received.
    """
    rng = np.random.default_rng(seed)

    entries: list[TokenKV] = []
    entries_by_position: dict[int, TokenKV] = {}

    for pos in range(n_text_tokens):
        entry = _make_entry(pos, Modality.TEXT, kv_elements_per_token)
        entries.append(entry)
        entries_by_position[pos] = entry

    visual_start = n_text_tokens
    for pos in range(visual_start, visual_start + n_visual_tokens):
        entry = _make_entry(pos, Modality.VISUAL, kv_elements_per_token)
        entries.append(entry)
        entries_by_position[pos] = entry

    visual_positions = list(range(visual_start, visual_start + n_visual_tokens))
    n_salient = max(1, round(salient_visual_fraction * n_visual_tokens))
    salient_visual_positions = set(
        rng.choice(visual_positions, size=n_salient, replace=False).tolist()
    )

    session = TraceSession(entries=entries, salient_visual_positions=salient_visual_positions)

    decode_start = visual_start + n_visual_tokens
    for step in range(n_decode_steps):
        new_pos = decode_start + step
        new_entry = _make_entry(new_pos, Modality.TEXT, kv_elements_per_token)
        session.entries.append(new_entry)
        entries_by_position[new_pos] = new_entry

        active_positions = list(entries_by_position.keys())
        weights = _attention_weights(
            active_positions=active_positions,
            current_position=new_pos,
            salient_visual_positions=salient_visual_positions,
            recency_bias=recency_bias,
            rng=rng,
        )
        _update_importance_ema(entries_by_position, weights, ema_decay)

        session.steps.append(
            DecodeStep(step=step, new_entry_position=new_pos, attention_weights=weights)
        )

    return session
