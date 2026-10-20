"""Build a ``TraceSession`` from real VLM attention weights (Milestone 5).

Torch-free on purpose: ``experiments/extract_vlm_trace.py`` does the model
run and saves plain arrays; this module only converts them, so the
simulator and policies never depend on a deep-learning stack.

Alignment with the simulator's model (prefill entries, then one new entry
per decode step, each step attending over every entry that exists):

- The last prompt token acts as the first decode step's query, so prefill
  is the first ``L - 1`` prompt positions.
- Decode step ``t`` has ``new_entry_position = L - 1 + t`` and attends over
  positions ``0 .. L - 1 + t`` (length ``L + t``), including itself, exactly
  like the synthetic trace.

The synthetic trace's ``salient_visual_positions`` ground truth does not
exist for real traces, so it is left empty.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kvtier.kv_entry import Modality, Tier, TokenKV
from kvtier.tiers import HOT_BYTES_PER_ELEMENT
from kvtier.trace import DEFAULT_KV_ELEMENTS_PER_TOKEN, DecodeStep, TraceSession


def trace_from_attention(
    is_visual: np.ndarray,
    step_weights: list[np.ndarray],
    kv_elements_per_token: int = DEFAULT_KV_ELEMENTS_PER_TOKEN,
) -> TraceSession:
    """Convert modality flags + per-step attention vectors into a session.

    ``is_visual`` has one bool per *prefill* position (length ``L - 1``).
    ``step_weights[t]`` has length ``len(is_visual) + t + 1``; each is
    renormalised to sum to 1 so downstream quality metrics are well-defined.
    """
    n_prefill = len(is_visual)
    size_bytes = kv_elements_per_token * HOT_BYTES_PER_ELEMENT

    def make(pos: int, modality: Modality) -> TokenKV:
        return TokenKV(position=pos, modality=modality, size_bytes=size_bytes, tier=Tier.HOT)

    session = TraceSession(
        entries=[
            make(i, Modality.VISUAL if v else Modality.TEXT) for i, v in enumerate(is_visual)
        ]
    )
    for t, w in enumerate(step_weights):
        expected = n_prefill + t + 1
        if len(w) != expected:
            raise ValueError(f"step {t}: expected {expected} weights, got {len(w)}")
        w = np.clip(np.asarray(w, dtype=float), 0.0, None)
        w = w / w.sum()
        new_pos = n_prefill + t
        session.entries.append(make(new_pos, Modality.TEXT))
        session.steps.append(
            DecodeStep(
                step=t,
                new_entry_position=new_pos,
                attention_weights={p: float(x) for p, x in enumerate(w)},
            )
        )
    return session


def save_trace(path: str | Path, is_visual: np.ndarray, step_weights: list[np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        is_visual=np.asarray(is_visual, dtype=bool),
        weights=np.concatenate([np.asarray(w, dtype=np.float32) for w in step_weights]),
        lengths=np.array([len(w) for w in step_weights]),
    )


def load_trace(path: str | Path, **kwargs) -> TraceSession:
    with np.load(path) as z:
        is_visual = z["is_visual"]
        bounds = np.cumsum(z["lengths"])[:-1]
        step_weights = np.split(z["weights"], bounds)
    return trace_from_attention(is_visual, step_weights, **kwargs)
