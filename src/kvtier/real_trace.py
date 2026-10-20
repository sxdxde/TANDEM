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


def trace_from_layer_attention(
    is_visual: np.ndarray,
    step_layer_weights: list[np.ndarray],
    kv_elements_per_token: int = DEFAULT_KV_ELEMENTS_PER_TOKEN,
    layer_stride: int = 1,
) -> TraceSession:
    """Per-layer variant (Milestone 6): one KV entry per (token, layer).

    ``step_layer_weights[t]`` has shape ``(n_layers, n_prefill + t + 1)``.
    Entry position id is ``token * n_layers_kept + layer_index``. Each
    layer's row is renormalised to sum to 1 and the step's weights are
    divided by the number of layers, so total attention mass per step is 1
    and the quality metrics keep their meaning (mean over layers).
    ``layer_stride > 1`` keeps every k-th layer, to bound simulation cost.
    Each per-layer entry has the same abstract size as a single-entry token,
    so absolute energies are ~n_layers x the layer-averaged traces; compare
    ratios, not absolutes, across the two trace types.
    """
    n_prefill = len(is_visual)
    size_bytes = kv_elements_per_token * HOT_BYTES_PER_ELEMENT
    first = np.asarray(step_layer_weights[0])[::layer_stride]
    n_layers = first.shape[0]

    def make(token: int, layer: int, modality: Modality) -> TokenKV:
        return TokenKV(
            position=token * n_layers + layer,
            modality=modality,
            size_bytes=size_bytes,
            tier=Tier.HOT,
        )

    session = TraceSession(entries=[])
    for tok, v in enumerate(is_visual):
        for layer in range(n_layers):
            session.entries.append(make(tok, layer, Modality.VISUAL if v else Modality.TEXT))

    for t, w in enumerate(step_layer_weights):
        w = np.clip(np.asarray(w, dtype=float)[::layer_stride], 0.0, None)
        expected = n_prefill + t + 1
        if w.shape != (n_layers, expected):
            raise ValueError(f"step {t}: expected {(n_layers, expected)}, got {w.shape}")
        w = w / w.sum(axis=1, keepdims=True) / n_layers
        tok = n_prefill + t
        new_positions = []
        for layer in range(n_layers):
            e = make(tok, layer, Modality.TEXT)
            session.entries.append(e)
            new_positions.append(e.position)
        session.steps.append(
            DecodeStep(
                step=t,
                new_entry_position=new_positions[0],
                extra_new_positions=new_positions[1:],
                attention_weights={
                    token * n_layers + layer: float(w[layer, token])
                    for token in range(expected)
                    for layer in range(n_layers)
                },
            )
        )
    return session


def load_layer_trace(path: str | Path, **kwargs) -> TraceSession:
    with np.load(path) as z:
        is_visual = z["is_visual"]
        n_layers = int(z["n_layers"])
        lengths = z["lengths"]
        flat = z["weights"].astype(np.float32)
    steps, i = [], 0
    for ln in lengths:
        steps.append(flat[i : i + n_layers * ln].reshape(n_layers, ln))
        i += n_layers * ln
    return trace_from_layer_attention(is_visual, steps, **kwargs)
