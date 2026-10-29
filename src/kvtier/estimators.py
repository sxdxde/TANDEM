"""Imperfect importance estimators (Milestone 8).

M3-M7 feed the importance monitor the true attention weight of every entry at every step, which a
real system only gets by paying for (or approximating) the attention scores. An estimator sits
between the true weights and the monitor. ``observe(step, weights, ema_decay)`` returns
``(weights_for_monitor | None, decay | None)``: ``None`` weights = the monitor sees nothing this
step. Quality and access costs in the simulator always use the *true* weights.

All estimators here are free (no energy/latency is charged for computing them); the experiment
reports how much estimator overhead the measured saving can absorb instead.
"""

from __future__ import annotations

import numpy as np


class Oracle:
    name = "oracle"

    def observe(self, step, weights, ema_decay):
        return weights, None


class Noisy:
    """Multiplicative log-normal noise, w * exp(sigma * N(0,1)), independent per entry and step."""

    def __init__(self, sigma: float, seed: int = 0):
        self.sigma, self.rng, self.name = sigma, np.random.default_rng(seed), f"noise{sigma:g}"

    def observe(self, step, weights, ema_decay):
        keys = list(weights)
        f = self.rng.lognormal(0.0, self.sigma, len(keys))
        return {k: weights[k] * x for k, x in zip(keys, f)}, None


class Subsample:
    """Observe attention only every m-th step; the EMA decay is raised to m to keep its time constant."""

    def __init__(self, m: int):
        self.m, self.name = m, f"sub{m}"

    def observe(self, step, weights, ema_decay):
        return (weights, ema_decay ** self.m) if step % self.m == 0 else (None, None)


class Delay:
    """The monitor sees the weights from ``d`` steps ago (a stale / pipelined signal)."""

    def __init__(self, d: int):
        self.d, self.buf, self.name = d, [], f"delay{d}"

    def observe(self, step, weights, ema_decay):
        self.buf.append(weights)
        return (self.buf[-1 - self.d], None) if len(self.buf) > self.d else (None, None)


class TokenShared:
    """One score per token (the layer-mean) shared by all of its per-layer entries: the cheap
    estimator a real system would use, since it needs one score per token, not per (token, layer).
    Entry positions are ``token * n_layers + layer``."""

    def __init__(self, n_layers: int):
        self.n, self.name = n_layers, "token_shared"

    def observe(self, step, weights, ema_decay):
        n = self.n
        mean: dict[int, float] = {}
        for p, w in weights.items():
            mean[p // n] = mean.get(p // n, 0.0) + w / n
        return {p: mean[p // n] for p in weights}, None


class Chain:
    """Apply estimators in order (each sees the previous one's output)."""

    def __init__(self, *stages, name: str):
        self.stages, self.name = stages, name

    def observe(self, step, weights, ema_decay):
        decay = None
        for s in self.stages:
            weights, d = s.observe(step, weights, ema_decay)
            decay = d if d is not None else decay
            if weights is None:
                return None, None
        return weights, decay


def make(name: str, n_layers: int, seed: int):
    if name == "oracle":
        return Oracle()
    if name.startswith("noise") and "_" not in name:
        return Noisy(float(name[5:]), seed)
    if name.startswith("sub") and "_" not in name:
        return Subsample(int(name[3:]))
    if name.startswith("delay"):
        return Delay(int(name[5:]))
    if name == "token_shared":
        return TokenShared(n_layers)
    if name == "shared_sub4":
        return Chain(TokenShared(n_layers), Subsample(4), name=name)
    if name == "noise0.5_sub4":
        return Chain(Noisy(0.5, seed), Subsample(4), name=name)
    raise ValueError(name)
