"""HOT/WARM per-access latency and energy cost model.

Cost model shape: each tier charges a fixed per-access overhead plus a cost
that scales linearly with the entry's *effective* size in that tier:

    latency(entry) = base_latency_ns + latency_per_byte_ns * effective_size_bytes
    energy(entry)  = base_energy_pj  + energy_per_byte_pj  * effective_size_bytes

``TokenKV.size_bytes`` is fixed at creation and represents the entry's size
at HOT (FP16) precision. A tier's ``effective_size_bytes`` rescales that by
the tier's own bytes-per-element, so WARM's reduced precision (INT4) shrinks
the effective size *and* gets a lower per-byte cost — the two effects
compound, matching how a real compute-in-memory tier would both quantize and
move less data per access.

Default constants below are deliberately named and centralized so the
tier-cost story can be re-argued by changing one place (or overridden via
``configs/default.yaml`` through ``load_tier_config``):

- WARM_ENERGY_RATIO = 0.1: WARM's per-byte access energy is ~10x lower than
  HOT's, an order-of-magnitude figure consistent with commonly cited
  compute-in-memory / near-memory energy savings over conventional
  DRAM/HBM access + separate compute (data movement dominates DRAM access
  energy; keeping compute in the array removes most of that movement).
- WARM_LATENCY_RATIO = 1.5: WARM is ~1.5x slower per access than HOT,
  reflecting the common CIM tradeoff where analog/near-memory compute adds
  overhead even as it saves energy and movement — latency does not improve
  for free the way energy does.
- HOT stores entries at FP16 (2 bytes/element), WARM at INT4
  (0.5 bytes/element), consistent with KV-cache quantization results
  reported in the literature this project responds to (e.g. HybridKV,
  ZipVL).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from kvtier.kv_entry import Tier, TokenKV

# --- Named cost-model constants (HOT baseline + WARM ratios) --------------

HOT_BASE_LATENCY_NS: float = 20.0
HOT_LATENCY_NS_PER_BYTE: float = 0.01
HOT_BASE_ENERGY_PJ: float = 5.0
HOT_ENERGY_PJ_PER_BYTE: float = 8.0
HOT_BYTES_PER_ELEMENT: float = 2.0  # FP16

WARM_ENERGY_RATIO: float = 0.1   # ~10x lower per-byte access energy than HOT
WARM_LATENCY_RATIO: float = 1.5  # ~1.5x higher per-access latency than HOT
WARM_BYTES_PER_ELEMENT: float = 0.5  # INT4

# Relative quantization error of a value read from each tier (Milestone 4).
# HOT is FP16 (treated as lossless reference); WARM is INT4. 0.05 is an
# assumed order-of-magnitude relative error for per-group INT4 KV
# quantization, NOT a measured value -- see NOTES.md. Only the ratio to
# other tiers matters for policy comparisons, but sweeps should vary it.
HOT_QUANT_ERROR: float = 0.0
WARM_QUANT_ERROR: float = 0.05


@dataclass(frozen=True)
class TierResource:
    """Per-byte access-cost model for one memory tier."""

    tier: Tier
    base_latency_ns: float
    latency_per_byte_ns: float
    base_energy_pj: float
    energy_per_byte_pj: float
    bytes_per_element: float
    quant_error: float = 0.0  # relative error of reads from this tier

    def effective_size_bytes(self, entry: TokenKV) -> float:
        """Rescale ``entry.size_bytes`` (stored at HOT/FP16 granularity) to
        this tier's precision."""
        return entry.size_bytes * (self.bytes_per_element / HOT_BYTES_PER_ELEMENT)

    def access_cost(self, entry: TokenKV) -> tuple[float, float]:
        """Return ``(latency_ns, energy_pj)`` for one access to ``entry``
        while it resides in this tier."""
        size = self.effective_size_bytes(entry)
        latency = self.base_latency_ns + self.latency_per_byte_ns * size
        energy = self.base_energy_pj + self.energy_per_byte_pj * size
        return latency, energy


def _default_hot() -> TierResource:
    return TierResource(
        tier=Tier.HOT,
        base_latency_ns=HOT_BASE_LATENCY_NS,
        latency_per_byte_ns=HOT_LATENCY_NS_PER_BYTE,
        base_energy_pj=HOT_BASE_ENERGY_PJ,
        energy_per_byte_pj=HOT_ENERGY_PJ_PER_BYTE,
        bytes_per_element=HOT_BYTES_PER_ELEMENT,
        quant_error=HOT_QUANT_ERROR,
    )


def _default_warm() -> TierResource:
    return TierResource(
        tier=Tier.WARM,
        base_latency_ns=HOT_BASE_LATENCY_NS * WARM_LATENCY_RATIO,
        latency_per_byte_ns=HOT_LATENCY_NS_PER_BYTE * WARM_LATENCY_RATIO,
        base_energy_pj=HOT_BASE_ENERGY_PJ * WARM_ENERGY_RATIO,
        energy_per_byte_pj=HOT_ENERGY_PJ_PER_BYTE * WARM_ENERGY_RATIO,
        bytes_per_element=WARM_BYTES_PER_ELEMENT,
        quant_error=WARM_QUANT_ERROR,
    )


#: Default tier registry, built from the named constants above. These match
#: ``configs/default.yaml`` — see ``load_tier_config`` to load overrides.
TIERS: dict[Tier, TierResource] = {Tier.HOT: _default_hot(), Tier.WARM: _default_warm()}


def get_tier(tier: Tier) -> TierResource:
    return TIERS[tier]


def load_tier_config(path: str | Path) -> dict[Tier, TierResource]:
    """Load a ``tiers:`` block (see ``configs/default.yaml``) into a
    ``{Tier: TierResource}`` registry, overriding the built-in defaults."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    tiers_cfg = raw["tiers"]
    result: dict[Tier, TierResource] = {}
    for name, tier in (("hot", Tier.HOT), ("warm", Tier.WARM)):
        cfg = tiers_cfg[name]
        result[tier] = TierResource(
            tier=tier,
            base_latency_ns=cfg["base_latency_ns"],
            latency_per_byte_ns=cfg["latency_per_byte_ns"],
            base_energy_pj=cfg["base_energy_pj"],
            energy_per_byte_pj=cfg["energy_per_byte_pj"],
            bytes_per_element=cfg["bytes_per_element"],
            quant_error=cfg.get("quant_error", 0.0),
        )
    return result
