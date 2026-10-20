"""Core data model for a single KV-cache entry and the enums that classify it."""

from enum import Enum, auto
from dataclasses import dataclass


class Modality(Enum):
    TEXT = auto()
    VISUAL = auto()


class Tier(Enum):
    HOT = auto()
    WARM = auto()


@dataclass
class TokenKV:
    """One token's KV-cache entry and its live placement/importance state.

    ``size_bytes`` is the entry's size at HOT (FP16) precision; tier-specific
    access costs (see ``tiers.py``) rescale it by the tier's bytes-per-element
    to reflect WARM's reduced precision, rather than storing a separate size
    per tier.
    """

    position: int
    modality: Modality
    size_bytes: float
    tier: Tier                     # current tier, mutable as placement changes
    importance: float = 0.0        # EMA of attention weight received, updated live
    tier_changes: int = 0          # count of promotions/demotions this entry underwent
    observed_steps: int = 0        # decode steps this entry's importance has been updated
    steps_since_move: int = 0      # decode steps since creation or last tier change
    raw_importance: float = 0.0    # uncorrected live EMA (see monitor.py bias_correct)
