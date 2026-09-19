"""Aggregate energy/latency/footprint metrics from a simulated session.

``total_latency_ns`` sums every entry's per-access latency at each decode
step, treating accesses within a step as fully serialized. That is a
total-work figure (an upper bound on wall-clock time), not a claim that a
real system couldn't pipeline or bank-parallelize those accesses — flagged
here since it drives how the headline latency numbers should be read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from kvtier.kv_entry import Tier


@dataclass
class StepMetrics:
    """Metrics for a single decode step."""

    step: int
    total_latency_ns: float
    total_energy_pj: float
    tier_latency_ns: dict[Tier, float]
    tier_energy_pj: dict[Tier, float]
    tier_footprint_bytes: dict[Tier, float]
    tier_access_counts: dict[Tier, int]
    # Populated from Milestone 3 onward; zero for the static M2 policies.
    reconfig_latency_ns: float = 0.0
    reconfig_energy_pj: float = 0.0
    n_promotions: int = 0
    n_demotions: int = 0


@dataclass
class SessionMetrics:
    """Full time series of per-step metrics for one policy run."""

    policy_name: str
    steps: list[StepMetrics] = field(default_factory=list)

    @property
    def total_latency_ns(self) -> float:
        return sum(s.total_latency_ns for s in self.steps)

    @property
    def total_energy_pj(self) -> float:
        return sum(s.total_energy_pj for s in self.steps)

    @property
    def total_reconfig_latency_ns(self) -> float:
        return sum(s.reconfig_latency_ns for s in self.steps)

    @property
    def total_reconfig_energy_pj(self) -> float:
        return sum(s.reconfig_energy_pj for s in self.steps)

    @property
    def total_promotions(self) -> int:
        return sum(s.n_promotions for s in self.steps)

    @property
    def total_demotions(self) -> int:
        return sum(s.n_demotions for s in self.steps)

    def total_latency_by_tier(self) -> dict[Tier, float]:
        return {
            tier: sum(s.tier_latency_ns.get(tier, 0.0) for s in self.steps)
            for tier in Tier
        }

    def total_energy_by_tier(self) -> dict[Tier, float]:
        return {
            tier: sum(s.tier_energy_pj.get(tier, 0.0) for s in self.steps)
            for tier in Tier
        }

    def final_footprint_bytes(self) -> dict[Tier, float]:
        if not self.steps:
            return {tier: 0.0 for tier in Tier}
        return dict(self.steps[-1].tier_footprint_bytes)

    def to_summary_dict(self) -> dict:
        """One flat row of headline numbers, suitable for a comparison table."""
        latency_by_tier = self.total_latency_by_tier()
        energy_by_tier = self.total_energy_by_tier()
        footprint = self.final_footprint_bytes()
        return {
            "policy": self.policy_name,
            "total_latency_ns": self.total_latency_ns,
            "total_energy_pj": self.total_energy_pj,
            "hot_latency_ns": latency_by_tier[Tier.HOT],
            "warm_latency_ns": latency_by_tier[Tier.WARM],
            "hot_energy_pj": energy_by_tier[Tier.HOT],
            "warm_energy_pj": energy_by_tier[Tier.WARM],
            "reconfig_latency_ns": self.total_reconfig_latency_ns,
            "reconfig_energy_pj": self.total_reconfig_energy_pj,
            "n_promotions": self.total_promotions,
            "n_demotions": self.total_demotions,
            "final_hot_footprint_bytes": footprint[Tier.HOT],
            "final_warm_footprint_bytes": footprint[Tier.WARM],
        }

    def to_timeseries_df(self) -> pd.DataFrame:
        """Per-step time series, one row per decode step (for M4 timeline plots)."""
        rows = []
        for s in self.steps:
            rows.append(
                {
                    "step": s.step,
                    "total_latency_ns": s.total_latency_ns,
                    "total_energy_pj": s.total_energy_pj,
                    "hot_latency_ns": s.tier_latency_ns.get(Tier.HOT, 0.0),
                    "warm_latency_ns": s.tier_latency_ns.get(Tier.WARM, 0.0),
                    "hot_energy_pj": s.tier_energy_pj.get(Tier.HOT, 0.0),
                    "warm_energy_pj": s.tier_energy_pj.get(Tier.WARM, 0.0),
                    "hot_footprint_bytes": s.tier_footprint_bytes.get(Tier.HOT, 0.0),
                    "warm_footprint_bytes": s.tier_footprint_bytes.get(Tier.WARM, 0.0),
                    "hot_access_count": s.tier_access_counts.get(Tier.HOT, 0),
                    "warm_access_count": s.tier_access_counts.get(Tier.WARM, 0),
                    "reconfig_latency_ns": s.reconfig_latency_ns,
                    "reconfig_energy_pj": s.reconfig_energy_pj,
                    "n_promotions": s.n_promotions,
                    "n_demotions": s.n_demotions,
                }
            )
        return pd.DataFrame(rows)


def save_timeseries_csv(metrics: SessionMetrics, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_timeseries_df().to_csv(path, index=False)


def combine_summaries(all_metrics: list[SessionMetrics]) -> pd.DataFrame:
    """One row per policy run, for side-by-side comparison."""
    return pd.DataFrame([m.to_summary_dict() for m in all_metrics])


def print_summary(metrics: SessionMetrics) -> None:
    s = metrics.to_summary_dict()
    print(f"--- {s['policy']} ---")
    print(f"total latency:   {s['total_latency_ns']:,.1f} ns")
    print(f"total energy:    {s['total_energy_pj']:,.1f} pJ")
    print(f"  HOT:  {s['hot_latency_ns']:,.1f} ns  |  {s['hot_energy_pj']:,.1f} pJ")
    print(f"  WARM: {s['warm_latency_ns']:,.1f} ns  |  {s['warm_energy_pj']:,.1f} pJ")
    if s["n_promotions"] or s["n_demotions"]:
        print(
            f"reconfig:        {s['reconfig_latency_ns']:,.1f} ns  |  "
            f"{s['reconfig_energy_pj']:,.1f} pJ  "
            f"({s['n_promotions']} promotions, {s['n_demotions']} demotions)"
        )
    print(
        f"final footprint: HOT {s['final_hot_footprint_bytes']:,.0f} B  |  "
        f"WARM {s['final_warm_footprint_bytes']:,.0f} B"
    )
