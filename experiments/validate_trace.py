"""Milestone 1 checkpoint: validate that the synthetic trace matches the
literature's reported pattern — visual tokens' average importance should
decline over decode while a small salient minority stays high — before any
simulator or policy is built on top of it.

Usage:
    python experiments/validate_trace.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from kvtier.kv_entry import Modality
from kvtier.trace import generate_session

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_trace_config(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return raw["trace"]


def importance_by_modality_over_decode(session, ema_decay: float) -> dict[str, list[float]]:
    """Replay decode step by step, recording average importance per group
    *as of that step* (not just the final value), so the trend is visible."""
    entries_by_position = {e.position: e for e in session.entries}
    salient = session.salient_visual_positions

    text_avg: list[float] = []
    visual_avg: list[float] = []
    visual_salient_avg: list[float] = []
    visual_nonsalient_avg: list[float] = []

    # Re-run the EMA forward pass ourselves is unnecessary: generate_session
    # already leaves each entry at its *final* importance. To show a trend
    # over decode we instead recompute a running EMA snapshot per step from
    # the recorded per-step attention weights.
    running: dict[int, float] = {pos: 0.0 for pos in entries_by_position}

    for step in session.steps:
        for pos, w in step.attention_weights.items():
            running[pos] = ema_decay * running[pos] + (1.0 - ema_decay) * w

        text_vals = [
            running[pos] for pos, e in entries_by_position.items()
            if e.modality is Modality.TEXT and pos in step.attention_weights
        ]
        visual_vals = [
            running[pos] for pos, e in entries_by_position.items()
            if e.modality is Modality.VISUAL and pos in step.attention_weights
        ]
        visual_salient_vals = [
            running[pos] for pos in salient if pos in step.attention_weights
        ]
        visual_nonsalient_vals = [
            running[pos] for pos, e in entries_by_position.items()
            if e.modality is Modality.VISUAL and pos not in salient and pos in step.attention_weights
        ]

        text_avg.append(sum(text_vals) / len(text_vals) if text_vals else 0.0)
        visual_avg.append(sum(visual_vals) / len(visual_vals) if visual_vals else 0.0)
        visual_salient_avg.append(
            sum(visual_salient_vals) / len(visual_salient_vals) if visual_salient_vals else 0.0
        )
        visual_nonsalient_avg.append(
            sum(visual_nonsalient_vals) / len(visual_nonsalient_vals)
            if visual_nonsalient_vals
            else 0.0
        )

    return {
        "TEXT (avg)": text_avg,
        "VISUAL (avg, all)": visual_avg,
        "VISUAL (salient subset)": visual_salient_avg,
        "VISUAL (non-salient)": visual_nonsalient_avg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "importance_by_modality.png"))
    args = parser.parse_args()

    trace_cfg = load_trace_config(Path(args.config))
    session = generate_session(**trace_cfg)

    series = importance_by_modality_over_decode(session, ema_decay=trace_cfg["ema_decay"])
    steps = list(range(1, len(session.steps) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in series.items():
        ax.plot(steps, values, label=label)

    ax.set_xlabel("Decode step")
    ax.set_ylabel("Running EMA importance")
    ax.set_title("Importance by modality over decode (Milestone 1 validation)")
    ax.legend()
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    n_salient = len(session.salient_visual_positions)
    n_visual = sum(1 for e in session.entries if e.modality is Modality.VISUAL)
    print(f"Salient visual subset: {n_salient}/{n_visual} entries")
    print(f"Final avg importance — TEXT: {series['TEXT (avg)'][-1]:.4f}, "
          f"VISUAL all: {series['VISUAL (avg, all)'][-1]:.4f}, "
          f"VISUAL salient: {series['VISUAL (salient subset)'][-1]:.4f}, "
          f"VISUAL non-salient: {series['VISUAL (non-salient)'][-1]:.4f}")


if __name__ == "__main__":
    main()
