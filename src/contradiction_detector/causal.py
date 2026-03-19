"""Simple heuristic causal inference for contradictions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List

from contradiction_detector.detector import Contradiction
from contradiction_detector.embedder import SimpleEmbedder
from contradiction_detector.parser import ExperimentRecord


@dataclass
class Confounder:
    """A potential confounding variable."""

    name: str
    description: str
    strength: float  # 0-1, how strongly it co-varies


@dataclass
class CausalAnalysis:
    """Causal analysis result for a contradiction."""

    contradiction: Contradiction
    confounders: List[Confounder]
    summary: str


def identify_confounders(
    contradiction: Contradiction,
    all_experiments: List[ExperimentRecord],
) -> CausalAnalysis:
    """Identify potential confounders for a contradiction.

    Uses heuristics to find variables that co-vary with both the change
    and the outcome.
    """
    confounders: List[Confounder] = []

    # Check memory_gb as confounder
    mem_diff = abs(contradiction.exp_a.memory_gb - contradiction.exp_b.memory_gb)
    if mem_diff > 0.5:
        strength = min(mem_diff / 5.0, 1.0)
        confounders.append(Confounder(
            name="memory_usage",
            description=(
                f"Memory differs by {mem_diff:.1f}GB "
                f"(A={contradiction.exp_a.memory_gb:.1f}GB, "
                f"B={contradiction.exp_b.memory_gb:.1f}GB), "
                f"suggesting different model sizes or batch sizes."
            ),
            strength=round(strength, 3),
        ))

    # Check description keywords as confounders
    words_a = set(contradiction.exp_a.description.lower().split())
    words_b = set(contradiction.exp_b.description.lower().split())
    unique_a = words_a - words_b
    unique_b = words_b - words_a

    context_keywords = {
        "small", "large", "big", "tiny", "deep", "shallow",
        "wide", "narrow", "fast", "slow", "short", "long",
    }

    for word in unique_a | unique_b:
        if word in context_keywords:
            confounders.append(Confounder(
                name=f"context_{word}",
                description=(
                    f"Keyword '{word}' appears in only one experiment, "
                    f"suggesting a contextual difference that may confound results."
                ),
                strength=0.6,
            ))

    # Check if other experiments with the same change type show a pattern
    same_type = [
        e for e in all_experiments
        if e.change_type == contradiction.change_type
        and e.commit not in (contradiction.exp_a.commit, contradiction.exp_b.commit)
    ]

    if same_type:
        metrics = [e.metric_value for e in same_type]
        avg_metric = sum(metrics) / len(metrics)
        metric_variance = sum((m - avg_metric) ** 2 for m in metrics) / len(metrics)

        if metric_variance > 0.01:
            confounders.append(Confounder(
                name="high_variance_change_type",
                description=(
                    f"Other experiments with {contradiction.change_type} show "
                    f"high variance (var={metric_variance:.4f}), suggesting "
                    f"this change type is inherently context-dependent."
                ),
                strength=round(min(metric_variance * 10, 1.0), 3),
            ))

    # Sort by strength
    confounders.sort(key=lambda c: c.strength, reverse=True)

    summary = _build_summary(contradiction, confounders)

    return CausalAnalysis(
        contradiction=contradiction,
        confounders=confounders,
        summary=summary,
    )


def _build_summary(
    contradiction: Contradiction,
    confounders: List[Confounder],
) -> str:
    """Build a human-readable causal analysis summary."""
    if not confounders:
        return (
            f"No obvious confounders found for {contradiction.change_type} "
            f"contradiction between {contradiction.exp_a.commit} and "
            f"{contradiction.exp_b.commit}. The contradiction may be due to "
            f"randomness or unmeasured variables."
        )

    lines = [
        f"Causal analysis for {contradiction.change_type} contradiction "
        f"({contradiction.exp_a.commit} vs {contradiction.exp_b.commit}):",
    ]
    for conf in confounders:
        lines.append(f"  - [{conf.strength:.0%}] {conf.name}: {conf.description}")

    return "\n".join(lines)
