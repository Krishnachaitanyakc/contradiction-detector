"""Detect contradictory experiment results."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List

from autoresearch_contradict.parser import ExperimentParser, ExperimentRecord


@dataclass
class Contradiction:
    """A detected contradiction between two experiments."""

    exp_a: ExperimentRecord
    exp_b: ExperimentRecord
    change_type: str
    direction_a: str
    direction_b: str
    context_diff: str


class ContradictionDetector:
    """Detect contradictions in experiment results."""

    def __init__(self, baseline_metric: float = 0.80) -> None:
        self.baseline_metric = baseline_metric
        self._parser = ExperimentParser()

    def find_contradictions(
        self, experiments: List[ExperimentRecord]
    ) -> List[Contradiction]:
        """Find pairs of experiments with contradictory results.

        Groups experiments by change_type, then finds pairs within each group
        where one improved and the other worsened relative to baseline.
        """
        if len(experiments) < 2:
            return []

        # Group by change type
        groups: dict = defaultdict(list)
        for exp in experiments:
            if exp.change_type != "other":
                groups[exp.change_type].append(exp)

        contradictions: List[Contradiction] = []

        for change_type, group_exps in groups.items():
            if len(group_exps) < 2:
                continue

            # Find pairs with opposite directions
            for i in range(len(group_exps)):
                for j in range(i + 1, len(group_exps)):
                    exp_a = group_exps[i]
                    exp_b = group_exps[j]

                    dir_a = self._parser.determine_direction(
                        exp_a.metric_value, self.baseline_metric
                    )
                    dir_b = self._parser.determine_direction(
                        exp_b.metric_value, self.baseline_metric
                    )

                    if dir_a == "neutral" or dir_b == "neutral":
                        continue
                    if dir_a == dir_b:
                        continue

                    # Found a contradiction
                    context_diff = self._compute_context_diff(exp_a, exp_b)
                    contradictions.append(
                        Contradiction(
                            exp_a=exp_a,
                            exp_b=exp_b,
                            change_type=change_type,
                            direction_a=dir_a,
                            direction_b=dir_b,
                            context_diff=context_diff,
                        )
                    )

        return contradictions

    def _compute_context_diff(
        self, exp_a: ExperimentRecord, exp_b: ExperimentRecord
    ) -> str:
        """Compute a textual diff of context between two experiments."""
        words_a = set(exp_a.description.lower().split())
        words_b = set(exp_b.description.lower().split())

        only_a = words_a - words_b
        only_b = words_b - words_a

        parts = []
        if only_a:
            parts.append(f"A mentions: {', '.join(sorted(only_a))}")
        if only_b:
            parts.append(f"B mentions: {', '.join(sorted(only_b))}")

        return "; ".join(parts) if parts else "No obvious context difference"
