"""Detect contradictory experiment results."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

from autoresearch_contradict.embedder import SimpleEmbedder
from autoresearch_contradict.parser import ExperimentParser, ExperimentRecord

CHANGE_TYPE_SPECIFICITY = {
    "lr_change": 0.9,
    "batch_size_change": 0.85,
    "warmup_change": 0.8,
    "weight_init_change": 0.8,
    "loss_function_change": 0.85,
    "regularization_change": 0.7,
    "optimizer_change": 0.7,
    "architecture_change": 0.6,
    "data_augmentation_change": 0.6,
    "data_preprocessing_change": 0.5,
    "other": 0.3,
}


@dataclass
class Contradiction:
    """A detected contradiction between two experiments."""

    exp_a: ExperimentRecord
    exp_b: ExperimentRecord
    change_type: str
    direction_a: str
    direction_b: str
    context_diff: str
    confidence: float = field(default=0.5)


class ContradictionDetector:
    """Detect contradictions in experiment results."""

    def __init__(self, baseline_metric: float = 0.80) -> None:
        self.baseline_metric = baseline_metric
        self._parser = ExperimentParser()
        self._embedder = SimpleEmbedder()

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
                    confidence = self._compute_confidence(exp_a, exp_b, change_type)
                    contradictions.append(
                        Contradiction(
                            exp_a=exp_a,
                            exp_b=exp_b,
                            change_type=change_type,
                            direction_a=dir_a,
                            direction_b=dir_b,
                            context_diff=context_diff,
                            confidence=confidence,
                        )
                    )

        return contradictions

    def _compute_confidence(
        self, exp_a: ExperimentRecord, exp_b: ExperimentRecord, change_type: str
    ) -> float:
        """Compute confidence score from description similarity, metric divergence, and change specificity."""
        # (a) Cosine similarity of descriptions
        vectors = self._embedder.embed([exp_a.description, exp_b.description])
        if len(vectors) == 2:
            desc_sim = self._embedder.cosine_similarity(vectors[0], vectors[1])
        else:
            desc_sim = 0.5

        # (b) Magnitude of metric divergence (normalized)
        metric_diff = abs(exp_a.metric_value - exp_b.metric_value)
        metric_divergence = min(metric_diff / max(self.baseline_metric, 0.01), 1.0)

        # (c) Specificity of change type
        specificity = CHANGE_TYPE_SPECIFICITY.get(change_type, 0.3)

        # Weighted average: similarity 0.3, divergence 0.4, specificity 0.3
        confidence = 0.3 * desc_sim + 0.4 * metric_divergence + 0.3 * specificity
        return round(min(max(confidence, 0.0), 1.0), 3)

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
