"""Template-based hypothesis generation for contradictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from autoresearch_contradict.detector import Contradiction


@dataclass
class Hypothesis:
    """A hypothesis explaining a contradiction."""

    explanation: str
    confidence: float
    suggested_experiment: str


HYPOTHESIS_TEMPLATES = {
    "lr_change": [
        {
            "explanation": (
                "The learning rate change helped in experiment A "
                "(metric={metric_a:.4f}) but hurt in experiment B "
                "(metric={metric_b:.4f}). Hypothesis: optimal learning rate "
                "depends on context ({context_diff})."
            ),
            "confidence": 0.7,
            "suggested_experiment": (
                "Try intermediate learning rate values, controlling for "
                "the context difference: {context_diff}"
            ),
        },
        {
            "explanation": (
                "Learning rate sensitivity may vary with model scale or "
                "training stage. The same LR change produced opposite results."
            ),
            "confidence": 0.5,
            "suggested_experiment": (
                "Run a learning rate sweep across different model sizes "
                "to map the optimal LR landscape."
            ),
        },
    ],
    "regularization_change": [
        {
            "explanation": (
                "Regularization helped in experiment A (metric={metric_a:.4f}) "
                "but hurt in experiment B (metric={metric_b:.4f}). "
                "Hypothesis: the model in B was already under-fitting, "
                "so additional regularization was harmful."
            ),
            "confidence": 0.6,
            "suggested_experiment": (
                "Check training loss curves for both experiments. "
                "If B shows high training loss, reduce regularization strength."
            ),
        },
    ],
    "optimizer_change": [
        {
            "explanation": (
                "The optimizer change improved experiment A but degraded B. "
                "Hypothesis: the optimizer interacts with the loss landscape, "
                "which differs between the two setups ({context_diff})."
            ),
            "confidence": 0.5,
            "suggested_experiment": (
                "Try the optimizer with different hyperparameter settings "
                "(lr, beta1, beta2) for each setup."
            ),
        },
    ],
    "architecture_change": [
        {
            "explanation": (
                "The architecture change helped in A but hurt in B. "
                "Hypothesis: the architectural modification's effectiveness "
                "depends on the scale or complexity of the task ({context_diff})."
            ),
            "confidence": 0.5,
            "suggested_experiment": (
                "Test the architecture change with varying model depths "
                "and widths to find the sweet spot."
            ),
        },
    ],
}

DEFAULT_TEMPLATE = [
    {
        "explanation": (
            "The same type of change ({change_type}) produced opposite results: "
            "experiment A (metric={metric_a:.4f}, {direction_a}) vs "
            "experiment B (metric={metric_b:.4f}, {direction_b}). "
            "Context difference: {context_diff}"
        ),
        "confidence": 0.4,
        "suggested_experiment": (
            "Run ablation study to isolate which factors cause the "
            "divergent outcomes for {change_type}."
        ),
    },
]


class HypothesisGenerator:
    """Generate hypotheses explaining contradictions."""

    def generate_hypotheses(
        self, contradiction: Contradiction
    ) -> List[Hypothesis]:
        """Generate hypotheses for a contradiction."""
        templates = HYPOTHESIS_TEMPLATES.get(
            contradiction.change_type, DEFAULT_TEMPLATE
        )

        format_vars = {
            "metric_a": contradiction.exp_a.metric_value,
            "metric_b": contradiction.exp_b.metric_value,
            "direction_a": contradiction.direction_a,
            "direction_b": contradiction.direction_b,
            "context_diff": contradiction.context_diff,
            "change_type": contradiction.change_type,
            "desc_a": contradiction.exp_a.description,
            "desc_b": contradiction.exp_b.description,
        }

        hypotheses = []
        for template in templates:
            hypotheses.append(
                Hypothesis(
                    explanation=template["explanation"].format(**format_vars),
                    confidence=template["confidence"],
                    suggested_experiment=template["suggested_experiment"].format(
                        **format_vars
                    ),
                )
            )

        return hypotheses
