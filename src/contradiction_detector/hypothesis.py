"""Template-based hypothesis generation for contradictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from contradiction_detector.detector import Contradiction


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
        {
            "explanation": (
                "The divergence ({metric_pct_change:.1f}% metric difference) "
                "suggests the learning rate interacts with other hyperparameters. "
                "Memory usage: A={memory_a:.1f}GB, B={memory_b:.1f}GB."
            ),
            "confidence": 0.6,
            "suggested_experiment": (
                "Binary search: try LR midpoint between experiments A and B "
                "values. Test with both configurations to isolate the interaction."
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
        {
            "explanation": (
                "The {metric_pct_change:.1f}% metric difference suggests "
                "regularization strength needs to be tuned per-dataset. "
                "Memory footprint difference: {memory_a:.1f}GB vs {memory_b:.1f}GB."
            ),
            "confidence": 0.5,
            "suggested_experiment": (
                "Sweep regularization coefficient in [0.01, 0.1, 0.3, 0.5] "
                "for both setups. Plot validation curves to find optimal range."
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
        {
            "explanation": (
                "Optimizer sensitivity ({metric_pct_change:.1f}% divergence) "
                "may indicate different convergence requirements. "
                "Memory: A={memory_a:.1f}GB, B={memory_b:.1f}GB."
            ),
            "confidence": 0.45,
            "suggested_experiment": (
                "Run both optimizers with a grid of learning rates "
                "[1e-4, 5e-4, 1e-3, 5e-3] to separate optimizer from LR effects."
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
        {
            "explanation": (
                "Architecture change produced {metric_pct_change:.1f}% divergence. "
                "This may indicate scale-dependent benefits. "
                "Memory: A={memory_a:.1f}GB vs B={memory_b:.1f}GB."
            ),
            "confidence": 0.45,
            "suggested_experiment": (
                "Test architecture at 3 scales: 0.5x, 1x, 2x the current "
                "parameter count. Monitor both metric and memory."
            ),
        },
    ],
    "batch_size_change": [
        {
            "explanation": (
                "Batch size change improved A (metric={metric_a:.4f}) but "
                "degraded B (metric={metric_b:.4f}). Hypothesis: optimal batch "
                "size depends on learning rate and model capacity ({context_diff})."
            ),
            "confidence": 0.65,
            "suggested_experiment": (
                "Try linear scaling rule: adjust LR proportionally to batch size "
                "change. Test batch sizes at powers of 2 between A and B values."
            ),
        },
    ],
    "warmup_change": [
        {
            "explanation": (
                "Warmup schedule change helped A but hurt B. "
                "Hypothesis: warmup duration should scale with dataset size "
                "and learning rate ({context_diff})."
            ),
            "confidence": 0.6,
            "suggested_experiment": (
                "Test warmup durations of [100, 500, 1000, 5000] steps "
                "with both configurations. Monitor loss stability."
            ),
        },
    ],
    "weight_init_change": [
        {
            "explanation": (
                "Weight initialization change produced opposite results "
                "(A={metric_a:.4f}, B={metric_b:.4f}). Hypothesis: initialization "
                "strategy interacts with architecture depth ({context_diff})."
            ),
            "confidence": 0.55,
            "suggested_experiment": (
                "Compare Xavier, Kaiming, and orthogonal initialization "
                "across different network depths [2, 4, 8, 16 layers]."
            ),
        },
    ],
    "loss_function_change": [
        {
            "explanation": (
                "Loss function change improved A but hurt B. "
                "Hypothesis: the loss function's effectiveness depends on "
                "class distribution or task type ({context_diff})."
            ),
            "confidence": 0.6,
            "suggested_experiment": (
                "Run ablation with both loss functions on datasets with "
                "varying class imbalance ratios [1:1, 1:10, 1:100]."
            ),
        },
    ],
    "data_preprocessing_change": [
        {
            "explanation": (
                "Data preprocessing change produced opposite results. "
                "Hypothesis: preprocessing interacts with data distribution "
                "or model architecture ({context_diff})."
            ),
            "confidence": 0.5,
            "suggested_experiment": (
                "Apply each preprocessing step independently and measure "
                "impact. Check data distribution statistics before/after."
            ),
        },
    ],
    "data_augmentation_change": [
        {
            "explanation": (
                "Data augmentation improved A but degraded B. "
                "Hypothesis: augmentation may not preserve task-relevant "
                "features in certain data distributions ({context_diff})."
            ),
            "confidence": 0.55,
            "suggested_experiment": (
                "Test individual augmentation techniques separately. "
                "Measure augmentation impact on both training and validation distributions."
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

        metric_diff = abs(
            contradiction.exp_a.metric_value - contradiction.exp_b.metric_value
        )
        baseline_avg = (
            contradiction.exp_a.metric_value + contradiction.exp_b.metric_value
        ) / 2
        metric_pct_change = (metric_diff / max(baseline_avg, 0.001)) * 100

        format_vars = {
            "metric_a": contradiction.exp_a.metric_value,
            "metric_b": contradiction.exp_b.metric_value,
            "direction_a": contradiction.direction_a,
            "direction_b": contradiction.direction_b,
            "context_diff": contradiction.context_diff,
            "change_type": contradiction.change_type,
            "desc_a": contradiction.exp_a.description,
            "desc_b": contradiction.exp_b.description,
            "memory_a": contradiction.exp_a.memory_gb,
            "memory_b": contradiction.exp_b.memory_gb,
            "metric_pct_change": metric_pct_change,
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

        # Feature 5: Auto experiment suggestions - parameter sweep
        hypotheses.append(self._generate_sweep_suggestion(contradiction, format_vars))

        return hypotheses

    def _generate_sweep_suggestion(
        self, contradiction: Contradiction, format_vars: dict
    ) -> Hypothesis:
        """Generate a binary search parameter sweep suggestion."""
        metric_a = contradiction.exp_a.metric_value
        metric_b = contradiction.exp_b.metric_value
        midpoint = (metric_a + metric_b) / 2

        return Hypothesis(
            explanation=(
                f"Parameter sweep suggestion: the contradicting experiments "
                f"({format_vars['desc_a']!r} vs {format_vars['desc_b']!r}) "
                f"achieved metrics {metric_a:.4f} and {metric_b:.4f}. "
                f"A binary search between their parameter values may find "
                f"the transition point."
            ),
            confidence=0.35,
            suggested_experiment=(
                f"Binary search: test parameter values at the midpoint between "
                f"the two experiments' settings. Target metric region around "
                f"{midpoint:.4f}. Iterate 3-5 times to narrow the transition boundary."
            ),
        )
