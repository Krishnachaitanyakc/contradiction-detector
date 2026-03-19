"""LLM-powered hypothesis generation using Anthropic Claude."""

from __future__ import annotations

import os
from typing import List

from autoresearch_contradict.detector import Contradiction
from autoresearch_contradict.hypothesis import Hypothesis, HypothesisGenerator


def generate_llm_hypotheses(
    contradiction: Contradiction,
    api_key: str | None = None,
) -> List[Hypothesis]:
    """Generate hypotheses using Claude LLM.

    Falls back to template-based generation if LLM is unavailable.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return _fallback(contradiction)

    try:
        import anthropic
    except ImportError:
        return _fallback(contradiction)

    prompt = (
        f"You are a machine learning research assistant. Analyze this contradiction "
        f"in experiment results and generate hypotheses.\n\n"
        f"Change type: {contradiction.change_type}\n"
        f"Experiment A: {contradiction.exp_a.description} "
        f"(metric={contradiction.exp_a.metric_value:.4f}, direction={contradiction.direction_a}, "
        f"memory={contradiction.exp_a.memory_gb:.1f}GB)\n"
        f"Experiment B: {contradiction.exp_b.description} "
        f"(metric={contradiction.exp_b.metric_value:.4f}, direction={contradiction.direction_b}, "
        f"memory={contradiction.exp_b.memory_gb:.1f}GB)\n"
        f"Context difference: {contradiction.context_diff}\n\n"
        f"Provide exactly 2 hypotheses. For each, give:\n"
        f"1. A one-paragraph explanation of why these results contradict\n"
        f"2. A confidence score between 0.0 and 1.0\n"
        f"3. A specific suggested experiment to resolve the contradiction\n\n"
        f"Format each hypothesis as:\n"
        f"HYPOTHESIS:\n"
        f"EXPLANATION: <explanation>\n"
        f"CONFIDENCE: <float>\n"
        f"EXPERIMENT: <suggestion>\n"
    )

    try:
        client = anthropic.Anthropic(api_key=key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text
        return _parse_llm_response(text)
    except Exception:
        return _fallback(contradiction)


def _parse_llm_response(text: str) -> List[Hypothesis]:
    """Parse LLM response into Hypothesis objects."""
    hypotheses = []
    blocks = text.split("HYPOTHESIS:")[1:]  # Skip text before first HYPOTHESIS

    for block in blocks:
        explanation = ""
        confidence = 0.5
        experiment = ""

        for line in block.strip().split("\n"):
            line = line.strip()
            if line.startswith("EXPLANATION:"):
                explanation = line[len("EXPLANATION:"):].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line[len("CONFIDENCE:"):].strip())
                    confidence = min(max(confidence, 0.0), 1.0)
                except ValueError:
                    confidence = 0.5
            elif line.startswith("EXPERIMENT:"):
                experiment = line[len("EXPERIMENT:"):].strip()

        if explanation:
            hypotheses.append(
                Hypothesis(
                    explanation=explanation,
                    confidence=confidence,
                    suggested_experiment=experiment or "Further investigation needed.",
                )
            )

    return hypotheses if hypotheses else _fallback_empty()


def _fallback(contradiction: Contradiction) -> List[Hypothesis]:
    """Fallback to template-based generation."""
    gen = HypothesisGenerator()
    return gen.generate_hypotheses(contradiction)


def _fallback_empty() -> List[Hypothesis]:
    """Return a minimal hypothesis when parsing fails."""
    return [
        Hypothesis(
            explanation="LLM response could not be parsed. Manual analysis recommended.",
            confidence=0.2,
            suggested_experiment="Review the contradiction manually and design a targeted experiment.",
        )
    ]
