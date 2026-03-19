"""Full pipeline: parse -> detect contradictions -> generate hypotheses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from autoresearch_contradict.detector import Contradiction, ContradictionDetector
from autoresearch_contradict.hypothesis import Hypothesis, HypothesisGenerator
from autoresearch_contradict.parser import ExperimentParser, ExperimentRecord


@dataclass
class AnalysisReport:
    """Complete analysis report."""

    experiments: List[ExperimentRecord]
    contradictions: List[Contradiction]
    hypotheses: List[Hypothesis]
    markdown: str


class ContradictionAnalyzer:
    """Full pipeline for contradiction analysis."""

    def __init__(self, baseline_metric: float = 0.80) -> None:
        self.parser = ExperimentParser()
        self.detector = ContradictionDetector(baseline_metric=baseline_metric)
        self.hypothesis_gen = HypothesisGenerator()

    def analyze_tsv(self, tsv_content: str) -> AnalysisReport:
        """Run full analysis on TSV content."""
        experiments = self.parser.parse_tsv(tsv_content)
        return self._analyze(experiments)

    def analyze_file(self, filepath: str) -> AnalysisReport:
        """Run full analysis on a TSV file."""
        experiments = self.parser.parse_tsv_file(filepath)
        return self._analyze(experiments)

    def _analyze(self, experiments: List[ExperimentRecord]) -> AnalysisReport:
        """Core analysis logic."""
        contradictions = self.detector.find_contradictions(experiments)

        all_hypotheses: List[Hypothesis] = []
        for contradiction in contradictions:
            hypotheses = self.hypothesis_gen.generate_hypotheses(contradiction)
            all_hypotheses.extend(hypotheses)

        markdown = self._generate_markdown(experiments, contradictions, all_hypotheses)

        return AnalysisReport(
            experiments=experiments,
            contradictions=contradictions,
            hypotheses=all_hypotheses,
            markdown=markdown,
        )

    def _generate_markdown(
        self,
        experiments: List[ExperimentRecord],
        contradictions: List[Contradiction],
        hypotheses: List[Hypothesis],
    ) -> str:
        """Generate a markdown report."""
        lines = ["# Contradiction Analysis Report\n"]

        lines.append(f"## Summary\n")
        lines.append(f"- Total experiments: {len(experiments)}")
        lines.append(f"- Contradictions found: {len(contradictions)}")
        lines.append(f"- Hypotheses generated: {len(hypotheses)}")
        lines.append("")

        if contradictions:
            lines.append("## Contradictions\n")
            for i, c in enumerate(contradictions, 1):
                lines.append(f"### Contradiction {i}: {c.change_type}\n")
                lines.append(
                    f"- **Experiment A** [{c.exp_a.commit}]: "
                    f"{c.exp_a.description} (metric={c.exp_a.metric_value:.4f}, {c.direction_a})"
                )
                lines.append(
                    f"- **Experiment B** [{c.exp_b.commit}]: "
                    f"{c.exp_b.description} (metric={c.exp_b.metric_value:.4f}, {c.direction_b})"
                )
                lines.append(f"- **Context diff**: {c.context_diff}")
                lines.append("")

        if hypotheses:
            lines.append("## Hypotheses\n")
            for i, h in enumerate(hypotheses, 1):
                lines.append(f"### Hypothesis {i} (confidence: {h.confidence:.0%})\n")
                lines.append(f"{h.explanation}\n")
                lines.append(f"**Suggested experiment**: {h.suggested_experiment}\n")

        return "\n".join(lines)
