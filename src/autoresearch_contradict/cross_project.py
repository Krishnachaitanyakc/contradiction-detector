"""Cross-project contradiction detection across multiple results files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from autoresearch_contradict.detector import Contradiction, ContradictionDetector
from autoresearch_contradict.parser import ExperimentParser, ExperimentRecord


@dataclass
class CrossProjectContradiction:
    """A contradiction found across different project files."""

    source_file_a: str
    source_file_b: str
    contradiction: Contradiction


def cross_analyze(
    filepaths: List[str],
    baseline_metric: float = 0.80,
) -> List[CrossProjectContradiction]:
    """Find contradictions across multiple results.tsv files.

    Compares experiments from different files to find cross-project
    contradictions where the same type of change produced opposite results.
    """
    parser = ExperimentParser()
    detector = ContradictionDetector(baseline_metric=baseline_metric)

    # Parse all files and tag with source
    file_experiments: List[tuple[str, List[ExperimentRecord]]] = []
    for filepath in filepaths:
        try:
            exps = parser.parse_tsv_file(filepath)
            file_experiments.append((filepath, exps))
        except Exception:
            continue

    cross_contradictions: List[CrossProjectContradiction] = []

    # Compare experiments across different files
    for i in range(len(file_experiments)):
        for j in range(i + 1, len(file_experiments)):
            file_a, exps_a = file_experiments[i]
            file_b, exps_b = file_experiments[j]

            combined = exps_a + exps_b
            contradictions = detector.find_contradictions(combined)

            for c in contradictions:
                a_in_file_a = c.exp_a in exps_a
                b_in_file_b = c.exp_b in exps_b
                a_in_file_b = c.exp_a in exps_b
                b_in_file_a = c.exp_b in exps_a

                is_cross = (a_in_file_a and b_in_file_b) or (a_in_file_b and b_in_file_a)

                if is_cross:
                    src_a = file_a if a_in_file_a else file_b
                    src_b = file_b if b_in_file_b else file_a
                    cross_contradictions.append(
                        CrossProjectContradiction(
                            source_file_a=src_a,
                            source_file_b=src_b,
                            contradiction=c,
                        )
                    )

    return cross_contradictions


def format_cross_analysis(results: List[CrossProjectContradiction]) -> str:
    """Format cross-project analysis results."""
    if not results:
        return "No cross-project contradictions found."

    lines = [f"# Cross-Project Contradiction Analysis\n"]
    lines.append(f"Found {len(results)} cross-project contradiction(s).\n")

    for i, cpc in enumerate(results, 1):
        c = cpc.contradiction
        lines.append(f"## Contradiction {i}: {c.change_type}")
        lines.append(f"- File A: {cpc.source_file_a}")
        lines.append(f"  Experiment: [{c.exp_a.commit}] {c.exp_a.description} ({c.direction_a})")
        lines.append(f"- File B: {cpc.source_file_b}")
        lines.append(f"  Experiment: [{c.exp_b.commit}] {c.exp_b.description} ({c.direction_b})")
        lines.append(f"- Confidence: {c.confidence:.3f}")
        lines.append("")

    return "\n".join(lines)
