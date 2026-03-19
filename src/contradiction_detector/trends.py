"""Time-series analysis of contradiction emergence."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from contradiction_detector.detector import Contradiction, ContradictionDetector
from contradiction_detector.parser import ExperimentRecord


def parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    """Parse a timestamp string in common formats."""
    if not ts:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def sort_by_time(experiments: List[ExperimentRecord]) -> List[ExperimentRecord]:
    """Sort experiments by timestamp, putting None-timestamp ones first."""
    def sort_key(exp: ExperimentRecord) -> Tuple[int, str]:
        ts = parse_timestamp(exp.timestamp)
        if ts is None:
            return (0, "")
        return (1, ts.isoformat())
    return sorted(experiments, key=sort_key)


def analyze_trends(
    experiments: List[ExperimentRecord],
    baseline_metric: float = 0.80,
) -> List[dict]:
    """Analyze how contradictions emerge over time.

    Returns a list of dicts with:
    - timestamp: when the contradiction was first detectable
    - contradiction: the Contradiction object
    - cumulative_count: running count of contradictions
    """
    sorted_exps = sort_by_time(experiments)
    detector = ContradictionDetector(baseline_metric=baseline_metric)

    timeline = []
    seen_pairs = set()
    cumulative = 0

    for i in range(2, len(sorted_exps) + 1):
        subset = sorted_exps[:i]
        contradictions = detector.find_contradictions(subset)

        for c in contradictions:
            pair_key = (c.exp_a.commit, c.exp_b.commit)
            reverse_key = (c.exp_b.commit, c.exp_a.commit)
            if pair_key not in seen_pairs and reverse_key not in seen_pairs:
                seen_pairs.add(pair_key)
                cumulative += 1

                new_exp = sorted_exps[i - 1]
                ts = parse_timestamp(new_exp.timestamp)

                timeline.append({
                    "timestamp": ts.isoformat() if ts else None,
                    "new_experiment": new_exp.commit,
                    "contradiction": c,
                    "cumulative_count": cumulative,
                })

    return timeline


def format_trends_report(timeline: List[dict]) -> str:
    """Format trends analysis as a human-readable report."""
    if not timeline:
        return "No contradiction trends detected."

    lines = ["# Contradiction Trends Over Time\n"]
    lines.append(f"Total contradictions emerged: {len(timeline)}\n")

    for entry in timeline:
        ts_str = entry["timestamp"] or "unknown time"
        c = entry["contradiction"]
        lines.append(
            f"- [{ts_str}] New contradiction #{entry['cumulative_count']}: "
            f"{c.change_type} between {c.exp_a.commit} and {c.exp_b.commit} "
            f"(triggered by {entry['new_experiment']})"
        )

    return "\n".join(lines)
