"""Tests for cross-project contradiction detection."""

import pytest

from autoresearch_contradict.cross_project import (
    cross_analyze, format_cross_analysis, CrossProjectContradiction,
)
from autoresearch_contradict.detector import Contradiction
from autoresearch_contradict.parser import ExperimentRecord


TSV_PROJECT_A = """commit\tmetric_value\tmemory_gb\tstatus\tdescription
a1\t0.90\t4.0\tsuccess\tIncreased learning rate to 0.01 small model
a2\t0.85\t4.0\tsuccess\tAdded dropout 0.5
"""

TSV_PROJECT_B = """commit\tmetric_value\tmemory_gb\tstatus\tdescription
b1\t0.70\t4.0\tsuccess\tIncreased learning rate to 0.01 large model
b2\t0.88\t4.0\tsuccess\tAdded batch normalization
"""


class TestCrossAnalyze:
    def test_finds_cross_project_contradictions(self, tmp_path):
        f_a = tmp_path / "project_a.tsv"
        f_b = tmp_path / "project_b.tsv"
        f_a.write_text(TSV_PROJECT_A)
        f_b.write_text(TSV_PROJECT_B)

        results = cross_analyze([str(f_a), str(f_b)], baseline_metric=0.80)
        # Should find lr_change contradiction across files
        assert isinstance(results, list)
        lr_contradictions = [r for r in results if r.contradiction.change_type == "lr_change"]
        assert len(lr_contradictions) >= 1

    def test_no_cross_contradictions_same_direction(self, tmp_path):
        tsv_c = """commit\tmetric_value\tmemory_gb\tstatus\tdescription
c1\t0.85\t4.0\tsuccess\tIncreased learning rate
"""
        tsv_d = """commit\tmetric_value\tmemory_gb\tstatus\tdescription
d1\t0.88\t4.0\tsuccess\tIncreased learning rate too
"""
        f_c = tmp_path / "project_c.tsv"
        f_d = tmp_path / "project_d.tsv"
        f_c.write_text(tsv_c)
        f_d.write_text(tsv_d)

        results = cross_analyze([str(f_c), str(f_d)], baseline_metric=0.80)
        assert len(results) == 0

    def test_empty_files(self, tmp_path):
        f = tmp_path / "empty.tsv"
        f.write_text("commit\tmetric_value\tmemory_gb\tstatus\tdescription\n")
        results = cross_analyze([str(f)], baseline_metric=0.80)
        assert results == []


class TestFormatCrossAnalysis:
    def test_no_results(self):
        result = format_cross_analysis([])
        assert "No cross-project" in result

    def test_with_results(self):
        results = [
            CrossProjectContradiction(
                source_file_a="a.tsv",
                source_file_b="b.tsv",
                contradiction=Contradiction(
                    exp_a=ExperimentRecord("c1", 0.90, 4.0, "success", "lr increase", "lr_change"),
                    exp_b=ExperimentRecord("c2", 0.70, 4.0, "success", "lr increase", "lr_change"),
                    change_type="lr_change",
                    direction_a="positive",
                    direction_b="negative",
                    context_diff="test",
                    confidence=0.7,
                ),
            ),
        ]
        output = format_cross_analysis(results)
        assert "Cross-Project" in output
        assert "a.tsv" in output
        assert "b.tsv" in output
