"""Tests for full pipeline analyzer."""

import pytest
from autoresearch_contradict.analyzer import ContradictionAnalyzer, AnalysisReport


SAMPLE_TSV = """commit\tmetric_value\tmemory_gb\tstatus\tdescription
abc123\t0.90\t4.2\tsuccess\tIncreased learning rate to 0.01 for small model
def456\t0.70\t3.8\tsuccess\tIncreased learning rate to 0.01 for large model
ghi789\t0.85\t5.1\tsuccess\tAdded dropout regularization small dataset
jkl012\t0.75\t4.5\tsuccess\tAdded dropout regularization large dataset
"""


class TestAnalysisReport:
    def test_creation(self):
        report = AnalysisReport(
            experiments=[],
            contradictions=[],
            hypotheses=[],
            markdown="",
        )
        assert report.experiments == []


class TestContradictionAnalyzer:
    def setup_method(self):
        self.analyzer = ContradictionAnalyzer(baseline_metric=0.80)

    def test_analyze_from_string(self):
        report = self.analyzer.analyze_tsv(SAMPLE_TSV)
        assert isinstance(report, AnalysisReport)
        assert len(report.experiments) == 4
        assert len(report.contradictions) >= 1
        assert len(report.hypotheses) >= 1

    def test_analyze_from_file(self, tmp_path):
        f = tmp_path / "results.tsv"
        f.write_text(SAMPLE_TSV)
        report = self.analyzer.analyze_file(str(f))
        assert isinstance(report, AnalysisReport)
        assert len(report.experiments) == 4

    def test_generate_markdown_report(self):
        report = self.analyzer.analyze_tsv(SAMPLE_TSV)
        assert "# Contradiction Analysis" in report.markdown
        assert len(report.markdown) > 50

    def test_empty_data(self):
        report = self.analyzer.analyze_tsv(
            "commit\tmetric_value\tmemory_gb\tstatus\tdescription\n"
        )
        assert report.experiments == []
        assert report.contradictions == []

    def test_no_contradictions(self):
        tsv = """commit\tmetric_value\tmemory_gb\tstatus\tdescription
c1\t0.85\t4.0\tsuccess\tIncreased learning rate
c2\t0.88\t4.0\tsuccess\tIncreased learning rate more
"""
        report = self.analyzer.analyze_tsv(tsv)
        assert len(report.contradictions) == 0
