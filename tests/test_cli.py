"""Tests for CLI."""

import pytest
from click.testing import CliRunner
from contradiction_detector.cli import cli


SAMPLE_TSV = """commit\tmetric_value\tmemory_gb\tstatus\tdescription
abc123\t0.90\t4.2\tsuccess\tIncreased learning rate to 0.01 for small model
def456\t0.70\t3.8\tsuccess\tIncreased learning rate to 0.01 for large model
ghi789\t0.85\t5.1\tsuccess\tAdded dropout regularization small dataset
jkl012\t0.75\t4.5\tsuccess\tAdded dropout regularization large dataset
"""


class TestCLI:
    def setup_method(self):
        self.runner = CliRunner()

    def test_cli_help(self):
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_analyze_command(self, tmp_path):
        f = tmp_path / "results.tsv"
        f.write_text(SAMPLE_TSV)
        result = self.runner.invoke(cli, ["analyze", str(f)])
        assert result.exit_code == 0

    def test_contradictions_command(self, tmp_path):
        f = tmp_path / "results.tsv"
        f.write_text(SAMPLE_TSV)
        result = self.runner.invoke(cli, ["contradictions", str(f)])
        assert result.exit_code == 0

    def test_hypotheses_command(self, tmp_path):
        f = tmp_path / "results.tsv"
        f.write_text(SAMPLE_TSV)
        result = self.runner.invoke(cli, ["hypotheses", str(f)])
        assert result.exit_code == 0

    def test_report_command(self, tmp_path):
        f = tmp_path / "results.tsv"
        f.write_text(SAMPLE_TSV)
        result = self.runner.invoke(cli, ["report", str(f)])
        assert result.exit_code == 0

    def test_missing_file(self):
        result = self.runner.invoke(cli, ["analyze", "/nonexistent/file.tsv"])
        assert result.exit_code != 0 or "error" in result.output.lower() or "not found" in result.output.lower()
