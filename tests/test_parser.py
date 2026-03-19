"""Tests for experiment parser."""

import pytest
from contradiction_detector.parser import ExperimentParser, ExperimentRecord


SAMPLE_TSV = """commit\tmetric_value\tmemory_gb\tstatus\tdescription
abc123\t0.85\t4.2\tsuccess\tIncreased learning rate from 0.001 to 0.01
def456\t0.82\t3.8\tsuccess\tAdded dropout regularization p=0.5
ghi789\t0.75\t5.1\tsuccess\tIncreased learning rate from 0.001 to 0.01 with larger model
jkl012\t0.88\t4.5\tsuccess\tSwitched to Adam optimizer
mno345\t0.78\t4.0\tsuccess\tAdded batch normalization layers
pqr678\t0.90\t4.3\tsuccess\tReduced learning rate to 0.0001
"""


class TestExperimentRecord:
    def test_creation(self):
        rec = ExperimentRecord(
            commit="abc",
            metric_value=0.85,
            memory_gb=4.2,
            status="success",
            description="test",
            change_type="lr_change",
        )
        assert rec.commit == "abc"
        assert rec.change_type == "lr_change"


class TestExperimentParser:
    def setup_method(self):
        self.parser = ExperimentParser()

    def test_parse_tsv(self):
        records = self.parser.parse_tsv(SAMPLE_TSV)
        assert len(records) == 6
        assert records[0].commit == "abc123"
        assert records[0].metric_value == 0.85

    def test_parse_tsv_file(self, tmp_path):
        f = tmp_path / "results.tsv"
        f.write_text(SAMPLE_TSV)
        records = self.parser.parse_tsv_file(str(f))
        assert len(records) == 6

    def test_categorize_lr_change(self):
        records = self.parser.parse_tsv(SAMPLE_TSV)
        lr_records = [r for r in records if r.change_type == "lr_change"]
        assert len(lr_records) >= 2

    def test_categorize_regularization(self):
        records = self.parser.parse_tsv(SAMPLE_TSV)
        reg_records = [r for r in records if r.change_type == "regularization_change"]
        assert len(reg_records) >= 1

    def test_categorize_optimizer(self):
        records = self.parser.parse_tsv(SAMPLE_TSV)
        opt_records = [r for r in records if r.change_type == "optimizer_change"]
        assert len(opt_records) >= 1

    def test_categorize_architecture(self):
        records = self.parser.parse_tsv(SAMPLE_TSV)
        arch_records = [r for r in records if r.change_type == "architecture_change"]
        assert len(arch_records) >= 1

    def test_empty_tsv(self):
        records = self.parser.parse_tsv("commit\tmetric_value\tmemory_gb\tstatus\tdescription\n")
        assert records == []

    def test_determine_direction_positive(self):
        direction = self.parser.determine_direction(0.85, 0.80)
        assert direction == "positive"

    def test_determine_direction_negative(self):
        direction = self.parser.determine_direction(0.75, 0.80)
        assert direction == "negative"

    def test_determine_direction_neutral(self):
        direction = self.parser.determine_direction(0.80, 0.80)
        assert direction == "neutral"
