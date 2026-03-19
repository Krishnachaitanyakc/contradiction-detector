"""Tests for new change type categories."""

import pytest
from autoresearch_contradict.parser import ExperimentParser


class TestNewChangeCategories:
    def setup_method(self):
        self.parser = ExperimentParser()

    def test_batch_size_change(self):
        tsv = "commit\tmetric_value\tmemory_gb\tstatus\tdescription\nc1\t0.85\t4.0\tsuccess\tIncreased batch size to 64\n"
        records = self.parser.parse_tsv(tsv)
        assert records[0].change_type == "batch_size_change"

    def test_warmup_change(self):
        tsv = "commit\tmetric_value\tmemory_gb\tstatus\tdescription\nc1\t0.85\t4.0\tsuccess\tAdded warmup schedule for 1000 steps\n"
        records = self.parser.parse_tsv(tsv)
        assert records[0].change_type == "warmup_change"

    def test_weight_init_change(self):
        tsv = "commit\tmetric_value\tmemory_gb\tstatus\tdescription\nc1\t0.85\t4.0\tsuccess\tSwitched to xavier weight initialization\n"
        records = self.parser.parse_tsv(tsv)
        assert records[0].change_type == "weight_init_change"

    def test_kaiming_init(self):
        tsv = "commit\tmetric_value\tmemory_gb\tstatus\tdescription\nc1\t0.85\t4.0\tsuccess\tUsed kaiming initialization\n"
        records = self.parser.parse_tsv(tsv)
        assert records[0].change_type == "weight_init_change"

    def test_loss_function_change(self):
        tsv = "commit\tmetric_value\tmemory_gb\tstatus\tdescription\nc1\t0.85\t4.0\tsuccess\tSwitched to focal loss function\n"
        records = self.parser.parse_tsv(tsv)
        assert records[0].change_type == "loss_function_change"

    def test_cross_entropy_loss(self):
        tsv = "commit\tmetric_value\tmemory_gb\tstatus\tdescription\nc1\t0.85\t4.0\tsuccess\tUsed cross entropy loss\n"
        records = self.parser.parse_tsv(tsv)
        assert records[0].change_type == "loss_function_change"

    def test_data_preprocessing_change(self):
        tsv = "commit\tmetric_value\tmemory_gb\tstatus\tdescription\nc1\t0.85\t4.0\tsuccess\tChanged data preprocessing pipeline\n"
        records = self.parser.parse_tsv(tsv)
        assert records[0].change_type == "data_preprocessing_change"

    def test_tokenizer_preprocessing(self):
        tsv = "commit\tmetric_value\tmemory_gb\tstatus\tdescription\nc1\t0.85\t4.0\tsuccess\tChanged tokenizer settings\n"
        records = self.parser.parse_tsv(tsv)
        assert records[0].change_type == "data_preprocessing_change"

    def test_mini_batch_change(self):
        tsv = "commit\tmetric_value\tmemory_gb\tstatus\tdescription\nc1\t0.85\t4.0\tsuccess\tReduced mini-batch count\n"
        records = self.parser.parse_tsv(tsv)
        assert records[0].change_type == "batch_size_change"

    def test_timestamp_parsing(self):
        tsv = "commit\tmetric_value\tmemory_gb\tstatus\tdescription\ttimestamp\nc1\t0.85\t4.0\tsuccess\ttest\t2024-01-15\n"
        records = self.parser.parse_tsv(tsv)
        assert records[0].timestamp == "2024-01-15"

    def test_no_timestamp(self):
        tsv = "commit\tmetric_value\tmemory_gb\tstatus\tdescription\nc1\t0.85\t4.0\tsuccess\ttest\n"
        records = self.parser.parse_tsv(tsv)
        assert records[0].timestamp is None
