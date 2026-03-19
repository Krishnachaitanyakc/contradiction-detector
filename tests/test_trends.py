"""Tests for time-series analysis."""

import pytest
from autoresearch_contradict.trends import (
    parse_timestamp, sort_by_time, analyze_trends, format_trends_report,
)
from autoresearch_contradict.parser import ExperimentRecord


class TestParseTimestamp:
    def test_date_format(self):
        ts = parse_timestamp("2024-01-15")
        assert ts is not None
        assert ts.year == 2024
        assert ts.month == 1
        assert ts.day == 15

    def test_datetime_format(self):
        ts = parse_timestamp("2024-01-15 10:30:00")
        assert ts is not None
        assert ts.hour == 10

    def test_iso_format(self):
        ts = parse_timestamp("2024-01-15T10:30:00")
        assert ts is not None

    def test_none_input(self):
        assert parse_timestamp(None) is None

    def test_invalid_format(self):
        assert parse_timestamp("not-a-date") is None


class TestSortByTime:
    def test_sorts_by_timestamp(self):
        exps = [
            ExperimentRecord("c2", 0.85, 4.0, "success", "test", "lr_change", "2024-01-20"),
            ExperimentRecord("c1", 0.85, 4.0, "success", "test", "lr_change", "2024-01-10"),
            ExperimentRecord("c3", 0.85, 4.0, "success", "test", "lr_change", "2024-01-15"),
        ]
        sorted_exps = sort_by_time(exps)
        assert sorted_exps[0].commit == "c1"
        assert sorted_exps[1].commit == "c3"
        assert sorted_exps[2].commit == "c2"

    def test_none_timestamps_first(self):
        exps = [
            ExperimentRecord("c2", 0.85, 4.0, "success", "test", "lr_change", "2024-01-10"),
            ExperimentRecord("c1", 0.85, 4.0, "success", "test", "lr_change", None),
        ]
        sorted_exps = sort_by_time(exps)
        assert sorted_exps[0].commit == "c1"


class TestAnalyzeTrends:
    def test_finds_contradiction_emergence(self):
        exps = [
            ExperimentRecord("c1", 0.90, 4.0, "success",
                           "Increased learning rate small", "lr_change", "2024-01-01"),
            ExperimentRecord("c2", 0.70, 4.0, "success",
                           "Increased learning rate large", "lr_change", "2024-01-10"),
        ]
        timeline = analyze_trends(exps, baseline_metric=0.80)
        assert len(timeline) >= 1
        assert timeline[0]["cumulative_count"] == 1

    def test_empty_experiments(self):
        timeline = analyze_trends([], baseline_metric=0.80)
        assert timeline == []

    def test_no_contradictions(self):
        exps = [
            ExperimentRecord("c1", 0.85, 4.0, "success",
                           "Increased learning rate", "lr_change", "2024-01-01"),
            ExperimentRecord("c2", 0.88, 4.0, "success",
                           "Increased learning rate more", "lr_change", "2024-01-10"),
        ]
        timeline = analyze_trends(exps, baseline_metric=0.80)
        assert timeline == []


class TestFormatTrendsReport:
    def test_empty_timeline(self):
        result = format_trends_report([])
        assert "No contradiction trends" in result

    def test_non_empty_timeline(self):
        from autoresearch_contradict.detector import Contradiction
        timeline = [{
            "timestamp": "2024-01-10T00:00:00",
            "new_experiment": "c2",
            "contradiction": Contradiction(
                exp_a=ExperimentRecord("c1", 0.90, 4.0, "success", "test a", "lr_change"),
                exp_b=ExperimentRecord("c2", 0.70, 4.0, "success", "test b", "lr_change"),
                change_type="lr_change",
                direction_a="positive",
                direction_b="negative",
                context_diff="test",
                confidence=0.7,
            ),
            "cumulative_count": 1,
        }]
        result = format_trends_report(timeline)
        assert "Contradiction Trends" in result
        assert "lr_change" in result
