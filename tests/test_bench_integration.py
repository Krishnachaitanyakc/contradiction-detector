"""Tests for bench integration."""

import pytest
from unittest.mock import patch, MagicMock

from contradiction_detector.bench_integration import (
    search_related_papers, format_paper_context,
)
from contradiction_detector.detector import Contradiction
from contradiction_detector.parser import ExperimentRecord


def _make_contradiction():
    return Contradiction(
        exp_a=ExperimentRecord("c1", 0.90, 4.0, "success",
                             "Increased learning rate", "lr_change"),
        exp_b=ExperimentRecord("c2", 0.70, 4.0, "success",
                             "Increased learning rate large", "lr_change"),
        change_type="lr_change",
        direction_a="positive",
        direction_b="negative",
        context_diff="model size",
        confidence=0.7,
    )


class TestBenchIntegration:
    def test_returns_empty_when_not_installed(self):
        c = _make_contradiction()
        with patch.dict("sys.modules", {"autoresearch_bench": None, "autoresearch_bench.papers": None}):
            result = search_related_papers(c)
        assert isinstance(result, list)

    def test_format_empty_papers(self):
        result = format_paper_context([])
        assert result == ""

    def test_format_papers(self):
        papers = [
            {"title": "Test Paper", "abstract": "An abstract", "url": "http://example.com"},
        ]
        result = format_paper_context(papers)
        assert "Test Paper" in result
        assert "Related literature" in result
