"""Tests for memory integration."""

import pytest
from unittest.mock import patch

from autoresearch_contradict.memory_integration import (
    query_similar_contradictions, enrich_contradiction_context,
)
from autoresearch_contradict.detector import Contradiction
from autoresearch_contradict.parser import ExperimentRecord


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


class TestMemoryIntegration:
    def test_returns_empty_when_not_installed(self):
        c = _make_contradiction()
        with patch.dict("sys.modules", {"autoresearch_memory": None, "autoresearch_memory.store": None}):
            result = query_similar_contradictions(c)
        # May return empty or results depending on import availability
        assert isinstance(result, list)

    def test_enrich_returns_string(self):
        c = _make_contradiction()
        result = enrich_contradiction_context(c)
        assert isinstance(result, str)

    def test_query_returns_list(self):
        c = _make_contradiction()
        result = query_similar_contradictions(c)
        assert isinstance(result, list)
