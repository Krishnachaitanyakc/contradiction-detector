"""Tests for LLM hypothesis generation."""

import pytest
from unittest.mock import patch, MagicMock

from autoresearch_contradict.llm_hypothesis import (
    generate_llm_hypotheses, _parse_llm_response, _fallback,
)
from autoresearch_contradict.hypothesis import Hypothesis
from autoresearch_contradict.detector import Contradiction
from autoresearch_contradict.parser import ExperimentRecord


def _make_contradiction():
    return Contradiction(
        exp_a=ExperimentRecord("c1", 0.90, 4.0, "success",
                             "Increased learning rate for small model", "lr_change"),
        exp_b=ExperimentRecord("c2", 0.70, 4.0, "success",
                             "Increased learning rate for large model", "lr_change"),
        change_type="lr_change",
        direction_a="positive",
        direction_b="negative",
        context_diff="model size differs",
        confidence=0.7,
    )


class TestParseResponse:
    def test_parse_valid_response(self):
        text = (
            "HYPOTHESIS:\n"
            "EXPLANATION: The learning rate is too high for large models.\n"
            "CONFIDENCE: 0.8\n"
            "EXPERIMENT: Try lower LR for large model.\n"
            "HYPOTHESIS:\n"
            "EXPLANATION: Model capacity affects optimal LR.\n"
            "CONFIDENCE: 0.6\n"
            "EXPERIMENT: Sweep LR across model sizes.\n"
        )
        result = _parse_llm_response(text)
        assert len(result) == 2
        assert result[0].confidence == 0.8
        assert "learning rate" in result[0].explanation.lower()

    def test_parse_empty_response(self):
        result = _parse_llm_response("No hypotheses here")
        assert len(result) >= 1  # Falls back


class TestFallback:
    def test_fallback_returns_template_hypotheses(self):
        c = _make_contradiction()
        result = _fallback(c)
        assert len(result) >= 1
        assert all(isinstance(h, Hypothesis) for h in result)


class TestGenerateLLMHypotheses:
    def test_no_api_key_falls_back(self):
        c = _make_contradiction()
        with patch.dict("os.environ", {}, clear=True):
            result = generate_llm_hypotheses(c, api_key=None)
        assert len(result) >= 1
        assert all(isinstance(h, Hypothesis) for h in result)

    def test_with_api_key_calls_anthropic(self):
        c = _make_contradiction()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=(
            "HYPOTHESIS:\n"
            "EXPLANATION: LR too high for large models.\n"
            "CONFIDENCE: 0.75\n"
            "EXPERIMENT: Lower the LR.\n"
        ))]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = generate_llm_hypotheses(c, api_key="test-key")

        assert len(result) >= 1
        assert result[0].confidence == 0.75

    def test_api_failure_falls_back(self):
        c = _make_contradiction()
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.side_effect = Exception("API error")

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = generate_llm_hypotheses(c, api_key="test-key")
        assert len(result) >= 1
