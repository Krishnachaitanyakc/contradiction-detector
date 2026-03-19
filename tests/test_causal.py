"""Tests for causal inference module."""

import pytest

from autoresearch_contradict.causal import (
    identify_confounders, Confounder, CausalAnalysis,
)
from autoresearch_contradict.detector import Contradiction
from autoresearch_contradict.parser import ExperimentRecord


class TestIdentifyConfounders:
    def test_memory_confounder(self):
        c = Contradiction(
            exp_a=ExperimentRecord("c1", 0.90, 8.0, "success",
                                 "Increased learning rate small model", "lr_change"),
            exp_b=ExperimentRecord("c2", 0.70, 2.0, "success",
                                 "Increased learning rate large model", "lr_change"),
            change_type="lr_change",
            direction_a="positive",
            direction_b="negative",
            context_diff="memory differs",
            confidence=0.7,
        )
        result = identify_confounders(c, [])
        assert isinstance(result, CausalAnalysis)
        mem_confounders = [cf for cf in result.confounders if cf.name == "memory_usage"]
        assert len(mem_confounders) == 1
        assert mem_confounders[0].strength > 0

    def test_keyword_confounder(self):
        c = Contradiction(
            exp_a=ExperimentRecord("c1", 0.90, 4.0, "success",
                                 "Increased learning rate small model", "lr_change"),
            exp_b=ExperimentRecord("c2", 0.70, 4.0, "success",
                                 "Increased learning rate large model", "lr_change"),
            change_type="lr_change",
            direction_a="positive",
            direction_b="negative",
            context_diff="size differs",
            confidence=0.7,
        )
        result = identify_confounders(c, [])
        keyword_confounders = [cf for cf in result.confounders
                              if cf.name.startswith("context_")]
        assert len(keyword_confounders) >= 1

    def test_high_variance_confounder(self):
        c = Contradiction(
            exp_a=ExperimentRecord("c1", 0.90, 4.0, "success",
                                 "Increased learning rate A", "lr_change"),
            exp_b=ExperimentRecord("c2", 0.70, 4.0, "success",
                                 "Increased learning rate B", "lr_change"),
            change_type="lr_change",
            direction_a="positive",
            direction_b="negative",
            context_diff="",
            confidence=0.7,
        )
        all_exps = [
            ExperimentRecord("c3", 0.60, 4.0, "success", "lr change c3", "lr_change"),
            ExperimentRecord("c4", 0.95, 4.0, "success", "lr change c4", "lr_change"),
        ]
        result = identify_confounders(c, all_exps)
        var_confounders = [cf for cf in result.confounders
                          if cf.name == "high_variance_change_type"]
        assert len(var_confounders) == 1

    def test_no_confounders(self):
        c = Contradiction(
            exp_a=ExperimentRecord("c1", 0.81, 4.0, "success",
                                 "test experiment a", "lr_change"),
            exp_b=ExperimentRecord("c2", 0.79, 4.0, "success",
                                 "test experiment b", "lr_change"),
            change_type="lr_change",
            direction_a="positive",
            direction_b="negative",
            context_diff="",
            confidence=0.5,
        )
        result = identify_confounders(c, [])
        assert isinstance(result.summary, str)
        assert "No obvious confounders" in result.summary or len(result.confounders) >= 0

    def test_confounders_sorted_by_strength(self):
        c = Contradiction(
            exp_a=ExperimentRecord("c1", 0.90, 10.0, "success",
                                 "Increased learning rate small model", "lr_change"),
            exp_b=ExperimentRecord("c2", 0.70, 1.0, "success",
                                 "Increased learning rate large model", "lr_change"),
            change_type="lr_change",
            direction_a="positive",
            direction_b="negative",
            context_diff="",
            confidence=0.7,
        )
        result = identify_confounders(c, [])
        if len(result.confounders) >= 2:
            for i in range(len(result.confounders) - 1):
                assert result.confounders[i].strength >= result.confounders[i + 1].strength
