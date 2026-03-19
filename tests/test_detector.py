"""Tests for contradiction detector."""

import pytest
from autoresearch_contradict.detector import ContradictionDetector, Contradiction
from autoresearch_contradict.parser import ExperimentRecord


class TestContradiction:
    def test_creation(self):
        c = Contradiction(
            exp_a=ExperimentRecord("c1", 0.85, 4.0, "success", "test a", "lr_change"),
            exp_b=ExperimentRecord("c2", 0.75, 4.0, "success", "test b", "lr_change"),
            change_type="lr_change",
            direction_a="positive",
            direction_b="negative",
            context_diff="Different model sizes",
        )
        assert c.change_type == "lr_change"
        assert c.direction_a == "positive"
        assert c.direction_b == "negative"


class TestContradictionDetector:
    def setup_method(self):
        self.detector = ContradictionDetector(baseline_metric=0.80)

    def test_find_contradictions_basic(self):
        experiments = [
            ExperimentRecord("c1", 0.90, 4.0, "success",
                           "Increased learning rate to 0.01 for small model", "lr_change"),
            ExperimentRecord("c2", 0.70, 4.0, "success",
                           "Increased learning rate to 0.01 for large model", "lr_change"),
        ]
        contradictions = self.detector.find_contradictions(experiments)
        assert len(contradictions) >= 1
        c = contradictions[0]
        assert c.change_type == "lr_change"
        assert c.direction_a != c.direction_b

    def test_no_contradictions_same_direction(self):
        experiments = [
            ExperimentRecord("c1", 0.85, 4.0, "success",
                           "Increased learning rate", "lr_change"),
            ExperimentRecord("c2", 0.88, 4.0, "success",
                           "Increased learning rate more", "lr_change"),
        ]
        contradictions = self.detector.find_contradictions(experiments)
        assert len(contradictions) == 0

    def test_contradictions_different_types_no_contradiction(self):
        experiments = [
            ExperimentRecord("c1", 0.90, 4.0, "success",
                           "Increased learning rate", "lr_change"),
            ExperimentRecord("c2", 0.70, 4.0, "success",
                           "Added dropout", "regularization_change"),
        ]
        contradictions = self.detector.find_contradictions(experiments)
        assert len(contradictions) == 0

    def test_find_contradictions_empty(self):
        contradictions = self.detector.find_contradictions([])
        assert contradictions == []

    def test_find_contradictions_single(self):
        experiments = [
            ExperimentRecord("c1", 0.85, 4.0, "success", "test", "lr_change"),
        ]
        contradictions = self.detector.find_contradictions(experiments)
        assert contradictions == []

    def test_multiple_contradictions(self):
        experiments = [
            ExperimentRecord("c1", 0.90, 4.0, "success",
                           "Increased learning rate small", "lr_change"),
            ExperimentRecord("c2", 0.70, 4.0, "success",
                           "Increased learning rate large", "lr_change"),
            ExperimentRecord("c3", 0.85, 4.0, "success",
                           "Added dropout small model", "regularization_change"),
            ExperimentRecord("c4", 0.75, 4.0, "success",
                           "Added dropout large model", "regularization_change"),
        ]
        contradictions = self.detector.find_contradictions(experiments)
        assert len(contradictions) >= 2
