"""Tests for hypothesis generator."""

import pytest
from contradiction_detector.hypothesis import HypothesisGenerator, Hypothesis
from contradiction_detector.detector import Contradiction
from contradiction_detector.parser import ExperimentRecord


class TestHypothesis:
    def test_creation(self):
        h = Hypothesis(
            explanation="LR scales inversely with model size",
            confidence=0.7,
            suggested_experiment="Try LR=0.005 with medium model",
        )
        assert h.confidence == 0.7


class TestHypothesisGenerator:
    def setup_method(self):
        self.generator = HypothesisGenerator()

    def test_generate_hypothesis_lr(self):
        contradiction = Contradiction(
            exp_a=ExperimentRecord("c1", 0.90, 4.0, "success",
                                 "Increased learning rate for small model", "lr_change"),
            exp_b=ExperimentRecord("c2", 0.70, 4.0, "success",
                                 "Increased learning rate for large model", "lr_change"),
            change_type="lr_change",
            direction_a="positive",
            direction_b="negative",
            context_diff="model size differs",
        )
        hypotheses = self.generator.generate_hypotheses(contradiction)
        assert len(hypotheses) >= 1
        assert all(h.explanation != "" for h in hypotheses)
        assert all(h.suggested_experiment != "" for h in hypotheses)
        assert all(0 <= h.confidence <= 1 for h in hypotheses)

    def test_generate_hypothesis_regularization(self):
        contradiction = Contradiction(
            exp_a=ExperimentRecord("c1", 0.85, 4.0, "success",
                                 "Added dropout small dataset", "regularization_change"),
            exp_b=ExperimentRecord("c2", 0.75, 4.0, "success",
                                 "Added dropout large dataset", "regularization_change"),
            change_type="regularization_change",
            direction_a="positive",
            direction_b="negative",
            context_diff="dataset size differs",
        )
        hypotheses = self.generator.generate_hypotheses(contradiction)
        assert len(hypotheses) >= 1

    def test_generate_hypothesis_optimizer(self):
        contradiction = Contradiction(
            exp_a=ExperimentRecord("c1", 0.90, 4.0, "success",
                                 "Switched to Adam", "optimizer_change"),
            exp_b=ExperimentRecord("c2", 0.70, 4.0, "success",
                                 "Switched to Adam", "optimizer_change"),
            change_type="optimizer_change",
            direction_a="positive",
            direction_b="negative",
            context_diff="different tasks",
        )
        hypotheses = self.generator.generate_hypotheses(contradiction)
        assert len(hypotheses) >= 1

    def test_generate_hypothesis_architecture(self):
        contradiction = Contradiction(
            exp_a=ExperimentRecord("c1", 0.88, 4.0, "success",
                                 "Added skip connections", "architecture_change"),
            exp_b=ExperimentRecord("c2", 0.72, 4.0, "success",
                                 "Added skip connections", "architecture_change"),
            change_type="architecture_change",
            direction_a="positive",
            direction_b="negative",
            context_diff="different depths",
        )
        hypotheses = self.generator.generate_hypotheses(contradiction)
        assert len(hypotheses) >= 1

    def test_hypothesis_confidence_range(self):
        contradiction = Contradiction(
            exp_a=ExperimentRecord("c1", 0.90, 4.0, "success", "test a", "lr_change"),
            exp_b=ExperimentRecord("c2", 0.70, 4.0, "success", "test b", "lr_change"),
            change_type="lr_change",
            direction_a="positive",
            direction_b="negative",
            context_diff="",
        )
        hypotheses = self.generator.generate_hypotheses(contradiction)
        for h in hypotheses:
            assert 0.0 <= h.confidence <= 1.0
