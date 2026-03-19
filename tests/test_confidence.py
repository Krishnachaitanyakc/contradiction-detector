"""Tests for confidence scoring on contradictions."""

import pytest
from autoresearch_contradict.detector import ContradictionDetector, Contradiction
from autoresearch_contradict.parser import ExperimentRecord


class TestConfidenceScoring:
    def setup_method(self):
        self.detector = ContradictionDetector(baseline_metric=0.80)

    def test_contradiction_has_confidence(self):
        experiments = [
            ExperimentRecord("c1", 0.90, 4.0, "success",
                           "Increased learning rate to 0.01 for small model", "lr_change"),
            ExperimentRecord("c2", 0.70, 4.0, "success",
                           "Increased learning rate to 0.01 for large model", "lr_change"),
        ]
        contradictions = self.detector.find_contradictions(experiments)
        assert len(contradictions) >= 1
        for c in contradictions:
            assert hasattr(c, "confidence")
            assert 0.0 <= c.confidence <= 1.0

    def test_high_metric_divergence_increases_confidence(self):
        # Large metric difference
        exps_large = [
            ExperimentRecord("c1", 0.95, 4.0, "success",
                           "Increased learning rate small", "lr_change"),
            ExperimentRecord("c2", 0.60, 4.0, "success",
                           "Increased learning rate large", "lr_change"),
        ]
        # Small metric difference
        exps_small = [
            ExperimentRecord("c3", 0.81, 4.0, "success",
                           "Increased learning rate small", "lr_change"),
            ExperimentRecord("c4", 0.79, 4.0, "success",
                           "Increased learning rate large", "lr_change"),
        ]
        c_large = self.detector.find_contradictions(exps_large)
        c_small = self.detector.find_contradictions(exps_small)

        assert len(c_large) >= 1
        assert len(c_small) >= 1
        assert c_large[0].confidence > c_small[0].confidence

    def test_specific_change_type_higher_confidence(self):
        # lr_change is more specific than architecture_change
        exps_lr = [
            ExperimentRecord("c1", 0.90, 4.0, "success",
                           "Increased learning rate A", "lr_change"),
            ExperimentRecord("c2", 0.70, 4.0, "success",
                           "Increased learning rate B", "lr_change"),
        ]
        exps_arch = [
            ExperimentRecord("c3", 0.90, 4.0, "success",
                           "Changed architecture A", "architecture_change"),
            ExperimentRecord("c4", 0.70, 4.0, "success",
                           "Changed architecture B", "architecture_change"),
        ]

        c_lr = self.detector.find_contradictions(exps_lr)
        c_arch = self.detector.find_contradictions(exps_arch)

        assert len(c_lr) >= 1
        assert len(c_arch) >= 1
        assert c_lr[0].confidence > c_arch[0].confidence

    def test_confidence_bounded(self):
        experiments = [
            ExperimentRecord("c1", 1.0, 10.0, "success",
                           "Increased learning rate for small model", "lr_change"),
            ExperimentRecord("c2", 0.0, 0.1, "success",
                           "Increased learning rate for large model", "lr_change"),
        ]
        contradictions = self.detector.find_contradictions(experiments)
        assert len(contradictions) >= 1
        assert 0.0 <= contradictions[0].confidence <= 1.0
