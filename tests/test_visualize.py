"""Tests for cluster visualization."""

import os
import pytest
import numpy as np

from contradiction_detector.visualize import compute_embeddings_2d, plot_clusters
from contradiction_detector.detector import Contradiction
from contradiction_detector.parser import ExperimentRecord


class TestComputeEmbeddings2D:
    def test_returns_2d_coords(self):
        experiments = [
            ExperimentRecord("c1", 0.90, 4.0, "success", "learning rate increase", "lr_change"),
            ExperimentRecord("c2", 0.70, 4.0, "success", "dropout regularization", "regularization_change"),
            ExperimentRecord("c3", 0.85, 4.0, "success", "batch normalization added", "architecture_change"),
        ]
        coords = compute_embeddings_2d(experiments, method="pca")
        assert coords.shape == (3, 2)

    def test_single_experiment(self):
        experiments = [
            ExperimentRecord("c1", 0.90, 4.0, "success", "test", "lr_change"),
        ]
        coords = compute_embeddings_2d(experiments, method="pca")
        assert coords.shape == (1, 2)

    def test_empty_experiments(self):
        coords = compute_embeddings_2d([], method="pca")
        assert coords.shape == (0, 2)

    def test_tsne_method(self):
        experiments = [
            ExperimentRecord(f"c{i}", 0.80 + i * 0.01, 4.0, "success",
                           f"experiment {i} with learning rate", "lr_change")
            for i in range(6)
        ]
        coords = compute_embeddings_2d(experiments, method="tsne")
        assert coords.shape == (6, 2)


class TestPlotClusters:
    def test_plot_saves_file(self, tmp_path):
        experiments = [
            ExperimentRecord("c1", 0.90, 4.0, "success", "learning rate increase", "lr_change"),
            ExperimentRecord("c2", 0.70, 4.0, "success", "learning rate decrease", "lr_change"),
            ExperimentRecord("c3", 0.85, 4.0, "success", "added dropout", "regularization_change"),
        ]
        contradictions = [
            Contradiction(
                exp_a=experiments[0], exp_b=experiments[1],
                change_type="lr_change", direction_a="positive",
                direction_b="negative", context_diff="different directions",
                confidence=0.7,
            ),
        ]
        output = str(tmp_path / "test_plot.png")
        plot_clusters(experiments, contradictions, output_path=output)
        assert os.path.exists(output)

    def test_plot_empty_experiments(self, tmp_path):
        output = str(tmp_path / "empty_plot.png")
        plot_clusters([], [], output_path=output)
        assert not os.path.exists(output)
