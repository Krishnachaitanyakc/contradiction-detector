"""Tests for embedder."""

import pytest
from contradiction_detector.embedder import SimpleEmbedder


class TestSimpleEmbedder:
    def setup_method(self):
        self.embedder = SimpleEmbedder()

    def test_embed_texts(self):
        texts = [
            "Increased learning rate to 0.01",
            "Added dropout regularization",
            "Changed learning rate schedule",
        ]
        vectors = self.embedder.embed(texts)
        assert len(vectors) == 3
        assert len(vectors[0]) > 0

    def test_cosine_similarity_identical(self):
        texts = ["learning rate increase", "learning rate increase"]
        vectors = self.embedder.embed(texts)
        sim = self.embedder.cosine_similarity(vectors[0], vectors[1])
        assert abs(sim - 1.0) < 0.01

    def test_cosine_similarity_different(self):
        texts = [
            "learning rate increase",
            "completely unrelated topic about cooking pasta",
        ]
        vectors = self.embedder.embed(texts)
        sim = self.embedder.cosine_similarity(vectors[0], vectors[1])
        assert sim < 0.5

    def test_pairwise_similarity(self):
        texts = [
            "Increased learning rate",
            "Changed learning rate",
            "Added dropout",
        ]
        vectors = self.embedder.embed(texts)
        sim_matrix = self.embedder.pairwise_similarity(vectors)
        assert sim_matrix.shape == (3, 3)
        assert abs(sim_matrix[0, 0] - 1.0) < 0.01
        assert sim_matrix[0, 1] > sim_matrix[0, 2]

    def test_embed_single_text(self):
        vectors = self.embedder.embed(["single text"])
        assert len(vectors) == 1

    def test_embed_empty_list(self):
        vectors = self.embedder.embed([])
        assert len(vectors) == 0
