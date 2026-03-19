"""TF-IDF vectorization and cosine similarity for experiments."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

import numpy as np


class SimpleEmbedder:
    """Embed texts using TF-IDF vectorization."""

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of texts into TF-IDF vectors."""
        if not texts:
            return []

        tokenized = [self._tokenize(t) for t in texts]

        # Build vocabulary
        vocab: dict = {}
        for tokens in tokenized:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        if not vocab:
            return [np.zeros(1) for _ in texts]

        # Compute IDF
        n_docs = len(texts)
        doc_freq: Counter = Counter()
        for tokens in tokenized:
            for token in set(tokens):
                doc_freq[token] += 1

        idf = {}
        for token, freq in doc_freq.items():
            idf[token] = math.log((n_docs + 1) / (1 + freq)) + 1

        # Build TF-IDF vectors
        vectors = []
        for tokens in tokenized:
            vec = np.zeros(len(vocab))
            tf = Counter(tokens)
            for token, count in tf.items():
                if token in vocab:
                    idx = vocab[token]
                    vec[idx] = (count / max(len(tokens), 1)) * idf.get(token, 1.0)
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)

        return vectors

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def pairwise_similarity(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        n = len(vectors)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = self.cosine_similarity(vectors[i], vectors[j])
        return matrix

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase words."""
        return re.findall(r"[a-z0-9]+", text.lower())
