"""Cluster visualization for experiments and contradictions."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from contradiction_detector.detector import Contradiction
from contradiction_detector.embedder import SimpleEmbedder
from contradiction_detector.parser import ExperimentRecord


def compute_embeddings_2d(
    experiments: List[ExperimentRecord],
    method: str = "pca",
) -> np.ndarray:
    """Embed experiment descriptions into 2D space using PCA or t-SNE."""
    if len(experiments) < 2:
        return np.zeros((len(experiments), 2))

    embedder = SimpleEmbedder()
    descriptions = [exp.description for exp in experiments]
    vectors = embedder.embed(descriptions)
    matrix = np.array(vectors)

    n_components = min(2, matrix.shape[1], matrix.shape[0])
    if n_components < 2:
        result = np.zeros((len(experiments), 2))
        if n_components == 1:
            reduced = PCA(n_components=1).fit_transform(matrix)
            result[:, 0] = reduced[:, 0]
        return result

    if method == "tsne" and len(experiments) >= 5:
        perplexity = min(30, len(experiments) - 1)
        coords = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(matrix)
    else:
        coords = PCA(n_components=2, random_state=42).fit_transform(matrix)

    return coords


def plot_clusters(
    experiments: List[ExperimentRecord],
    contradictions: List[Contradiction],
    method: str = "pca",
    output_path: Optional[str] = None,
) -> None:
    """Plot experiments as scatter with contradiction pairs connected by red lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not experiments:
        return

    coords = compute_embeddings_2d(experiments, method=method)

    # Map experiments to indices
    exp_to_idx = {id(exp): i for i, exp in enumerate(experiments)}

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by change type
    change_types = list(set(exp.change_type for exp in experiments))
    color_map = {}
    cmap = matplotlib.colormaps.get_cmap("tab10")
    for i, ct in enumerate(change_types):
        color_map[ct] = cmap(i % 10)

    for i, exp in enumerate(experiments):
        color = color_map[exp.change_type]
        ax.scatter(coords[i, 0], coords[i, 1], c=[color], s=60, zorder=3)
        ax.annotate(
            exp.commit[:7],
            (coords[i, 0], coords[i, 1]),
            fontsize=7,
            ha="center",
            va="bottom",
        )

    # Draw contradiction lines in red
    for c in contradictions:
        idx_a = exp_to_idx.get(id(c.exp_a))
        idx_b = exp_to_idx.get(id(c.exp_b))
        if idx_a is not None and idx_b is not None:
            ax.plot(
                [coords[idx_a, 0], coords[idx_b, 0]],
                [coords[idx_a, 1], coords[idx_b, 1]],
                "r-",
                alpha=0.5,
                linewidth=1.5,
                zorder=2,
            )

    # Legend for change types
    for ct, color in color_map.items():
        ax.scatter([], [], c=[color], label=ct)
    ax.legend(loc="best", fontsize=8)

    ax.set_title("Experiment Clusters with Contradictions")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        fig.savefig("contradictions_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
