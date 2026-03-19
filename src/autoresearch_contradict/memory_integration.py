"""Optional integration with autoresearch-memory for context enrichment."""

from __future__ import annotations

from typing import Any, List

from autoresearch_contradict.detector import Contradiction


def query_similar_contradictions(
    contradiction: Contradiction,
    k: int = 5,
) -> List[dict[str, Any]]:
    """Query autoresearch-memory for similar past contradictions.

    Returns an empty list if autoresearch-memory is not installed.
    """
    try:
        from autoresearch_memory.store import SimpleMemoryStore
    except ImportError:
        return []

    store = SimpleMemoryStore()

    query = (
        f"{contradiction.change_type}: "
        f"{contradiction.exp_a.description} vs {contradiction.exp_b.description}"
    )
    return store.query_similar(query, k=k)


def enrich_contradiction_context(
    contradiction: Contradiction,
    k: int = 3,
) -> str:
    """Enrich contradiction context with similar past experiments.

    Returns additional context string, or empty string if memory unavailable.
    """
    similar = query_similar_contradictions(contradiction, k=k)
    if not similar:
        return ""

    parts = ["Past similar experiments:"]
    for exp in similar:
        desc = exp.get("description", "")
        metric = exp.get("metric_value", 0.0)
        parts.append(f"  - {desc} (metric={metric:.4f})")

    return "\n".join(parts)
