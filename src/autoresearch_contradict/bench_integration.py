"""Optional integration with autoresearch-bench for paper search."""

from __future__ import annotations

from typing import Any, List

from autoresearch_contradict.detector import Contradiction


def search_related_papers(
    contradiction: Contradiction,
    limit: int = 5,
) -> List[dict[str, str]]:
    """Search for papers addressing similar contradictions.

    Returns an empty list if autoresearch-bench is not installed.
    """
    try:
        from autoresearch_bench.papers import PaperFetcher
    except ImportError:
        return []

    query = f"{contradiction.change_type} contradictory results {contradiction.context_diff}"
    fetcher = PaperFetcher()

    try:
        papers = fetcher.fetch_papers(query, limit=limit)
        return [
            {
                "title": p.title,
                "abstract": p.abstract[:200] if p.abstract else "",
                "url": p.url,
            }
            for p in papers
        ]
    except Exception:
        return []


def format_paper_context(papers: List[dict[str, str]]) -> str:
    """Format papers into a context string for hypothesis enrichment."""
    if not papers:
        return ""

    lines = ["Related literature:"]
    for p in papers:
        lines.append(f"  - {p['title']}")
        if p.get("abstract"):
            lines.append(f"    {p['abstract'][:150]}...")

    return "\n".join(lines)
