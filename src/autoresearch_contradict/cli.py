"""CLI for autoresearch-contradict."""

from __future__ import annotations

import os

import click

from autoresearch_contradict.analyzer import ContradictionAnalyzer


@click.group()
def cli() -> None:
    """autoresearch-contradict: Detect contradictory experiment results."""


def _get_analyzer() -> ContradictionAnalyzer:
    return ContradictionAnalyzer(baseline_metric=0.80)


@cli.command()
@click.argument("filepath")
@click.option("--baseline", default=0.80, help="Baseline metric value")
def analyze(filepath: str, baseline: float) -> None:
    """Run full contradiction analysis on a results.tsv file."""
    if not os.path.exists(filepath):
        click.echo(f"Error: File not found: {filepath}")
        raise SystemExit(1)

    analyzer = ContradictionAnalyzer(baseline_metric=baseline)
    report = analyzer.analyze_file(filepath)
    click.echo(f"Analyzed {len(report.experiments)} experiments")
    click.echo(f"Found {len(report.contradictions)} contradictions")
    click.echo(f"Generated {len(report.hypotheses)} hypotheses")


@cli.command()
@click.argument("filepath")
@click.option("--baseline", default=0.80, help="Baseline metric value")
def contradictions(filepath: str, baseline: float) -> None:
    """List found contradictions."""
    if not os.path.exists(filepath):
        click.echo(f"Error: File not found: {filepath}")
        raise SystemExit(1)

    analyzer = ContradictionAnalyzer(baseline_metric=baseline)
    report = analyzer.analyze_file(filepath)

    if not report.contradictions:
        click.echo("No contradictions found.")
        return

    for i, c in enumerate(report.contradictions, 1):
        click.echo(f"\nContradiction {i} ({c.change_type}):")
        click.echo(f"  A: [{c.exp_a.commit}] {c.exp_a.description} -> {c.direction_a}")
        click.echo(f"  B: [{c.exp_b.commit}] {c.exp_b.description} -> {c.direction_b}")


@cli.command()
@click.argument("filepath")
@click.option("--baseline", default=0.80, help="Baseline metric value")
def hypotheses(filepath: str, baseline: float) -> None:
    """Show generated hypotheses."""
    if not os.path.exists(filepath):
        click.echo(f"Error: File not found: {filepath}")
        raise SystemExit(1)

    analyzer = ContradictionAnalyzer(baseline_metric=baseline)
    report = analyzer.analyze_file(filepath)

    if not report.hypotheses:
        click.echo("No hypotheses generated.")
        return

    for i, h in enumerate(report.hypotheses, 1):
        click.echo(f"\nHypothesis {i} (confidence: {h.confidence:.0%}):")
        click.echo(f"  {h.explanation}")
        click.echo(f"  Suggestion: {h.suggested_experiment}")


@cli.command()
@click.argument("filepath")
@click.option("--baseline", default=0.80, help="Baseline metric value")
def report(filepath: str, baseline: float) -> None:
    """Generate markdown report."""
    if not os.path.exists(filepath):
        click.echo(f"Error: File not found: {filepath}")
        raise SystemExit(1)

    analyzer = ContradictionAnalyzer(baseline_metric=baseline)
    report_data = analyzer.analyze_file(filepath)
    click.echo(report_data.markdown)


if __name__ == "__main__":
    cli()
