"""CLI for contradiction-detector."""

from __future__ import annotations

import os

import click

from contradiction_detector.analyzer import ContradictionAnalyzer


@click.group()
def cli() -> None:
    """contradiction-detector: Detect contradictory experiment results."""


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
        click.echo(f"\nContradiction {i} ({c.change_type}, confidence={c.confidence:.3f}):")
        click.echo(f"  A: [{c.exp_a.commit}] {c.exp_a.description} -> {c.direction_a}")
        click.echo(f"  B: [{c.exp_b.commit}] {c.exp_b.description} -> {c.direction_b}")


@cli.command()
@click.argument("filepath")
@click.option("--baseline", default=0.80, help="Baseline metric value")
@click.option("--llm", is_flag=True, default=False, help="Use LLM for hypothesis generation")
def hypotheses(filepath: str, baseline: float, llm: bool) -> None:
    """Show generated hypotheses."""
    if not os.path.exists(filepath):
        click.echo(f"Error: File not found: {filepath}")
        raise SystemExit(1)

    analyzer = ContradictionAnalyzer(baseline_metric=baseline)
    report = analyzer.analyze_file(filepath)

    if llm and report.contradictions:
        from contradiction_detector.llm_hypothesis import generate_llm_hypotheses

        all_hypotheses = []
        for c in report.contradictions:
            all_hypotheses.extend(generate_llm_hypotheses(c))

        if not all_hypotheses:
            click.echo("No hypotheses generated (LLM mode).")
            return

        for i, h in enumerate(all_hypotheses, 1):
            click.echo(f"\nHypothesis {i} (confidence: {h.confidence:.0%}) [LLM]:")
            click.echo(f"  {h.explanation}")
            click.echo(f"  Suggestion: {h.suggested_experiment}")
        return

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


@cli.command()
@click.argument("filepath")
@click.option("--baseline", default=0.80, help="Baseline metric value")
@click.option("--method", default="pca", type=click.Choice(["pca", "tsne"]),
              help="Dimensionality reduction method")
@click.option("--output", default=None, help="Output image path")
def visualize(filepath: str, baseline: float, method: str, output: str) -> None:
    """Generate cluster visualization of experiments."""
    if not os.path.exists(filepath):
        click.echo(f"Error: File not found: {filepath}")
        raise SystemExit(1)

    analyzer = ContradictionAnalyzer(baseline_metric=baseline)
    report_data = analyzer.analyze_file(filepath)

    from contradiction_detector.visualize import plot_clusters

    plot_clusters(
        report_data.experiments,
        report_data.contradictions,
        method=method,
        output_path=output,
    )
    out = output or "contradictions_plot.png"
    click.echo(f"Visualization saved to {out}")


@cli.command()
@click.argument("filepath")
@click.option("--baseline", default=0.80, help="Baseline metric value")
def trends(filepath: str, baseline: float) -> None:
    """Show contradiction emergence over time."""
    if not os.path.exists(filepath):
        click.echo(f"Error: File not found: {filepath}")
        raise SystemExit(1)

    from contradiction_detector.parser import ExperimentParser
    from contradiction_detector.trends import analyze_trends, format_trends_report

    parser = ExperimentParser()
    experiments = parser.parse_tsv_file(filepath)
    timeline = analyze_trends(experiments, baseline_metric=baseline)
    click.echo(format_trends_report(timeline))


@cli.command(name="cross-analyze")
@click.argument("filepaths", nargs=-1, required=True)
@click.option("--baseline", default=0.80, help="Baseline metric value")
def cross_analyze_cmd(filepaths: tuple, baseline: float) -> None:
    """Detect contradictions across multiple results files."""
    for fp in filepaths:
        if not os.path.exists(fp):
            click.echo(f"Error: File not found: {fp}")
            raise SystemExit(1)

    from contradiction_detector.cross_project import cross_analyze, format_cross_analysis

    results = cross_analyze(list(filepaths), baseline_metric=baseline)
    click.echo(format_cross_analysis(results))


if __name__ == "__main__":
    cli()
