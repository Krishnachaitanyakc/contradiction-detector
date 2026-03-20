# contradiction-detector

**Your experiments disagree. Here's why.**

Automatically detect contradictions in ML experiment results, generate hypotheses explaining *why* they happened, and suggest the next experiment to run. Stop wasting weeks debugging conflicting results manually.

---

## The Problem

You ran 50 experiments. Learning rate 0.001 improved accuracy in one setup but *destroyed* it in another. Dropout helped on Monday, hurt on Friday. Your results.tsv is a minefield of contradictions -- and you have no idea which results to trust.

**contradiction-detector** finds these contradictions automatically, explains them with causal hypotheses, and tells you exactly what to try next.

## Quick Demo

```bash
$ contradiction-detector analyze results.tsv

Analyzed 47 experiments
Found 3 contradictions
Generated 9 hypotheses

Contradiction 1 (lr_change, confidence=0.847):
  A: [abc1234] "Increased LR to 0.01" -> positive (0.8650)
  B: [def5678] "Increased LR to 0.01 with large batch" -> negative (0.7200)

  Hypothesis (confidence: 70%):
    Optimal learning rate depends on batch size context.
  Suggested experiment:
    Try intermediate LR values, controlling for batch size.
```

## Features

- **Contradiction Detection** -- Groups experiments by change type (learning rate, architecture, optimizer, etc.) and finds pairs where the same change produced opposite outcomes. Confidence scoring combines description similarity, metric divergence, and change-type specificity.

- **Hypothesis Generation** -- Template-based engine covers 10 change categories (LR, regularization, optimizer, architecture, batch size, warmup, weight init, loss function, data preprocessing, data augmentation) with domain-specific explanations and experiment suggestions.

- **LLM-Powered Hypotheses** -- Optional Claude integration generates deeper, context-aware hypotheses when templates aren't enough. Falls back gracefully to templates if unavailable.

- **Causal Analysis** -- Identifies confounding variables (memory usage differences, contextual keywords, high-variance change types) that may explain why the same change produced different results.

- **Trend Analysis** -- Tracks how contradictions emerge over time as new experiments are added. See which experiment introduced a contradiction and how the contradiction count grows.

- **Cross-Project Analysis** -- Compare results across multiple TSV files to find contradictions that span different projects or teams.

- **Cluster Visualization** -- PCA/t-SNE scatter plots with contradiction pairs connected by red lines. Color-coded by change type for instant pattern recognition.

- **Ecosystem Integration** -- Optional hooks into `paper-benchmark` for literature search and `autoresearch-memory` for historical context enrichment.

## Installation

```bash
pip install contradiction-detector
```

For LLM-powered hypotheses:
```bash
pip install contradiction-detector[llm]
```

## Usage

### CLI

```bash
# Full analysis pipeline
contradiction-detector analyze results.tsv --baseline 0.80

# List contradictions with confidence scores
contradiction-detector contradictions results.tsv

# Generate hypotheses (template-based)
contradiction-detector hypotheses results.tsv

# Generate hypotheses (LLM-powered, requires ANTHROPIC_API_KEY)
contradiction-detector hypotheses results.tsv --llm

# Markdown report
contradiction-detector report results.tsv

# Visualize experiment clusters with contradiction lines
contradiction-detector visualize results.tsv --method pca --output clusters.png

# Track contradiction emergence over time
contradiction-detector trends results.tsv

# Cross-project contradiction detection
contradiction-detector cross-analyze project_a/results.tsv project_b/results.tsv
```

### Python API

```python
from contradiction_detector.analyzer import ContradictionAnalyzer

analyzer = ContradictionAnalyzer(baseline_metric=0.80)
report = analyzer.analyze_file("results.tsv")

print(f"Found {len(report.contradictions)} contradictions")
for c in report.contradictions:
    print(f"  {c.change_type}: {c.exp_a.commit} vs {c.exp_b.commit} (confidence={c.confidence:.3f})")

for h in report.hypotheses:
    print(f"  [{h.confidence:.0%}] {h.explanation}")
    print(f"  -> Try: {h.suggested_experiment}")

# Full markdown report
print(report.markdown)
```

```python
# Causal analysis
from contradiction_detector.causal import identify_confounders

for contradiction in report.contradictions:
    analysis = identify_confounders(contradiction, report.experiments)
    for conf in analysis.confounders:
        print(f"  Confounder: {conf.name} (strength={conf.strength:.0%})")

# Cross-project analysis
from contradiction_detector.cross_project import cross_analyze

results = cross_analyze(["project_a/results.tsv", "project_b/results.tsv"])
for r in results:
    print(f"  Cross-project: {r.source_file_a} vs {r.source_file_b}")
```

### Input Format

Tab-separated file with columns: `commit`, `metric_value`, `memory_gb`, `status`, `description`, and optionally `timestamp`.

```
commit	metric_value	memory_gb	status	description	timestamp
abc1234	0.8650	4.2	success	Increased learning rate to 0.01	2025-01-15
def5678	0.7200	8.1	success	Increased learning rate to 0.01 with large batch	2025-01-16
```

## How It Works

```
results.tsv
    |
    v
[Parser] -- Categorize experiments by change type (10 categories via regex)
    |
    v
[Embedder] -- TF-IDF vectorization of experiment descriptions
    |
    v
[Detector] -- Group by change type, find opposite-outcome pairs
    |          Confidence = 0.3*similarity + 0.4*divergence + 0.3*specificity
    v
[Hypothesis Generator] -- Template-based or LLM-powered explanations
    |
    v
[Causal Analyzer] -- Identify confounding variables
    |
    v
[Report / Visualization] -- Markdown report, PCA/t-SNE cluster plots
```

**Key modules:**
- `parser.py` -- Regex-based categorization into 10 change types
- `embedder.py` -- TF-IDF vectorization with cosine similarity
- `detector.py` -- Pairwise contradiction detection with confidence scoring
- `hypothesis.py` -- 25+ hypothesis templates across all change categories
- `llm_hypothesis.py` -- Claude-powered hypothesis generation
- `causal.py` -- Confounder identification (memory, context, variance)
- `trends.py` -- Time-series contradiction emergence analysis
- `cross_project.py` -- Multi-file cross-project comparison
- `visualize.py` -- PCA/t-SNE cluster visualization with matplotlib

## Comparison

| Feature | Manual Analysis | W&B | contradiction-detector |
|---|---|---|---|
| Detect contradictions | Manual spreadsheet review | No | Automatic |
| Explain *why* results conflict | Researcher intuition | No | Template + LLM hypotheses |
| Suggest next experiment | Manual | No | Auto-generated suggestions |
| Cross-project comparison | Copy-paste | Dashboard only | Built-in cross-analyze |
| Causal confounders | Manual | No | Automatic identification |
| Time-series trends | No | Run history | Contradiction emergence tracking |
| Visualization | Custom scripts | Built-in charts | Contradiction-aware cluster plots |
| Cost | Weeks of researcher time | $$ subscription | Free, open source |

## Contributing

Contributions are welcome. To get started:

```bash
git clone https://github.com/autoresearch/contradiction-detector.git
cd contradiction-detector
pip install -e ".[dev]"
pytest
```

Areas where we'd love help:
- Additional hypothesis templates for new change categories
- Improved embedding methods (sentence transformers, etc.)
- Integration with more experiment tracking platforms
- Interactive visualization (Plotly, web dashboard)

## License

MIT
