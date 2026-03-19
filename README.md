# contradiction-detector

Detect contradictory experiment results and generate hypotheses explaining the contradictions.

## Features

- Parse experiment results from TSV files
- Categorize experiments by change type (learning rate, architecture, etc.)
- Embed experiments using TF-IDF vectorization
- Detect contradictions: similar changes with opposite outcomes
- Generate template-based hypotheses explaining contradictions
- Produce structured markdown reports

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
# Run full analysis pipeline
contradiction-detector analyze results.tsv

# List found contradictions
contradiction-detector contradictions results.tsv

# Show generated hypotheses
contradiction-detector hypotheses results.tsv

# Generate markdown report
contradiction-detector report results.tsv
```

### Python API

```python
from contradiction_detector.analyzer import ContradictionAnalyzer

analyzer = ContradictionAnalyzer()
report = analyzer.analyze("results.tsv")
print(report.contradictions)
print(report.hypotheses)
```

## Architecture

- `parser.py` - Parse results.tsv and categorize experiments
- `embedder.py` - TF-IDF vectorization and cosine similarity
- `detector.py` - Contradiction detection using clustering
- `hypothesis.py` - Template-based hypothesis generation
- `analyzer.py` - Full pipeline orchestration
