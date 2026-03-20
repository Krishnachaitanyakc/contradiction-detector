# Contradiction Detector Demo

Run `bash demo/run_demo.sh` to see contradiction detection in action.

## What the demo does

1. Analyzes 20 ML experiments from `sample_results.tsv` for contradictions
2. Lists all detected contradictions with confidence scores
3. Generates hypotheses explaining why contradictions occurred
4. Runs cross-project analysis comparing two experiment files

## Sample data

- `sample_results.tsv` -- 20 experiments with learning rate, architecture, regularization, and optimizer changes
- `sample_results_v2.tsv` -- 14 experiments from a second "project" for cross-project contradiction detection

## Requirements

```bash
pip install numpy scikit-learn click matplotlib
```
