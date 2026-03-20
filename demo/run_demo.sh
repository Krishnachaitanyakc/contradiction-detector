#!/bin/bash
# Contradiction Detector Demo
# Run this script from any directory: bash demo/run_demo.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Find a python3 with numpy installed
PYTHON=""
for candidate in python3 /usr/bin/python3 /usr/local/bin/python3; do
    if command -v "$candidate" &>/dev/null && "$candidate" -c "import numpy" 2>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Could not find python3 with numpy installed."
    echo "Please run: pip3 install numpy scikit-learn click matplotlib"
    exit 1
fi

echo "Using Python: $PYTHON"

echo "============================================"
echo "   Contradiction Detector Demo"
echo "============================================"
echo ""
echo "This demo analyzes ML experiment results to find"
echo "contradictions where the same type of change produced"
echo "opposite outcomes in different contexts."
echo ""
echo "--------------------------------------------"
echo "Step 1: Full Analysis (20 experiments)"
echo "--------------------------------------------"
echo ""

cd "$PROJECT_DIR"
PYTHONPATH=src "$PYTHON" -c "
from contradiction_detector.analyzer import ContradictionAnalyzer

analyzer = ContradictionAnalyzer(baseline_metric=0.80)
report = analyzer.analyze_file('demo/sample_results.tsv')

print(f'Analyzed {len(report.experiments)} experiments')
print(f'Found {len(report.contradictions)} contradictions')
print(f'Generated {len(report.hypotheses)} hypotheses')
"

echo ""
echo "--------------------------------------------"
echo "Step 2: Listing Contradictions"
echo "--------------------------------------------"
echo ""

PYTHONPATH=src "$PYTHON" -c "
from contradiction_detector.analyzer import ContradictionAnalyzer

analyzer = ContradictionAnalyzer(baseline_metric=0.80)
report = analyzer.analyze_file('demo/sample_results.tsv')

for i, c in enumerate(report.contradictions, 1):
    print(f'Contradiction {i} ({c.change_type}, confidence={c.confidence:.3f}):')
    print(f'  A: [{c.exp_a.commit}] {c.exp_a.description} -> {c.direction_a}')
    print(f'  B: [{c.exp_b.commit}] {c.exp_b.description} -> {c.direction_b}')
    print()
"

echo "--------------------------------------------"
echo "Step 3: Generating Hypotheses"
echo "--------------------------------------------"
echo ""

PYTHONPATH=src "$PYTHON" -c "
from contradiction_detector.analyzer import ContradictionAnalyzer

analyzer = ContradictionAnalyzer(baseline_metric=0.80)
report = analyzer.analyze_file('demo/sample_results.tsv')

for i, h in enumerate(report.hypotheses[:8], 1):
    print(f'Hypothesis {i} (confidence: {h.confidence:.0%}):')
    print(f'  {h.explanation}')
    print(f'  Suggestion: {h.suggested_experiment}')
    print()
"

echo "--------------------------------------------"
echo "Step 4: Cross-Project Analysis"
echo "--------------------------------------------"
echo ""

PYTHONPATH=src "$PYTHON" -c "
from contradiction_detector.cross_project import cross_analyze, format_cross_analysis

results = cross_analyze(
    ['demo/sample_results.tsv', 'demo/sample_results_v2.tsv'],
    baseline_metric=0.80,
)
print(format_cross_analysis(results))
"

echo ""
echo "============================================"
echo "   Demo Complete!"
echo "============================================"
echo ""
echo "To run on your own data, create a TSV file with columns:"
echo "  commit  metric_value  memory_gb  status  description"
echo ""
echo "Then run:"
echo "  contradiction-detector analyze your_results.tsv --baseline 0.80"
