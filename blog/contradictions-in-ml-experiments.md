# I Built a Tool That Finds Contradictions in Your ML Experiments

You have run 50 experiments. Two of them used identical learning rates. One improved accuracy by 3%. The other crashed it by 5%. Why?

If you have spent any meaningful time training models, you know this feeling. You stare at your experiment tracker, scanning rows of results, trying to reconcile why the same change -- the exact same hyperparameter tweak -- produced completely opposite outcomes. Maybe the batch size was different. Maybe the model was larger. Maybe it was a Tuesday. You spend hours digging through logs, diffing configs, and second-guessing your pipeline. Sometimes you find the answer. Often you don't.

I built `contradiction-detector` because I got tired of doing this by hand. It is a Python tool that automatically scans your experiment results, finds pairs of experiments that contradict each other, and generates hypotheses about why the contradiction exists -- along with concrete suggestions for resolving it.

## The Problem: Contradictions Hide in Plain Sight

Machine learning experimentation is fundamentally messy. You change one thing, observe a result, change something else, observe another result. Over weeks and months, you accumulate hundreds of experiment records. Buried in those records are contradictions -- cases where the same type of change led to opposite outcomes.

These contradictions are not bugs. They are signal. A learning rate increase that helps a small model but hurts a large one tells you something important about your optimization landscape. Dropout that improves performance on one dataset but degrades it on another reveals something about your model's capacity relative to the data. These are the kinds of insights that lead to real understanding of your system.

The problem is finding them.

Existing experiment tracking tools -- Weights & Biases, MLflow, Neptune -- are excellent at recording metrics, visualizing training curves, and organizing runs. But none of them will tap you on the shoulder and say, "Hey, experiment #37 and experiment #52 both added skip connections, but one improved accuracy by 9% while the other dropped it by 6%. You should look into that."

Manual analysis does not scale. With 10 experiments, you can eyeball the results. With 100, you might catch the obvious contradictions. With 500, you are guaranteed to miss critical patterns. And in a team setting, where multiple researchers are running experiments on the same codebase, contradictions across projects are nearly invisible.

This is what `contradiction-detector` was built to solve.

## The Solution: contradiction-detector

`contradiction-detector` takes your experiment results in a simple TSV format and does three things:

1. **Detects contradictions** -- finds pairs of experiments where the same type of change (learning rate, architecture, regularization, etc.) produced opposite outcomes relative to a baseline.

2. **Explains them** -- generates hypotheses about why the contradiction occurred, using template-based reasoning that considers the context differences between experiments.

3. **Suggests resolution** -- proposes specific follow-up experiments to resolve each contradiction, from parameter sweeps to controlled ablations.

Under the hood, it works by first categorizing each experiment into one of 10 change types using regex pattern matching (learning rate, architecture, regularization, optimizer, data augmentation, batch size, warmup, weight initialization, loss function, and data preprocessing). Then it groups experiments by change type and finds pairs where one experiment improved and the other worsened relative to your baseline metric.

Each contradiction gets a confidence score computed from three factors: the cosine similarity of the experiment descriptions (using TF-IDF vectorization), the magnitude of the metric divergence, and the specificity of the change type. A contradiction between two nearly identical learning rate experiments with large metric differences scores higher than a vague architecture change with a small effect.

The CLI is straightforward:

```bash
# Full analysis
contradiction-detector analyze results.tsv --baseline 0.80

# Just the contradictions
contradiction-detector contradictions results.tsv --baseline 0.80

# Hypotheses with suggested experiments
contradiction-detector hypotheses results.tsv --baseline 0.80

# Cross-project analysis
contradiction-detector cross-analyze project_a.tsv project_b.tsv

# Cluster visualization
contradiction-detector visualize results.tsv --method pca --output clusters.png
```

## Live Demo: Finding Contradictions in 20 Experiments

Let me walk you through what this looks like in practice. I created a dataset of 20 realistic ML experiments -- the kind you might accumulate over a few weeks of model development. The experiments include learning rate changes, architecture modifications, regularization tweaks, and optimizer switches.

Running the analyzer:

```
$ contradiction-detector analyze demo/sample_results.tsv --baseline 0.80

Analyzed 20 experiments
Found 15 contradictions
Generated 51 hypotheses
```

Fifteen contradictions in 20 experiments. That is not unusual -- once you have multiple experiments per change type, contradictions emerge quickly. Let me look at the most interesting ones.

```
Contradiction 1 (lr_change, confidence=0.555):
  A: [a1b2c3d] Increased learning rate from 0.001 to 0.01 -> positive
  B: [e4f5g6h] Increased learning rate from 0.001 to 0.01 on larger model -> negative
```

This is a classic. The exact same learning rate change -- 0.001 to 0.01 -- improved the metric in one experiment (0.835) but degraded it in another (0.748). The key context difference: experiment B was on a "larger model." The confidence score of 0.555 reflects high description similarity (the descriptions are nearly identical) combined with meaningful metric divergence.

Here is another good one:

```
Contradiction 14 (regularization_change, confidence=0.499):
  A: [o1p2q3r] Increased weight decay from 1e-4 to 1e-3 -> positive
  B: [s4t5u6v] Increased weight decay from 1e-4 to 1e-3 on small dataset -> negative
```

Same weight decay change, opposite results. The context: a "small dataset." This makes intuitive sense -- stronger regularization can help prevent overfitting on large datasets but may suppress learning on small ones.

Now let's see what the tool suggests:

```
Hypothesis 1 (confidence: 70%):
  The learning rate change helped in experiment A (metric=0.8350) but
  hurt in experiment B (metric=0.7480). Hypothesis: optimal learning
  rate depends on context (B mentions: larger, model, on).
  Suggestion: Try intermediate learning rate values, controlling for
  the context difference: B mentions: larger, model, on

Hypothesis 2 (confidence: 50%):
  Learning rate sensitivity may vary with model scale or training
  stage. The same LR change produced opposite results.
  Suggestion: Run a learning rate sweep across different model sizes
  to map the optimal LR landscape.

Hypothesis 3 (confidence: 60%):
  The divergence (11.0% metric difference) suggests the learning rate
  interacts with other hyperparameters. Memory usage: A=4.2GB, B=4.1GB.
  Suggestion: Binary search: try LR midpoint between experiments A and
  B values. Test with both configurations to isolate the interaction.
```

Each contradiction generates multiple hypotheses at different confidence levels, each with a specific actionable suggestion. The first hypothesis identifies the context difference. The second generalizes to a broader pattern. The third proposes a concrete binary search strategy.

The cross-project analysis is equally powerful. When you have two separate result files -- say, from two team members or two phases of development -- the tool finds contradictions that span across them:

```
$ contradiction-detector cross-analyze project_a.tsv project_b.tsv

# Cross-Project Contradiction Analysis

Found 29 cross-project contradiction(s).

## Contradiction 1: lr_change
- File A: project_a.tsv
  Experiment: [a1b2c3d] Increased learning rate from 0.001 to 0.01 (positive)
- File B: project_b.tsv
  Experiment: [p1q2r3s] Increased learning rate from 0.001 to 0.01 with small batch (negative)
- Confidence: 0.551
```

This is the kind of insight that is almost impossible to catch manually, especially on a team where experiments are spread across different trackers or branches.

## Under the Hood

The technical implementation has a few components worth highlighting.

**TF-IDF Similarity.** Experiment descriptions are vectorized using a custom TF-IDF implementation that tokenizes descriptions into lowercase words, computes term frequency and inverse document frequency, and produces L2-normalized vectors. Cosine similarity between these vectors measures how "similar" two experiments are -- higher similarity between contradicting experiments means higher confidence that the contradiction is real rather than a false positive from comparing unrelated experiments.

**10 Change Categories.** The parser uses regex pattern matching to classify each experiment into one of 10 change types. Each category has a specificity weight that factors into the confidence score. Learning rate changes have high specificity (0.9) because they are very targeted -- if two LR experiments contradict, that is meaningful. Architecture changes have lower specificity (0.6) because "architecture change" is a broad category that might group dissimilar modifications.

**Confidence Scoring.** The confidence formula combines three signals with a weighted average: description similarity (weight 0.3), metric divergence normalized by baseline (weight 0.4), and change type specificity (weight 0.3). This means high-confidence contradictions are those where similar experiments with specific change types produced large metric differences.

**LLM Integration.** For richer hypotheses, the tool optionally integrates with Claude via the Anthropic API. Pass `--llm` to the hypotheses command and it will send each contradiction to Claude for analysis, returning structured hypotheses with explanations, confidence scores, and suggested experiments. Without an API key, it falls back gracefully to the template-based generator.

**Visualization.** The tool can generate scatter plots of experiments in 2D space (via PCA or t-SNE on the TF-IDF vectors), with contradiction pairs connected by red lines. This gives you an at-a-glance view of which experiment clusters are producing contradictions.

## What Makes This Novel

I have looked for existing tools that do this and come up empty. Experiment trackers record data. Hyperparameter optimizers search for optimal configurations. But nothing systematically detects and explains contradictions across experiment histories.

This is a gap that matters. Contradictions are where the interesting science lives. They reveal interaction effects, context dependencies, and implicit assumptions in your experimental setup. A tool that surfaces them automatically changes the way you think about your experiments -- instead of treating each run as an isolated data point, you start seeing the relationships and tensions between them.

There is potential here for deeper research work. The causal analysis module (currently heuristic-based) could be extended with proper causal inference methods. The hypothesis generator could incorporate findings from the ML literature about known hyperparameter interactions. And the cross-project analysis could scale to organization-wide experiment databases, finding contradictions across teams and codebases.

## Try It Yourself

Installation:

```bash
pip install contradiction-detector
```

For LLM-powered hypotheses:

```bash
pip install "contradiction-detector[llm]"
export ANTHROPIC_API_KEY=your-key-here
```

Quick start with sample data:

```bash
# Clone the repo
git clone https://github.com/your-org/contradiction-detector.git
cd contradiction-detector

# Run the demo
bash demo/run_demo.sh

# Or analyze your own data
contradiction-detector analyze your_results.tsv --baseline 0.80
contradiction-detector hypotheses your_results.tsv --baseline 0.80 --llm
contradiction-detector visualize your_results.tsv --output my_clusters.png
```

Your input file is a TSV with five columns:

```
commit    metric_value    memory_gb    status    description
a1b2c3d   0.8350          4.2          improved  Increased learning rate from 0.001 to 0.01
```

The `description` field is where the magic happens -- it is what the tool uses to categorize changes and compute similarity. Be descriptive. "Changed LR" will work. "Increased learning rate from 0.001 to 0.01 on ResNet-50 with CIFAR-10" will work much better.

## What's Next

The roadmap includes:

- **W&B plugin** -- pull experiment data directly from Weights & Biases instead of TSV files
- **Automated monitoring** -- watch for new contradictions as experiments are logged
- **Causal inference** -- move beyond heuristics to proper causal analysis of confounders
- **Paper search** -- automatically find published research addressing similar contradictions
- **Team dashboards** -- cross-researcher contradiction tracking with Slack notifications

## Get Involved

If you have ever lost hours debugging a contradiction that a tool could have caught in seconds, this project is for you.

- Star the repo on GitHub
- Try it on your own experiment history
- Open an issue if you find edge cases or have feature requests
- Contribute -- the hypothesis templates in particular benefit from domain expertise

The worst contradictions are the ones you never notice. Stop missing them.
