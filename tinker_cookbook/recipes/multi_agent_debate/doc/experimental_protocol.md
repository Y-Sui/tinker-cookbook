# Multi-Agent Debate: Experimental Protocol & Reproducibility Guide

**Purpose:** This document provides standardized experimental protocols for multi-agent debate research, ensuring reproducibility and fair comparisons across different configurations.

---

## 1. Standard Experimental Setups

### 1.1 Baseline Configuration

**Purpose:** Default setup for initial experiments and ablations.

```python
# Model
model_name = "Qwen/Qwen3-8B"
renderer_name = "qwen3_disable_thinking"
max_tokens = 8196
enable_sequence_extension = True

# Debate
num_agents = 3
max_rounds = 3  # 9 total turns

# Training
batch_size = 16
learning_rate = 3e-5
num_train_datapoints = 1024
epoch = 1

# Rewards
lambda_gen = 1.0
lambda_judge = 1.0
enable_format_penalty = True

# Evaluation
eval_every = 50
max_parallel_evals = 64
disable_eval = False
```

**When to use:** First run on any new dataset, baseline for all ablations.

### 1.2 Fast Iteration Configuration

**Purpose:** Quick debugging and hyperparameter search.

```python
# Reduced scale for faster iteration
batch_size = 4
num_train_datapoints = 64
max_tokens = 4096

# Simplified debate
num_agents = 2
max_rounds = 2  # 4 total turns

# More frequent evaluation
eval_every = 10
num_groups_to_log = 2
log_full_transcript = True  # For debugging

# Faster evaluation
max_parallel_evals = 16
```

**When to use:** Debugging code, testing new features, quick sanity checks.

### 1.3 Production Configuration

**Purpose:** Final training runs for publication.

```python
# Full scale
batch_size = 32
num_train_datapoints = 2048
max_tokens = 8196

# Extended debate
num_agents = 5
max_rounds = 3  # 15 total turns

# Careful evaluation
eval_every = 100
max_parallel_evals = 64

# Logging
log_full_transcript = False  # Too slow for production
num_groups_to_log = 0  # Disable transcript logging
```

**When to use:** Final experiments, publication results, model release.

---

## 2. Dataset-Specific Configurations

### 2.1 GSM8K (Grade School Math)

**Difficulty:** Easy
**Expected base accuracy:** 30-60%
**Target accuracy after training:** 60-75%

```python
dataset_path = "tinker_cookbook/data/gsm8k.jsonl"
problem_field = "question"
answer_field = "answer"

# Configuration
num_agents = 3
max_rounds = 2  # 6 turns (simpler problems need less debate)
batch_size = 32
learning_rate = 5e-5
num_train_datapoints = 2048

# Expected training time: 2-4 hours on 1x H100
```

**Metrics to track:**
- `train_gsm8k/correct`: Direct accuracy
- `train_gsm8k/pass@3`: Best-of-3 accuracy
- `reward/gen/mean`: Should be positive (good solutions rewarded)

### 2.2 MATH (High School Math)

**Difficulty:** Medium to Hard
**Expected base accuracy:** 10-30%
**Target accuracy after training:** 25-40%

```python
dataset_path = "tinker_cookbook/data/math.jsonl"
problem_field = "problem"
answer_field = "answer"

# Configuration
num_agents = 4
max_rounds = 3  # 12 turns
batch_size = 16
learning_rate = 3e-5
num_train_datapoints = 1536

# Expected training time: 6-12 hours on 1x H100
```

**Metrics to track:**
- `train_math/correct`: Overall accuracy
- `train_math/algebra/correct`: Per-category accuracy
- `train_math/pass@4`: Best-of-4 accuracy
- `v2/total_votes`: Should increase over time (better judgment)

### 2.3 AIME (Competition Math)

**Difficulty:** Very Hard
**Expected base accuracy:** 0-5%
**Target accuracy after training:** 12-18% (with curriculum)

```python
dataset_path = "tinker_cookbook/data/aime2024_sample.jsonl"
problem_field = "problem"
answer_field = "answer"

# Configuration (REQUIRES CURRICULUM - see Section 4)
num_agents = 5
max_rounds = 3  # 15 turns
batch_size = 8  # Smaller batches for stability
learning_rate = 1e-5  # Lower LR for fine-tuning
num_train_datapoints = 512

# Expected training time: 8-16 hours on 1x H100 (after curriculum)
```

**Special considerations:**
- ⚠️ DO NOT train directly on AIME without curriculum
- ⚠️ Requires supervised pre-training or curriculum from GSM8K → MATH → AIME
- Track `v2/comparison_lines_invalid` (should be < 30%)

### 2.4 LongWriter-6K (Non-Verifiable)

**Difficulty:** Subjective
**Expected base accuracy:** N/A (no ground truth)
**Target:** Improved peer agreement and quality

```python
dataset_path = "tinker_cookbook/data/longwriter_6k_sample.jsonl"
problem_field = "query"
# No answer_field (non-verifiable)

# Configuration
num_agents = 4
max_rounds = 3  # 12 turns
batch_size = 16
learning_rate = 3e-5
num_train_datapoints = 1024

# Expected training time: 4-8 hours on 1x H100
```

**Metrics to track:**
- `v2/total_votes`: Number of comparisons made
- `v2/missing_comparisons`: Should decrease
- Reward variance: Should stabilize over training

---

## 3. Ablation Studies

### 3.1 Reward System Ablations

**A1: Generator-Only Rewards**
```python
lambda_gen = 1.0
lambda_judge = 0.0  # Disable judge rewards
```
**Hypothesis:** Generation quality improves, judgment quality doesn't.

**A2: Judge-Only Rewards**
```python
lambda_gen = 0.0  # Disable generator rewards
lambda_judge = 1.0
```
**Hypothesis:** Judgment quality improves, generation quality doesn't.

**A3: Balanced Rewards (Baseline)**
```python
lambda_gen = 1.0
lambda_judge = 1.0
```

**A4: Judge-Emphasized**
```python
lambda_gen = 0.5
lambda_judge = 1.5
```
**Hypothesis:** Better judges → better rewards → better generators (indirect effect).

**A5: No Format Penalty**
```python
enable_format_penalty = False
```
**Hypothesis:** Removes constraint, may improve content quality or cause format collapse.

### 3.2 Debate Structure Ablations

**B1: Number of Agents**
```python
# Vary: num_agents ∈ {2, 3, 5, 7}
# Keep: max_rounds = 3, other params constant
```
**Hypothesis:** More agents → more diverse solutions, but slower training.

**B2: Number of Rounds**
```python
# Vary: max_rounds ∈ {1, 2, 3, 4}
# Keep: num_agents = 3, other params constant
```
**Hypothesis:** More rounds → better refinement, but diminishing returns.

**B3: Persona Diversity**
```python
# Condition 1: All agents same persona
# Condition 2: Different personas (baseline)
```
**Hypothesis:** Diverse personas improve exploration.

### 3.3 Training Dynamics Ablations

**C1: Learning Rate**
```python
# Vary: learning_rate ∈ {1e-5, 3e-5, 5e-5, 1e-4}
```
**Expected:** Inverted-U relationship (too low = no learning, too high = instability).

**C2: Batch Size**
```python
# Vary: batch_size ∈ {8, 16, 32, 64}
```
**Expected:** Larger batches = more stable gradients, but slower iteration.

**C3: Sequence Extension**
```python
# Condition 1: enable_sequence_extension = True (O(T) scaling)
# Condition 2: enable_sequence_extension = False (O(T²) scaling)
```
**Expected:** True is ~9x faster for 9-turn debates, same final accuracy.

---

## 4. Curriculum Learning Protocol

### 4.1 Three-Stage Curriculum (Recommended)

**Stage 1: GSM8K (Foundation)**
```python
# Duration: 1000-2000 batches or until 60% accuracy
dataset = "gsm8k"
num_agents = 3
max_rounds = 2
learning_rate = 5e-5
batch_size = 32

# Stopping criterion: train_gsm8k/correct >= 0.60
```

**Stage 2: MATH (Intermediate)**
```python
# Duration: 1500-3000 batches or until 30% accuracy
dataset = "math"
num_agents = 4
max_rounds = 3
learning_rate = 3e-5
batch_size = 16

# Load checkpoint from Stage 1
# Stopping criterion: train_math/correct >= 0.30
```

**Stage 3: AIME (Advanced)**
```python
# Duration: 1000-2000 batches or until convergence
dataset = "aime"
num_agents = 5
max_rounds = 3
learning_rate = 1e-5  # Lower LR for fine-tuning
batch_size = 8

# Load checkpoint from Stage 2
# Stopping criterion: train_aime/pass@5 >= 0.15
```

### 4.2 Mixed-Difficulty Curriculum (Alternative)

**Concept:** Sample from multiple datasets with changing proportions.

```python
# Batch composition over time
batch_distribution = [
    # Batches 0-500: 70% GSM8K, 30% MATH
    (0, 500): {"gsm8k": 0.7, "math": 0.3},

    # Batches 500-1500: 50% GSM8K, 40% MATH, 10% AIME
    (500, 1500): {"gsm8k": 0.5, "math": 0.4, "aime": 0.1},

    # Batches 1500+: 20% GSM8K, 50% MATH, 30% AIME
    (1500, float('inf')): {"gsm8k": 0.2, "math": 0.5, "aime": 0.3},
]
```

**Advantages:**
- Maintains skills on easier problems
- Smoother transition between difficulty levels

**Disadvantages:**
- More complex implementation
- Harder to interpret results

---

## 5. Evaluation Protocol

### 5.1 Online Evaluation (During Training)

**Frequency:** Every 50-100 batches
**Purpose:** Track progress without stopping training

```python
eval_every = 50  # Evaluate every 50 batches
max_parallel_evals = 64  # Fast evaluation
num_groups_to_log = 2  # Minimal logging
```

**Metrics to save:**
- Accuracy (`eval/debate/{dataset}/correct`)
- Pass@k (`eval/debate/{dataset}/pass@k`)
- Consensus (`eval/debate/{dataset}/cons@k`)
- Format validity

### 5.2 Offline Evaluation (Checkpoint Analysis)

**Purpose:** Detailed analysis of saved checkpoints.

```python
# Load checkpoint
checkpoint_path = "tinker://workspace/checkpoint_batch_1000"

# Run full evaluation
python -m tinker_cookbook.recipes.multi_agent_debate.scripts.eval \
    checkpoint_path=$checkpoint_path \
    dataset_path="tinker_cookbook/data/aime2024_sample.jsonl" \
    max_parallel_evals=16 \
    log_full_transcript=True \
    num_groups_to_log=50  # Log many examples
```

**Analysis:**
1. Compute standard metrics (accuracy, pass@k)
2. Manual inspection of transcripts (quality, reasoning)
3. Error analysis (categorize failure modes)
4. Judgment accuracy (compare to ground truth)

### 5.3 Cross-Dataset Generalization

**Purpose:** Test if improvements transfer to unseen problems.

```python
# Train on MATH
train_dataset = "math"

# Evaluate on:
eval_datasets = [
    "math_test",       # Held-out MATH problems
    "gsm8k",           # Easier (should be perfect)
    "aime",            # Harder (generalization test)
    "mathbench",       # Different distribution
]
```

**Expected patterns:**
- Perfect on easier datasets (GSM8K)
- Moderate improvement on held-out test set
- Limited transfer to much harder datasets (AIME)

---

## 6. Reproducibility Checklist

### 6.1 Required Information for Reproduction

When reporting results, include:

**Model:**
- [ ] Model name and version (e.g., "Qwen/Qwen3-8B")
- [ ] Renderer name (e.g., "qwen3_disable_thinking")
- [ ] Sequence extension enabled/disabled

**Data:**
- [ ] Dataset name and version
- [ ] Number of training examples
- [ ] Train/test split (if applicable)
- [ ] Data preprocessing steps

**Hyperparameters:**
- [ ] Number of agents
- [ ] Number of rounds
- [ ] Batch size
- [ ] Learning rate
- [ ] Reward weights (λ_gen, λ_judge)
- [ ] Format penalty enabled/disabled

**Training:**
- [ ] Number of batches/epochs
- [ ] Optimizer (default: Tinker RL)
- [ ] Stopping criterion
- [ ] Random seed (if set)

**Hardware:**
- [ ] GPU type (e.g., H100, A100)
- [ ] Number of GPUs
- [ ] Wall-clock time

**Results:**
- [ ] Final metrics with confidence intervals
- [ ] Learning curves (accuracy vs. batches)
- [ ] Example outputs (qualitative analysis)

### 6.2 Random Seed Management

**For reproducibility:**
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set before training
set_seed(42)
```

**Note:** Perfect reproducibility is difficult in distributed RL due to:
- Async policy rollouts
- Non-deterministic GPU operations
- Floating-point arithmetic

Expected variance: ±2-3% in accuracy across runs with same hyperparameters.

### 6.3 Versioning

**Code:**
```bash
git rev-parse HEAD  # Record git commit hash
```

**Dependencies:**
```bash
pip freeze > requirements.txt  # Save exact versions
```

**Data:**
```bash
md5sum dataset.jsonl  # Record data checksum
```

---

## 7. Statistical Significance Testing

### 7.1 Comparing Two Configurations

**Scenario:** Does configuration A outperform configuration B?

**Protocol:**
1. Run each configuration 3-5 times with different random seeds
2. Compute mean and standard deviation of accuracy
3. Perform paired t-test (if distributions are normal) or Wilcoxon signed-rank test

**Example:**
```python
from scipy import stats

config_a_accuracies = [0.45, 0.47, 0.46, 0.48, 0.45]  # 5 runs
config_b_accuracies = [0.42, 0.43, 0.41, 0.44, 0.42]  # 5 runs

t_stat, p_value = stats.ttest_rel(config_a_accuracies, config_b_accuracies)
print(f"p-value: {p_value}")

# If p < 0.05, difference is statistically significant
```

**Reporting:**
```
Configuration A: 46.2% ± 1.1%
Configuration B: 42.4% ± 1.1%
Improvement: 3.8 percentage points (p < 0.05)
```

### 7.2 Ablation Study Analysis

**Scenario:** Does removing component X hurt performance?

**Protocol:**
1. Run baseline (all components) 3-5 times
2. Run ablation (component X removed) 3-5 times
3. Compare using paired t-test

**Example:**
```python
baseline = [0.46, 0.47, 0.45]  # With judge rewards
ablation = [0.41, 0.42, 0.40]  # Without judge rewards

t_stat, p_value = stats.ttest_rel(baseline, ablation)
print(f"Effect of judge rewards: +{np.mean(baseline) - np.mean(ablation):.2%} (p={p_value:.3f})")
```

---

## 8. Common Pitfalls & How to Avoid Them

### 8.1 Overfitting to Evaluation Set

**Pitfall:** Evaluating on the same problems used for training.

**Solution:**
- For non-online-TTL: Always use held-out test set
- For online-TTL: Accept that train = eval (by design)
- Report both train and test metrics clearly

### 8.2 Cherry-Picking Checkpoints

**Pitfall:** Reporting best checkpoint instead of final checkpoint.

**Solution:**
- Pre-specify checkpoint selection criterion (e.g., "best eval accuracy")
- Report both best and final checkpoints
- Use validation set for checkpoint selection, test set for final evaluation

### 8.3 Hyperparameter Overfitting

**Pitfall:** Tuning hyperparameters on test set.

**Solution:**
- Use train/val/test split
- Tune hyperparameters on validation set only
- Evaluate on test set exactly once (at the end)

### 8.4 Ignoring Variance

**Pitfall:** Reporting single run without confidence intervals.

**Solution:**
- Run at least 3 trials with different seeds
- Report mean ± std
- Show error bars in plots

### 8.5 Unfair Comparisons

**Pitfall:** Comparing methods with different compute budgets.

**Solution:**
- Match number of forward passes (not wall-clock time)
- Report compute (e.g., "1000 batches × 16 problems × 3 agents = 48K forward passes")
- Compare methods at iso-compute

---

## 9. Logging & Visualization

### 9.1 Weights & Biases Integration

**Setup:**
```python
wandb_project = "multi-agent-debate"
wandb_name = f"aime_{num_agents}agents_{max_rounds}rounds"
```

**Recommended logging:**
- Loss: Every batch
- Metrics: Every eval (50-100 batches)
- Histograms: Every 500 batches (reward distribution)
- Examples: Every eval (sample debate transcripts)

### 9.2 Visualization Scripts

**Learning curves:**
```python
import matplotlib.pyplot as plt

plt.plot(batches, train_accuracy, label='Train')
plt.plot(batches, eval_accuracy, label='Eval')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('learning_curve.png')
```

**Reward distributions:**
```python
plt.hist(gen_rewards, alpha=0.5, label='Generator')
plt.hist(judge_rewards, alpha=0.5, label='Judge')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('reward_distribution.png')
```

---

## 10. Checklist for New Experiments

Before running a new experiment:

**Setup:**
- [ ] Dataset prepared and validated
- [ ] Hyperparameters chosen (baseline or ablation)
- [ ] Random seed set (for reproducibility)
- [ ] Logging configured (W&B or local)

**Sanity checks:**
- [ ] Run 1-2 batches to verify no crashes
- [ ] Check metrics are in reasonable range
- [ ] Verify GPU utilization (should be >80%)
- [ ] Inspect sample outputs (format, content)

**During training:**
- [ ] Monitor loss (should decrease)
- [ ] Monitor reward variance (should stabilize)
- [ ] Check for NaN/Inf values
- [ ] Verify checkpoints are saving

**After training:**
- [ ] Run final evaluation on test set
- [ ] Save results (metrics, plots, examples)
- [ ] Document hyperparameters and observations
- [ ] Archive checkpoint if promising

---

## Appendix A: Useful Commands

### Training
```bash
# Baseline
python -m tinker_cookbook.recipes.multi_agent_debate.scripts.train \
    env="verifiable" \
    dataset_path="tinker_cookbook/data/aime2024_sample.jsonl"

# With custom config
python -m tinker_cookbook.recipes.multi_agent_debate.scripts.train \
    num_agents=5 \
    max_rounds=3 \
    batch_size=32 \
    learning_rate=3e-5 \
    wandb_project="my-experiments"
```

### Evaluation
```bash
# Evaluate checkpoint
python -m tinker_cookbook.recipes.multi_agent_debate.scripts.eval \
    checkpoint_path="tinker://workspace/checkpoint_batch_1000" \
    dataset_path="tinker_cookbook/data/aime2024_sample.jsonl" \
    log_full_transcript=True

# Fast evaluation (no logging)
python -m tinker_cookbook.recipes.multi_agent_debate.scripts.eval \
    checkpoint_path="tinker://workspace/checkpoint_batch_1000" \
    dataset_path="tinker_cookbook/data/aime2024_sample.jsonl" \
    max_parallel_evals=128 \
    num_groups_to_log=0
```

### OpenRouter Self-Play (No Training)
```bash
export OPENROUTER_API_KEY=sk-or-...
python -m tinker_cookbook.recipes.multi_agent_debate.scripts.openrouter_selfplay \
    env="verifiable" \
    policy_model="openai/gpt-4o-mini" \
    dataset_path="tinker_cookbook/data/aime2024_sample.jsonl" \
    max_questions=10
```

---

**Document Version:** 1.0
**Last Updated:** January 17, 2025
**Maintained by:** Thinking Machines Lab
