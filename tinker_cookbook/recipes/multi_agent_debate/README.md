# Multi-Agent Debate

Multi-agent debate with pairwise comparison rewards for reinforcement learning. Agents engage in structured debates where they propose solutions, evaluate peers, and perform pairwise comparisons. The reward system incentivizes both high-quality solutions and accurate peer evaluation.

## Features

- **Dual Environment Support**:
  - Verifiable (math): Ground-truth answers with automated grading
  - Non-verifiable: Open-ended questions with peer evaluation only

- **V2 Reward System**:
  - **Generator rewards** (for `<solution>` and `<evaluation>` tokens): Soft vote ratio providing dense gradient signal
  - **Judge rewards** (for `<comparison>` tokens): Consensus alignment rewarding accurate peer evaluation

- **Parallel Generation**: All agents generate simultaneously within each round for faster training and independent judgments

- **Agent Personas**: 5 distinct reasoning styles (Methodical Analyst, Creative Problem-Solver, Devil's Advocate, Synthesizer, First Principles Thinker) with different temperature settings

- **Online Test-Time Learning**: Train and evaluate on the same dataset for verifiable environments

- **Sequence Extension**: O(T) compute scaling via KV-cache preservation (vs O(T²) without)

## Quick Start

### 1. Training (Verifiable Math)

Train on math problems with ground-truth verification:

```bash
python -m tinker_cookbook.recipes.multi_agent_debate.scripts.train \
    env="verifiable" \
    dataset_path="tinker_cookbook/data/aime2024_sample.jsonl" \
    problem_field="problem" \
    answer_field="answer" \
    num_agents=3 \
    max_rounds=3 \
    batch_size=16 \
    learning_rate=3e-5 \
    wandb_project="multi-agent-debate"
```

### 2. Training (Non-Verifiable)

Train on open-ended questions:

```bash
python -m tinker_cookbook.recipes.multi_agent_debate.train \
    env="non-verifiable" \
    dataset_path="tinker_cookbook/data/longwriter_6k_sample.jsonl" \
    problem_field="query" \
    num_agents=3 \
    max_rounds=3
```

### 3. Evaluation Only

Evaluate a trained checkpoint without training:

```bash
python -m tinker_cookbook.recipes.multi_agent_debate.scripts.eval \
    checkpoint_path="tinker://workspace-id/path/to/checkpoint" \
    dataset_path="tinker_cookbook/data/aime2024_sample.jsonl" \
    problem_field="problem" \
    answer_field="answer" \
    num_agents=3 \
    max_rounds=3 \
    max_parallel_evals=16
```

### 4. OpenRouter Self-Play (No Training)

Test debate dynamics without training using OpenRouter models:

```bash
export OPENROUTER_API_KEY=sk-or-...
python -m tinker_cookbook.recipes.multi_agent_debate.scripts.openrouter_selfplay \
    env="verifiable" \
    policy_model="openai/gpt-4o-mini" \
    dataset_path="tinker_cookbook/data/aime2024_sample.jsonl" \
    num_agents=3 \
    max_rounds=3 \
    max_questions=5
```

## Architecture

### Debate Protocol

Each debate consists of multiple rounds (cycles), with all agents generating in parallel within each round:

```
Round 1: Agent 0 | Agent 1 | Agent 2  (parallel)
         ↓        ↓        ↓
Round 2: Agent 0 | Agent 1 | Agent 2  (parallel, sees Round 1 responses)
         ↓        ↓        ↓
Round 3: Agent 0 | Agent 1 | Agent 2  (parallel, sees Round 1-2 responses)
```

**Blind Review**: Agents see `<solution>` and `<evaluation>` from previous rounds, but NOT `<comparison>` tags. This ensures independent judgment and prevents bandwagon effects.

### Response Format

Each agent response has three sections:

```xml
<solution>
Your answer to the query.
For math problems, must include \boxed{answer} format.
</solution>

<evaluation>
N/A (if first to respond)
Otherwise: Critique of other agents' solutions and reasoning.
</evaluation>

<comparison>
N/A (if fewer than 2 other agents have responded)
Otherwise: Pairwise comparisons, e.g.:
Agent 0 > Agent 1
Agent 2 > Agent 0
</comparison>
```

### Reward System V2

The v2 reward system assigns separate rewards to generator and judge tokens:

#### Generator Rewards (for `<solution>` and `<evaluation>` tokens)
- **Computation**: Soft vote ratio = `sum(votes_for) / count(votes)`
- **Range**: [-1, +1]
- **Interpretation**:
  - +1 = unanimously preferred
  - 0 = neutral (equal wins/losses)
  - -1 = unanimously rejected
- **Benefit**: Dense gradient signal at every step

#### Judge Rewards (for `<comparison>` tokens)
- **Computation**: Consensus alignment
  1. Build majority consensus for each pair across all votes
  2. Reward judges who align with consensus (+1) or disagree (-1)
  3. Ties receive 0 reward (no signal)
- **Benefit**: Incentivizes accurate peer evaluation

#### Format Penalties
- Missing comparisons when expected: -0.5 reward
- Turns 0-1 exempt (not enough agents to compare)
- Requires `enable_format_penalty=True` (default)

### Agent Personas

Each agent has a distinct reasoning style with suggested temperature:

| Agent ID | Persona | Style | Temperature |
|----------|---------|-------|-------------|
| 0 | Methodical Analyst | Systematic, step-by-step verification | 0.6 |
| 1 | Creative Problem-Solver | Unconventional approaches, pattern recognition | 1.0 |
| 2 | Devil's Advocate | Skeptical, looks for counterexamples | 0.9 |
| 3 | Synthesizer | Combines ideas from multiple sources | 1.0 |
| 4 | First Principles Thinker | Goes back to fundamentals | 0.8 |

Personas cycle if `num_agents > 5`.

## Configuration

### Key Hyperparameters

See `train.py` for full configuration options:

```python
@chz.chz
class CLIConfig:
    # Model
    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str = "qwen3_disable_thinking"
    max_tokens: int = 8196
    enable_sequence_extension: bool = True  # O(T) vs O(T²)

    # Debate
    num_agents: int = 3
    max_rounds: int = 3

    # Training
    batch_size: int = 16
    learning_rate: float = 3e-5
    use_cosine_lr_schedule: bool = False

    # Reward weights
    lambda_gen: float = 1.0    # Weight for generator rewards
    lambda_judge: float = 1.0  # Weight for judge rewards
    enable_format_penalty: bool = True

    # Evaluation (verifiable only)
    disable_eval: bool = False
    eval_every: int = 50  # Evaluate every N batches
    max_parallel_evals: int = 64

    # Optional summarization (uses OpenRouter)
    summarize_history: bool = False
    summarize_model: str = "openai/gpt-4o-mini"
```

### Recommended Settings

**Fast iteration** (debugging):
```bash
batch_size=4 num_train_datapoints=32 max_tokens=4096 disable_eval=True
```

**Production training** (math):
```bash
batch_size=16 num_train_datapoints=1024 max_tokens=8196 eval_every=50
```

**Accuracy vs Speed**:
- More agents (5-7): Better consensus, slower training
- Fewer agents (2-3): Faster training, noisier signal
- More rounds (4-5): More refinement, longer episodes

## Evaluation Metrics

### Training Metrics (verifiable environment)

Per-dataset metrics:
- `train_{dataset}/format`: Fraction of valid responses (has `\boxed{}`)
- `train_{dataset}/correct`: Fraction of correct final answers

Multi-agent aggregation (k = num_agents):
- `train_{dataset}/pass@k`: At least one agent got it right (optimistic)
- `train_{dataset}/avg@k`: Average accuracy across all k agents
- `train_{dataset}/cons@k`: Majority of agents got it right (consensus)

Reward metrics:
- `reward/gen/mean`: Mean generator reward
- `reward/judge/mean`: Mean judge reward
- `v2/total_votes`: Number of pairwise comparisons used
- `v2/missing_comparisons`: Number of format penalties applied

### Evaluation Metrics

Dual-mode evaluation (verifiable only):
- `eval/direct/{dataset}/format`: Direct (single-turn) format accuracy
- `eval/direct/{dataset}/correct`: Direct correctness
- `eval/debate/{dataset}/format`: Debate (multi-turn) format accuracy
- `eval/debate/{dataset}/correct`: Debate correctness
- `eval/debate/{dataset}/pass@k`: Best-of-k accuracy
- `eval/debate/{dataset}/cons@k`: Consensus accuracy

## File Structure

```
multi_agent_debate/
├── __init__.py                    # Public API exports
├── README.md                      # This file
├── POLISH_RECOMMENDATIONS.md      # Code organization recommendations
├── utils.py                       # Logging and utilities
│
├── core/                          # Core debate logic
│   ├── coordinator.py             # Turn-taking coordination
│   └── prompts.py                 # Prompts and response parsing
│
├── environments/                  # Environment implementations
│   ├── base.py                    # Base classes
│   ├── debate.py                  # Non-verifiable debate
│   └── verifiable.py              # Verifiable math debate
│
├── evaluation/                    # Evaluation components
│   └── evaluator.py               # Custom dual-mode evaluator
│
├── data/                          # Data loading
│   └── loaders.py                 # JSONL data loaders
│
├── scripts/                       # Runnable scripts
│   ├── train.py                   # Main training script
│   ├── eval.py                    # Evaluation-only script
│   └── openrouter_selfplay.py     # OpenRouter self-play (no training)
│
└── doc/                           # Algorithm documentation
    ├── algorithm_flowchart.md     # Visual protocol overview
    └── math_formulation.md        # Mathematical formulation
```

## Advanced Usage

### Custom Datasets

**Verifiable (math) dataset format** (JSONL):
```json
{"problem": "Solve x^2 = 4", "answer": "2"}
{"problem": "What is 2+2?", "answer": "4"}
```

**Non-verifiable dataset format** (JSONL):
```json
{"query": "Explain quantum computing"}
{"query": "Compare Python and Rust"}
```

### History Summarization

For long debates, enable history summarization using OpenRouter:

```bash
export OPENROUTER_API_KEY=sk-or-...
python -m tinker_cookbook.recipes.multi_agent_debate.train \
    summarize_history=True \
    summarize_model="openai/gpt-4o-mini"
```

### Sequence Extension

**Enabled (default)**: Preserves thinking tags in history → O(T) compute
**Disabled**: Strips thinking from history → O(T²) compute (HuggingFace behavior)

Only works with `renderer_name="qwen3_disable_thinking"`.

### Logging

Control transcript logging:
```bash
# Log first 4 training groups per batch
num_groups_to_log=4 log_full_transcript=True

# Log first 2 evaluation groups
eval_num_groups_to_log=2

# Disable all transcript logging (faster)
num_groups_to_log=0 log_full_transcript=False
```

## Troubleshooting

### Common Issues

**Issue**: `ValueError: summarize_history=True but no _summarizer_policy provided`
- **Solution**: Set `OPENROUTER_API_KEY` environment variable

**Issue**: Evaluation is slow
- **Solution**: Reduce `max_parallel_evals` (e.g., 16 or 32) to control concurrency

**Issue**: Agents produce invalid comparisons
- **Solution**: Check `v2/comparison_lines_invalid` metric. May need to adjust prompts or increase max_tokens

**Issue**: Rewards are all zero
- **Solution**: Agents may not be producing comparisons. Check `v2/total_votes` metric and transcript logs

**Issue**: Out of memory
- **Solution**: Reduce `batch_size`, `max_tokens`, or `num_agents`

### Performance Tips

1. **Use sequence extension**: `enable_sequence_extension=True` (9x faster for 9-turn episodes)
2. **Parallel eval**: `max_parallel_evals=64` for faster evaluation
3. **Disable eval during training**: `disable_eval=True` for faster iteration
4. **Use smaller models for debugging**: `model_name="Qwen/Qwen3-1.5B"`

## Research Documentation

Comprehensive research-style documentation is available in the `doc/` directory:

### Algorithm & Theory

- **`doc/algorithm_flowchart.md`**: Visual overview of the complete training flow
  - Turn-by-turn execution diagram
  - Coordinator synchronization logic
  - Reward computation pipeline

- **`doc/math_formulation.md`**: Mathematical formulation and implementation details
  - Formal problem definition
  - V2 reward system equations (soft vote ratio + consensus alignment)
  - Current behavior documentation (including edge cases)

### Research Guide

- **`doc/research_overview.md`**: Full research paper-style overview
  - Abstract, introduction, and motivation
  - Detailed method description (debate protocol, reward system v2)
  - Results analysis and identified issues
  - Future directions and open questions
  - **Recommended reading for researchers**

### Training Analysis

- **`doc/training_analysis_aime.md`**: Why training doesn't improve on AIME 2024
  - Root cause analysis (task difficulty, reward signal quality, exploration)
  - Diagnostic questions and experimental protocols
  - Concrete solutions (curriculum learning, ground-truth mixing, warm-start)
  - Action plan with expected outcomes
  - **Must-read if training on hard mathematical reasoning tasks**

### Experimental Protocol

- **`doc/experimental_protocol.md`**: Standardized experimental procedures
  - Standard configurations (baseline, fast iteration, production)
  - Dataset-specific hyperparameters (GSM8K, MATH, AIME)
  - Ablation study templates
  - Curriculum learning protocols
  - Reproducibility checklist
  - **Essential for running systematic experiments**

## Citation

If you use this recipe, please cite:

```bibtex
@software{multi_agent_debate_tinker,
  title = {Multi-Agent Debate for Reinforcement Learning},
  author = {Thinking Machines Lab},
  year = {2025},
  url = {https://github.com/thinking-machines/tinker-cookbook}
}
```

## See Also

- `../math_rl/`: Single-agent math RL for comparison
- `../../docs/rl.mdx`: General RL documentation for Tinker
- `../../docs/training-sampling.mdx`: Training and sampling basics
