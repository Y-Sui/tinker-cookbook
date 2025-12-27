# Math Multi-Agent Debate Integration Summary

## Overview

This integration combines the **multi-agent debate framework** with **verifiable math problem-solving**, creating a powerful RL environment where agents:

1. **Collaborate** to solve challenging math problems
2. **Learn** from both peer evaluations (debate rewards) and correctness metrics
3. **Adapt** their strategies based on multi-turn feedback

## What Was Created

### New Files

#### Core Implementation
1. **`tinker_cookbook/recipes/multi_agent_debate/math_debate_prompts.py`**
   - Math-focused system and user prompts
   - Follows `math_rl` style while preserving debate structure
   - Emphasizes \\boxed{} format and step-by-step reasoning

2. **`tinker_cookbook/recipes/multi_agent_debate/math_debate_datasets.py`**
   - Dataset loaders for training (MATH, GSM8K, Polaris)
   - Dataset loaders for evaluation (MATH-500, AIME 2024, AIME 2025)
   - Generic JSONL loader for custom datasets
   - Dataset registry for easy access

3. **`tinker_cookbook/recipes/multi_agent_debate/math_debate_env.py`**
   - `MathDebateEnv`: Environment for one agent in a debate
   - `MathDebateEnvGroupBuilder`: Builder for environment groups
   - Dual reward modes:
     - **Debate rewards** (training): Learn from peer comparisons
     - **Correctness rewards** (evaluation): Measure accuracy

4. **`tinker_cookbook/recipes/multi_agent_debate/math_debate_dataset.py`**
   - `MathDebateDataset`: RL dataset implementation
   - `MathDebateDatasetBuilder`: Unified dataset builder
   - Pre-configured builders:
     - `MathTrainingBuilder`
     - `GSM8KTrainingBuilder`
     - `PolarisTrainingBuilder`
     - `MATH500EvalBuilder`
     - `AIME2024EvalBuilder`
     - `AIME2025EvalBuilder`

#### Documentation
5. **`tinker_cookbook/recipes/multi_agent_debate/MATH_DEBATE_README.md`**
   - Comprehensive documentation
   - Usage examples
   - Configuration guide
   - Tips and best practices

6. **`data/README.md`**
   - Dataset format specification
   - Instructions for adding custom datasets

#### Example Scripts
7. **`tinker_cookbook/recipes/multi_agent_debate/examples/train_math_debate.py`**
   - Training examples on different datasets
   - Debate vs. correctness reward comparison
   - Multi-round debate configuration

8. **`tinker_cookbook/recipes/multi_agent_debate/examples/eval_math_debate.py`**
   - Evaluation on AIME and MATH-500
   - Single vs. multi-agent comparison
   - Comprehensive benchmark suite

#### Data
9. **`data/aime_2024_example.jsonl`**
   - Example AIME 2024 problems
   - Template for full dataset

10. **`data/aime_2025_example.jsonl`**
    - Example AIME 2025 problems
    - Template for full dataset

## Key Features

### 1. Dual Reward System

**Training (Debate Rewards)**
- Agents learn to produce solutions that peers rank highly
- Encourages collaborative problem-solving
- Reward = (wins - losses) from pairwise comparisons

**Evaluation (Correctness Rewards)**
- Measures mathematical accuracy
- Format penalty for missing \\boxed{}
- Direct verification via sympy or math_verify

### 2. Flexible Dataset Support

**Training Datasets (Self-Play)**
- MATH: 12k competition problems
- GSM8K: 7.5k grade school problems
- Polaris: 53k diverse problems

**Evaluation Datasets (Fixed Opponents)**
- MATH-500: Standard benchmark (500 problems)
- AIME 2024: Competition problems (30 problems)
- AIME 2025: Competition problems (30 problems)

**Custom Datasets**
- Load from JSONL files
- Easy integration via dataset registry

### 3. Math-Focused Prompts

Prompts redesigned to match `math_rl` style:
- Clear problem statement
- Step-by-step reasoning emphasis
- Mandatory \\boxed{} format
- Structured evaluation and comparison

### 4. Configurable Debate Structure

```python
num_agents=3         # Number of debating agents
max_rounds=3         # Rounds of debate (3×3 = 9 turns)
history_rounds=2     # Recent history shown
summarize_history=True  # Summarize old turns
```

## How It Works

### Training Flow

```
1. Load training dataset (MATH/GSM8K/Polaris)
2. For each batch:
   a. Sample problems
   b. Initialize multi-agent debate (self-play)
   c. Agents take turns proposing solutions
   d. Each agent evaluates and compares others
   e. Compute debate rewards from peer comparisons
   f. Update policy via RL (forward_backward/PPO/etc.)
3. Periodically evaluate on test set with correctness metrics
```

### Evaluation Flow

```
1. Load evaluation dataset (MATH-500/AIME)
2. Initialize fixed opponent policies
3. For each problem:
   a. Run multi-agent debate
   b. Policy being evaluated plays against fixed opponents
   c. Extract final answers (\\boxed{})
   d. Grade answers for correctness
   e. Compute accuracy metrics
4. Report results (accuracy, format compliance, etc.)
```

## Usage Examples

### Quick Start: Train on MATH

```python
import asyncio
from tinker_cookbook.recipes.multi_agent_debate.math_debate_dataset import MathTrainingBuilder
from tinker_cookbook.rl.train import train_rl

async def main():
    builder = MathTrainingBuilder(
        batch_size=4,
        num_train_datapoints=100,
        num_test_datapoints=20,
        num_agents=3,
        max_rounds=3,
        model_name="Qwen/Qwen3-8B-Instruct",
        renderer_name="qwen3",
        train_reward_mode="debate",
        eval_reward_mode="correctness",
    )

    await train_rl(
        dataset_builder=builder,
        base_model="Qwen/Qwen3-8B-Instruct",
        n_batches=25,
        lora_rank=64,
        lr=5e-5,
        loss_fn="forward_backward",
    )

asyncio.run(main())
```

### Evaluate on AIME 2024

```python
from tinker_cookbook.recipes.multi_agent_debate.math_debate_dataset import AIME2024EvalBuilder

async def main():
    builder = AIME2024EvalBuilder(
        batch_size=1,
        num_test_datapoints=30,
        num_agents=3,
        max_rounds=4,  # More rounds for hard problems
        model_name="path/to/trained/model",
        renderer_name="qwen3",
        log_full_transcript=True,
    )

    await train_rl(
        dataset_builder=builder,
        base_model="path/to/trained/model",
        n_batches=0,  # Eval only
        eval_every=1,
    )
```

## Configuration Guide

### For Training

```python
MathDebateDatasetBuilder(
    dataset_name="math",           # or "gsm8k", "polaris"
    batch_size=4,
    num_train_datapoints=100,
    num_test_datapoints=20,

    num_agents=3,                  # 3-5 agents recommended
    max_rounds=3,                  # 2-4 rounds for training
    history_rounds=2,              # Show last 2 rounds

    model_name="Qwen/Qwen3-8B-Instruct",
    renderer_name="qwen3",

    train_reward_mode="debate",    # Learn collaboration
    eval_reward_mode="correctness", # Measure accuracy

    log_full_transcript=True,      # Debug early on
)
```

### For Evaluation

```python
AIME2024EvalBuilder(
    batch_size=1,                  # One problem at a time
    num_test_datapoints=30,

    num_agents=3,
    max_rounds=4,                  # More rounds for hard problems
    history_rounds=3,
    summarize_history=True,        # Manage long contexts

    model_name="path/to/model",
    eval_reward_mode="correctness",
    log_full_transcript=True,      # Analyze failure modes
)
```

## Dataset Setup

### Using Built-in Datasets

No setup needed! Built-in datasets (MATH, GSM8K, Polaris, MATH-500) are loaded from HuggingFace automatically.

### Adding AIME Datasets

1. Place AIME problems in JSONL files:
   ```bash
   data/aime_2024.jsonl
   data/aime_2025.jsonl
   ```

2. Format (one problem per line):
   ```json
   {"problem": "Problem text here", "answer": "42", "id": "aime_2024_1"}
   ```

3. Use pre-configured builders:
   ```python
   builder = AIME2024EvalBuilder(...)
   ```

### Custom Datasets

1. Create JSONL file:
   ```bash
   data/my_dataset.jsonl
   ```

2. Load with generic loader:
   ```python
   from tinker_cookbook.recipes.multi_agent_debate.math_debate_datasets import load_jsonl_math_problems

   problems = load_jsonl_math_problems(
       "data/my_dataset.jsonl",
       problem_field="problem",
       answer_field="answer",
       dataset_name="my_dataset",
   )
   ```

3. Use with MathDebateDatasetBuilder:
   ```python
   builder = MathDebateDatasetBuilder(
       dataset_name="my_dataset",  # After registering in DATASET_LOADERS
       ...
   )
   ```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Math Debate Training Pipeline                 │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
         ┌──────▼──────┐            ┌──────▼──────┐
         │  Training   │            │ Evaluation  │
         │  Datasets   │            │  Datasets   │
         └──────┬──────┘            └──────┬──────┘
                │                           │
      ┌─────────┼─────────┐        ┌───────┼────────┐
      │         │         │        │       │        │
   MATH      GSM8K    Polaris   MATH-500  AIME    AIME
  (12k)     (7.5k)    (53k)      (500)   2024    2025
                │                           │
                └─────────────┬─────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ MathDebateDataset │
                    │     Builder       │
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
       ┌──────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
       │   Agent 0   │ │  Agent 1  │ │   Agent 2   │
       │  (Policy)   │ │ (Policy)  │ │  (Policy)   │
       └──────┬──────┘ └─────┬─────┘ └──────┬──────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                   ┌─────────▼──────────┐
                   │   Coordinator      │
                   │  (Turn Management) │
                   └─────────┬──────────┘
                             │
               ┌─────────────┼─────────────┐
               │             │             │
          Solution      Evaluation    Comparison
         (\\boxed{})    (Critique)    (Rankings)
               │             │             │
               └─────────────┼─────────────┘
                             │
                  ┌──────────▼──────────┐
                  │   Reward Computer   │
                  └──────────┬──────────┘
                             │
                  ┌──────────┼──────────┐
                  │          │          │
             Debate      Correctness   Mixed
            (Training)   (Evaluation)  (Both)
```

## Differences from Original `verifiable_env.py`

### Improvements

1. **Math-focused prompts**: Match `math_rl` style for better problem-solving
2. **Flexible dataset loading**: Support for both HuggingFace and JSONL
3. **Dual reward modes**: Debate (training) vs. Correctness (evaluation)
4. **Pre-configured builders**: Easy setup for common use cases
5. **Better documentation**: Comprehensive guide and examples
6. **Example scripts**: Ready-to-run training and evaluation

### Backward Compatibility

The original `verifiable_env.py` remains unchanged. The new implementation is in separate files:
- `math_debate_*.py` - New implementation
- `verifiable_env.py` - Original implementation (still works)

## Next Steps

### For Training

1. **Set up datasets**: Ensure AIME datasets are in `data/` if you want to use them for evaluation
2. **Configure training**: Edit `examples/train_math_debate.py` to match your setup
3. **Run training**: `python examples/train_math_debate.py`
4. **Monitor progress**: Check logs and transcripts for debate quality
5. **Tune hyperparameters**: Adjust `num_agents`, `max_rounds`, learning rate, etc.

### For Evaluation

1. **Prepare test sets**: Place AIME JSONL files in `data/`
2. **Configure evaluation**: Edit `examples/eval_math_debate.py`
3. **Run evaluation**: `python examples/eval_math_debate.py --checkpoint path/to/model --dataset aime_2024`
4. **Analyze results**: Review transcripts to understand failure modes
5. **Compare baselines**: Run single-agent and multi-agent evaluations

### For Research

1. **Ablation studies**: Compare debate vs. correctness rewards
2. **Scaling studies**: Test with different numbers of agents and rounds
3. **Dataset transfer**: Train on one dataset, evaluate on others
4. **Prompt engineering**: Experiment with different prompt formats
5. **Hybrid rewards**: Combine debate and correctness rewards

## Performance Tips

### Training
- Start with small datasets (100-200 problems) to iterate quickly
- Use `log_full_transcript=True` early to debug prompt issues
- Begin with `num_agents=3` and `max_rounds=3` (9 total turns)
- Monitor format compliance - increase `format_coef` if needed

### Evaluation
- Use `batch_size=1` to see individual problem results
- Enable full transcript logging for analysis
- Increase `max_rounds` for harder problems (AIME, olympiad)
- Compare single-agent vs. multi-agent performance

### Resource Management
- Enable `summarize_history=True` for long debates (>15 turns)
- Reduce `history_rounds` if context gets too long
- Use smaller models for summarization (Qwen3-4B)
- Batch size affects memory - reduce if OOM errors occur

## Troubleshooting

### Common Issues

**Format errors (missing \\boxed{})**
- Increase `format_coef` penalty (try 0.2 or 0.5)
- Check prompts emphasize \\boxed{} format
- Add few-shot examples showing correct format

**Low debate quality**
- Increase `num_agents` for more diverse perspectives
- Adjust `max_rounds` for more refinement
- Review transcripts to see what agents are doing

**Context length issues**
- Enable `summarize_history=True`
- Reduce `history_rounds` (try 1 or 2)
- Shorten `max_rounds` if context fills up

**Poor evaluation accuracy**
- Train longer (more batches)
- Use larger base model
- Increase `max_rounds` during evaluation
- Check if grading is working correctly

## Citation

If you use this integration in your research, please cite:

```bibtex
@software{math_debate_integration,
  title={Math Multi-Agent Debate Integration for Tinker Cookbook},
  author={Tinker Cookbook Contributors},
  year={2025},
  url={https://github.com/anthropics/tinker-cookbook}
}
```

## Future Enhancements

Potential improvements for future versions:

1. **Dynamic debate length**: Terminate debates early if agents converge
2. **Confidence-based weighting**: Weight comparisons by agent confidence
3. **Expert iteration**: Use best solutions as training data
4. **Multi-task learning**: Train on mixed datasets simultaneously
5. **Debate visualization**: Tools to visualize debate dynamics
6. **Advanced grading**: Support for proof verification, symbolic math
7. **Meta-learning**: Learn debate strategies across problems

## Contact

For questions, issues, or contributions:
- GitHub Issues: https://github.com/anthropics/tinker-cookbook/issues
- Documentation: See `MATH_DEBATE_README.md` for detailed guide
