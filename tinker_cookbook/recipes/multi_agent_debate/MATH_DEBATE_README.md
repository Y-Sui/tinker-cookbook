# Math Multi-Agent Debate Environment

This module integrates multi-agent debate with verifiable math problem-solving, combining collaborative reasoning with mathematical correctness evaluation.

## Overview

The math debate environment enables multiple agents to:
1. **Collaborate** on solving challenging math problems
2. **Critique** each other's solutions and reasoning
3. **Compare** different solution approaches
4. **Learn** from both debate rewards (training) and correctness metrics (evaluation)

## Key Features

### Training Mode
- **Self-play**: All agents are the policy being trained
- **Debate rewards**: Agents receive rewards based on peer evaluations and comparisons
- **Datasets**: MATH, GSM8K, Polaris (via HuggingFace)
- **Goal**: Learn to collaborate and solve problems effectively

### Evaluation Mode
- **Fixed opponents**: Use pre-trained models as opponents
- **Correctness rewards**: Agents rewarded for mathematical accuracy
- **Datasets**: MATH-500, AIME 2024, AIME 2025
- **Goal**: Measure problem-solving accuracy on held-out test sets

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Math Debate Environment                 │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐           │
│  │  Agent 0  │  │  Agent 1  │  │  Agent 2  │           │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘           │
│        │              │              │                   │
│        └──────────────┼──────────────┘                   │
│                       │                                  │
│              ┌────────▼────────┐                         │
│              │  Coordinator    │                         │
│              │  (Turn-taking)  │                         │
│              └────────┬────────┘                         │
│                       │                                  │
│         ┌─────────────┼─────────────┐                    │
│         │             │             │                    │
│    ┌────▼────┐   ┌───▼────┐   ┌───▼────┐               │
│    │Solution │   │ Eval   │   │Compare │               │
│    │(\\boxed) │   │(Crit.) │   │(Rank)  │               │
│    └─────────┘   └────────┘   └────────┘               │
│                                                           │
│         Training: Debate Rewards ← Comparisons           │
│         Eval: Correctness ← Ground Truth                 │
└─────────────────────────────────────────────────────────┘
```

## Files

- `math_debate_prompts.py`: Math-focused system and user prompts
- `math_debate_datasets.py`: Dataset loaders for training and evaluation
- `math_debate_env.py`: Environment implementation with both reward modes
- `math_debate_dataset.py`: Unified dataset builder with pre-configured builders
- `MATH_DEBATE_README.md`: This file

## Quick Start

### 1. Training on MATH with Debate Rewards

```python
import asyncio
from tinker_cookbook.recipes.multi_agent_debate.math_debate_dataset import MathTrainingBuilder
from tinker_cookbook.rl.train import train_rl

async def main():
    # Configure dataset
    dataset_builder = MathTrainingBuilder(
        batch_size=4,
        num_train_datapoints=100,
        num_test_datapoints=20,
        num_agents=3,
        max_rounds=3,
        model_name="Qwen/Qwen3-8B-Instruct",
        renderer_name="qwen3",
        train_reward_mode="debate",  # Learn from peer evaluations
        eval_reward_mode="correctness",  # Test on accuracy
        log_full_transcript=True,
    )

    # Train
    await train_rl(
        dataset_builder=dataset_builder,
        base_model="Qwen/Qwen3-8B-Instruct",
        n_batches=25,
        lora_rank=64,
        lr=5e-5,
        loss_fn="forward_backward",
    )

asyncio.run(main())
```

### 2. Evaluating on AIME 2024

```python
import asyncio
from tinker_cookbook.recipes.multi_agent_debate.math_debate_dataset import AIME2024EvalBuilder
from tinker_cookbook.rl.train import train_rl

async def main():
    dataset_builder = AIME2024EvalBuilder(
        batch_size=1,
        num_test_datapoints=30,  # All AIME 2024 problems
        num_agents=3,
        max_rounds=3,
        model_name="Qwen/Qwen3-8B-Instruct",
        renderer_name="qwen3",
        eval_reward_mode="correctness",
        log_full_transcript=True,
    )

    # Run evaluation only (n_batches=0 for training)
    await train_rl(
        dataset_builder=dataset_builder,
        base_model="path/to/trained/model",
        n_batches=0,  # Eval only
        eval_every=1,
    )

asyncio.run(main())
```

### 3. Mixed Training (Multiple Datasets)

```python
# You can train on multiple datasets by using the generic builder

from tinker_cookbook.recipes.multi_agent_debate.math_debate_dataset import MathDebateDatasetBuilder

# Train on GSM8K
gsm8k_builder = MathDebateDatasetBuilder(
    dataset_name="gsm8k",
    batch_size=8,
    num_train_datapoints=200,
    num_test_datapoints=50,
    num_agents=3,
    max_rounds=3,
    model_name="Qwen/Qwen3-8B-Instruct",
    renderer_name="qwen3",
    train_reward_mode="debate",
)

# Evaluate on MATH-500
eval_builder = MATH500EvalBuilder(
    num_test_datapoints=500,
    num_agents=3,
    max_rounds=3,
    model_name="Qwen/Qwen3-8B-Instruct",
    renderer_name="qwen3",
)
```

## Configuration Options

### Dataset Builder Parameters

#### Core Settings
- `dataset_name`: Dataset to use (math, gsm8k, polaris, math_500, aime_2024, aime_2025)
- `batch_size`: Number of problem groups per batch
- `num_train_datapoints`: Number of training problems
- `num_test_datapoints`: Number of evaluation problems

#### Debate Settings
- `num_agents`: Number of agents in debate (default: 3)
- `max_rounds`: Number of debate rounds (default: 3, so 9 total turns for 3 agents)
- `history_rounds`: Number of recent rounds to show in context (default: 2)
- `summarize_history`: Whether to summarize old history to save context (default: False)
- `summarize_model`: Model for summarization (default: Qwen3-4B)

#### Model Settings
- `model_name`: Base model for agents
- `renderer_name`: Renderer matching model family (llama3, qwen3, etc.)
- `opponent_model_name`: Model for opponents in eval (default: same as model_name)

#### Reward Settings
- `train_reward_mode`: "debate" (learn from comparisons) or "correctness" (learn from accuracy)
- `eval_reward_mode`: "debate" or "correctness" (typically "correctness" for eval)
- `format_coef`: Penalty for missing \\boxed{} format (default: 0.1)
- `grader`: "sympy" or "math_verify" for answer verification (default: "sympy")
- `grade_timeout`: Timeout for grading in seconds (default: 1.0)

#### Data Settings
- `max_problems`: Limit on problems to load (default: None = all)
- `shuffle_seed`: Seed for problem shuffling (default: 42)

## Prompt Format

### System Prompt
Agents receive a math-focused system prompt that emphasizes:
- Showing complete step-by-step reasoning
- Including final answer in `\\boxed{}` format
- Reviewing other agents' solutions for errors
- Comparing solution approaches constructively

### User Prompt Structure

**First Turn:**
```
Problem: [problem statement]

This is Turn 1 of 9 (Cycle 1/3).

You are the first agent to respond. Provide your solution to this problem.
...
```

**Later Turns:**
```
Problem: [problem statement]

Previous Solutions:
--- Turn 1 (Agent 0) ---
[Agent 0's solution]

--- Turn 2 (Agent 1) ---
[Agent 1's solution]
...

This is Turn 3 of 9 (Cycle 1/3).

Previous agents have proposed solutions. Your tasks:
1. In <solution>: Provide your best solution (must include \boxed{} answer)
2. In <evaluation>: Review recent solutions and identify errors
3. In <comparison>: Compare OTHER agents' solutions and rank quality
```

### Agent Output Format

```xml
<solution>
Let's solve this step by step:

Step 1: ...
Step 2: ...
...

Therefore, the answer is \boxed{42}.
</solution>

<evaluation>
Agent 0's solution correctly identifies... However, there's an error in step 3...
Agent 1's approach using... is valid, but...
</evaluation>

<comparison>
Agent 0 > Agent 1  (Agent 0's solution is more complete and has correct final answer)
</comparison>
```

## Reward Functions

### Debate Reward (Training)
Based on pairwise comparisons between agents:
- Win: +1 point when Agent A > Agent B
- Loss: -1 point when Agent A < Agent B
- Tie: 0 points when Agent A = Agent B
- Final reward: (wins - losses) / total_comparisons

Encourages agents to produce solutions that peers find superior.

### Correctness Reward (Evaluation)
Based on mathematical accuracy:
- Correct answer with proper format: +1.0
- Correct format (has \\boxed{}), wrong answer: -0.1 * format_coef
- Missing format: -1.0 * format_coef

Directly measures problem-solving ability.

## Training Datasets

### MATH (Hendrycks)
- ~12,000 competition math problems
- Topics: algebra, geometry, number theory, counting, probability
- Difficulty: AMC 10/12 to AIME level
- Loaded from: `EleutherAI/hendrycks_math`

### GSM8K
- ~7,500 grade school math word problems
- Focus: arithmetic reasoning, multi-step problems
- Loaded from: `openai/gsm8k`

### Polaris
- ~53,000 math problems
- Diverse difficulty and topics
- Loaded from: `POLARIS-Project/Polaris-Dataset-53K`

## Evaluation Datasets

### MATH-500
- 500 held-out problems from Hendrycks MATH
- Standard benchmark for math reasoning
- Loaded from: `HuggingFaceH4/MATH-500`

### AIME 2024 / 2025
- American Invitational Mathematics Examination
- 30 problems per year
- High difficulty (top 5% of high school students)
- Loaded from: `data/aime_2024.jsonl`, `data/aime_2025.jsonl`

## Adding Custom Datasets

### From JSONL Files

```python
from tinker_cookbook.recipes.multi_agent_debate.math_debate_datasets import load_jsonl_math_problems

problems = load_jsonl_math_problems(
    file_path="data/my_dataset.jsonl",
    problem_field="problem",
    answer_field="answer",
    dataset_name="my_dataset",
    max_problems=100,
)
```

JSONL format:
```json
{"problem": "What is 2+2?", "answer": "4"}
{"problem": "Solve x^2 = 16", "answer": "4"}
```

### Register New Dataset

```python
# In math_debate_datasets.py
def load_my_dataset() -> list[MathProblem]:
    # Your loading logic
    return problems

# Register it
DATASET_LOADERS["my_dataset"] = load_my_dataset
```

## Advanced Usage

### Custom Reward Functions

You can extend `MathDebateEnvGroupBuilder` to implement custom reward functions:

```python
from tinker_cookbook.recipes.multi_agent_debate.math_debate_env import MathDebateEnvGroupBuilder

class CustomRewardBuilder(MathDebateEnvGroupBuilder):
    async def compute_group_rewards(self, trajectory_group, env_group):
        # Your custom reward logic
        # Can combine debate rewards, correctness, and other signals
        return rewards_and_metrics
```

### Hybrid Reward Training

```python
# Use debate rewards during training but periodically evaluate with correctness
dataset_builder = MathDebateDatasetBuilder(
    dataset_name="math",
    train_reward_mode="debate",  # Learn collaboration
    eval_reward_mode="correctness",  # Measure accuracy
    ...
)
```

### Adjusting History Context

For longer debates or complex problems:
```python
dataset_builder = MathDebateDatasetBuilder(
    history_rounds=3,  # Show more history
    summarize_history=True,  # Summarize old turns to save context
    max_rounds=5,  # More rounds for complex problems
    ...
)
```

## Tips and Best Practices

### For Training
1. Start with `num_agents=3` and `max_rounds=3` (9 total turns)
2. Use `train_reward_mode="debate"` to encourage collaboration
3. Log transcripts (`log_full_transcript=True`) to debug early
4. Use smaller batches (4-8) with larger group_size for better exploration

### For Evaluation
1. Use `eval_reward_mode="correctness"` for accurate metrics
2. Set `batch_size=1` for evaluation datasets to see individual problem results
3. Enable full transcript logging to analyze failure modes
4. Consider using a stronger `opponent_model_name` for harder evaluations

### Performance Tuning
1. If context is too long, enable `summarize_history=True`
2. Reduce `history_rounds` if agents get distracted by old solutions
3. Increase `max_rounds` for very hard problems (AIME, olympiad)
4. Adjust `format_coef` if models struggle with \\boxed{} formatting

### Common Issues

**Issue**: Agents repeat each other's solutions
- **Solution**: Increase `num_agents` or `max_rounds` for more diversity

**Issue**: Format errors (missing \\boxed{})
- **Solution**: Increase `format_coef` penalty or add few-shot examples

**Issue**: Low accuracy on evaluation
- **Solution**: Train longer, use larger model, or increase `max_rounds`

**Issue**: Out of context during long debates
- **Solution**: Enable `summarize_history=True` and reduce `history_rounds`

## Examples

See `examples/` directory for complete training and evaluation scripts:
- `train_math_debate.py`: Full training pipeline on MATH
- `eval_aime.py`: Evaluation on AIME datasets
- `multi_dataset_training.py`: Training on multiple datasets

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{tinker_cookbook_math_debate,
  title={Math Multi-Agent Debate Environment},
  author={Tinker Cookbook Contributors},
  year={2025},
  url={https://github.com/anthropics/tinker-cookbook}
}
```

## Contributing

To contribute improvements:
1. Follow the existing code style
2. Add tests for new dataset loaders
3. Document any new configuration options
4. Provide example usage in docstrings
