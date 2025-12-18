# Multi-Agent Conversation Preference RL

This recipe implements a multi-agent conversation system with self-comparison rewards for reinforcement learning.

## Overview

This implementation features:

- **Multi-agent coordination**: N agents (default 3) take turns in a conversation via a Coordinator pattern
- **Structured output**: Each agent produces reflection, evaluation, and solution using pydantic models
- **Self-comparison rewards**: The policy model judges its own outputs (LLM-as-judge) without external models
- **Tournament-style comparison**: Pairwise comparisons within each conversation determine rewards

## Architecture

### Episode Flow

1. Create N agents sharing one `MultiAgentCoordinator`
2. Agents take turns (0→1→2→0...) generating `AgentResponse` (reflection, evaluation, solution)
3. Coordinator manages turn-taking via `asyncio.Condition` (like TwoPlayerCoordinator from text_arena)
4. After max_rounds: extract solutions, run pairwise tournament
5. `PolicySelfComparisonJudge` compares pairs: "Which solution is better?"
6. Win-loss accumulation (from preference_envs.py) produces per-agent rewards
7. Standard RL loop trains on these rewards

### Key Components

- **`AgentResponse`**: Pydantic model with `reflection`, `evaluation`, `solution` fields
- **`MultiAgentCoordinator`**: Synchronizes turn-taking for N agents using asyncio
- **`MultiAgentConversationEnv`**: Individual agent's environment (multiple share one coordinator)
- **`PolicySelfComparisonJudge`**: Uses policy to compare solutions (LLM-as-judge)
- **`MultiAgentConversationEnvGroupBuilder`**: Creates agent groups and computes tournament rewards

## Usage

### Basic Training with Custom Prompts

```bash
python -m tinker_cookbook.recipes.conversation_preference_rl.train \
    model_name=meta-llama/Llama-3.2-1B \
    num_agents=3 \
    max_rounds=5 \
    batch_size=12 \
    train_prompts_path=/path/to/prompts.jsonl \
    log_path=/tmp/conv-pref-rl/run1
```

### Training with HuggingFace Dataset
# meta-llama/Llama-3.2-1B
```bash
python -m tinker_cookbook.recipes.conversation_preference_rl.train \
    model_name=Qwen/Qwen3-8B\
    hf_dataset_name=openai/gsm8k \
    hf_dataset_name_config=main \
    hf_prompt_column=question \
    num_agents=3 \
    max_rounds=3 \
    batch_size=24 \
    max_train_samples=1000 \
    learning_rate=1e-5 \
    wandb_project=conv-pref-rl
```

### Training with Direct Prompt List

```bash
python -m tinker_cookbook.recipes.conversation_preference_rl.train \
    model_name=Qwen/Qwen2.5-7B \
    'train_prompts=["What is 2+2?","Explain gravity.","How do computers work?"]' \
    num_agents=4 \
    max_rounds=4 \
    batch_size=16
```

## Configuration Parameters

### Model Settings

- `model_name`: Base model to use (e.g., `meta-llama/Llama-3.2-1B`)
- `renderer_name`: Tokenizer renderer (auto-detected if None)

### Multi-Agent Parameters

- `num_agents`: Number of agents in each conversation (default: 3)
- `max_rounds`: Maximum conversation rounds (default: 5)

### Training Parameters

- `batch_size`: Training batch size (must be divisible by num_agents, default: 128)
- `learning_rate`: Learning rate (auto-computed if None using `hyperparam_utils.get_lr`)
- `max_tokens`: Maximum tokens per generation (default: 512)
- `num_substeps`: Number of substeps per batch (default: 1)
- `max_training_datapoints`: Total training datapoints (default: 131072)

### Dataset Sources

Pick one or combine multiple:

- `train_prompts`: Direct list of prompts
- `train_prompts_path`: Path to JSONL file with `{"query": "..."}` entries
- `test_prompts_path`: Path to JSONL file for test set
- `hf_dataset_name`: HuggingFace dataset name
- `hf_prompt_column`: Column name for prompts (default: "query")
- `max_train_samples`: Limit training samples
- `max_test_samples`: Limit test samples (default: 100)

### Logging

- `log_path`: Directory for logs and checkpoints
- `eval_every`: Evaluation frequency (default: 5)
- `save_every`: Checkpoint save frequency (default: 20)
- `wandb_project`: Weights & Biases project name
- `wandb_name`: W&B run name
- `behavior_if_exists`: What to do if log_path exists ("ask", "delete", "resume", "raise")

## Data Format

### Prompt Files (JSONL)

```jsonl
{"query": "What is the capital of France?"}
{"query": "Explain quantum entanglement."}
{"query": "How do I make a cake?"}
```

Alternative keys: `"prompt"` or `"text"` instead of `"query"`

### Agent Personas

The default implementation has 3 agent roles:

1. **Agent 0 - The Innovator**: Creative, novel solutions
2. **Agent 1 - The Critic**: Rigorous, analytical solutions
3. **Agent 2 - The Synthesizer**: Balanced, integrative solutions

You can customize these by editing `AGENT_SYSTEM_PROMPTS` in `env.py`.

## How It Works

### Turn-Taking Mechanism

The `MultiAgentCoordinator` uses `asyncio.Condition` to synchronize agents:

```
Initialize: current_turn=0, current_round=1, episode_done=False

wait_for_turn(agent_id):
  1. Acquire condition lock
  2. Wait until: current_turn == agent_id OR episode_done
  3. Release lock

submit_response(agent_id, response):
  1. Acquire condition lock
  2. Verify current_turn == agent_id
  3. Append response to conversation_history
  4. Advance: current_turn = (current_turn + 1) % num_agents
  5. If current_turn wrapped to 0: current_round += 1
  6. If current_round > max_rounds: episode_done = True
  7. Notify all waiting agents
  8. Release lock
```

### Tournament Reward Computation

After an episode completes:

```python
# Extract final solutions from each agent
solutions = [agent_0_solution, agent_1_solution, agent_2_solution]

# Create pairwise comparisons (ALL_PAIRS_BOTH_WAYS)
comparison_pairs = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]

# For each pair, judge compares and returns score
for (i, j) in comparison_pairs:
    score = judge.compare(query, solutions[i], solutions[j])
    # score ∈ [-1, 1]: -1 means A wins, 0 means tie, +1 means B wins
    win_loss[j] += score
    win_loss[i] -= score

# Normalize by matchup count
rewards = [win_loss[i] / matchup_count[i] for i in range(num_agents)]
```

This creates a **zero-sum tournament** where winning agents get positive rewards and losing agents get negative rewards.

### Structured Output Parsing

Agents produce JSON output with multiple fallback parsing strategies:

1. **Direct JSON parse**: `{"reflection": "...", "evaluation": "...", "solution": "..."}`
2. **Markdown code block**: ` ```json {...} ``` `
3. **Plain code block**: ` ``` {...} ``` `
4. **XML-style**: `<reflection>...</reflection>...`
5. **Unstructured fallback**: Use entire text as solution

## Metrics Tracked

- `episode_length`, `num_turns`: Episode statistics
- `reward/mean`, `reward/std`, `reward/max`, `reward/min`: Reward distribution
- `win_minus_loss/mean`, `matchup_count`: Tournament statistics
- `agent_{id}/reward`, `agent_{id}/win_rate`: Per-agent performance
- `judge/preference_A_rate`, `judge/preference_B_rate`, `judge/tie_rate`: Judge behavior

## Hyperparameter Guidance

### Learning Rate

The script auto-computes learning rate using `hyperparam_utils.get_lr(model_name)`. For LoRA fine-tuning:

- Llama-3.2-1B: ~3e-4
- Llama-3.1-8B: ~1e-4
- Qwen2.5-7B: ~1e-4

You can override with `learning_rate=1e-5`.

### Batch Size

Must be divisible by `num_agents`. Recommended:

- 3 agents: batch_size ∈ {12, 24, 48, 96, 192}
- 4 agents: batch_size ∈ {16, 32, 64, 128}

Larger batches = more stable gradients but slower iteration.

### Number of Rounds

- **3-5 rounds**: Quick convergence, less computation
- **5-10 rounds**: More deliberation, richer conversation
- **>10 rounds**: Diminishing returns, risk of repetition

### Number of Agents

- **2 agents**: Simple debate (one proposes, one critiques)
- **3 agents**: Balanced (propose, critique, synthesize)
- **4+ agents**: More diverse perspectives but more computation

## Advanced Customization

### Custom Agent Prompts

Edit `AGENT_SYSTEM_PROMPTS` in `env.py`:

```python
AGENT_SYSTEM_PROMPTS = {
    0: "You are a domain expert. Provide technical solutions...",
    1: "You are a skeptic. Challenge assumptions...",
    2: "You are a pragmatist. Focus on practical implementation...",
}
```

### Custom Judge Prompt

Edit `JUDGE_SYSTEM_PROMPT` in `env.py`:

```python
JUDGE_SYSTEM_PROMPT = """You are evaluating solutions based on:
1. Technical accuracy
2. Clarity
3. Completeness
..."""
```

### Tournament Pattern

Change comparison pattern in `MultiAgentConversationEnvGroupBuilder`:

```python
# env.py line 458
tournament_pattern: TournamentPattern = TournamentPattern.ALL_PAIRS_ONE_WAY
```

- `ALL_PAIRS_BOTH_WAYS`: Every pair compared in both directions (more comparisons)
- `ALL_PAIRS_ONE_WAY`: Each pair compared once (faster)

## Troubleshooting

### Error: "batch_size must be divisible by num_agents"

Solution: Choose batch_size as a multiple of num_agents (e.g., for 3 agents: 12, 24, 48, ...)

### Error: "Must provide either train_prompts, train_prompts_path, or hf_dataset_name"

Solution: Provide at least one data source:

```bash
train_prompts_path=/path/to/data.jsonl
# OR
hf_dataset_name=openai/gsm8k
# OR
'train_prompts=["prompt1", "prompt2"]'
```

### Parsing Failures

Check logs for `parsing_failures` metric. If high:

1. Verify model is producing JSON format
2. Try adjusting prompts to emphasize JSON structure
3. Fallback parsers will handle most cases

### Reward Variance Too High

If rewards are too noisy:

1. Increase `num_agents` for more comparisons per episode
2. Use `ALL_PAIRS_BOTH_WAYS` for more pairwise matchups
3. Monitor `judge/tie_rate` - high tie rate may indicate weak differentiation

### Judge Always Returns Tie

If judge isn't discriminating:

1. Make judge prompt more specific about evaluation criteria
2. Try larger model (8B vs 1B)
3. Add few-shot examples to judge prompt

## References

- **Coordinator pattern**: `tinker_cookbook/recipes/multiplayer_rl/text_arena/env.py`
- **Tournament rewards**: `tinker_cookbook/rl/preference_envs.py`
- **Pydantic models**: `tinker_cookbook/preference/prompted_preference.py`

## License

Same as tinker-cookbook.
