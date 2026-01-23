# CANT Evaluation with Inspect AI

Generic CANT protocol solver for evaluating multi-agent reasoning on any Inspect AI benchmark.

## Prerequisites

```bash
pip install inspect_evals
```

## Quick Start

### Inline Evaluation During Training

```bash
python -m tinker_cookbook.recipes.cant.train \
    model_name=Qwen/Qwen3-8B \
    train_datasets=aime2024 \
    eval_every=50 \
    eval_tasks=inspect_evals/gsm8k,inspect_evals/math \
    eval_use_cant_protocol=true \
    eval_num_agents=4 \
    eval_limit=100
```

### Offline Evaluation on Checkpoint

```bash
# CANT Protocol
python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
    model_path=tinker://user/run/step-500 \
    task_names=inspect_evals/gsm8k,inspect_evals/math \
    use_cant_protocol=true \
    num_agents=4

# Baseline (for comparison)
python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
    model_path=tinker://user/run/step-500 \
    task_names=inspect_evals/gsm8k,inspect_evals/math \
    use_cant_protocol=false
```

## How It Works

The CANT protocol solver injects 4-round multi-agent discussion into any Inspect AI task:

1. **Round 0**: Each agent proposes initial solution
2. **Round 1**: Agents rank and critique solutions
3. **Round 2**: Agents revise based on feedback
4. **Round 3**: Agents provide final rankings

The **highest-ranked solution** is evaluated using the task's original scorer.

**Toggle modes with one flag**: `eval_use_cant_protocol=true/false`

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eval_tasks` | `None` | Comma-separated task names (e.g., `inspect_evals/gsm8k,inspect_evals/math`) |
| `eval_use_cant_protocol` | `True` | CANT protocol vs baseline |
| `eval_num_agents` | `None` | Number of agents (defaults to `num_agents`) |
| `eval_every` | `0` | Evaluate every N steps (0 = disabled) |
| `eval_limit` | `None` | Max samples per task (None = all) |
| `eval_temperature` | `0.0` | Sampling temperature |
| `eval_max_tokens` | `2048` | Max tokens per generation |

### Offline Evaluation Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `model_path` | Checkpoint path | `tinker://user/run/step-500` |
| `task_names` | Comma-separated tasks | `inspect_evals/gsm8k,inspect_evals/math` |
| `use_cant_protocol` | CANT vs baseline | `true` or `false` |
| `num_agents` | Number of agents | `4` |

## Supported Tasks

**Any Inspect AI task works!** Examples:

- **Math**: `inspect_evals/gsm8k`, `inspect_evals/math`
- **Knowledge**: `inspect_evals/mmlu`
- **Instructions**: `inspect_evals/ifeval`
- **Reasoning**: `inspect_evals/arc`, `inspect_evals/hellaswag`
- **Code**: `inspect_evals/humaneval`
- **Custom**: `your_module:your_task`

**Find tasks**: `inspect list tasks` or visit https://inspect.aisi.org.uk/evals.html

## Common Use Cases

### Compare CANT vs Baseline

```bash
# CANT protocol
python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
    model_path=tinker://user/run/step-500 \
    task_names=inspect_evals/gsm8k \
    use_cant_protocol=true \
    num_agents=4

# Baseline
python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
    model_path=tinker://user/run/step-500 \
    task_names=inspect_evals/gsm8k \
    use_cant_protocol=false
```

### Ablate Number of Agents

```bash
for num_agents in 2 4 8; do
    python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
        model_path=tinker://user/run/step-500 \
        task_names=inspect_evals/gsm8k \
        use_cant_protocol=true \
        num_agents=${num_agents}
done
```

### Track Training Progress

```bash
for step in 100 200 300 400 500; do
    python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
        model_path=tinker://user/run/step-${step} \
        task_names=inspect_evals/gsm8k \
        use_cant_protocol=true \
        num_agents=4
done
```

## Troubleshooting

**Task not found**: Install `inspect_evals` package
```bash
pip install inspect_evals
```

**Too slow**: Reduce sample count
```bash
eval_limit=50
```

**Out of memory**: Reduce agents or tokens
```bash
eval_num_agents=2
eval_max_tokens=1024
```

## Architecture

### Solver Injection Pattern

```python
# CANT protocol mode: inject solver via eval_async
if use_cant_protocol:
    eval_params["solver"] = cant_protocol_solver(num_agents)

results = await eval_async(**eval_params)
```

### Selection Strategy

```python
# Convert rankings to scores (lower rank = higher score)
score = num_agents - rank + 1

# Select agent with highest cumulative score
best_agent = max(agent_scores)
```

## Benefits

✅ Works with ANY Inspect AI task  
✅ Single flag controls CANT vs baseline  
✅ Preserves task's original scorer  
✅ No code changes for new benchmarks  
✅ Clean, maintainable architecture  

## References

- [Inspect AI Docs](https://inspect.aisi.org.uk/)
- [Task Listing](https://inspect.aisi.org.uk/evals.html)
- [Tinker Evals Guide](../../../docs/evals.mdx)
