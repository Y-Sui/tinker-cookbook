# CANT Evaluation Quick Start Guide

This guide shows you how to quickly start using the generic CANT protocol solver with Inspect AI for evaluation.

## Prerequisites

**Install the `inspect_evals` package:**

```bash
pip install inspect_evals
```

This package contains the standard evaluation tasks (GSM8K, MATH, MMLU, IFEval, etc.).

## 🚀 Quick Start (5 minutes)

### Step 1: Train with Inline Evaluation

Add evaluation to your training command:

```bash
python -m tinker_cookbook.recipes.cant.train \
    model_name=Qwen/Qwen3-8B \
    train_datasets=aime2024 \
    batch_size=8 \
    num_train_datapoints=100 \
    eval_every=20 \
    eval_tasks=inspect_evals/gsm8k \
    eval_use_cant_protocol=true \
    eval_num_agents=4 \
    eval_limit=50
```

**What happens:** Every 20 training steps, the model will be evaluated on 50 GSM8K problems using the CANT 4-round discussion protocol.

### Step 2: Evaluate a Checkpoint Offline

After training, evaluate a saved checkpoint:

```bash
python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
    model_path=tinker://your-user/your-run-id/step-100 \
    model_name=Qwen/Qwen3-8B \
    renderer_name=qwen3 \
    task_names=inspect_evals/gsm8k,inspect_evals/math \
    use_cant_protocol=true \
    num_agents=4
```


python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
    model_name=Qwen/Qwen3-8B \
    renderer_name=qwen3 \
    task_names=inspect_evals/aime2024 \
    use_cant_protocol=true \
    num_agents=4 \
    max_tokens=20000 \
    temperature=1.0 \
    limit=5

python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
    model_name=Qwen/Qwen3-8B \
    renderer_name=qwen3 \
    task_names=inspect_evals/aime2024 \
    use_cant_protocol=false \
    max_tokens=8096 \
    temperature=1.0 \


python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
    model_name=Qwen/Qwen3-8B \
    renderer_name=qwen3_disable_thinking \
    task_names=inspect_evals/aime2024 \
    use_cant_protocol=false \
    max_tokens=8096 \
    temperature=1.0

    

**What happens:** Evaluates your checkpoint on GSM8K and MATH using CANT protocol with the tasks' native scorers.

### Step 3: Compare Against Baseline

Run the same evaluation in baseline mode (single-turn):

```bash
python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
    model_path=tinker://your-user/your-run-id/step-100 \
    model_name=Qwen/Qwen3-8B \
    renderer_name=qwen3 \
    task_names=inspect_evals/gsm8k,inspect_evals/math \
    use_cant_protocol=false
```

**What happens:** Evaluates the same checkpoint using standard single-turn prompting. Compare the results to see if CANT protocol improves performance!

## 📊 Example Output

```
================================================================================
CANT Evaluation Results (CANT Protocol)
================================================================================
  gsm8k/accuracy: 0.7850
  math/accuracy: 0.4200
================================================================================
```

vs.

```
================================================================================
CANT Evaluation Results (Baseline)
================================================================================
  gsm8k/accuracy: 0.7200
  math/accuracy: 0.3800
================================================================================
```

In this example, CANT protocol improved accuracy by ~6% on GSM8K and ~4% on MATH!

## 🎯 Common Use Cases

### Use Case 1: Quick Evaluation During Training

**Goal:** Track progress on key benchmarks without slowing down training too much.

```bash
eval_every=50                          # Evaluate every 50 steps
eval_tasks=inspect_evals/gsm8k        # Just one fast benchmark
eval_limit=100                         # Only 100 samples (faster)
```

### Use Case 2: Comprehensive Final Evaluation

**Goal:** Thorough evaluation of final checkpoint on multiple benchmarks.

```bash
python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
    model_path=tinker://user/run/final \
    task_names=inspect_evals/gsm8k,inspect_evals/math,inspect_evals/mmlu \
    use_cant_protocol=true \
    num_agents=4
    # No limit = evaluate on all samples
```

### Use Case 3: Ablation Study on Number of Agents

**Goal:** See how performance changes with different numbers of agents.

```bash
# 2 agents
python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
    model_path=tinker://user/run/step-500 \
    task_names=inspect_evals/gsm8k \
    use_cant_protocol=true \
    num_agents=2

# 4 agents
python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
    model_path=tinker://user/run/step-500 \
    task_names=inspect_evals/gsm8k \
    use_cant_protocol=true \
    num_agents=4

# 8 agents
python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
    model_path=tinker://user/run/step-500 \
    task_names=inspect_evals/gsm8k \
    use_cant_protocol=true \
    num_agents=8
```

### Use Case 4: Compare Multiple Checkpoints

**Goal:** Track improvement across training.

```bash
for step in 100 200 300 400 500; do
    echo "Evaluating step ${step}..."
    python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
        model_path=tinker://user/run/step-${step} \
        task_names=inspect_evals/gsm8k \
        use_cant_protocol=true \
        num_agents=4 \
        eval_limit=200
done
```

### Use Case 5: Evaluate on ANY Inspect AI Task

**Goal:** Test CANT protocol on different types of benchmarks.

```bash
# Instruction following
python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
    model_path=tinker://user/run/step-500 \
    task_names=inspect_evals/ifeval \
    use_cant_protocol=true

# General knowledge
python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
    model_path=tinker://user/run/step-500 \
    task_names=inspect_evals/mmlu \
    use_cant_protocol=true

# Commonsense reasoning
python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
    model_path=tinker://user/run/step-500 \
    task_names=inspect_evals/hellaswag \
    use_cant_protocol=true
```

## ⚙️ Key Parameters Explained

### Training Parameters

| Parameter | What It Does | Typical Values |
|-----------|--------------|----------------|
| `eval_tasks` | Which Inspect AI tasks to evaluate | `"inspect_evals/gsm8k"`, `"inspect_evals/gsm8k,inspect_evals/math"` |
| `eval_use_cant_protocol` | CANT vs baseline | `true` (CANT), `false` (baseline) |
| `eval_num_agents` | How many agents in discussion | `2`, `4`, `8` |
| `eval_every` | How often to evaluate | `20`, `50`, `100` steps |
| `eval_limit` | How many samples per task | `50`, `100`, `None` (all) |
| `eval_temperature` | Sampling temperature | `0.0` (greedy), `0.7` (sampling) |

### Offline Evaluation Parameters

| Parameter | What It Does | Example |
|-----------|--------------|---------|
| `model_path` | Checkpoint to evaluate | `tinker://user/run/step-500` |
| `task_names` | Inspect AI tasks (comma-separated) | `inspect_evals/gsm8k,inspect_evals/math` |
| `use_cant_protocol` | CANT vs baseline | `true` or `false` |
| `num_agents` | Agents for CANT protocol | `4` |

## 🔍 Finding Task Names

### List Available Tasks

```bash
# List all Inspect AI tasks
inspect list tasks

# Search for specific tasks
inspect list tasks | grep gsm8k
inspect list tasks | grep math
```

### Common Task Names

- **Math**: `inspect_evals/gsm8k`, `inspect_evals/math`
- **MMLU**: `inspect_evals/mmlu`
- **Instruction Following**: `inspect_evals/ifeval`
- **Reasoning**: `inspect_evals/arc`, `inspect_evals/hellaswag`

**Full list**: https://inspect.aisi.org.uk/evals.html

## 🔧 Troubleshooting

### Problem: Evaluation is too slow

**Solution 1:** Reduce sample count
```bash
eval_limit=50  # Instead of evaluating all samples
```

**Solution 2:** Reduce agents
```bash
eval_num_agents=2  # Instead of 4
```

**Solution 3:** Evaluate less frequently
```bash
eval_every=100  # Instead of every 20 steps
```

### Problem: Out of memory

**Solution 1:** Reduce max tokens
```bash
eval_max_tokens=1024  # Instead of 2048
```

**Solution 2:** Reduce agents
```bash
eval_num_agents=2  # Fewer agents = less memory
```

### Problem: Task not found

**Solution:** Verify task name format

```bash
# WRONG - short name doesn't work
eval_tasks=gsm8k  # ❌

# CORRECT - use full Inspect AI task name
eval_tasks=inspect_evals/gsm8k  # ✅
```

Check task existence:
```bash
inspect list tasks | grep gsm8k
```

### Problem: Want to use custom dataset

**Solution:** Create an Inspect AI task:

```python
# my_tasks.py
from inspect_ai import task, Task

@task
def my_custom_task():
    return Task(
        dataset=my_dataset,
        solver=my_solver(),
        scorer=my_scorer(),
    )
```

Then use:
```bash
task_names=my_tasks:my_custom_task
```

## 📚 Supported Task Types

The generic CANT solver works with **ANY** Inspect AI task:

### ✅ Math & Reasoning
- GSM8K, MATH, AIME
- ARC (AI2 Reasoning Challenge)
- HellaSwag (commonsense)

### ✅ General Knowledge
- MMLU (general knowledge)
- TriviaQA

### ✅ Instruction Following
- IFEval

### ✅ Code & Logic
- HumanEval
- MBPP

### ✅ Your Custom Tasks
- Any task you create with Inspect AI!

**The CANT protocol can be applied to all of these!**

## 🎓 Next Steps

1. **Read Full Documentation**: See `README.md` in this directory
2. **Explore Inspect AI Tasks**: https://inspect.aisi.org.uk/evals.html
3. **Create Custom Tasks**: Follow Inspect AI task creation guide
4. **Run Experiments**: Compare CANT protocol vs baseline on various tasks

## 💡 Pro Tips

1. **Start small**: Use `eval_limit=50` for fast iteration
2. **Compare modes**: Always run both CANT and baseline to measure improvement
3. **Track agents**: Try different `num_agents` to find optimal collaboration
4. **Greedy sampling**: Use `eval_temperature=0.0` for deterministic evaluation
5. **Save compute**: Evaluate less frequently early in training, more frequently near the end
6. **Universal solver**: The CANT protocol works on ANY task - experiment!

## 🌟 Key Advantage: Universal Compatibility

**The generic CANT solver means you can evaluate ANY benchmark!**

```bash
# Today: Math benchmarks
task_names=inspect_evals/gsm8k

# Tomorrow: General knowledge
task_names=inspect_evals/mmlu

# Next week: Code generation
task_names=inspect_evals/humaneval

# Your custom task
task_names=my_module:my_task
```

**No code changes needed - just change the task name!**

## 🤝 Getting Help

- **Documentation**: Check `README.md` in this directory
- **Task names**: Visit https://inspect.aisi.org.uk/evals.html
- **Code Examples**: See `inspect_tasks.py` for implementation details
- **Issues**: Report problems on the GitHub repo

---

**Happy Evaluating! 🎉**

**Remember**: The CANT protocol is now a universal solver that works with ANY Inspect AI task!
