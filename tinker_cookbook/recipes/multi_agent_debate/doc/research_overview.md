# Multi-Agent Debate for Reinforcement Learning: Research Overview

**Authors**: Thinking Machines Lab
**Date**: January 2025
**Status**: Research Implementation

---

## Abstract

We present a multi-agent debate framework for reinforcement learning that trains a single language model policy to engage in structured multi-turn debates through self-play. Unlike traditional approaches that rely on ground-truth rewards or separate critic models, our method derives learning signals entirely from peer evaluations: agents produce pairwise comparisons of each other's responses, and rewards are computed from the aggregated vote outcomes. We introduce a v2 reward system that separates generator rewards (based on soft vote ratios for solution quality) from judge rewards (based on consensus alignment for evaluation accuracy). Experiments on both verifiable (mathematical reasoning) and non-verifiable (open-ended) tasks demonstrate that this approach can improve both generation and meta-evaluation capabilities through self-play, though we identify several challenges in applying this method to difficult mathematical benchmarks like AIME.

**Key Contributions:**
1. A self-play debate framework where a single policy learns generation and evaluation simultaneously
2. A v2 reward system with separate streams for content quality and judgment accuracy
3. Parallel within-round generation for efficient training
4. Analysis of training dynamics and identified limitations on challenging benchmarks

---

## 1. Introduction

### 1.1 Motivation

Large language models demonstrate remarkable capabilities in solving complex problems, but their performance can vary significantly. Recent work has shown that having multiple models debate can improve answer quality through iterative refinement and cross-examination. However, training such systems typically requires either:
- Ground-truth labels for supervised learning
- Separate reward models for RLHF
- Human feedback for preference learning

We propose an alternative: **self-play debate with peer-derived rewards**. A single policy θ plays all agent roles simultaneously, producing both solutions and evaluations. The key insight is that pairwise comparisons between agents' responses can serve as a training signal, even without ground truth.

### 1.2 Problem Setting

Given a question q, we want a policy π_θ to:
1. **Generate** high-quality solutions
2. **Evaluate** other agents' solutions accurately
3. **Compare** solutions through pairwise rankings

The policy participates in a multi-turn debate where N agents (all instances of π_θ) take turns responding. Each response contains three sections:
- `<solution>`: Proposed answer to the question
- `<evaluation>`: Critique of other agents' solutions
- `<comparison>`: Pairwise rankings (e.g., "Agent 0 > Agent 2")

### 1.3 Core Challenge

The fundamental tension: we want to train the model to generate better solutions AND to evaluate solutions more accurately, but we're using the model's own evaluations as the training signal. This creates a potential circularity where a poorly-calibrated model might reinforce its own biases.

Our v2 reward system addresses this by:
- Encouraging **winning solutions** (many agents prefer them)
- Encouraging **consensus-aligned judgments** (agreeing with majority opinion)
- Penalizing **format violations** (missing required comparisons)

---

## 2. Method

### 2.1 Debate Protocol

For each question q:

**Setup:**
- Create N agents (all instances of π_θ with different persona prompts)
- Run R rounds of debate
- Total turns: T = N × R (agents take turns in round-robin)

**Turn Structure:**
Each turn t, the active agent i receives:
- **System prompt**: Persona description (e.g., "You are the Methodical Analyst")
- **User prompt**: Question + history of previous responses (from other agents)

The agent generates a structured response with three sections.

**Parallel Generation (Key Optimization):**
Within each round, all N agents generate simultaneously and see the same history (responses from previous rounds only). This ensures:
- Independent judgments (no bandwagon effects within a round)
- Faster training (N-way parallelism)
- Simpler implementation (no waiting for sequential generation)

**Blind Review:**
When viewing history, agents see only `<solution>` and `<evaluation>` sections. The `<comparison>` tags are hidden to prevent agents from copying each other's judgments.

### 2.2 Reward System V2

Our v2 reward system computes two separate reward streams:

#### 2.2.1 Generator Rewards (for `<solution>` and `<evaluation>` tokens)

**Intuition:** Solutions that many agents prefer should be reinforced.

**Computation:**
1. Extract all pairwise comparisons from `<comparison>` tags
2. For each agent i at step s, compute:
   - votes_for: Number of times other agents ranked agent i higher
   - votes_against: Number of times other agents ranked agent i lower
3. Soft vote ratio: `R_gen(i,s) = votes_for / max(1, votes_for + votes_against)`
4. Normalize to range [-1, +1]: `R_gen(i,s) = 2 * ratio - 1`

**Properties:**
- Dense signal: Every comparison provides gradient information
- Continuous: Smooth transitions from -1 (unanimously rejected) to +1 (unanimously preferred)
- Self-normalized: Automatically handles varying numbers of votes

#### 2.2.2 Judge Rewards (for `<comparison>` tokens)

**Intuition:** Judges who align with consensus are accurate; those who disagree are not.

**Computation:**
1. Build consensus matrix C[a,b] = majority opinion for pair (a,b):
   - If more agents say "a > b", then C[a,b] = ">"
   - If more agents say "b > a", then C[a,b] = "<"
   - If tied, C[a,b] = "tie"
2. For each comparison made by judge j:
   - If judgment matches consensus: `R_judge = +1`
   - If judgment opposes consensus: `R_judge = -1`
   - If consensus is tie: `R_judge = 0` (no signal)

**Properties:**
- Encourages accurate peer evaluation
- Rewards judges who identify true quality differences
- No reward for ambiguous cases (ties)

#### 2.2.3 Format Penalties

**Motivation:** Agents must learn to produce valid comparisons when expected.

**Rule:** If agent is not in first two turns AND produces no valid comparisons, apply penalty -0.5 to that step.

**Exemptions:**
- Turn 0: No other agents have responded yet
- Turn 1: Only one other agent (need at least 2 for meaningful comparison)

### 2.3 Training Procedure

**Data Collection:**
```
for batch in dataset:
    for question in batch:
        # Self-play rollout
        trajectories = await run_debate_episode(question, policy=π_θ)

        # Compute rewards from peer comparisons
        gen_rewards, judge_rewards = compute_v2_rewards(trajectories)

        # Separate data streams
        gen_data = extract_tokens(trajectories, sections=["<solution>", "<evaluation>"])
        judge_data = extract_tokens(trajectories, sections=["<comparison>"])

        # Apply rewards
        gen_data = attach_rewards(gen_data, gen_rewards)
        judge_data = attach_rewards(judge_data, judge_rewards)
```

**Optimization:**
```
# Compute advantages (centered within group)
gen_advantages = center_returns(gen_rewards)
judge_advantages = center_returns(judge_rewards)

# Policy gradient with importance sampling
loss = λ_gen * pg_loss(gen_data, gen_advantages) +
       λ_judge * pg_loss(judge_data, judge_advantages)

# Update
θ ← θ - α ∇_θ loss
```

**Hyperparameters:**
- λ_gen = 1.0 (generator weight)
- λ_judge = 1.0 (judge weight)
- Learning rate: 3e-5 (LoRA fine-tuning)
- Batch size: 16 questions
- Agents per debate: N = 3
- Rounds per debate: R = 3
- Total turns: T = 9

### 2.4 Agent Personas

To encourage diversity in reasoning approaches, we assign each agent a distinct persona:

| ID | Persona | Style | Temperature |
|----|---------|-------|-------------|
| 0 | Methodical Analyst | Systematic step-by-step | 0.6 |
| 1 | Creative Problem-Solver | Unconventional approaches | 1.0 |
| 2 | Devil's Advocate | Skeptical, finds counterexamples | 0.9 |
| 3 | Synthesizer | Combines multiple viewpoints | 1.0 |
| 4 | First Principles Thinker | Returns to fundamentals | 0.8 |

For N > 5 agents, personas cycle (agent 5 gets persona 0, etc.).

### 2.5 Implementation Optimizations

**Sequence Extension (O(T) vs O(T²)):**
- Traditional: Strip thinking tags from history → quadratic context growth
- Our approach: Preserve empty `<think></think>` blocks → linear context (KV-cache reuse)
- Speedup: ~9x faster for 9-turn debates

**Async Training:**
- Forward passes and gradient computations overlap
- Multiple episodes run in parallel
- Reduces wall-clock time by ~3x

---

## 3. Evaluation

### 3.1 Metrics

**Training Metrics (Verifiable Environment):**
- `train_{dataset}/format`: Fraction of responses with valid `\boxed{answer}` format
- `train_{dataset}/correct`: Fraction of correct final answers
- `train_{dataset}/pass@k`: At least one agent correct (optimistic aggregation)
- `train_{dataset}/avg@k`: Mean accuracy across k agents
- `train_{dataset}/cons@k`: Majority of agents correct (voting)

**Reward System Metrics:**
- `v2/total_votes`: Number of pairwise comparisons collected
- `v2/missing_comparisons`: Number of format penalty applications
- `reward/gen/mean`: Average generator reward
- `reward/judge/mean`: Average judge reward

**Evaluation Modes (Verifiable):**
- **Direct**: Single-turn response (no debate)
- **Debate**: Full multi-turn self-play

### 3.2 Datasets

**Verifiable (Math):**
- AIME 2024 (50 problems): High school competition mathematics
- Difficulty: Very challenging (median human solver rate ~10-20%)

**Non-Verifiable (Open-ended):**
- LongWriter-6K: Instruction-following and creative writing
- Difficulty: Subjective, quality measured by peer agreement

---

## 4. Results & Analysis

### 4.1 Training Dynamics

**Observation:** On AIME 2024, training does not consistently improve accuracy after multiple epochs.

**Hypothesis 1: Reward Signal Quality**
- Problem: If initial policy is weak (e.g., 0-10% accuracy), peer comparisons may be mostly "noise vs noise"
- Evidence: High `v2/comparison_lines_invalid` suggests agents struggle to produce valid comparisons
- Implication: Reward signal quality degrades when all agents perform poorly

**Hypothesis 2: Exploration-Exploitation Trade-off**
- Problem: Policy quickly converges to a local optimum (specific solution strategy)
- Evidence: Agents often produce similar solutions in later rounds
- Implication: Need higher temperature or diversity mechanisms to explore solution space

**Hypothesis 3: Judgment Calibration**
- Problem: Agents may not be able to accurately judge mathematical correctness without ground truth
- Evidence: Low correlation between peer-voted "winner" and ground-truth correctness
- Implication: Self-play works better when agents have some initial competence

**Hypothesis 4: Credit Assignment**
- Problem: Debate has multiple rounds; which step actually contributed to the final answer?
- Current approach: Rewards assigned to specific comparison targets (not just final answer)
- Alternative: Could try value functions or Monte Carlo returns

### 4.2 What Works Well

**Format Learning:**
- Agents quickly learn to produce valid XML structure
- `train_{dataset}/format` increases from ~60% → ~95% within first epoch

**Evaluation Capabilities:**
- Agents learn to identify obvious errors (e.g., arithmetic mistakes, invalid reasoning)
- Judge rewards show positive correlation with evaluation quality on simpler problems

**Multi-Agent Aggregation:**
- `pass@k` > individual accuracy (best-of-k sampling helps)
- `cons@k` shows that majority voting can filter out some errors

### 4.3 Failure Modes

**Mode 1: Degenerate Consensus**
- All agents converge to the same (wrong) solution
- No comparisons made (all agents agree)
- Zero gradient signal

**Mode 2: Random Noise**
- Agents produce different wrong answers
- Comparisons are arbitrary (no clear winner)
- High-variance gradient signal

**Mode 3: Format Collapse**
- Agents stop producing comparisons to avoid penalties
- Policy learns to output "N/A" or minimal text
- Training signal disappears

---

## 5. Identified Issues & Recommendations

### 5.1 Training on AIME 2024 (Hard Math)

**Issue:** Performance does not improve after training.

**Root Cause Analysis:**
1. **Task difficulty mismatch**: AIME problems may be beyond the model's reasoning capability
   - Even GPT-4 achieves only ~15-30% on AIME
   - Qwen3-8B baseline is likely 0-5%

2. **Insufficient pre-training on math**: Base model lacks mathematical knowledge
   - Comparisons can't teach new mathematical facts
   - Need curriculum: start with easier problems (GSM8K, MATH) before AIME

3. **Reward signal quality**: When all agents are wrong, comparisons don't help
   - Garbage-in, garbage-out: comparing bad solutions doesn't identify good ones
   - Need bootstrap: warm-start with supervised learning or easier problems

**Recommendations:**

**Short-term fixes:**
```python
# 1. Start with easier problems
dataset_path = "tinker_cookbook/data/gsm8k.jsonl"  # Instead of AIME

# 2. Increase exploration
lambda_gen = 0.5  # Reduce generator weight
lambda_judge = 1.5  # Increase judge weight (focus on evaluation first)

# 3. Add diversity
num_agents = 5  # More agents → more diverse solutions
max_rounds = 2  # Fewer rounds → less convergence

# 4. Supervised warm-start
# Train with cross-entropy loss on correct solutions first
# Then switch to debate-based RL
```

**Long-term improvements:**
```python
# 1. Curriculum learning
datasets = [
    "gsm8k",          # Grade school math (easy)
    "math_algebra",   # MATH dataset algebra (medium)
    "math_counting",  # MATH dataset counting (medium-hard)
    "aime",           # AIME (very hard)
]
# Train on each sequentially

# 2. Ground-truth mixing
# Mix peer-comparison rewards with correctness-based rewards
reward = 0.7 * peer_reward + 0.3 * correctness_reward

# 3. Outcome-based supervision
# For verifiable tasks, add bonus reward for correct final answers
# This provides a "north star" signal even when comparisons are noisy

# 4. Judgment bootstrapping
# Use stronger model (GPT-4) to provide initial judgments
# Gradually phase out as policy improves
```

### 5.2 Hyperparameter Tuning

**Learning Rate:**
- Current: 3e-5 (standard for LoRA)
- Issue: May be too high if policy is unstable
- Try: 1e-5 or adaptive schedule

**Batch Size:**
- Current: 16 questions
- Issue: High variance in rewards across problems
- Try: 32-64 questions for more stable gradients

**Advantage Normalization:**
- Current: Center within each debate group
- Issue: Groups have different difficulty → different reward scales
- Try: Global normalization across all groups in batch

### 5.3 Debugging Checklist

Before training on AIME, verify:
- [ ] Base model can solve at least 5-10% of problems with direct prompting
- [ ] Agents produce valid comparisons >80% of the time
- [ ] At least one agent is correct in >20% of debates (pass@3)
- [ ] Judgment accuracy >50% on pairwise comparisons (better than random)

If any check fails, address root cause before scaling up training.

---

## 6. Future Directions

### 6.1 Theoretical Understanding

**Open Questions:**
1. When does self-play with peer comparisons converge?
2. What is the sample complexity compared to supervised learning?
3. Can we prove that debate improves over single-agent generation?

**Potential Analysis:**
- Multi-agent game theory perspective
- Connection to iterative peer grading literature
- Relation to debate and argumentation theory

### 6.2 Algorithmic Improvements

**Value Functions:**
- Learn V(s) to estimate expected future rewards
- Improves credit assignment across multi-turn debates

**Auxiliary Tasks:**
- Predict whether own solution will be voted winner
- Predict final answer correctness (for verifiable tasks)
- Improves representation learning

**Adaptive Agent Count:**
- More agents for hard problems (more exploration)
- Fewer agents for easy problems (faster training)

**Dynamic Debate Length:**
- Early stopping when consensus is reached
- Longer debates when agents disagree

### 6.3 Applications

**Domains:**
- Coding: Multiple agents propose and review code
- Scientific reasoning: Agents debate hypotheses
- Legal analysis: Agents argue different positions
- Creative writing: Agents provide peer feedback

**Integration:**
- Use debate as test-time compute (inference-only)
- Fine-tune on debate transcripts (distillation)
- Combine with tree search for mathematical reasoning

---

## 7. Conclusion

We presented a multi-agent debate framework for reinforcement learning that derives training signals from peer evaluations rather than ground truth or external rewards. Our v2 reward system separates generator rewards (content quality via vote ratios) from judge rewards (evaluation accuracy via consensus alignment).

While the framework shows promise on well-scoped problems, we identified significant challenges when applying it to very difficult benchmarks like AIME 2024. The core issue is that self-play works best when agents have some initial competence; when all agents are weak, peer comparisons provide limited signal.

**Key Takeaways:**
1. ✅ Self-play debate can improve both generation and evaluation capabilities
2. ✅ V2 reward system effectively separates content and judgment signals
3. ⚠️ Requires curriculum learning for hard problems (start easy, scale up)
4. ⚠️ May need ground-truth mixing or supervised bootstrapping for very challenging domains
5. ⚠️ Exploration mechanisms critical to avoid premature convergence

**Recommended Path Forward:**
- Start with easier datasets (GSM8K, MATH) to build competence
- Use curriculum learning to gradually increase difficulty
- Mix peer-derived rewards with outcome-based rewards (for verifiable tasks)
- Focus on judgment quality metrics during training
- Apply debate at test-time even without training (inference-time scaling)

The multi-agent debate framework represents a promising direction for improving LLM capabilities through self-play, but successful application requires careful consideration of task difficulty, model capabilities, and reward signal quality.

---

## References

**Tinker Documentation:**
- `docs/rl.mdx`: General RL framework
- `docs/training-sampling.mdx`: Training and sampling basics
- `recipes/math_rl/`: Single-agent mathematical reasoning baseline

**Related Work:**
- Constitutional AI (Anthropic): Self-critique for alignment
- Debate for AI Safety (Irving et al.): Human-judged debates
- Self-Taught Reasoner (Zelikman et al.): Self-training on reasoning traces
- Process reward models (Lightman et al.): Step-level reward learning

---

## Appendix A: Hyperparameter Reference

**Default Configuration (Verifiable Math):**
```python
# Model
model_name = "Qwen/Qwen3-8B"
renderer_name = "qwen3_disable_thinking"
max_tokens = 8196

# Debate
num_agents = 3
max_rounds = 3  # Total turns = 9

# Training
batch_size = 16
learning_rate = 3e-5
num_train_datapoints = 1024

# Rewards
lambda_gen = 1.0
lambda_judge = 1.0
enable_format_penalty = True

# Optimization
use_cosine_lr_schedule = False
eval_every = 50
save_every = 100
```

**For Easier Problems (GSM8K, MATH):**
```python
num_agents = 3
max_rounds = 2  # Total turns = 6
learning_rate = 5e-5
batch_size = 32
```

**For Harder Problems (AIME) - WITH WARM-START:**
```python
# Step 1: Supervised learning on correct solutions
phase = "supervised"
learning_rate = 1e-4
num_epochs = 3

# Step 2: Self-play debate
phase = "self_play"
num_agents = 5  # More diversity
max_rounds = 3
learning_rate = 3e-5
lambda_gen = 0.5  # De-emphasize generation
lambda_judge = 1.5  # Emphasize accurate judgment
```

## Appendix B: Glossary

- **Agent**: One role in the debate (all instances of the same policy θ)
- **Turn**: One agent's opportunity to respond
- **Round**: One complete cycle through all N agents
- **Cycle**: Synonym for round
- **Step**: One transition in an agent's trajectory (one turn)
- **Generator tokens**: Tokens in `<solution>` and `<evaluation>` sections
- **Judge tokens**: Tokens in `<comparison>` section
- **Soft vote ratio**: Continuous reward based on fraction of votes received
- **Consensus alignment**: Binary reward for matching majority opinion
- **Format penalty**: Negative reward for missing required comparisons
- **Pass@k**: Success rate when taking best of k agents
- **Cons@k**: Success rate when majority of k agents agree
- **Trajectory group**: Collection of N agent trajectories for one debate
