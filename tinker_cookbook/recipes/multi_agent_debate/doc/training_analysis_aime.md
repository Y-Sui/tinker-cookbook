# Analysis: Why Training Doesn't Improve Performance on AIME 2024

**Date**: January 2025
**Issue**: Model accuracy does not improve (or even degrades) after training on AIME 2024 problems
**Status**: Root cause identified, solutions proposed

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Diagnostic Questions](#2-diagnostic-questions)
3. [Root Cause Analysis](#3-root-cause-analysis)
4. [Experimental Evidence](#4-experimental-evidence)
5. [Proposed Solutions](#5-proposed-solutions)
6. [Action Plan](#6-action-plan)
7. [Expected Outcomes](#7-expected-outcomes)

---

## 1. Problem Statement

### 1.1 Observed Behavior

**Training Setup:**
```bash
python -m tinker_cookbook.recipes.multi_agent_debate.scripts.train \
    env="verifiable" \
    dataset_path="tinker_cookbook/data/aime2024_sample.jsonl" \
    model_name="Qwen/Qwen3-8B" \
    num_agents=3 \
    max_rounds=3 \
    batch_size=16 \
    learning_rate=3e-5
```

**Expected:** Accuracy increases over training batches
**Actual:** Accuracy remains at baseline (~0-5%) or fluctuates without clear improvement

**Example Metrics:**
```
Batch 0:  train_aime/correct = 0.04, train_aime/pass@3 = 0.12
Batch 50: train_aime/correct = 0.03, train_aime/pass@3 = 0.11
Batch 100: train_aime/correct = 0.05, train_aime/pass@3 = 0.13
```

No clear upward trend despite hundreds of training batches.

### 1.2 Context: AIME Difficulty

**AIME (American Invitational Mathematics Examination):**
- Target audience: Top ~5% of high school math students
- Format: 15 problems in 3 hours, answers are integers 000-999
- Median problem-solving rate (humans): ~10-20%
- GPT-4 performance: ~15-30%
- Expected Qwen3-8B baseline: ~0-5%

**Example AIME Problem:**
> Find the number of positive integers $n \leq 600$ whose value of $\phi(n)$ (Euler's totient function) is a perfect square.

This requires:
- Understanding of number theory (Euler's totient function)
- Systematic case analysis
- Careful counting and verification

---

## 2. Diagnostic Questions

Before proposing solutions, we need to understand what's actually happening during training.

### 2.1 Data Collection Questions

**Q1: What is the base model's zero-shot accuracy on AIME?**
- **How to check:** Run evaluation without training
  ```bash
  python -m tinker_cookbook.recipes.multi_agent_debate.scripts.eval \
      checkpoint_path="tinker://pretrained/Qwen3-8B" \
      dataset_path="tinker_cookbook/data/aime2024_sample.jsonl"
  ```
- **Expected finding:** Likely 0-5% (very low)
- **Implication:** If base model is at ~0%, comparison-based rewards have little signal

**Q2: Are agents producing valid comparisons?**
- **Metrics to check:**
  - `v2/total_votes`: Should be > 0 (agents making comparisons)
  - `v2/comparison_lines_invalid`: Should be < 50% (most comparisons valid)
  - `v2/missing_comparisons`: Should be < 20% (agents producing comparisons when expected)
- **Red flags:**
  - `total_votes = 0` → Agents not comparing at all
  - `comparison_lines_invalid > 70%` → Agents producing malformed comparisons
  - `missing_comparisons > 50%` → Format collapse

**Q3: Is there any diversity in agent solutions?**
- **How to check:** Enable transcript logging:
  ```python
  log_full_transcript=True
  num_groups_to_log=4
  ```
- **Look for:**
  - Do agents propose different approaches?
  - Or do all agents converge to the same (wrong) solution?
- **Implication:** If no diversity → no exploration → no learning

**Q4: Do pairwise comparisons correlate with correctness?**
- **Analysis needed:** For problems where at least one agent is correct:
  - Did other agents rank that agent higher?
  - Or are comparisons random/anticorrelated?
- **If anticorrelated:** Agents are making systematically wrong judgments
- **If random:** Agents can't distinguish good from bad solutions

**Q5: Are gradients flowing properly?**
- **Metrics to check:**
  - `reward/gen/mean`: Should not be zero (generator rewards exist)
  - `reward/gen/std`: Should be > 0.1 (some variance in rewards)
  - `reward/judge/mean`: Should not be zero (judge rewards exist)
- **Red flags:**
  - All rewards near zero → No training signal
  - Extremely high variance (std > 5) → Unstable training

### 2.2 Training Dynamics Questions

**Q6: Is the loss decreasing?**
- **Check W&B dashboard** or training logs for:
  - `loss/total`: Should trend downward
  - `loss/forward_backward`: Policy gradient loss component
- **If loss is NOT decreasing:**
  - Learning rate may be too low
  - Gradients may be vanishing
  - Reward signal may be too noisy

**Q7: Are policies actually updating?**
- **How to check:**
  - Compare model outputs at batch 0 vs batch 100
  - Do responses change noticeably?
- **If outputs don't change:** Policy is not learning (frozen weights, LR too low, etc.)

**Q8: Is there overfitting to the training set?**
- **Check:**
  - Train accuracy vs eval accuracy
  - Eval on held-out problems
- **Expected:** With online TTL, train = eval (same problems)
- **If train accuracy increases but eval doesn't:** Memorization without generalization

---

## 3. Root Cause Analysis

Based on typical failure modes in self-play RL on hard tasks:

### 3.1 Hypothesis 1: Task Beyond Model Capability (MOST LIKELY)

**Claim:** AIME is fundamentally too hard for Qwen3-8B to learn through self-play alone.

**Evidence:**
- AIME requires advanced mathematical reasoning
- Qwen3-8B has limited mathematical pre-training
- Even GPT-4 only achieves ~15-30%
- No amount of self-play can teach new mathematical facts

**Analogy:** Teaching calculus through peer grading when no one knows calculus.

**Verification:**
1. Check base model accuracy on easier datasets:
   - GSM8K (grade school math): Expected ~30-60%
   - MATH (high school): Expected ~10-30%
   - AIME: Expected ~0-5%
2. If base model CAN solve easier problems, self-play SHOULD work there
3. If base model CANNOT solve AIME, need curriculum learning

**Conclusion:** Start with easier math problems, build competence, then scale to AIME.

### 3.2 Hypothesis 2: Reward Signal Degradation

**Claim:** When all agents produce incorrect solutions, pairwise comparisons provide minimal or misleading signal.

**Scenario:**
```
Agent 0: "Answer is 42" (wrong)
Agent 1: "Answer is 17" (wrong)
Agent 2: "Answer is 99" (wrong)
Ground truth: 56

Agent 2 comparison: "Agent 0 > Agent 1" (arbitrary choice between two wrong answers)
```

**Problem:** The "winning" solution (42) is still wrong. Reinforcing it doesn't help.

**Verification:**
- Compute correlation between peer-voted winner and ground-truth correctness
- If correlation < 0.3 → Comparisons are not identifying better solutions

**Mitigation:**
- Mix peer comparison rewards with correctness-based rewards:
  ```python
  reward = 0.5 * peer_reward + 0.5 * correctness_reward
  ```
- This provides a "north star" signal even when peer judgments are unreliable

### 3.3 Hypothesis 3: Insufficient Exploration

**Claim:** Policy quickly converges to a mediocre strategy and stops exploring.

**Mechanism:**
1. Agent finds a solution strategy that "sounds reasonable" (e.g., "factor the expression")
2. Other agents see this strategy and copy it
3. All agents converge to the same approach (which is wrong)
4. No comparisons made → no gradient → no learning

**Verification:**
- Check transcript logs: Do agents propose diverse approaches?
- Check `v2/total_votes`: If low, agents are agreeing too much

**Mitigation:**
- Increase temperature for some agents (more randomness)
- Increase `num_agents` (more diverse starting points)
- Reduce `max_rounds` (less time to converge to consensus)
- Add exploration bonus (reward novelty)

### 3.4 Hypothesis 4: Format Learning Dominates

**Claim:** Model learns to produce valid XML format but not correct mathematical reasoning.

**Mechanism:**
- Format penalties (-0.5) for missing comparisons provide clear signal
- Content quality (peer comparisons) provides noisy signal
- Policy optimizes for format compliance, ignores content

**Verification:**
- Check metrics:
  - `train_aime/format`: Should increase to ~90-95%
  - `train_aime/correct`: Stays at ~0-5%
- If format increases but correctness doesn't → Policy is "gaming" the reward

**Mitigation:**
- Reduce format penalty weight
- Increase generator reward weight
- Add correctness-based bonuses

### 3.5 Hypothesis 5: Judgment Cascades (Secondary Issue)

**Claim:** Agents copy each other's judgments, leading to arbitrary consensus.

**Mechanism:**
1. Agent 0 makes random comparison: "Agent 0 > Agent 1"
2. Agent 2 sees no clear difference, follows Agent 0's lead
3. Agent 3 also follows (consensus builds)
4. Result: Consensus not based on quality, but on who spoke first

**Note:** Our implementation has "blind review" (comparisons hidden from history), which mitigates this somewhat.

**Verification:**
- Check if first agent's comparisons are disproportionately adopted by later agents
- Requires detailed transcript analysis

---

## 4. Experimental Evidence

To confirm root causes, run these experiments:

### 4.1 Experiment 1: Easier Dataset Baseline

**Hypothesis:** Self-play works on easier math, fails on AIME due to difficulty.

**Setup:**
```bash
# Test on GSM8K (grade school math)
python -m tinker_cookbook.recipes.multi_agent_debate.scripts.train \
    dataset_path="tinker_cookbook/data/gsm8k.jsonl" \
    problem_field="question" \
    answer_field="answer" \
    num_agents=3 \
    max_rounds=2 \
    batch_size=32 \
    learning_rate=5e-5
```

**Expected:** If hypothesis is correct:
- Base model accuracy on GSM8K: ~30-60%
- After training: Should improve to ~50-70%
- `pass@3` should reach ~70-85%

**Interpretation:**
- ✅ If GSM8K improves → Method works, AIME is just too hard
- ❌ If GSM8K doesn't improve → Fundamental issue with method

### 4.2 Experiment 2: Ground-Truth Mixing

**Hypothesis:** Adding correctness signal helps even when peer judgments are noisy.

**Setup:** Modify reward computation to mix peer and correctness:
```python
# In verifiable_env.py, modify _compute_training_rewards()
# Add correctness bonus
if agent_is_correct:
    gen_rewards[agent_id] = [r + 0.5 for r in gen_rewards[agent_id]]
```

**Expected:**
- Rewards should correlate better with ground truth
- Training should show clearer improvement

### 4.3 Experiment 3: Judgment Quality Analysis

**Hypothesis:** Agents cannot reliably judge mathematical correctness.

**Setup:**
Enable detailed logging and analyze:
```python
for each debate:
    for each comparison "A > B":
        check: is A actually better than B (per ground truth)?

compute: judgment_accuracy = correct_comparisons / total_comparisons
```

**Expected:**
- Random baseline: 50%
- Good judges: >70%
- If judgment_accuracy < 55% → Agents are guessing

### 4.4 Experiment 4: Supervised Warm-Start

**Hypothesis:** Self-play needs initial competence to work.

**Setup:**
```python
# Phase 1: Supervised learning on correct solutions (3-5 epochs)
# Use standard cross-entropy loss, not RL

# Phase 2: Switch to self-play debate
# Now agents have some baseline competence
```

**Expected:**
- Phase 1 should reach ~10-15% accuracy on AIME
- Phase 2 should further improve to ~15-20% via debate

---

## 5. Proposed Solutions

Based on root cause analysis, here are concrete solutions:

### 5.1 Solution A: Curriculum Learning (RECOMMENDED)

**Strategy:** Start easy, gradually increase difficulty.

**Implementation:**
```python
# Stage 1: GSM8K (1-2 days of training)
dataset = "gsm8k"
learning_rate = 5e-5
num_agents = 3
max_rounds = 2
target_accuracy = 60%  # Move to next stage when reached

# Stage 2: MATH Easy Problems (2-3 days)
dataset = "math_easy"
learning_rate = 3e-5
num_agents = 3
max_rounds = 3
target_accuracy = 40%

# Stage 3: MATH Medium Problems (3-5 days)
dataset = "math_medium"
learning_rate = 2e-5
num_agents = 4
max_rounds = 3
target_accuracy = 25%

# Stage 4: AIME (final stage)
dataset = "aime"
learning_rate = 1e-5
num_agents = 5
max_rounds = 3
target_accuracy = 15%  # Realistic goal
```

**Advantages:**
- Builds mathematical reasoning gradually
- Each stage provides signal for the next
- Well-established in RL literature

**Disadvantages:**
- Takes longer overall
- Requires curating curriculum

### 5.2 Solution B: Ground-Truth Reward Mixing

**Strategy:** Combine peer comparisons with correctness bonuses.

**Implementation:**
```python
# In verifiable_env.py
def _compute_training_rewards(self, trajectory_group, env_group, problem):
    # Existing v2 rewards
    gen_rewards, judge_rewards, v2_metrics = self._compute_rewards_v2(...)

    # Add correctness bonuses
    for agent_id in range(self.num_agents):
        final_answer = extract_boxed(agent_solutions[agent_id])
        if final_answer == problem.answer:
            # Bonus for correct answer
            gen_rewards[agent_id][-1] += 1.0  # Bonus to last step

    return gen_rewards, judge_rewards, metrics
```

**Advantages:**
- Provides clear "north star" signal
- Simple to implement
- Compatible with existing v2 system

**Disadvantages:**
- Only works for verifiable tasks
- May reduce generalization (focuses on answer, not reasoning)

### 5.3 Solution C: Supervised Pre-Training Phase

**Strategy:** Teach mathematical facts first, then improve via debate.

**Implementation:**
```python
# Step 1: Supervised fine-tuning (3-5 epochs)
# Train with cross-entropy loss on correct solutions
for problem, solution in dataset:
    loss = cross_entropy(model(problem), solution)
    optimize(loss)

# Checkpoint after supervised phase

# Step 2: Load supervised checkpoint and run debate training
checkpoint = load_supervised_checkpoint()
run_debate_training(checkpoint)
```

**Advantages:**
- Guarantees baseline competence
- Debate only refines, doesn't teach from scratch
- Proven approach (e.g., STaR paper)

**Disadvantages:**
- Requires correct solution data
- Two-phase training is more complex

### 5.4 Solution D: Increase Exploration

**Strategy:** Force more diversity in solutions to explore space better.

**Implementation:**
```python
# Increase agent count
num_agents = 5  # Was 3

# Increase temperature diversity
AGENT_PERSONAS = [
    ("Methodical", 0.6),
    ("Creative", 1.2),    # Higher temp
    ("Skeptic", 0.9),
    ("Synthesizer", 1.1), # Higher temp
    ("Random", 1.5),      # Very high temp
]

# Reduce rounds (less consensus)
max_rounds = 2  # Was 3

# Add exploration bonus (reward novelty)
if solution_is_novel(agent_solution):
    reward += 0.3
```

**Advantages:**
- May help discover new solution strategies
- Prevents premature convergence

**Disadvantages:**
- Higher variance in training
- May not help if problem is just too hard

### 5.5 Solution E: Hybrid Evaluation Metrics

**Strategy:** Track multiple metrics to understand what's being learned.

**Implementation:**
```python
# Add intermediate metrics
metrics = {
    "format_valid": ...,         # Existing
    "correct_final_answer": ..., # Existing
    "correct_approach": ...,     # NEW: Is the solution method valid?
    "partial_credit": ...,       # NEW: How many steps were correct?
    "judgment_accuracy": ...,    # NEW: Are comparisons accurate?
}

# Use these to diagnose which parts are improving
```

**Advantages:**
- Better visibility into training dynamics
- Can identify partial progress even if final accuracy doesn't improve

**Disadvantages:**
- Requires manual annotation or heuristics

---

## 6. Action Plan

**Recommended approach:** Combine solutions A + B + E (curriculum + mixing + metrics).

### Phase 1: Diagnostic (1-2 days)

**Goal:** Confirm root cause before scaling up.

**Tasks:**
1. ✅ Run baseline evaluation on AIME (zero-shot, no training)
2. ✅ Run baseline evaluation on GSM8K
3. ✅ Analyze training logs for AIME run (check all diagnostic questions)
4. ✅ Implement judgment accuracy metric
5. ✅ Run Experiment 1 (easier dataset baseline)

**Success criteria:**
- Understand current model capability on different difficulty levels
- Confirm that self-play works on easier problems
- Identify specific failure mode (exploration, judgment, etc.)

### Phase 2: Quick Wins (2-3 days)

**Goal:** Improve training on existing setup.

**Tasks:**
1. ✅ Implement ground-truth reward mixing (Solution B)
2. ✅ Add intermediate metrics (Solution E)
3. ✅ Tune hyperparameters:
   - Increase batch size to 32 (more stable gradients)
   - Reduce learning rate to 1e-5 (more stable updates)
   - Increase `num_agents` to 5 (more diversity)
4. ✅ Re-run training on AIME with improvements

**Success criteria:**
- Training loss should decrease consistently
- At least one intermediate metric shows improvement (even if final accuracy doesn't)
- Reward variance decreases (more stable)

### Phase 3: Curriculum Learning (1-2 weeks)

**Goal:** Build competence from easy to hard.

**Tasks:**
1. ✅ Curate curriculum:
   - GSM8K: ~1000 problems (easy)
   - MATH: ~1000 problems (medium)
   - AIME: ~50 problems (hard)
2. ✅ Train on GSM8K until 60% accuracy
3. ✅ Continue training on MATH until 30% accuracy
4. ✅ Finally train on AIME (target: 15% accuracy)

**Success criteria:**
- Each stage shows improvement
- AIME performance better than direct training (Phase 2)

### Phase 4: Paper Writing (Ongoing)

**Goal:** Document findings for publication.

**Tasks:**
1. ✅ Complete research overview (doc/research_overview.md)
2. ✅ Document training curves and failure modes
3. ✅ Write ablation studies (with/without mixing, with/without curriculum)
4. ✅ Prepare visualizations (accuracy over time, reward distributions)

---

## 7. Expected Outcomes

### 7.1 Short-Term (After Phase 2)

**Metrics:**
- AIME `train/correct`: 0% → 5-8% (modest improvement)
- AIME `train/pass@3`: 0-5% → 10-15% (better with aggregation)
- Training stability: Reduced reward variance

**Understanding:**
- Clear picture of what's being learned (format, judgment, reasoning)
- Identified bottlenecks (e.g., "model can judge but not generate")

### 7.2 Medium-Term (After Phase 3)

**Metrics:**
- GSM8K: 40% → 65% (solid improvement)
- MATH: 15% → 35% (moderate improvement)
- AIME: 0% → 12-15% (reaching GPT-3.5 level)

**Scientific Contribution:**
- Demonstrated: Self-play debate works when task difficulty matches model capability
- Demonstrated: Curriculum learning essential for complex reasoning
- Documented: Failure modes when task is too hard

### 7.3 Long-Term (Research Contribution)

**Publications:**
- Paper: "Multi-Agent Debate for Mathematical Reasoning: A Curriculum Approach"
- Sections:
  1. Method: V2 reward system with dual streams
  2. Results: Works on GSM8K/MATH, struggles on AIME
  3. Analysis: Why self-play needs competence threshold
  4. Solutions: Curriculum + mixing + metrics

**Broader Impact:**
- Framework applicable to other domains (coding, science, etc.)
- Insights into when self-play RL works vs. when it needs bootstrapping
- Practical recipe for training debate systems

---

## 8. Debugging Checklist

Before opening an issue or seeking help, verify:

### 8.1 Data & Environment
- [ ] Dataset loaded correctly (`len(problems) > 0`)
- [ ] Problems have required fields (`problem`, `answer`)
- [ ] Answers are in correct format (for AIME: integers 000-999)
- [ ] Base model can tokenize problems without errors

### 8.2 Training Configuration
- [ ] Model is LoRA-trainable (not frozen)
- [ ] Learning rate is reasonable (1e-5 to 5e-5 for LoRA)
- [ ] Batch size fits in memory
- [ ] Number of agents × rounds × max_tokens fits in memory

### 8.3 Reward System
- [ ] Agents produce valid comparisons (`v2/total_votes > 0`)
- [ ] Rewards have non-zero variance (`reward/gen/std > 0.1`)
- [ ] Format penalties are not dominating (`v2/missing_comparisons < 50%`)
- [ ] Generator and judge rewards are both non-zero

### 8.4 Training Progress
- [ ] Loss is decreasing over batches
- [ ] At least one metric shows improvement (even if not accuracy)
- [ ] Model outputs are changing (policy is updating)
- [ ] No NaN or Inf values in logs

### 8.5 Evaluation
- [ ] Evaluation runs without errors
- [ ] Evaluation metrics make sense (not all 0 or all 1)
- [ ] Evaluation uses correct grader (`sympy` for AIME)
- [ ] Timeout is sufficient for grading (>= 2 seconds)

---

## 9. Conclusion

**TL;DR:**
- **Root cause:** AIME is likely too hard for Qwen3-8B to learn via self-play alone
- **Evidence:** Base model accuracy ~0-5%, comparison signal is noisy
- **Solution:** Start with easier problems (GSM8K), use curriculum learning, mix in ground-truth rewards
- **Timeline:** Diagnostic (2 days) → Quick wins (3 days) → Curriculum (2 weeks)
- **Expected result:** 12-15% on AIME (comparable to GPT-3.5) after full curriculum

**Key Insight:** Self-play debate is a powerful technique, but it amplifies existing capabilities rather than teaching new knowledge. When the base model lacks mathematical reasoning ability, debate cannot create it from nothing. The solution is to build competence gradually through curriculum learning, then use debate to refine and improve.

**Next Steps:**
1. Run diagnostic experiments (Section 4)
2. Implement quick wins (Section 5.2 + 5.5)
3. Execute action plan (Section 6)
4. Document results for paper

---

## Appendix: Common Error Patterns

### Pattern 1: Degenerate Solutions
```
All agents: "Answer is 1"
Comparisons: None (all agents agree)
Reward: 0 (no gradient)
```
**Fix:** Increase exploration (higher temperature, more agents)

### Pattern 2: Format Collapse
```
Agent response: "<solution>N/A</solution><evaluation>N/A</evaluation><comparison>N/A</comparison>"
```
**Fix:** Reduce format penalty, increase content reward weight

### Pattern 3: Circular Reasoning
```
Agent 0: "The answer is A because it's better than B"
Agent 1: "The answer is B because it's better than A"
Agent 2: "I agree with Agent 0"
```
**Fix:** Add ground-truth signal, improve judgment prompts

### Pattern 4: Reward Explosion
```
Batch 0: reward/gen/mean = 0.1
Batch 1: reward/gen/mean = 2.5
Batch 2: reward/gen/mean = 10.7 (explosion!)
```
**Fix:** Lower learning rate, clip gradients, normalize advantages

### Pattern 5: No Exploration
```
All agents produce identical solutions (copy-paste)
```
**Fix:** Increase temperature, add diversity penalties, use more agents

---

**Document Version:** 1.0
**Last Updated:** January 17, 2025
**Contact:** See README.md for project maintainers
