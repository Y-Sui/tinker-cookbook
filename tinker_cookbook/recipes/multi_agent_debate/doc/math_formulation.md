# Multi-Agent Debate: Current Implementation (Scope → Details → Math)

## Plain-language summary

A single language model is trained (or evaluated) in self-play by having it personate multiple “agents” who debate the same question over several rounds. Agents take turns. On each turn an agent must produce three parts: (1) a proposed solution, (2) a critique/meta-critique of the discussion, and (3) pairwise comparisons that rank other agents (e.g., “Agent 0 > Agent 2”).

The core learning signal comes from these peer comparisons rather than from task correctness. Every time an agent states a comparison “A > B”, the system treats it as a win for A and a loss for B. Across the whole debate, each agent accumulates positive and negative comparison points based on how often they are ranked above or below others. Optionally, agents are also penalized when they fail to provide any valid comparisons on turns where comparisons are expected (with early turns exempt to allow “warm-up”). The idea here is that the judge comparison is based on both generation, and evaluation, if the agent's evaluation has some issues, they might also been judged as bad completion. By doing this, we also judge the judge capabilities.

Rewards are not given immediately while the agents are generating text. Instead, the full transcript is collected first, then rewards are computed afterward from all comparisons and penalties. The rewards are distributed to the corresponding steps been compared. For example, if agent 2's pairwise comparison shows that agent 0's completion is better than agent 1's completion. Then agent 0's completion will be rewarded.

For optimization, each agent’s total return is converted into an advantage by subtracting the average return within the same debate group (so training focuses on relative performance between agents on the same prompt). That advantage is applied to the tokens the agent generated, yielding a token-level policy-gradient style update with an importance-sampling correction between the policy that generated the data and the current policy.

There are two variants. In the non-verifiable version, the only training signal is the peer-comparison mechanism. In the verifiable math version, training is still driven by comparisons, but the system also checks whether each agent’s final answer matches ground truth (with a required boxed-answer format) and logs additional metrics such as best-of-N success (at least one agent correct), average correctness across agents, and majority-consensus correctness. The verifiable setup can also be run in evaluation modes that either perform a full debate or do a single direct answer attempt.

This note documents the *current* behavior (including non-obvious edge cases) of the multi-agent debate recipe as implemented under `tinker_cookbook/recipes/multi_agent_debate/`.

It is written to be copy/paste-able as “implementation details / methods” for a paper, but it intentionally stays faithful to the code, not an idealized design.


We formulate multi-agent debate as a self-play game where the unified policy π simultaneously generates N debate positions {s₁, s₂, ..., sₙ} and N² pairwise preference judgments {p_ij} for all pairs of positions. The reward signal for position i is derived from the aggregated peer evaluations: r_i = Σⱼ I(p_ji = 'prefer i') - Σⱼ I(p_ij = 'prefer j')

## What This Recipe Is (High-Level Scope)

The recipe trains (and/or evaluates) a single policy in *self-play* across multiple “roles” (“agents”) on the *same* question. The roles take turns producing:

- a proposed answer (`<solution>...</solution>`),
- a critique/meta-critique (`<evaluation>...</evaluation>`),
- and pairwise rankings of *other* agents (`<comparison>...</comparison>`).

Learning is driven by **comparison-derived step rewards**: when an agent ranks “Agent A > Agent B”, the recipe adds `+1` to A’s most recent step (before the ranking was made) and `-1` to B’s.

The recipe has two environment variants:

1) **Non-verifiable debate** (`env.py`): open-ended questions; reward comes only from peer comparisons.
2) **Verifiable math debate** (`verifiable_env.py`): math problems with ground truth; *training reward is still comparison-based*, but the system also logs correctness/format metrics and supports correctness-based evaluation modes.

## Files and Entry Points (Concrete Code Map)

Core multi-agent debate implementation:

- `tinker_cookbook/recipes/multi_agent_debate/coordinator.py`: shared turn-taking state (`MultiAgentCoordinator`).
- `tinker_cookbook/recipes/multi_agent_debate/base_env.py`: shared env logic + shared reward shaping (`BaseMultiAgentDebateEnv`, `BaseMultiAgentEnvGroupBuilder`).
- `tinker_cookbook/recipes/multi_agent_debate/env.py`: non-verifiable debate env + dataset builder.
- `tinker_cookbook/recipes/multi_agent_debate/verifiable_env.py`: verifiable math env + dataset builder + evaluation modes.
- `tinker_cookbook/recipes/multi_agent_debate/prompts.py`: system prompts + XML parsing (`parse_agent_response`).
- `tinker_cookbook/recipes/multi_agent_debate/utils.py`: stop conditions + reward indexing helpers + logging.

Supporting scripts:

- `tinker_cookbook/recipes/multi_agent_debate/train.py`: training CLI; wires the dataset builder into the generic RL trainer (`tinker_cookbook/rl/train.py`) and (for verifiable) adds a custom evaluator.
- `tinker_cookbook/recipes/multi_agent_debate/eval.py`: evaluation-only CLI for verifiable math, given a checkpoint.
- `tinker_cookbook/recipes/multi_agent_debate/evaluator.py`: verifiable evaluator (runs debate rollouts; direct mode exists but is currently commented out in `__call__`).
- `tinker_cookbook/recipes/multi_agent_debate/inference.py`: OpenRouter “self-play” script that exercises the same coordinator/parsing/reward logic without Tinker training.

The generic RL plumbing that consumes this environment:

- `tinker_cookbook/rl/rollouts.py`: runs one rollout per env (concurrently) and then calls `compute_group_rewards(...)`.
- `tinker_cookbook/rl/data_processing.py`: converts trajectories + returns into token-level training data and computes centered advantages.

## Runtime Objects and Control Flow

### Episode, turns, and trajectories

Fix:

- number of agents \(N = \texttt{num\_agents}\),
- maximum rounds \(R = \texttt{max\_rounds}\),
- maximum global turns \(T_{\max} = N \cdot R\).

For each question/problem \(p\), the group builder creates:

- one shared `MultiAgentCoordinator(question=p, num_agents=N, max_turns=T_max)`, and
- \(N\) env instances `BaseMultiAgentDebateEnv(agent_id=i, coordinator=...)`, one per role.

Each env produces a per-agent trajectory:
\[
\tau_i = \{(o_{i,0}, a_{i,0}, r_{i,0}), (o_{i,1}, a_{i,1}, r_{i,1}), \dots, (o_{i,R-1}, a_{i,R-1}, r_{i,R-1})\},
\]
where there is exactly one transition per “cycle” (one full pass over all agents), so each agent takes \(R\) actions total.

### Turn-taking (the coordinator)

`MultiAgentCoordinator` (in `coordinator.py`) maintains a shared `ConversationState`:

- `current_turn`: global turn index \(t \in \{0, 1, \dots, T_{\max}\}\),
- `current_agent_id = current_turn mod N`,
- `agent_responses`: list of parsed responses, ordered by global turn,
- `done`: whether the conversation has reached `max_turns`.

Turn order is enforced via an `asyncio.Condition`. Although `rl/rollouts.py` launches all env rollouts concurrently (`asyncio.gather`), each env blocks in `wait_for_turn(...)` until it is that agent’s turn, so only one agent effectively acts at a time.

## Prompting and Parsing (Exact Contract)

### What is sent to the model

Each env step constructs a two-message prompt:

- system: `AGENT_SYSTEM_PROMPT.format(agent_id=i)` or `VERIFIABLE_AGENT_SYSTEM_PROMPT.format(agent_id=i)`
- user: `get_observation_string()` (question + some amount of history + per-turn instruction)

The system prompt and user prompt both require the model to output exactly three XML tags, in order:

1. `<solution>...</solution>`
2. `<evaluation>...</evaluation>`
3. `<comparison>...</comparison>`

### Parsing behavior (`parse_agent_response`)

`parse_agent_response` (in `prompts.py`) is designed to be robust to common model wrappers and truncations:

1) Normalization:
   - Strips leading/trailing whitespace.
   - Removes a leading fenced block marker (e.g. ```xml) and a trailing ``` fence if present.
   - Extracts and removes `<think>...</think>` blocks (case-insensitive); the removed content is stored as `ParsedResponse.thinking` for logging/debugging.

2) Preferred parse path:
   - It searches for the **last** complete triple-tag block (`<solution>...<evaluation>...<comparison>...</comparison>`) using a multiline regex that requires the tags to start at the beginning of a line (or start-of-string).
   - If found, that last full block is used.

3) Fallback parse path:
   - If the full block is missing (e.g., max_tokens cutoff), it tries to extract each tag separately (again preferring the last match).
   - Missing/incomplete tags become placeholder strings like `[INCOMPLETE] ...` or `[PARSE_ERROR: Missing <solution> tag]`.

4) Comparison extraction:
   - From the `<comparison>` text it extracts all occurrences matching:
     \[
     \texttt{Agent\\s+(\\d+)\\s*([><])\\s*Agent\\s+(\\d+)}.
     \]
   - Any comparison that includes the author (self-comparison) is dropped and counted in `ParsedResponse.self_comparisons_dropped`.
   - The result is `comparisons: list[tuple[int, str, int]]`, where each entry is `(agent_a_id, op, agent_b_id)` and `op ∈ {">","<"}`.

Important: comparisons are extracted even if the agent includes explanatory prose around them; the extractor matches substrings anywhere in the `<comparison>` block.

## Observation / History Formatting

### Non-verifiable history (`env.py`)

Non-verifiable debate context is built from `ParsedResponse` objects and rendered turn-by-turn in `_format_turns(...)`. The prompt includes the *question* and then a “Previous turns of conversation” block.

Non-verifiable debates use the same history window semantics as the verifiable env:

- `history_turns < 0` → include all prior turns,
- `history_turns = 0` → include no prior turns,
- `history_turns = K > 0` → include only the last `K` prior turns.

If `summarize_history=True`, the entire history string (as constructed) is summarized by an auxiliary OpenRouter model (see below).

### Verifiable history (`verifiable_env.py`)

Verifiable debates implement the “expected” semantics:

- `history_turns < 0` → show all turns,
- `history_turns == 0` → show no turns,
- `history_turns > 0` → show last `history_turns` turns.

Turn numbering is preserved via a `start_offset`, so the displayed indices correspond to the original global turn numbers even when history is truncated.

### Optional summarization (shared across envs)

If `summarize_history=True`, `BaseMultiAgentEnvGroupBuilder` creates exactly one shared `OpenRouterMessageCompleter` per group and passes it into each env as `_summarizer_policy`.

- The summarizer uses `SUMMARIZER_SYSTEM_PROMPT` from `prompts.py`.
- It requires `OPENROUTER_API_KEY` to be set.
- The summarization happens inside `BaseMultiAgentDebateEnv._summarize(...)` and returns plain text which is then used as the “history” chunk in the next turn’s observation string.

## Sampling Stop Conditions

The recipe uses a debate-specific stop marker:

```
</comparison>
```

`get_debate_stop_condition(renderer)` (in `utils.py`) makes this renderer-compatible:

- if the renderer uses token-id stop sequences (`list[int]`), it returns the renderer-provided stop tokens unchanged,
- if the renderer uses string stop sequences (`list[str]`), it appends `</comparison>` (deduplicated).

The OpenRouter self-play script (`inference.py`) always passes `stop=["</comparison>"]` directly to the chat API.

## Reward Shaping (Current Implementation)

### Key design choice: rewards are step-wise and assigned post-hoc

During env stepping (`BaseMultiAgentDebateEnv.step`), the immediate per-step reward is always `0.0`.

After all rollouts complete, the group builder mutates the collected trajectories in-place using the full conversation transcript (`ParsedResponse` list) to assign **step-wise rewards**. There is no additional "final reward" in this recipe: the group builder returns final rewards of `0.0` for every agent, and all learning signal comes from the step rewards.

### Reward system configuration

The reward system is controlled by two configuration flags (`base_env.py`):

- `enable_reward_decay` (default: `True`): if enabled, distributes accumulated rewards across all agent steps using exponential decay weights; if disabled, assigns all accumulated reward to the agent's final step (legacy behavior).
- `enable_format_penalty` (default: `True`): if enabled, penalizes agents who fail to produce valid comparisons in eligible turns.

Constants (defined in `base_env.py`):

- `REWARD_DECAY_GAMMA = 0.7`: exponential decay factor for distributing rewards.
- `FORMAT_PENALTY = -0.5`: penalty applied per missing comparison.
- `FORMAT_EXEMPT_TURNS = 2`: turns 0 and 1 are exempt from format checking.

### Reward computation pipeline

The reward shaper (`BaseMultiAgentEnvGroupBuilder._populate_stepwise_rewards`) proceeds in five steps:

#### Step 1: Accumulate comparison rewards

For one debate instance (one question/problem), define the set of authored comparison events:
\[
\mathcal{C} = \{(u, t, a, \operatorname{op}, b)\},
\]
where:

- \(t\) is the global turn index of the authored response (0-based),
- \(u\) is the author agent id of that response,
- \(a,b\) are the two compared agents referenced in the comparison text,
- \(\operatorname{op} \in \{>,<\}\).

These events come from `ParsedResponse.comparisons`, which already removed self-comparisons at parse time (author \(u\) not allowed to appear in the extracted pairs).

Events are ignored if:

- \(a\) or \(b\) is out of bounds (`0 <= agent_id < N`),
- \(a=b\),
- \(\operatorname{op} \notin \{>,<\}\),
- one of \(a\) or \(b\) has not yet acted at least once *before* turn \(t\) (details below).

Each agent maintains a running **comparison reward accumulator** \(\rho_i^\text{comp}\) initialized to 0. For each valid comparison \((u, t, a, \operatorname{op}, b)\):

- If \(\operatorname{op} = >\): \(\rho_a^\text{comp} \mathrel{+}= 1\), \(\rho_b^\text{comp} \mathrel{-}= 1\).
- If \(\operatorname{op} = <\): \(\rho_a^\text{comp} \mathrel{-}= 1\), \(\rho_b^\text{comp} \mathrel{+}= 1\).

Count \(C_\text{valid}\) = number of valid comparisons processed.

#### Step 2: Accumulate format penalties (if enabled)

If `enable_format_penalty=True`, each agent maintains a **format penalty accumulator** \(\rho_i^\text{fmt}\) initialized to 0.

For each turn \(t \ge \texttt{FORMAT\_EXEMPT\_TURNS}\):
- Let \(u\) be the author of turn \(t\).
- If `ParsedResponse.comparisons` for turn \(t\) is empty: \(\rho_u^\text{fmt} \mathrel{+}= \texttt{FORMAT\_PENALTY}\).

Count \(M\) = number of missing comparisons (turns with empty comparison lists).

#### Step 3: Normalize accumulated rewards

Let \(T_\text{eligible} = \max(0, T_\text{total} - \texttt{FORMAT\_EXEMPT\_TURNS})\) be the number of turns eligible for format checking.

Define normalization factors:
\[
\alpha_\text{comp} = \begin{cases}
1/C_\text{valid} & \text{if } C_\text{valid} > 0, \\
1 & \text{otherwise},
\end{cases}
\quad
\alpha_\text{fmt} = \begin{cases}
1/T_\text{eligible} & \text{if } T_\text{eligible} > 0, \\
1 & \text{otherwise}.
\end{cases}
\]

For each agent \(i\), compute the **normalized total reward**:
\[
\hat{r}_i = \rho_i^\text{comp} \cdot \alpha_\text{comp} + \rho_i^\text{fmt} \cdot \alpha_\text{fmt}.
\]

#### Step 4: Distribute rewards to trajectory steps

Let agent \(i\)'s trajectory have \(R_i\) steps (transitions). If \(R_i = 0\), skip this agent.

**If `enable_reward_decay=True`** (default):

Compute exponential decay weights \(w_{i,s}\) for \(s \in \{0, 1, \dots, R_i-1\}\):
\[
w_{i,s} = \frac{\gamma^{R_i - 1 - s}}{\sum_{j=0}^{R_i-1} \gamma^{R_i - 1 - j}},
\]
where \(\gamma = \texttt{REWARD\_DECAY\_GAMMA}\) (0.7 by default).

The normalized weight vector satisfies \(\sum_{s=0}^{R_i-1} w_{i,s} = 1\), with \(w_{i,0}\) (earliest step) receiving the most decay and \(w_{i,R_i-1}\) (latest step) receiving the least decay.

Assign step rewards:
\[
r_{i,s} \mathrel{+}= \hat{r}_i \cdot w_{i,s}.
\]

**If `enable_reward_decay=False`** (legacy mode):

Assign all accumulated reward to the final step:
\[
r_{i,R_i-1} \mathrel{+}= \hat{r}_i.
\]

#### Step 5: Return summary metrics

The reward shaper returns:

- `stepwise_comparisons_used`: \(C_\text{valid}\), the number of valid comparison events processed.
- `missing_comparisons`: \(M\), the number of turns (after exemption period) with no valid comparisons.

### Comparison attribution: "Most recent action before the comparison"

Each agent \(i\) has one transition per cycle. The reward shaper attributes each comparison to the compared agents' **most recent step strictly before** the current authored turn. This is used only to validate that both agents have acted (step 1 above checks `agent_a_step_idx >= 0` and `agent_b_step_idx >= 0`); the actual reward distribution happens in step 4.

Concretely, the shaper computes:

- `agent_a_step_idx = get_step_idx_before_turn(a, turn_idx=t, num_agents=N)`
- `agent_b_step_idx = get_step_idx_before_turn(b, turn_idx=t, num_agents=N)`

where `get_step_idx_before_turn` (in `utils.py`) implements:

1) Find the most recent global turn index \(t' < t\) at which agent \(i\) acted.
2) Convert that global turn index to a per-agent step index via:
\[
s = \left\lfloor \frac{t'}{N} \right\rfloor.
\]

If an agent has not acted before \(t\), the function returns \(-1\), and the comparison event is skipped.

#### Worked example (3 agents, 2 cycles)

Let \(N=3\). Global turns proceed in fixed order and "cycle" means one full pass over all agents:

| Global turn \(t\) | Acting agent | Cycle \(c=\lfloor t/N \rfloor\) | That agent's step index |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 0 |
| 2 | 2 | 0 | 0 |
| 3 | 0 | 1 | 1 |
| 4 | 1 | 1 | 1 |
| 5 | 2 | 1 | 1 |

Now suppose at **turn \(t=4\)** (Agent 1's response) the `<comparison>` text contains `Agent 0 > Agent 2`.

Validation checks pass (both agents have acted), so:
- Comparison accumulator: \(\rho_0^\text{comp} \mathrel{+}= 1\), \(\rho_2^\text{comp} \mathrel{-}= 1\).
- \(C_\text{valid}\) increments by 1.

Similarly, at **turn \(t=5\)** (Agent 2's response), if it writes `Agent 1 < Agent 0`:
- Comparison accumulator: \(\rho_1^\text{comp} \mathrel{-}= 1\), \(\rho_0^\text{comp} \mathrel{+}= 1\).
- \(C_\text{valid}\) increments by 1.

After all turns, suppose \(C_\text{valid}=2\), \(T_\text{eligible}=4\), \(M=0\), and agent 0 has 2 steps. With `enable_reward_decay=True` and \(\gamma=0.7\):

\[
\hat{r}_0 = 2 \cdot (1/2) = 1.0,
\]
\[
w_{0,0} = \frac{0.7^1}{0.7 + 1.0} \approx 0.412, \quad w_{0,1} = \frac{1.0}{1.7} \approx 0.588,
\]
\[
r_{0,0} = 1.0 \cdot 0.412 = 0.412, \quad r_{0,1} = 1.0 \cdot 0.588 = 0.588.
\]

### Total return per agent

Because final rewards are always `0.0`, each agent's total return for the debate is:
\[
R_i = \sum_{s=0}^{R_i-1} r_{i,s}.
\]

If no valid comparisons occur and format penalties are disabled, all \(r_{i,s}=0\) and all returns are 0.

## RL Advantage Computation and Token-Level Objective (How Rewards Become Gradients)

This section is implemented outside the recipe directory (generic RL code), but it is part of the end-to-end behavior.

### Advantage: centered within each debate group

The RL pipeline computes advantages from total per-trajectory returns (step rewards + final reward) and centers them within each group:
\[
A_i = R_i - \frac{1}{N}\sum_{j=0}^{N-1} R_j.
\]

This is `tinker_cookbook/rl/data_processing.py::compute_advantages(...)`.

If all agents receive identical returns (e.g., no comparisons), all advantages are 0 and that group contributes no policy gradient (the trainer may warn and keep a singleton group).

### Trajectory → token-level training data

`tinker_cookbook/rl/data_processing.py::trajectory_to_data(...)` converts each trajectory into one or more `tinker.Datum`s:

- It merges consecutive (observation, action) pairs into a single sequence when each new observation is a strict prefix-extension of the prior observation+action context.
- If the observation is not a prefix-extension (possible when history formatting changes due to truncation or summarization), it closes the current datum and starts a new one.

For each datum, it builds per-token arrays:

- `logprobs`: sampler logprobs for action tokens, 0 for observation tokens,
- `advantages`: broadcasts the trajectory-level advantage \(A_i\) across action tokens, 0 for observation tokens,
- `mask`: 1 for action tokens, 0 for observation tokens (used locally for diagnostics).

### Optimization objective

Training uses Tinker’s built-in `loss_fn="importance_sampling"` (see `tinker_cookbook/rl/train.py`). Conceptually it is a token-level policy-gradient objective with an importance correction between the sampler policy \(q\) and the current policy \(\pi_\theta\).

## Verifiable Math Variant (What Is Different)

The verifiable environment (`verifiable_env.py`) requires that each `<solution>` contains a final answer in `\\boxed{...}` format. The recipe supports:

- **Training mode (`is_training=True`)**:
  - rewards: still *comparison-derived step rewards* (identical shaping rule as above),
  - metrics: logs per-agent and multi-agent aggregation metrics:
    - Per-agent metrics (one value per agent):
      - `train_<dataset>/format`: fraction of agent's responses with valid format (not `[INCOMPLETE]` or `[PARSE_ERROR]`),
      - `train_<dataset>/correct`: 1.0 if agent's latest response is correct, 0.0 otherwise.
    - Multi-agent aggregation metrics (same value for all agents in the group):
      - `train_<dataset>/pass@N`: 1.0 if at least one agent produced a correct answer (best-of-N),
      - `train_<dataset>/avg@N`: mean correctness across all N agents,
      - `train_<dataset>/cons@N`: 1.0 if more than half of the agents produced correct answers (majority vote).

- **Evaluation mode (`is_training=False`)**:
  - `eval_mode="debate"`: multi-turn debate, then compute `format`/`correct` from each agent's latest parsed solution.
  - `eval_mode="direct"`: a separate single-turn env `DirectMathEvaluationEnv` that prompts once and grades the response text.

### Multi-agent aggregation metrics (training only)

For a debate group with \(N\) agents, let \(y_i \in \{0,1\}\) denote whether agent \(i\)'s latest response is correct. Define:

- **pass@N** (optimistic best-of-N):
  \[
  \text{pass@N} = \begin{cases}
  1 & \text{if } \exists i: y_i = 1, \\
  0 & \text{otherwise}.
  \end{cases}
  \]

- **avg@N** (mean accuracy):
  \[
  \text{avg@N} = \frac{1}{N} \sum_{i=0}^{N-1} y_i.
  \]

- **cons@N** (majority consensus):
  \[
  \text{cons@N} = \begin{cases}
  1 & \text{if } \sum_{i=0}^{N-1} y_i > N/2, \\
  0 & \text{otherwise}.
  \end{cases}
  \]

These metrics are computed in `VerifiableMultiAgentEnvGroupBuilder._compute_training_rewards` (`verifiable_env.py:375-397`) and logged for every agent in the group (each agent receives the same aggregation metric values).

Correctness grading uses `tinker_cookbook/recipes/math_rl/math_grading.py` via `safe_grade(...)`, with a configurable `grader` (`"sympy"` or `"math_verify"`) and `grade_timeout`.

## Transcript Logging (Qualitative Debugging)

`utils.py` provides logtree helpers:

- `log_debate_transcript(coordinator)`: logs per-turn system prompt, observation string, parsed fields, and raw response.
- `log_debate_evaluation_final_solutions(...)` and `log_direct_evaluation(...)`: log verifiable evaluation details.

These are enabled via `log_full_transcript` and logtree’s outer logging context (e.g., the RL trainer’s `num_groups_to_log`).

## Notable Edge Cases / Non-Obvious Behaviors

- **Non-verifiable `history_turns=0` means "no history".** The prompt includes only the question and the per-turn instruction.
- **Reward accumulation and distribution are decoupled (since reward decay was introduced).** Comparisons accumulate into per-agent totals, which are then normalized and distributed across all agent steps with exponential decay weights. This differs from the legacy behavior (pre-decay) where each comparison directly updated a single step.
- **Reward normalization scales by episode size.** The comparison normalization factor \(\alpha_\text{comp} = 1/C_\text{valid}\) means that an episode with 10 comparisons contributes the same total magnitude of learning signal as an episode with 2 comparisons (all else equal). Similarly, format penalties scale by the number of eligible turns.
- **Format penalties apply only after turn 1 (0-indexed).** Turns 0 and 1 are exempt from format checking (`FORMAT_EXEMPT_TURNS=2`), allowing agents to warm up before penalties are applied.
- **Comparisons are substring-matched.** Only patterns of the form `Agent <int> > Agent <int>` (or `<`) are recognized; ties are not represented.
- **Self-comparisons are filtered at parse time.** They do not appear in `ParsedResponse.comparisons`, and the count is tracked in `ParsedResponse.self_comparisons_dropped`.
- **Summarization changes observation prefix structure.** When history is summarized, later observations may no longer be a prefix-extension of earlier observation+action sequences, causing `trajectory_to_data` to split into multiple data items.
- **Multi-agent aggregation metrics (pass@N, avg@N, cons@N) are logged identically for all agents in a group.** These are group-level metrics, not per-agent metrics. Each agent in the group receives the same value for these metrics in their logged output.
