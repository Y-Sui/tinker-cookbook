# Multi-Agent Debate: Current Implementation (Scope → Details → Math)

This note documents the *current* behavior (including non-obvious edge cases) of the multi-agent debate recipe as implemented under `tinker_cookbook/recipes/multi_agent_debate/`.

It is written to be copy/paste-able as “implementation details / methods” for a paper, but it intentionally stays faithful to the code, not an idealized design.

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

After all rollouts complete, the group builder mutates the collected trajectories in-place using the full conversation transcript (`ParsedResponse` list) to assign **step-wise rewards**. There is no additional “final reward” in this recipe: the group builder returns final rewards of `0.0` for every agent, and all learning signal comes from the step rewards.

### Comparison events

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

### Which step gets credited? “Most recent action before the comparison”

Each agent \(i\) has one transition per cycle. The reward shaper attributes each comparison to the compared agents’ **most recent step strictly before** the current authored turn.

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

Let \(N=3\). Global turns proceed in fixed order and “cycle” means one full pass over all agents:

| Global turn \(t\) | Acting agent | Cycle \(c=\lfloor t/N \rfloor\) | That agent’s step index |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 0 |
| 2 | 2 | 0 | 0 |
| 3 | 0 | 1 | 1 |
| 4 | 1 | 1 | 1 |
| 5 | 2 | 1 | 1 |

Now suppose at **turn \(t=4\)** (Agent 1’s response) the `<comparison>` text contains `Agent 0 > Agent 2`.

- For Agent 0: most recent action strictly before \(t=4\) was at \(t'=3\), so credited step is \(s_0=\lfloor 3/3 \rfloor = 1\).
- For Agent 2: most recent action strictly before \(t=4\) was at \(t'=2\), so credited step is \(s_2=\lfloor 2/3 \rfloor = 0\).

So this single comparison updates **Agent 0 step 1** by `+1` and **Agent 2 step 0** by `-1`. This “cross-cycle” attribution happens whenever the author is early in a cycle and one compared agent has not yet taken their turn in that same cycle.

Similarly, at **turn \(t=5\)** (Agent 2’s response), if it writes `Agent 1 < Agent 0`:

- Agent 1’s most recent action before \(t=5\) is \(t'=4\) → credited step \(s_1=\lfloor 4/3 \rfloor = 1\) (gets `-1`).
- Agent 0’s most recent action before \(t=5\) is \(t'=3\) → credited step \(s_0=\lfloor 3/3 \rfloor = 1\) (gets `+1`).

### Reward update rule

For a valid comparison event \((u, t, a, \operatorname{op}, b)\) with computed step indices \(s_a, s_b \ge 0\), the shaper updates the rewards stored in the trajectories’ transitions:

- If \(a > b\):
  \[
  r_{a,s_a} \mathrel{+}= 1,\quad r_{b,s_b} \mathrel{-}= 1.
  \]
- If \(a < b\):
  \[
  r_{a,s_a} \mathrel{-}= 1,\quad r_{b,s_b} \mathrel{+}= 1.
  \]

This is exactly `BaseMultiAgentEnvGroupBuilder._populate_stepwise_rewards(...)`.

### Total return per agent

Because final rewards are always `0.0`, each agent’s total return for the debate is:
\[
R_i = \sum_{s=0}^{R-1} r_{i,s}.
\]

If no valid comparisons occur, all \(r_{i,s}=0\) and all returns are 0.

### Logging for reward shaping

The reward shaper returns a single summary metric:

- `stepwise_comparisons_used`: sum of absolute non-zero reward increments assigned across all transitions.

This is computed by scanning all trajectories after mutation and summing `abs(transition.reward)` whenever a transition reward is non-zero. Note that if multiple comparisons hit the same step with opposite signs (e.g., `+1` then `-1`), they can cancel to 0 and will not contribute to this metric.

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
  - metrics: logs `train_<dataset>/format`, `train_<dataset>/correct`, and `train_<dataset>/pass@N`, computed using the latest response per agent (correctness) and the fraction of valid formatted responses across all of that agent’s turns (format).

- **Evaluation mode (`is_training=False`)**:
  - `eval_mode="debate"`: multi-turn debate, then compute `format`/`correct` from each agent’s latest parsed solution.
  - `eval_mode="direct"`: a separate single-turn env `DirectMathEvaluationEnv` that prompts once and grades the response text.

Correctness grading uses `tinker_cookbook/recipes/math_rl/math_grading.py` via `safe_grade(...)`, with a configurable `grader` (`"sympy"` or `"math_verify"`) and `grade_timeout`.

## Transcript Logging (Qualitative Debugging)

`utils.py` provides logtree helpers:

- `log_debate_transcript(coordinator)`: logs per-turn system prompt, observation string, parsed fields, and raw response.
- `log_debate_evaluation_final_solutions(...)` and `log_direct_evaluation(...)`: log verifiable evaluation details.

These are enabled via `log_full_transcript` and logtree’s outer logging context (e.g., the RL trainer’s `num_groups_to_log`).

## Notable Edge Cases / Non-Obvious Behaviors

- **Non-verifiable `history_turns=0` means “no history”.** The prompt includes only the question and the per-turn instruction.
- **Reward is attributed to the most recent step before a comparison, not the final answer.** A ranking made late in the debate credits (or debits) whatever the compared agent last said before that ranking.
- **Comparisons are substring-matched.** Only patterns of the form `Agent <int> > Agent <int>` (or `<`) are recognized; ties are not represented.
- **Self-comparisons are filtered at parse time.** They do not appear in `ParsedResponse.comparisons`, and the count is tracked in `ParsedResponse.self_comparisons_dropped`.
- **Summarization changes observation prefix structure.** When history is summarized, later observations may no longer be a prefix-extension of earlier observation+action sequences, causing `trajectory_to_data` to split into multiple data items.
