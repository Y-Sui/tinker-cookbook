# Multi-Agent Debate: Implementation + Math Formulation

This note formalizes the *exact* behavior (including edge cases) of the multi-agent debate recipe and its RL training loop, as implemented in:

- `tinker_cookbook/recipes/multi_agent_debate/env.py`
- `tinker_cookbook/recipes/multi_agent_debate/reward.py`
- `tinker_cookbook/recipes/multi_agent_debate/prompts.py`
- `tinker_cookbook/rl/rollouts.py`
- `tinker_cookbook/rl/data_processing.py`
- `tinker_cookbook/rl/train.py`

It is written to be directly reusable as “implementation details / methods” material for a paper.

## Notation

- Agents (roles): \(i \in \{0,1,\dots,N-1\}\) where \(N=\texttt{num\_agents}\).
- Questions / problems: \(p\). A training batch contains many questions \(p \in \mathcal{P}_k\) at iteration \(k\).
- A *group* \(p\) is one multi-agent episode (one question) containing \(N\) agent trajectories.
- Policy at iteration \(k\): \(\pi_{\theta_k}\).
- Sampler policy used to generate rollouts: \(q\). In the on-policy case, \(q \approx \pi_{\theta_k}\).
- For each agent \(i\) and question \(p\), a trajectory \(\tau_{p,i}\) is a list of transitions:
  \[
  \tau_{p,i} = \{(o^{(1)}_{p,i}, a^{(1)}_{p,i}), (o^{(2)}_{p,i}, a^{(2)}_{p,i}), \dots \}
  \]
  where:
  - \(o^{(t)}\) is the observation prompt at that agent’s \(t\)-th turn,
  - \(a^{(t)}\) is the sampled token sequence at that turn.
- Debate rounds: \(m \in \{1,\dots,\texttt{max\_rounds}\}\). Each round contains \(N\) global turns in fixed order \(0\to 1\to\dots\to N-1\).

We also use:

- Final per-agent group reward vector \(r_p \in \mathbb{R}^N\), with component \(r_{p,i}\).
- Advantage vector \(A_p \in \mathbb{R}^N\), with component \(A_{p,i}\).
- Parsed response objects \(R_{p,i,m}\) (agent \(i\)’s response in round \(m\)), which include text fields and extracted comparisons.

## System Decomposition (What Code Maps to What Math)

**(1) Prompting + parsing**
- `prompts.py` defines a strict XML contract and parses model outputs into structured fields.

**(2) Environment + coordinator**
- `env.py` defines:
  - a shared coordinator that enforces global turn order across agents,
  - per-agent envs that produce observations and accept actions,
  - how to construct “debate history” text for the prompt,
  - how to compute the final group reward from all parsed comparisons.

**(3) Rollouts**
- `rl/rollouts.py` runs `asyncio.gather` over agents, but coordination ensures only one agent acts at a time.

**(4) RL data processing**
- `rl/data_processing.py` converts each agent trajectory into one or more `tinker.Datum`s with token-level `logprobs`, `advantages`, and `target_tokens`.

**(5) RL training**
- `rl/train.py` implements the loop and calls Tinker `forward_backward_async(..., loss_fn="importance_sampling")` + `optim_step_async(...)`.

## Output Contract (XML Schema) and Parsing Details

### Required XML tags and order

Each agent’s action is expected to decode to text with *exactly* these tags in this order:

1. `<solution>...</solution>`
2. `<evaluation>...</evaluation>`
3. `<comparison>...</comparison>` (allowed to be `N/A` or empty)
4. `<consensus>YES/NO</consensus>`
5. `<consensus_reason>...</consensus_reason>`

The system prompt also imposes key constraints that affect reward correctness:

- Agents must not include themselves in the `<comparison>` section.
- Agents are asked to “reason before judging” to encourage consistent structure and reduce brittle output formats.

### Parser normalization (robustness to common wrappers)

Before extracting tags, the parser applies normalizations that matter for real model outputs:

- Strips Markdown code fences like:
  ```xml
  ...
  ```
- Removes Qwen-style `<think>...</think>` wrappers (case-insensitive), while preserving inner content.
- If there is preamble text, parsing begins at the first structured tag occurrence.

### Parsing comparisons (votes) with authorship

From the `<comparison>` block, the parser extracts zero or more lines matching:

```
Agent A > Agent B
Agent A = Agent B
```

Each extracted line yields a tuple:
\[
(a,\operatorname{op},b), \quad \operatorname{op}\in\{>,=\}.
\]

**Authorship is attached at submission time** (the coordinator knows which agent produced the response), yielding:
\[
\mathcal{C}_p = \{(u, a, \operatorname{op}, b)\},
\]
where \(u\) is the author agent id. Importantly:

- Any extracted comparison where \(a=u\) or \(b=u\) is dropped (enforcing “no self-comparison”).
- Malformed comparisons are ignored during reward computation (details below).

## Observation Construction (What Each Agent Sees)

Each agent env constructs an observation by rendering two messages:

- system: `AGENT_SYSTEM_PROMPT.format(agent_id=i)`
- user: a “debate context” string that includes the question, plus some recent transcript history.

### History truncation (`history_rounds`)

The env stores parsed responses by round. The history included in the prompt is round-based:

- `history_rounds = -1`: include *all* prior rounds.
- `history_rounds = 0`: include no history.
- `history_rounds = K > 0`: include only the last \(K\) rounds.

Each field (solution/evaluation/comparison/consensus_reason) is clipped to `max_chars_per_field` before insertion. This makes the observation string potentially **non-monotone** across turns (see “trajectory splitting” later).

### Round 1 bootstrapping logic (when comparisons are allowed)

Round 1 is special-cased to avoid “comparisons with no candidates”:

- If no one has spoken yet this round (Agent 0’s first turn):
  - instruct: propose solution; set evaluation and comparison to `N/A`.
- If exactly one agent has spoken (Agent 1’s first turn):
  - instruct: evaluate Agent 0; propose solution; no comparisons.
- Otherwise (Agent 2+ first-turn):
  - instruct: evaluate earlier agents; propose solution;
  - comparisons allowed only among agents who have already responded in the current round (and still excluding self).

From Round 2 onward, agents are instructed to evaluate prior *solutions + evaluations + comparisons* (meta-judging), and to compare other agents (excluding self), typically emphasizing recently available completions.

### Stop conditions (sampling termination)

The environment uses a debate-specific stop marker:

```
</consensus_reason>
```

Stop sequences are made renderer-compatible:

- If the renderer provides token-id stop sequences (token-based renderers), we cannot append strings, so we reuse the renderer stop tokens.
- If the renderer provides string stop sequences (text-based renderers), we append the debate-specific XML close tag (deduplicated).

This prevents “mixed stop types” bugs and ensures sampling terminates after the required fields are emitted.

## Episode Structure and Rewards

### Coordinator and turn-taking

For each question \(p\), self-play training constructs a single coordinator shared by all \(N\) envs. The coordinator stores:

- `current_agent_id` (global turn index modulo \(N\))
- all parsed responses by round
- `done` flag and `consensus_reached` flag

Agents block on a condition variable until it is their turn. Although rollouts run envs concurrently, effective action order is sequential.

### Termination

The episode ends at the end of a round if either:

1) Every agent in that round outputs `<consensus>YES</consensus>`, or
2) Round count reaches `max_rounds`.

### Per-step rewards and failure behavior

The environment returns a per-step reward:
\[
r^{\text{step}}_{p,i,t} =
\begin{cases}
-1 & \text{if parsing/turn-taking fails (and episode aborts)} \\
0 & \text{otherwise}
\end{cases}
\]

On error, the coordinator is aborted (to avoid deadlocks in other env tasks), and the episode is marked done.

### Final group reward

The main learning signal is a final *group reward* \(r_{p,i}\) computed from peer comparisons after the episode finishes. Total return per agent is:
\[
R_{p,i} = \sum_t r^{\text{step}}_{p,i,t} + r_{p,i}.
\]

In typical successful episodes, \(\sum_t r^{\text{step}}_{p,i,t}=0\), so \(R_{p,i}=r_{p,i}\).

## Pairwise Comparisons as Votes (Formal Definition)

Across all agents and all rounds, the set of authored comparisons is:
\[
\mathcal{C}_p = \{(u, a, \operatorname{op}, b)\}.
\]

**Validity checks.** A comparison is considered valid iff:

- \(a,b \in \{0,\dots,N-1\}\),
- \(a \neq b\),
- \(\operatorname{op} \in \{>,=\}\).

Invalid comparisons are discarded and counted as “malformed” in metrics.

If there are no valid comparisons, all final rewards default to 0.

## Reward Function 1: Win Rate (Leave-One-Out, `reward_mode=win_rate`)

This computes a leave-one-out win-rate for each target agent.

Define per-agent counters:

- wins: \(W_i \in \mathbb{R}_{\ge 0}\)
- votes: \(V_i \in \mathbb{R}_{\ge 0}\)

For each valid comparison \((u,a,\operatorname{op},b)\), update **for each target agent \(i\)** only if:

- \(u \neq i\) (leave-one-out), and
- \(i \in \{a,b\}\) (only comparisons involving \(i\) matter to \(i\)’s score).

Then:

- If \(\operatorname{op} = >\), the comparison says “\(a\) beats \(b\)”:
  \[
  W_a \mathrel{+}= 1,\quad V_a \mathrel{+}= 1,\quad V_b \mathrel{+}= 1.
  \]
- If \(\operatorname{op} = =\), the comparison says “\(a\) ties \(b\)”:
  \[
  W_a \mathrel{+}= \tfrac12,\quad W_b \mathrel{+}= \tfrac12,\quad V_a \mathrel{+}= 1,\quad V_b \mathrel{+}= 1.
  \]

The final reward is:
\[
r^{\text{win\_rate}}_{p,i} =
\begin{cases}
\dfrac{W_i}{V_i} & V_i > 0 \\
0 & V_i = 0
\end{cases}
\quad\in[0,1].
\]

Interpretation: “fraction of peer votes the agent wins, with ties = 0.5”.

## Reward Function 2: Win Minus Loss (Leave-One-Out, `reward_mode=win_minus_loss`)

This computes a leave-one-out signed outcome per matchup.

Define per-agent counters:

- signed score: \(S_i \in \mathbb{R}\)
- matchups: \(M_i \in \mathbb{R}_{\ge 0}\)

For each valid comparison \((u,a,\operatorname{op},b)\), update **for each target agent \(i\)** only if:

- \(u \neq i\), and
- \(i \in \{a,b\}\).

Always:
\[
M_a \mathrel{+}= 1,\quad M_b \mathrel{+}= 1.
\]

If \(\operatorname{op} = >\):
\[
S_a \mathrel{+}= 1,\quad S_b \mathrel{-}= 1.
\]

If \(\operatorname{op} = =\), no score change.

Final reward:
\[
r^{\text{wml}}_{p,i} =
\begin{cases}
\dfrac{S_i}{M_i} & M_i > 0 \\
0 & M_i = 0
\end{cases}
\quad\in[-1,1].
\]

Interpretation: “average signed margin per matchup; ties contribute 0”.

## Advantage Computation (Centered Within a Group)

The RL pipeline centers returns within each group (question) to form advantages:
\[
A_{p,i} = R_{p,i} - \frac{1}{N}\sum_{j=0}^{N-1} R_{p,j}.
\]

This acts as a per-group baseline and makes training invariant to adding a constant to all agent rewards for a question.

Key edge cases:

- If all returns are identical (e.g., no comparisons, no errors), then all \(A_{p,i}=0\) and that group produces zero policy gradient.
- If exactly one agent gets a parse error, its extra \(-1\) step reward creates variance in \(R_{p,i}\), producing non-zero \(A_{p,i}\) even when pairwise rewards are all 0.

## Converting Trajectories to Token-Level Training Data (`Datum`)

Each agent trajectory \(\tau_{p,i}\) is converted into one or more `tinker.Datum` objects. Each `Datum` carries:

- a `model_input` (token chunks)
- `loss_fn_inputs` with token-aligned vectors:
  - `target_tokens`
  - `logprobs` (under sampler \(q\))
  - `advantages` (scalar advantage broadcast across action tokens)
  - `mask` (1 for action tokens, 0 for observation tokens; used for local metrics)

### Full-sequence assembly with prefix checks

Each trajectory is a list of (observation, action-with-logprobs) pairs. The code constructs a “full sequence” by appending only the observation *delta* when the new observation is a strict extension of the previous context.

Let \(x\) be the running “full sequence” for the current `Datum` being built.

- If \(x\) is empty: set delta observation to the whole observation \(o\).
- Else if \(x\) is a prefix of the new observation \(o\): delta observation is the suffix \(o_{|x|:}\).
- Else: finalize the current `Datum` and start a new one from scratch.

This is necessary because the environment sometimes produces observations that are not prefixes of previous observation+action, e.g. due to:

- round-dependent instruction templates,
- history truncation (`history_rounds`),
- per-field clipping (`max_chars_per_field`).

### Per-token fields

For each `Datum` full sequence \(x\), define token-level arrays aligned to \(x\):

- sampler logprobs:
  - observation tokens get 0 (not sampled),
  - action tokens get the sampler logprobs from `TokensWithLogprobs`.
- advantages:
  - observation tokens get 0,
  - action tokens get the trajectory-level advantage \(A_{p,i}\) (constant per token).
- mask:
  - observation tokens get 0,
  - action tokens get 1.

Then \(x\) is shifted into next-token prediction form:

- input tokens: \((x_0, x_1, \dots, x_{T-2})\)
- target tokens: \((x_1, x_2, \dots, x_{T-1})\)

The logprobs/advantages/mask arrays are sliced to match this shifted alignment, dropping the first position.

### Mask handling

The RL training call strips `mask` before sending data to Tinker’s built-in loss, because the built-in loss does not consume it. Mask is still retained locally for diagnostics:

- `optim/kl_sample_train_*` and `optim/entropy` are computed over action tokens only (where mask==1).

## RL Objective Used (`loss_fn=importance_sampling`)

The default Tinker loss for RL is `importance_sampling`. At a high level, it implements a token-level policy-gradient objective for data collected under a sampler policy \(q\), while optimizing a training policy \(p=\pi_\theta\).

Conceptually:
\[
\mathcal{L}(\theta)
= -\sum_{p \in \mathcal{P}_k}\sum_{i=0}^{N-1}\sum_{t \in \text{action tokens of }\tau_{p,i}}
\rho_{p,i,t}(\theta)\; A_{p,i},
\]
where:
\[
\rho_{p,i,t}(\theta) =
\exp\Big(\log \pi_\theta(x_t \mid x_{<t}) - \log q(x_t \mid x_{<t})\Big).
\]

In the fully on-policy setting, \(q \approx \pi_{\theta_k}\) and \(\rho \approx 1\).

## Training Loop (Detailed)

### Batch structure

At iteration \(k\), the dataset yields a batch of `EnvGroupBuilder`s:

- batch size = number of questions per iteration = `batch_size`
- for each question \(p\), the group builder will create \(N\) env instances (one per role)

Thus, each training iteration samples \( |\mathcal{P}_k| \times N \) trajectories.

### Sampling policy and rollouts

The RL loop creates a `TinkerTokenCompleter` from a `SamplingClient` (sampler weights), then calls `do_group_rollout(builder, policy)` for each group.

Each group rollout returns:

- `trajectories_G`: list of \(N\) per-agent trajectories,
- `final_rewards_G`: list of \(N\) final group rewards,
- `metrics_G`: list of \(N\) per-trajectory metric dicts (includes debate reward diagnostics).

### Advantage + data assembly

For each trajectory group:

1) compute total returns per agent:
   \[
   R_{p,i} = \sum_t r^{\text{step}}_{p,i,t} + r_{p,i}
   \]
2) compute centered advantages \(A_{p,i}\)
3) convert trajectories to `Datum`s
4) concatenate all `Datum`s across all groups into a single training list \(D\)

### Optimization and pipelining

Training uses pipelined asynchronous calls to keep compute aligned to Tinker’s worker “clock cycles”:

1) enqueue `forward_backward_async(D_chunk)`
2) enqueue `optim_step_async`
3) while awaiting results of current chunk, enqueue the next chunk’s `forward_backward_async` and `optim_step_async`

This does not change the objective; it reduces idle time.

### Checkpointing and sampling-client refresh

At the end of an iteration, the loop obtains a fresh sampling client:

- either by saving a checkpoint when `save_every` triggers, or
- by calling `save_weights_and_get_sampling_client_async()`

This is crucial because existing sampling clients do not automatically pick up new weights.

### Evaluation mode: role-averaged vs fixed opponents

If a test dataset is built:

- training still uses self-play (all roles played by the current policy),
- evaluation uses `self_play=False`:
  - for each question, create \(N\) separate debates,
  - in debate \(i\), the learned policy controls role \(i\),
  - fixed opponent policies control the other roles.

This yields “role-averaged” evaluation, probing whether improvements are robust to role assignment.

## Metrics and Artifacts (What to Log and How to Interpret)

### Debate reward metrics (per trajectory; then averaged)

For each agent \(i\), the group reward computation emits:

- `pairwise_reward` (the selected reward value)
- `pairwise_reward_is_win_rate` (indicator)
- `pairwise_reward_is_win_minus_loss` (indicator)
- `pairwise_total_votes` (number of valid comparisons observed)
- `pairwise_malformed` (number of discarded malformed comparisons)
- `pairwise_any_votes` (1 if any valid comparisons exist, else 0)

### Per-step env metrics (per transition; then averaged)

- `consensus_reached` (1 if coordinator has reached group consensus by this step)
- `round` (current round index)
- `parse_error` (1 if parsing/turn-taking failed for that transition)

### RL aggregate metrics (computed from trajectory groups)

Common aggregate metrics include:

- token/episode stats:
  - `env/all/ac_tokens_per_turn`, `env/all/ob_tokens_per_turn`, `env/all/turns_per_episode`
  - `env/all/total_episodes`, `env/all/total_turns`
- reward stats:
  - `env/all/reward/total` (mean total return across all trajectories)
  - `env/all/by_group/frac_mixed` (fraction of groups with non-uniform returns across roles)

### Optimization diagnostics

Computed over action tokens only:

- `optim/kl_sample_train_v1`, `optim/kl_sample_train_v2`
- `optim/entropy`

### Logtree transcripts (qualitative debugging)

When logtree logging is enabled and `log_full_transcript=True`, each group can include a full transcript dump:

- question
- per round:
  - per agent: system prompt, observation string, extracted fields, raw output text

This is the most direct way to diagnose reward issues such as:

- low `pairwise_total_votes` (agents not producing comparisons),
- high `pairwise_malformed` (bad formatting),
- reward hacking (e.g., verbose but low-quality solutions receiving votes),
- prompt drift caused by history truncation/clipping.
