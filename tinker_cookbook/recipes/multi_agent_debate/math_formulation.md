# Multi-Agent Debate: Reward + RL Math Formulation

This note formalizes the reward functions and the RL training pipeline implemented by:

- `tinker_cookbook/recipes/multi_agent_debate/env.py`
- `tinker_cookbook/recipes/multi_agent_debate/reward.py`
- `tinker_cookbook/rl/train.py`
- `tinker_cookbook/rl/data_processing.py`

It is intended to match the code’s *actual* behavior (including edge cases).

## Notation

- Agents: \(i \in \{0,1,\dots,N-1\}\) where \(N=\texttt{num\_agents}\).
- Training iteration: \(k \in \{0,1,\dots\}\).
- A *problem* (question) index: \(p\). A training batch contains many questions \(p \in \mathcal{P}_k\).
- A *group* is a single multi-agent episode for a single question \(p\).
- Policy at iteration \(k\): \(\pi_{\theta_k}\).
- A trajectory for agent \(i\) in question \(p\): \(\tau_{p,i}\), consisting of multiple turns (transitions).
- Each transition has:
  - an observation (prompt) token sequence \(o\)
  - an action token sequence \(a\) (the model’s generated tokens for that turn)
  - sampled token logprobs under the sampler policy \(q\): \(\log q(a_t \mid o,a_{<t})\)

## Environment / Episode Structure

For each question \(p\), the environment creates \(N\) per-agent env instances that share a single coordinator (self-play).

- The debate proceeds in *rounds*. Each round is \(N\) turns, one per agent, in a fixed order.
- The episode terminates at the end of a round if either:
  1) every agent outputs `<consensus>YES</consensus>` in that round; or
  2) the maximum number of rounds is reached.

Per-turn reward returned by `MultiAgentDebateEnv.step()` is:

\[
r^{\text{step}}_{p,i,t} =
\begin{cases}
-1 & \text{if response parsing / turn-taking fails} \\
0 & \text{otherwise}
\end{cases}
\]

The *main* learning signal is a **final per-agent group reward** computed by the `EnvGroupBuilder` after the episode finishes.

## Pairwise Comparisons as “Votes”

Each agent produces a `<comparison>` block containing 0 or more lines of the form:

```
Agent A > Agent B
Agent A = Agent B
```

The parser extracts a multiset of comparisons **with authorship**:

\[
\mathcal{C}_p = \{(u, a, \operatorname{op}, b)\}
\]

where \(u\) is the author agent id, across **all agents and all rounds** for question \(p\).

Malformed comparisons are ignored (and counted in metrics):

- agent ids out of range
- \(a=b\)
- operator not in \(\{>,=\}\)

If there are no valid comparisons, rewards default to 0.

## Reward Function 1: Win Rate (Leave-One-Out, `reward_mode=win_rate`)

Define per-agent counters:

- wins: \(W_i\)
- votes/participations: \(V_i\)

Each valid comparison \((u,a,\operatorname{op},b)\) updates, **for each target agent \(i\)**, only if \(u \neq i\) and \(i \in \{a,b\}\):

- If \(\operatorname{op} = >\):
  \[
  W_a \mathrel{+}= 1,\quad
  V_a \mathrel{+}= 1,\quad
  V_b \mathrel{+}= 1
  \]
- If \(\operatorname{op} = =\):
  \[
  W_a \mathrel{+}= \tfrac12,\quad
  W_b \mathrel{+}= \tfrac12,\quad
  V_a \mathrel{+}= 1,\quad
  V_b \mathrel{+}= 1
  \]

The final reward (per target agent \(i\)) is:

\[
r^{\text{win\_rate}}_{p,i} =
\begin{cases}
\dfrac{W_i}{V_i} & V_i > 0 \\
0 & V_i = 0
\end{cases}
\qquad\in [0,1]
\]

Interpretation: “fraction of other-agents’ votes the agent wins (ties count as 0.5)”.

## Reward Function 2: Win Minus Loss (Leave-One-Out, `reward_mode=win_minus_loss`)

Define per-agent counters:

- signed score: \(S_i\)
- matchups/participations: \(M_i\)

Each valid comparison \((u,a,\operatorname{op},b)\) updates, **for each target agent \(i\)**, only if \(u \neq i\) and \(i \in \{a,b\}\):

- Always:
  \[
  M_a \mathrel{+}= 1,\quad
  M_b \mathrel{+}= 1
  \]
- If \(\operatorname{op} = >\):
  \[
  S_a \mathrel{+}= 1,\quad
  S_b \mathrel{-}= 1
  \]
- If \(\operatorname{op} = =\): no score change.

The final reward is:

\[
r^{\text{wml}}_{p,i} =
\begin{cases}
\dfrac{S_i}{M_i} & M_i > 0 \\
0 & M_i = 0
\end{cases}
\qquad\in [-1,1]
\]

Interpretation: “average signed margin per matchup from other agents’ votes” (zero-sum per matchup).

## Advantage Computation (Group-Centered)

The RL training code computes advantages by centering rewards within each group (question):

Let \(r_{p,i}\) be the chosen reward signal (win rate or win-minus-loss). Then:

\[
A_{p,i} = r_{p,i} - \frac{1}{N}\sum_{j=0}^{N-1} r_{p,j}
\]

This makes training invariant to a constant reward shift and implements a simple baseline.

Edge cases:

- If all agents receive the same reward (e.g., no votes → all \(r_{p,i}=0\)), then all advantages are 0 and the policy gradient is 0 for that group.

## Converting Trajectories to Training Data (Token-Level)

Each agent trajectory \(\tau_{p,i}\) is converted into one or more `tinker.Datum` objects.

### Token sequence construction

The code constructs a single “full sequence” by concatenating (observation delta) + (action tokens) turn by turn:

\[
x = o^{(1)} \,\Vert\, a^{(1)} \,\Vert\, o^{(2)} \,\Vert\, a^{(2)} \,\Vert\, \dots
\]

If observations are not strict prefixes of the previous observation+action (non-monotone contexts), the trajectory is split into multiple `Datum`s.

### Per-token fields

For each position \(t\) in the constructed full sequence:

- `logprobs[t]`: sampler logprob \(\log q(x_t \mid x_{<t})\), with zeros filled for observation tokens.
- `advantages[t]`:
  - \(A_{p,i}\) for action tokens
  - 0 for observation tokens
- `mask[t]`:
  - 1 for action tokens
  - 0 for observation tokens

Then the sequence is shifted into next-token prediction form:

- model input tokens are right-shifted: \((x_0, x_1, \dots, x_{T-2})\)
- targets are left-shifted: \((x_1, x_2, \dots, x_{T-1})\)

The server-side loss does **not** consume `mask` (it is stripped before sending); instead, observation tokens have advantage 0, so they contribute 0 to the RL objective.

## RL Objective Used by Training (`loss_fn=importance_sampling`)

By default, `tinker_cookbook/rl/train.py` uses:

- `loss_fn = "importance_sampling"`

Conceptually, this is a per-token policy-gradient objective using data sampled from \(q\) (the sampler policy used to generate trajectories) while updating \(p=\pi_{\theta}\) (the training policy):

\[
\mathcal{L}(\theta)
= -\sum_{p \in \mathcal{P}_k}\sum_{i=0}^{N-1}\sum_{t \in \text{action tokens of }\tau_{p,i}}
\rho_{p,i,t}(\theta)\; A_{p,i}
\]

where the importance ratio is:

\[
\rho_{p,i,t}(\theta)
= \exp\Big(\log \pi_\theta(x_t \mid x_{<t}) - \log q(x_t \mid x_{<t})\Big)
\]

In the fully on-policy case, \(q=\pi_{\theta_k}\) and (approximately) \(\rho \approx 1\).

## Full Training Loop (High-Level)

At each iteration \(k\):

1. **Construct env groups** for a batch of questions \(p \in \mathcal{P}_k\).
2. **Create a sampler policy** \(q\) from the current checkpointed weights.
3. **Roll out** each env group:
   - agents take turns producing XML responses
   - coordinator stores all parsed responses and comparisons
4. **Compute per-agent group reward** \(r_{p,i}\) from all comparisons \(\mathcal{C}_p\).
5. **Compute advantages** \(A_{p,i}\) by centering within the group.
6. **Convert trajectories to token-level training data** (`Datum`s) with per-token `logprobs` and `advantages`.
7. **Submit a Tinker training step**:
   - `forward_backward(..., loss_fn="importance_sampling")`
   - `optim_step(...)`
8. **(Optional)** run evaluation rollouts and log metrics.

## Metrics Produced by the Debate Reward

For each agent \(i\) in a group, the env logs:

- `pairwise_reward`: the chosen reward signal (depends on `reward_mode`)
- `pairwise_reward_is_win_rate`: 1 if `reward_mode=win_rate` else 0
- `pairwise_reward_is_win_minus_loss`: 1 if `reward_mode=win_minus_loss` else 0
- `pairwise_win_rate`: leave-one-out win-rate (even if not used as reward)
- `pairwise_win_minus_loss`: leave-one-out win-minus-loss (even if not used as reward)
- `pairwise_total_votes`, `pairwise_malformed`, `pairwise_any_votes`

Because these are all numeric, they can be safely aggregated by `dict_mean()` during evaluation.
