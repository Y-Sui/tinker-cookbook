# Multi-Agent Debate RL Algorithm Flow

This document provides a visual flowchart of the complete training algorithm for the multi-agent debate system.

## Overview

**Core Principle**: A single unified policy θ plays multiple agent roles in self-play debate, learning both generation and evaluation capabilities simultaneously.

**Key Features**:
- Single policy for all agents (no separate generator/judge policies)
- Step-wise rewards computed post-hoc from peer comparisons
- Turn-by-turn interleaved generation and evaluation
- Advantages centered across all steps in the group

---

## Complete Training Flow

```
╔═══════════════════════════════════════════════════════════════════╗
║                   Training Iteration N                            ║
║                   Policy State: θ_N                               ║
╚═══════════════════════════════════════════════════════════════════╝
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│ STEP 1: Sample Batch of Questions                                │
├───────────────────────────────────────────────────────────────────┤
│ Code: dataset.get_batch(i_batch)                                 │
│ Output: [Q1, Q2, ..., Q_batch_size]                              │
│                                                                   │
│ Example: batch_size=16                                            │
│   → 16 questions/problems for this iteration                     │
└───────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│ STEP 2: For Each Question, Run Debate Episode (Parallel)         │
├───────────────────────────────────────────────────────────────────┤
│ Code: await asyncio.gather(                                      │
│           *[do_group_rollout(builder, policy) for builder in P]  │
│       )                                                           │
│                                                                   │
│ Each episode uses the SAME policy θ_N for all agents            │
└───────────────────────────────────────────────────────────────────┘
                              ↓
        ╔═══════════════════════════════════════════════╗
        ║  Single Debate Episode (for one question)    ║
        ╚═══════════════════════════════════════════════╝
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│ STEP 2.1: Create Environment Group                               │
├───────────────────────────────────────────────────────────────────┤
│ Code: env_group_builder.make_envs()                              │
│                                                                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Shared Coordinator:                                         │ │
│ │   - question: "Solve: 2x + 3 = 11"                         │ │
│ │   - num_agents: 3                                           │ │
│ │   - max_turns: 9 (3 agents × 3 rounds)                     │ │
│ │   - state: ConversationState (tracks turn order)           │ │
│ │   - condition: asyncio.Condition (synchronization)         │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 3 Environment Instances (share coordinator):                │ │
│ │   ├─ MultiAgentDebateEnv(agent_id=0, coordinator=shared)   │ │
│ │   ├─ MultiAgentDebateEnv(agent_id=1, coordinator=shared)   │ │
│ │   └─ MultiAgentDebateEnv(agent_id=2, coordinator=shared)   │ │
│ └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│ STEP 2.2: Concurrent Agent Rollouts (Synchronized by Coordinator)│
├───────────────────────────────────────────────────────────────────┤
│ Code: await asyncio.gather(*[do_single_rollout(policy, env)     │
│                               for env in envs])                   │
│                                                                   │
│ Note: Although launched concurrently, agents take turns due to   │
│       coordinator.wait_for_turn() blocking                       │
└───────────────────────────────────────────────────────────────────┘

    ┌─────────────────────┬─────────────────────┬─────────────────────┐
    │    Agent 0          │    Agent 1          │    Agent 2          │
    │  (Policy θ_N)       │  (Policy θ_N)       │  (Policy θ_N)       │
    └─────────────────────┴─────────────────────┴─────────────────────┘
            │                     │                     │
            │ wait_for_turn()     │ wait_for_turn()     │
            │ (blocked)           │ (blocked)           │ (ready: id=0)
            ↓                     │                     │
    ╔═══════════════════╗         │                     │
    ║ Turn 0 (Round 0)  ║         │                     │
    ╠═══════════════════╣         │                     │
    ║ Agent 0 acts      ║         │                     │
    ╚═══════════════════╝         │                     │
            │                     │                     │
    ┌───────────────────────────────────────────────────┐
    │ Observation:                                      │
    │   System: "You are Agent 0... [persona]"         │
    │   User: "Question: 2x+3=11\nFirst turn, no hist" │
    └───────────────────────────────────────────────────┘
            ↓
    ┌───────────────────────────────────────────────────┐
    │ Policy Sampling:                                  │
    │   policy(observation, stop_condition)             │
    │   → tokens with logprobs                          │
    └───────────────────────────────────────────────────┘
            ↓
    ┌───────────────────────────────────────────────────┐
    │ Generated Action:                                 │
    │   <solution>                                      │
    │   2x = 11-3 = 8                                   │
    │   x = 4                                            │
    │   \\boxed{4}                                      │
    │   </solution>                                     │
    │   <evaluation>N/A</evaluation>                    │
    │   <comparison>N/A</comparison>                    │
    └───────────────────────────────────────────────────┘
            ↓
    ┌───────────────────────────────────────────────────┐
    │ env.step(action):                                 │
    │   1. coordinator.submit_response()                │
    │      → parse_agent_response()                     │
    │      → ParsedResponse stored                      │
    │   2. return StepResult(reward=0.0, done=False)    │
    │                        ↑ placeholder reward       │
    └───────────────────────────────────────────────────┘
            ↓
    ┌───────────────────────────────────────────────────┐
    │ Create Transition:                                │
    │   Transition(                                     │
    │     ob=observation,                               │
    │     ac=action_tokens_with_logprobs,               │
    │     reward=0.0,  ← placeholder                    │
    │     episode_done=False                            │
    │   )                                                │
    └───────────────────────────────────────────────────┘
            │                     │                     │
            │ notify_all()        │ wait_for_turn()     │
            │                     ↓ (ready: id=1)       │
            │             ╔═══════════════════╗         │
            │             ║ Turn 1 (Round 0)  ║         │
            │             ╠═══════════════════╣         │
            │             ║ Agent 1 acts      ║         │
            │             ╚═══════════════════╝         │
            │                     │                     │
            │                     ↓                     │
            │             (Similar process as Turn 0)   │
            │                     │                     │
            │                     │ notify_all()        │
            │                     │                     ↓
            │                     │             ╔═══════════════════╗
            │                     │             ║ Turn 2 (Round 0)  ║
            │                     │             ╠═══════════════════╣
            │                     │             ║ Agent 2 acts      ║
            │                     │             ╚═══════════════════╝
            │                     │                     │
            │                     │     ┌───────────────────────────┐
            │                     │     │ Observation (with history):│
            │                     │     │   History: S0, S1          │
            │                     │     │   Instruction: "Compare"   │
            │                     │     └───────────────────────────┘
            │                     │                     ↓
            │                     │     ┌───────────────────────────┐
            │                     │     │ Generated Action:          │
            │                     │     │   <solution>...</solution> │
            │                     │     │   <evaluation>             │
            │                     │     │   Agent 0's solution is... │
            │                     │     │   </evaluation>            │
            │                     │     │   <comparison>             │
            │                     │     │   Agent 0 > Agent 1        │
            │                     │     │   </comparison>            │
            │                     │     └───────────────────────────┘
            ↓                     ↓                     ↓
    (Continue to Round 1: Turns 3, 4, 5)
    (Continue to Round 2: Turns 6, 7, 8)
            ↓                     ↓                     ↓
    ┌─────────────────────────────────────────────────────────┐
    │ All Agents Complete (episode_done=True)                 │
    │                                                          │
    │ Agent 0 Trajectory:                                      │
    │   [Transition(ob_0, ac_0, r=0.0),  # Turn 0            │
    │    Transition(ob_1, ac_1, r=0.0),  # Turn 3            │
    │    Transition(ob_2, ac_2, r=0.0)]  # Turn 6            │
    │                                                          │
    │ Agent 1 Trajectory:                                      │
    │   [Transition(ob_0, ac_0, r=0.0),  # Turn 1            │
    │    Transition(ob_1, ac_1, r=0.0),  # Turn 4            │
    │    Transition(ob_2, ac_2, r=0.0)]  # Turn 7            │
    │                                                          │
    │ Agent 2 Trajectory:                                      │
    │   [Transition(ob_0, ac_0, r=0.0),  # Turn 2            │
    │    Transition(ob_1, ac_1, r=0.0),  # Turn 5            │
    │    Transition(ob_2, ac_2, r=0.0)]  # Turn 8            │
    │                                                          │
    │ Coordinator State:                                       │
    │   agent_responses = [9 ParsedResponses with comparisons]│
    └─────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│ STEP 2.3: Compute Group Rewards (Post-Hoc)                       │
├───────────────────────────────────────────────────────────────────┤
│ Code: env_group_builder.compute_group_rewards(trajectories, envs)│
│   └─ _populate_stepwise_rewards(trajectory_group, env_group)     │
└───────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────┐
        │ Step 2.3.1: Initialize Step Rewards         │
        ├─────────────────────────────────────────────┤
        │ step_rewards = [                            │
        │     [0.0, 0.0, 0.0],  # Agent 0's 3 steps  │
        │     [0.0, 0.0, 0.0],  # Agent 1's 3 steps  │
        │     [0.0, 0.0, 0.0],  # Agent 2's 3 steps  │
        │ ]                                            │
        └─────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Step 2.3.2: Process Comparison Rewards                  │
        ├─────────────────────────────────────────────────────────┤
        │ for turn_idx, response in enumerate(agent_responses):   │
        │     for (agent_a, op, agent_b) in response.comparisons: │
        │                                                          │
        │         # Find which step is being compared             │
        │         agent_a_step = get_step_idx_before_turn(        │
        │             agent_a, turn_idx, num_agents               │
        │         )                                                │
        │         agent_b_step = get_step_idx_before_turn(        │
        │             agent_b, turn_idx, num_agents               │
        │         )                                                │
        │                                                          │
        │         # Skip if agent hasn't acted yet                │
        │         if agent_a_step < 0 or agent_b_step < 0:        │
        │             continue                                     │
        │                                                          │
        │         # Assign rewards to specific steps              │
        │         if op == ">":                                    │
        │             step_rewards[agent_a][agent_a_step] += 1.0  │
        │             step_rewards[agent_b][agent_b_step] -= 1.0  │
        │         elif op == "<":                                  │
        │             step_rewards[agent_a][agent_a_step] -= 1.0  │
        │             step_rewards[agent_b][agent_b_step] += 1.0  │
        │                                                          │
        │ Example result:                                          │
        │   step_rewards = [                                       │
        │       [+2.0, +1.5, -0.5],  # Agent 0                    │
        │       [-1.0, +0.5, +1.0],  # Agent 1                    │
        │       [-1.0, -2.0, -0.5],  # Agent 2                    │
        │   ]                                                      │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Step 2.3.3: Format Penalties (if enabled)               │
        ├─────────────────────────────────────────────────────────┤
        │ if enable_format_penalty:                               │
        │     for turn_idx >= 2:  # Exempt turns 0-1             │
        │         if num_other_agents_acted >= 2:                 │
        │             if len(response.comparisons) == 0:          │
        │                 author_step = turn_idx // num_agents    │
        │                 step_rewards[author][author_step] += -0.5│
        │                                                          │
        │ Example: If Turn 5 has no comparisons                   │
        │   → step_rewards[2][1] += -0.5                          │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Step 2.3.4: Assign Rewards to Trajectories (In-Place)  │
        ├─────────────────────────────────────────────────────────┤
        │ for agent_id in range(num_agents):                      │
        │     trajectory = trajectory_group[agent_id]             │
        │     for step_idx, transition in enumerate(transitions): │
        │         transition.reward = step_rewards[agent_id][step]│
        │                                                          │
        │ Result: Trajectory objects modified in-place            │
        │   Agent 0: [Transition(r=+2.0), Transition(r=+1.5),    │
        │             Transition(r=-0.5)]                          │
        │   Agent 1: [Transition(r=-1.0), Transition(r=+0.5),    │
        │             Transition(r=+1.0)]                          │
        │   Agent 2: [Transition(r=-1.0), Transition(r=-2.0),    │
        │             Transition(r=-0.5)]                          │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────┐
        │ Return TrajectoryGroup:                     │
        │   trajectories_G = [Traj_0, Traj_1, Traj_2]│
        │   final_rewards_G = [0.0, 0.0, 0.0]         │
        │   metrics_G = [{...}, {...}, {...}]         │
        └─────────────────────────────────────────────┘

                              ↓
┌───────────────────────────────────────────────────────────────────┐
│ STEP 3: Collect All Episodes                                     │
├───────────────────────────────────────────────────────────────────┤
│ trajectory_groups_P = [TrajectoryGroup_Q1, ..., TrajectoryGroup_Q16]│
│                                                                   │
│ Each TrajectoryGroup contains:                                   │
│   - 3 trajectories (one per agent)                               │
│   - Each trajectory has 3 transitions with assigned rewards      │
└───────────────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│ STEP 4: Compute Stepwise Advantages                              │
├───────────────────────────────────────────────────────────────────┤
│ Code: compute_stepwise_advantages(trajectory_groups_P)           │
│ (from tinker_cookbook/rl/data_processing.py)                     │
└───────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Step 4.1: Flatten All Step Rewards                      │
        ├─────────────────────────────────────────────────────────┤
        │ For each trajectory_group:                              │
        │   all_step_rewards = []                                 │
        │   for trajectory in trajectories_G:                     │
        │       step_rewards = [t.reward for t in transitions]    │
        │       all_step_rewards.append(step_rewards)             │
        │                                                          │
        │ Example (one group):                                     │
        │   all_step_rewards = [                                   │
        │       [+2.0, +1.5, -0.5],  # Agent 0                    │
        │       [-1.0, +0.5, +1.0],  # Agent 1                    │
        │       [-1.0, -2.0, -0.5],  # Agent 2                    │
        │   ]                                                      │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Step 4.2: Compute Mean Across ALL Steps                │
        ├─────────────────────────────────────────────────────────┤
        │ all_rewards_flat = [r for agent_rewards in all_step_rewards│
        │                       for r in agent_rewards]            │
        │                                                          │
        │ Example:                                                 │
        │   all_rewards_flat = [+2.0, +1.5, -0.5,                 │
        │                       -1.0, +0.5, +1.0,                 │
        │                       -1.0, -2.0, -0.5]                 │
        │                                                          │
        │   mean_reward = sum(all_rewards_flat) / len(...)        │
        │               = 0.0 / 9 = 0.0                           │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Step 4.3: Center Each Step (Compute Advantages)        │
        ├─────────────────────────────────────────────────────────┤
        │ for step_rewards in all_step_rewards:                   │
        │     centered = [r - mean_reward for r in step_rewards]  │
        │     group_advantages.append(centered)                   │
        │                                                          │
        │ Example (mean=0.0, so no change):                       │
        │   advantages_G_S = [                                     │
        │       [+2.0, +1.5, -0.5],  # Agent 0                    │
        │       [-1.0, +0.5, +1.0],  # Agent 1                    │
        │       [-1.0, -2.0, -0.5],  # Agent 2                    │
        │   ]                                                      │
        │                                                          │
        │ Note: Centering ensures zero mean across all steps      │
        └─────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│ STEP 5: Assemble Token-Level Training Data                       │
├───────────────────────────────────────────────────────────────────┤
│ Code: assemble_training_data_stepwise(                           │
│           trajectory_groups_P, advantages_P_G_S                   │
│       )                                                           │
└───────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ For Each Trajectory:                                    │
        │   trajectory_to_data_stepwise(traj, step_advantages)    │
        ├─────────────────────────────────────────────────────────┤
        │ Agent 0, Step 0 (Turn 0):                               │
        │   Observation: [Q_tokens, sys_prompt_tokens]            │
        │   Action: [<solution>4</solution>... tokens]            │
        │   Step advantage: +2.0                                   │
        │                                                          │
        │   Build token arrays:                                    │
        │   ┌─────────────────────────────────────────────────┐  │
        │   │ Token Index:    0    1    2  ...  100  101  102 │  │
        │   │ Tokens:        [Q tokens...] [action tokens...] │  │
        │   │ Advantages:    [0.0  0.0  0.0] [+2.0 +2.0 +2.0]│  │
        │   │ Mask:          [0.0  0.0  0.0] [1.0  1.0  1.0] │  │
        │   │                ↑ obs: not trained  ↑ trained    │  │
        │   └─────────────────────────────────────────────────┘  │
        │                                                          │
        │ Agent 0, Step 1 (Turn 3):                               │
        │   Action tokens get advantage = +1.5                     │
        │                                                          │
        │ Agent 0, Step 2 (Turn 6):                               │
        │   Action tokens get advantage = -0.5                     │
        │                                                          │
        │ → Generate Datum objects for training                    │
        └─────────────────────────────────────────────────────────┘
                              ↓
┌───────────────────────────────────────────────────────────────────┐
│ STEP 6: Training Update                                          │
├───────────────────────────────────────────────────────────────────┤
│ Code: training_client.forward_backward_async(data, loss_fn)      │
│       training_client.optim_step_async(adam_params)              │
└───────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Loss Function (Importance Sampling or PPO):             │
        ├─────────────────────────────────────────────────────────┤
        │ For each action token t:                                │
        │                                                          │
        │   ratio_t = π_θ_new(token_t | context)                 │
        │             ─────────────────────────                   │
        │             π_θ_old(token_t | context)                 │
        │                                                          │
        │   loss_t = -advantage_t × log(ratio_t) × mask_t         │
        │                                                          │
        │ Total loss = Σ_t loss_t                                │
        │                                                          │
        │ Example:                                                 │
        │   Token 101 (action, advantage=+2.0):                   │
        │     loss = -2.0 × log(ratio) × 1.0                     │
        │     → Increase probability if ratio < 1                 │
        │                                                          │
        │   Token 201 (action, advantage=-0.5):                   │
        │     loss = -(-0.5) × log(ratio) × 1.0                  │
        │     → Decrease probability if ratio < 1                 │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Gradient Update:                                        │
        │   ∇_θ Loss → Adam optimizer → θ_N → θ_{N+1}           │
        │                                                          │
        │ Effect:                                                  │
        │   - Tokens with advantage > 0: 概率增加                 │
        │   - Tokens with advantage < 0: 概率降低                 │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Create New Sampling Client:                             │
        │   sampling_client = training_client.save_weights_and_   │
        │                     get_sampling_client_async()         │
        │                                                          │
        │ Updated Policy: θ_{N+1}                                 │
        └─────────────────────────────────────────────────────────┘
                              ↓
╔═══════════════════════════════════════════════════════════════════╗
║                   Training Iteration N+1                          ║
║                   Policy State: θ_{N+1}                          ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## Detailed Turn-by-Turn Example

### Configuration
- num_agents: 3
- max_rounds: 3
- history_turns: 2 (show last 2 turns)
- Total turns: 9 (3 agents × 3 rounds)

### Timeline

```
Turn 0 (Agent 0, Round 0):
┌────────────────────────────────────────────────────────────┐
│ Input Context:                                             │
│   System: "You are Agent 0, The Methodical Analyst..."    │
│   User: "Question: 2x+3=11\nFirst turn, no history"       │
│                                                            │
│ Policy Output:                                             │
│   <solution>2x = 8, x = 4, \\boxed{4}</solution>          │
│   <evaluation>N/A</evaluation>                             │
│   <comparison>N/A</comparison>                             │
│                                                            │
│ Coordinator State After:                                   │
│   current_turn: 0 → 1                                      │
│   current_agent_id: 0 → 1                                  │
│   agent_responses: [ParsedResponse_0]                      │
│                                                            │
│ Transition Stored:                                         │
│   Transition(ob, ac, reward=0.0, episode_done=False)       │
└────────────────────────────────────────────────────────────┘

Turn 1 (Agent 1, Round 0):
┌────────────────────────────────────────────────────────────┐
│ Input Context:                                             │
│   System: "You are Agent 1, The Creative Problem-Solver..."│
│   User: "Question: 2x+3=11\nFirst turn, no history"       │
│                                                            │
│ Policy Output:                                             │
│   <solution>x = (11-3)/2 = 4, \\boxed{4}</solution>       │
│   <evaluation>N/A</evaluation>                             │
│   <comparison>N/A</comparison>                             │
│                                                            │
│ Transition Stored: reward=0.0                              │
└────────────────────────────────────────────────────────────┘

Turn 2 (Agent 2, Round 0):
┌────────────────────────────────────────────────────────────┐
│ Input Context:                                             │
│   System: "You are Agent 2, The Devil's Advocate..."      │
│   User: "Question: 2x+3=11\n                               │
│          History (last 2 turns):                           │
│            Turn 1: Agent 0's solution: ...                 │
│            Turn 2: Agent 1's solution: ..."                │
│                                                            │
│ Policy Output:                                             │
│   <solution>2x = 11-3, x = 4, \\boxed{4}</solution>       │
│   <evaluation>                                             │
│   Agent 0 correctly solved but didn't verify.              │
│   Agent 1 showed clear steps.                              │
│   </evaluation>                                            │
│   <comparison>                                             │
│   Agent 1 > Agent 0                                        │
│   </comparison>                                            │
│                                                            │
│ Parsed Comparisons: [(1, ">", 0)]                         │
│ Transition Stored: reward=0.0                              │
└────────────────────────────────────────────────────────────┘

Turn 3 (Agent 0, Round 1):
┌────────────────────────────────────────────────────────────┐
│ Input Context:                                             │
│   History (last 2 turns): Turn 1-2 solutions               │
│                                                            │
│ Policy Output:                                             │
│   <solution>Verifying: 2(4)+3=8+3=11✓, \\boxed{4}</solution>│
│   <evaluation>All correct. Agent 1 most clear.</evaluation>│
│   <comparison>                                             │
│   Agent 1 > Agent 2                                        │
│   </comparison>                                            │
│                                                            │
│ Parsed Comparisons: [(1, ">", 2)]                         │
│ Transition Stored: reward=0.0                              │
└────────────────────────────────────────────────────────────┘

Turn 4-8: Continue similarly...

═══════════════════════════════════════════════════════════════
After Turn 8, All Agents Done
═══════════════════════════════════════════════════════════════

Coordinator State:
  agent_responses = [
      ParsedResponse(author=0, comparisons=[]),              # Turn 0
      ParsedResponse(author=1, comparisons=[]),              # Turn 1
      ParsedResponse(author=2, comparisons=[(1,">",0)]),     # Turn 2
      ParsedResponse(author=0, comparisons=[(1,">",2)]),     # Turn 3
      ParsedResponse(author=1, comparisons=[(0,">",2)]),     # Turn 4
      ParsedResponse(author=2, comparisons=[(1,">",0)]),     # Turn 5
      ParsedResponse(author=0, comparisons=[(1,">",2)]),     # Turn 6
      ParsedResponse(author=1, comparisons=[(0,">",2)]),     # Turn 7
      ParsedResponse(author=2, comparisons=[]),              # Turn 8 (no valid)
  ]
```

---

### **Reward Computation Example**

```
Processing Comparisons:
═══════════════════════════════════════════════════════════════

Turn 2: Agent 2 says "Agent 1 > Agent 0"
  agent_1_step = get_step_idx_before_turn(1, 2, 3) = 0
  agent_0_step = get_step_idx_before_turn(0, 2, 3) = 0
  → step_rewards[1][0] += 1.0  (Agent 1, Step 0)
  → step_rewards[0][0] -= 1.0  (Agent 0, Step 0)

Turn 3: Agent 0 says "Agent 1 > Agent 2"
  agent_1_step = get_step_idx_before_turn(1, 3, 3) = 0
  agent_2_step = get_step_idx_before_turn(2, 3, 3) = 0
  → step_rewards[1][0] += 1.0
  → step_rewards[2][0] -= 1.0

Turn 4: Agent 1 says "Agent 0 > Agent 2"
  agent_0_step = get_step_idx_before_turn(0, 4, 3) = 1
  agent_2_step = get_step_idx_before_turn(2, 4, 3) = 0
  → step_rewards[0][1] += 1.0
  → step_rewards[2][0] -= 1.0

Turn 5: Agent 2 says "Agent 1 > Agent 0"
  agent_1_step = get_step_idx_before_turn(1, 5, 3) = 1
  agent_0_step = get_step_idx_before_turn(0, 5, 3) = 1
  → step_rewards[1][1] += 1.0
  → step_rewards[0][1] -= 1.0

(Continue for Turn 6-8...)

Format Penalties:
  Turn 8 (Agent 2) has no comparisons (after exempt period)
  → step_rewards[2][2] += -0.5

Final step_rewards:
  Agent 0: [-1.0, 0.0, +1.0]
  Agent 1: [+2.0, +1.0, -0.5]
  Agent 2: [-1.0, -1.0, -0.5]
```

---

## Key Characteristics

### ✅ **Unified Policy (Single θ)**
```
All agents = Same policy θ_N
- Agent 0 at Turn 0: uses θ_N
- Agent 1 at Turn 1: uses θ_N
- Agent 2 at Turn 2: uses θ_N (judging Turn 0, 1)
→ Generator and Judge are the SAME model
```

### ✅ **Self-Play**
```
Policy plays against itself:
- Generates solutions
- Evaluates solutions generated by itself (in other positions)
- Learns from self-evaluation
```

### ✅ **Step-Wise Credit Assignment**
```
Different steps get different rewards:
- Step 0 (early solution): reward based on comparisons mentioning Turn 0
- Step 1 (improved solution): reward based on comparisons mentioning Turn 3
- Step 2 (final solution): reward based on comparisons mentioning Turn 6

NOT trajectory-level (all steps same reward)
```

### ✅ **Interleaved Generation and Evaluation**
```
Each turn = <solution> + <evaluation> + <comparison>
- Generation and judgment happen in the same forward pass
- Policy learns both capabilities simultaneously
- Knowledge transfer: judgment → better generation
```

### ⚠️ **Evaluation Coverage Imbalance**
```
With history_turns=2:
- Turn 0: evaluated ~1 time
- Turn 4: evaluated ~2 times
- Turn 8: evaluated ~0 times

This causes:
- Unequal training signal across steps
- Sparse consensus samples
- Potential instability
```

---

## Current Issues and Proposed Solutions

### Issue 1: Evaluation Imbalance
```
Problem: Early/late turns get fewer evaluations
Impact: Uneven training signal, unstable gradients

Solutions:
  A. history_turns = -1 (full history)
     → Turn 0: 1 eval → 7 evals
  B. num_agents = 5
     → More evaluators naturally
  C. Self-Reflection Round
     → +3 evals for all turns
```

### Issue 2: Sparse Consensus
```
Problem: Each solution pair judged 1-2 times (insufficient for consensus)
Impact: Cannot use consensus rewards effectively

Solutions:
  A. Full history + 5 agents
     → ~9 judgments per pair
  B. Self-Reflection Round
     → +3 judgments per pair guaranteed
```

### Issue 3: Cross-Episode Non-Stationarity
```
Problem: Policy θ updates → evaluation standards change
Impact: Same action gets different rewards across episodes

Solutions:
  A. Consensus Rewards
     → Multiple judges → more stable signal
  B. PPO loss (vs importance_sampling)
     → Clips large policy changes
  C. Curriculum (judge weight ramping)
     → Gradual introduction of judge training
```

---

## Training Configuration

```python
# Current typical config
num_agents = 3
max_rounds = 3
History is fixed to the last round (num_agents turns).
batch_size = 16
learning_rate = 3e-5
use_stepwise_advantages = True  # Enable per-step credit assignment
loss_fn = "importance_sampling"  # or "ppo"
```

---

## Metrics to Monitor

### Training Metrics
- `stepwise_comparisons_used`: Number of valid comparisons processed
- `missing_comparisons`: Turns without valid comparisons (format issues)
- `mean_reward_raw`: Average raw reward before centering

### Performance Metrics (Verifiable)
- `train_<dataset>/format`: Fraction of valid formatted responses
- `train_<dataset>/correct`: Correctness of latest solution
- `train_<dataset>/pass@N`: At least one agent correct
- `train_<dataset>/avg@N`: Average correctness across agents
- `train_<dataset>/cons@N`: Majority consensus correctness

### Stability Indicators
- Reward variance across steps
- Loss curve smoothness
- Gradient norms
- KL divergence (if computed)

---

## Files Reference

- `base_env.py`: `_populate_stepwise_rewards()` - reward computation logic
- `coordinator.py`: `MultiAgentCoordinator` - turn-taking synchronization
- `env.py`, `verifiable_env.py`: Environment implementations
- `prompts.py`: `parse_agent_response()` - XML parsing
- `utils.py`: `get_step_idx_before_turn()` - step indexing
- `train.py`: Training configuration and entry point
- `../rl/train.py`: Generic RL training loop
- `../rl/data_processing.py`: `compute_stepwise_advantages()`, `trajectory_to_data_stepwise()`
- `../rl/rollouts.py`: `do_group_rollout()` - episode execution

---

## Design Philosophy

This implementation embodies a **unified policy approach** where:

1. **Single Model, Multiple Roles**: The same policy θ plays all agent positions
2. **Joint Learning**: Generation and evaluation capabilities trained together
3. **Self-Improvement**: Policy learns to evaluate its own outputs (in different positions)
4. **Knowledge Transfer**: Judgment skills → improved generation quality

The goal is to create a policy that not only generates good solutions, but also understands what makes a solution good through the act of evaluation.
