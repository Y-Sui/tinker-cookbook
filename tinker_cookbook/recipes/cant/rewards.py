"""
Reward computation for CANT framework.

Implements three reward components:
1. r_disc: Persuasion reward (did critiques change others' scores?)
2. r_sol: Solution quality reward (final Bradley-Terry score)
3. r_meta: Majority alignment reward (agreement with majority preferences)
"""

import numpy as np
from collections import defaultdict, Counter
from typing import Sequence


def compute_persuasion_rewards(
    critiques: dict[int, list[int]],
    v_t0: np.ndarray,
    v_final: np.ndarray,
    beta: float = 5.0,
) -> dict[int, float]:
    """
    Compute r_disc for each agent based on critique effectiveness.

    Reward agents whose critiques successfully lowered target agents' scores.

    Args:
        critiques: Dict mapping {author_id: [target_ids]} showing who critiqued whom
        v_t0: Initial Bradley-Terry scores (from blind rankings)
        v_final: Final Bradley-Terry scores (after revision)
        beta: Scaling factor for persuasion rewards

    Returns:
        Dict mapping agent_id to persuasion reward

    Formula:
        r_disc(i) = Σ_{k ∈ targets_i} ReLU(β × (V_k^t0 - V_k^final))

    Where ReLU ensures we only reward successful critiques (score drops).
    """
    rewards = defaultdict(float)

    for author_id, targets in critiques.items():
        total_reward = 0.0
        for target_id in targets:
            # Calculate score drop
            delta = float(v_t0[target_id] - v_final[target_id])
            # ReLU: only reward if score dropped
            reward = max(0.0, beta * delta)
            total_reward += reward

        rewards[author_id] = total_reward

    return dict(rewards)


def compute_solution_rewards(
    v_final: np.ndarray,
) -> np.ndarray:
    """Compute r_sol as the final Bradley-Terry score.

    Args:
        v_final: Final Bradley-Terry scores for all agents

    Returns:
        Array of solution rewards in [0, 1].

    Note:
        We keep this reward in the natural Bradley-Terry scale for interpretability.
        Normalization (if desired) can be handled by downstream advantage processing.
    """
    return v_final.astype(np.float32)


def compute_consensus_rewards(
    final_rankings: dict[int, list[tuple[int, str, int]]],
) -> dict[int, float]:
    """
    Compute r_meta via majority vote alignment.

    Reward agents whose final rankings align with the majority opinion.

    Args:
        final_rankings: Dict mapping {author_id: [(agent_a, op, agent_b), ...]}

    Returns:
        Dict mapping agent_id to majority-alignment reward in range [0, 1]

    Formula:
        r_meta(i) = correct / total

    Where:
        - correct = number of pairwise comparisons matching majority
        - total = total number of comparisons made
    """
    # Build majority preferences for each pair
    pair_votes = defaultdict(list)  # {(A, B): [(author_id, winner), ...]}

    for author_id, rankings in final_rankings.items():
        for agent_a, op, agent_b in rankings:
            if op == "=":
                continue  # Skip ties for majority computation

            # Normalize pair to consistent order
            pair = tuple(sorted([agent_a, agent_b]))
            winner = agent_a if op == ">" else agent_b

            # Handle reversed pair
            if pair != (agent_a, agent_b):
                # Pair was reversed, so other agent won
                winner = agent_a if winner == agent_b else agent_b

            pair_votes[pair].append((author_id, winner))

    # Compute majority winner for each pair
    majority = {}
    for pair, votes in pair_votes.items():
        winner_counts = Counter(w for _, w in votes)
        if winner_counts:
            majority[pair] = winner_counts.most_common(1)[0][0]

    # Score each agent's alignment with majority
    rewards = {}
    for author_id, rankings in final_rankings.items():
        correct = 0
        total = 0

        for agent_a, op, agent_b in rankings:
            if op == "=":
                continue

            # Normalize pair
            pair = tuple(sorted([agent_a, agent_b]))
            if pair not in majority:
                continue  # Skip if no majority (shouldn't happen)

            # Determine this agent's vote
            winner = agent_a if op == ">" else agent_b
            if pair != (agent_a, agent_b):
                # Pair was reversed
                winner = agent_a if winner == agent_b else agent_b

            # Check if matches majority
            if winner == majority[pair]:
                correct += 1
            total += 1

        # Compute majority-alignment rate
        if total > 0:
            rewards[author_id] = float(correct) / total
        else:
            rewards[author_id] = 0.0

    return rewards


def compute_all_rewards(
    critiques: dict[int, list[int]],
    blind_rankings: dict[int, list[tuple[int, str, int]]],
    final_rankings: dict[int, list[tuple[int, str, int]]],
    num_agents: int,
    beta_disc: float = 5.0,
) -> dict[str, dict[int, float] | np.ndarray]:
    """Compute all reward components for a group of agents.

    Args:
        critiques: Dict mapping {author_id: [target_ids]}
        blind_rankings: Dict mapping {author_id: [(a, op, b), ...]} for Round 2
        final_rankings: Dict mapping {author_id: [(a, op, b), ...]} for Round 4
        num_agents: Total number of agents
        beta_disc: Scaling factor for persuasion rewards

    Returns:
        Dict containing:
            'r_disc': Dict[int, float] - Persuasion rewards
            'r_sol': np.ndarray - Solution quality rewards
            'r_meta': Dict[int, float] - Majority-alignment rewards
            'v_t0': np.ndarray - Initial BT scores
            'v_final': np.ndarray - Final BT scores
    """
    from tinker_cookbook.recipes.cant.bradley_terry import compute_scores_from_rankings

    # Compute Bradley-Terry scores
    v_t0 = compute_scores_from_rankings(blind_rankings, num_agents)
    v_final = compute_scores_from_rankings(final_rankings, num_agents)

    # Compute individual reward components
    r_disc = compute_persuasion_rewards(critiques, v_t0, v_final, beta_disc)
    r_sol = compute_solution_rewards(v_final)
    r_meta = compute_consensus_rewards(final_rankings)

    return {
        "r_disc": r_disc,
        "r_sol": r_sol,
        "r_meta": r_meta,
        "v_t0": v_t0,
        "v_final": v_final,
    }


def combine_rewards(
    r_disc: dict[int, float],
    r_sol: np.ndarray,
    r_meta: dict[int, float],
    weight_disc: float = 2.0,
    weight_sol: float = 1.0,
    weight_meta: float = 1.0,
) -> dict[int, dict[str, float]]:
    """Combine reward components with configurable weights.

    Args:
        r_disc: Persuasion rewards
        r_sol: Solution rewards
        r_meta: Majority-alignment rewards
        weight_disc: Weight for persuasion component
        weight_sol: Weight for solution component
        weight_meta: Weight for majority-alignment component

    Returns:
        Dict mapping agent_id to reward breakdown:
            {agent_id: {'disc': float, 'sol': float, 'meta': float}}
    """
    num_agents = len(r_sol)
    combined: dict[int, dict[str, float]] = {}

    for agent_id in range(num_agents):
        combined[agent_id] = {
            "disc": weight_disc * r_disc.get(agent_id, 0.0),
            "sol": weight_sol * float(r_sol[agent_id]),
            "meta": weight_meta * r_meta.get(agent_id, 0.0),
        }

    return combined
