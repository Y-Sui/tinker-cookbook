"""
Bradley-Terry model for computing skill parameters from pairwise comparisons.

Uses the choix library for efficient inference via Iterative Luce Spectral Ranking (ILSR).
"""

import numpy as np
from typing import Sequence

try:
    import choix
except ImportError:
    raise ImportError(
        "choix library is required for Bradley-Terry scoring. "
        "Install it with: pip install choix"
    )


def compute_bradley_terry_scores(
    comparisons: Sequence[tuple[int, int]],
    num_agents: int,
    alpha: float = 0.01,
) -> np.ndarray:
    """
    Compute Bradley-Terry skill parameters from pairwise comparisons.

    Args:
        comparisons: List of (winner, loser) pairs where agents are identified by integer IDs
        num_agents: Total number of agents in the system
        alpha: L2 regularization parameter for stability (default 0.01)

    Returns:
        scores: Array of length num_agents with skill parameters normalized to [0, 1].
                Higher values indicate better performance.

    Edge cases:
        - Empty comparisons: Returns uniform 0.5 for all agents
        - Agent missing from comparisons: Gets regularized score near 0.5
        - Single agent dominates: Sigmoid normalization prevents extreme values
    """
    if not comparisons:
        # No comparisons available - return neutral scores
        return np.full(num_agents, 0.5, dtype=np.float32)

    # Validate comparison indices
    for winner, loser in comparisons:
        if not (0 <= winner < num_agents and 0 <= loser < num_agents):
            raise ValueError(
                f"Invalid comparison ({winner}, {loser}): "
                f"indices must be in range [0, {num_agents})"
            )

    try:
        # Use choix's ILSR algorithm for maximum likelihood estimation
        # Returns log-skill parameters (unbounded)
        params = choix.ilsr_pairwise(
            n_items=num_agents,
            data=list(comparisons),
            alpha=alpha,
        )
    except Exception as e:
        # Fallback if choix fails (e.g., numerical issues)
        print(f"Warning: Bradley-Terry computation failed ({e}), using neutral scores")
        return np.full(num_agents, 0.5, dtype=np.float32)

    # Normalize to [0, 1] via sigmoid transformation
    # This maps unbounded log-skills to probabilities
    scores = 1.0 / (1.0 + np.exp(-params))

    return scores.astype(np.float32)


def pairwise_comparisons_to_winner_pairs(
    rankings: Sequence[tuple[int, str, int]]
) -> list[tuple[int, int]]:
    """
    Convert pairwise rankings (Agent X > Agent Y format) to (winner, loser) pairs.

    Args:
        rankings: List of (agent_a, operator, agent_b) tuples where operator is '>', '<', or '='

    Returns:
        pairs: List of (winner, loser) tuples suitable for Bradley-Terry computation

    Note:
        - '=' (tie) comparisons are skipped as Bradley-Terry doesn't handle them directly
        - '<' comparisons are converted to '>' by swapping the agents
    """
    pairs = []
    for agent_a, op, agent_b in rankings:
        if op == '>':
            pairs.append((agent_a, agent_b))
        elif op == '<':
            pairs.append((agent_b, agent_a))
        # Skip '=' ties - BT model assumes strict preferences

    return pairs


def compute_scores_from_rankings(
    rankings_by_agent: dict[int, list[tuple[int, str, int]]],
    num_agents: int,
    alpha: float = 0.01,
) -> np.ndarray:
    """
    Convenience function: convert agent rankings to Bradley-Terry scores.

    Args:
        rankings_by_agent: Dict mapping agent_id to their list of pairwise rankings
        num_agents: Total number of agents
        alpha: Regularization parameter

    Returns:
        scores: Bradley-Terry scores for each agent
    """
    # Aggregate all rankings from all agents
    all_rankings = []
    for rankings in rankings_by_agent.values():
        all_rankings.extend(rankings)

    # Convert to winner pairs
    pairs = pairwise_comparisons_to_winner_pairs(all_rankings)

    # Compute Bradley-Terry scores
    return compute_bradley_terry_scores(pairs, num_agents, alpha)
