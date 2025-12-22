from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .prompts import ParsedResponse


def _iter_comparisons_with_authors(agent_responses_rounds: list[list["ParsedResponse"]]):
    for round_responses in agent_responses_rounds:
        for resp in round_responses:
            author = getattr(resp, "author_id", -1)
            for agent_a, op, agent_b in resp.comparisons:
                yield author, agent_a, op, agent_b


def compute_pairwise_win_rates(
    agent_responses_rounds: list[list["ParsedResponse"]], num_agents: int
) -> tuple[list[float], dict[str, float]]:
    """
    Leave-one-out win-rate reward:

    For each target agent i, compute its win-rate using only comparisons authored by agents != i.
    Only comparisons that involve i contribute to i's score.

    Returns:
      - rewards_G: list[float] of length num_agents, each in [0, 1] (if any eligible votes exist)
      - summary metrics (global counts)
    """
    wins_G = [0.0 for _ in range(num_agents)]
    votes_G = [0.0 for _ in range(num_agents)]
    total_votes = 0
    total_malformed = 0

    for author, agent_a, op, agent_b in _iter_comparisons_with_authors(agent_responses_rounds):
        if not (0 <= agent_a < num_agents and 0 <= agent_b < num_agents):
            total_malformed += 1
            continue
        if agent_a == agent_b:
            total_malformed += 1
            continue
        if op not in {">", "="}:
            total_malformed += 1
            continue

        total_votes += 1

        # Update per-target scores, excluding self-authored votes.
        for i in range(num_agents):
            if author == i:
                continue
            if i != agent_a and i != agent_b:
                continue
            votes_G[i] += 1.0
            if op == ">":
                if i == agent_a:
                    wins_G[i] += 1.0
            elif op == "=":
                wins_G[i] += 0.5

    rewards_G = [
        (wins / votes if votes > 0 else 0.0) for wins, votes in zip(wins_G, votes_G, strict=True)
    ]
    metrics = {
        "pairwise_total_votes": float(total_votes),
        "pairwise_malformed": float(total_malformed),
        "pairwise_any_votes": 1.0 if total_votes > 0 else 0.0,
    }
    return rewards_G, metrics


def compute_pairwise_win_minus_loss(
    agent_responses_rounds: list[list["ParsedResponse"]], num_agents: int
) -> tuple[list[float], dict[str, float]]:
    """
    Leave-one-out win-minus-loss reward:

    For each target agent i, compute its mean (win-loss) score using only comparisons authored by agents != i.
    Only comparisons that involve i contribute to i's score.

    Returns:
      - rewards_G: list[float] of length num_agents, each in [-1, 1] (if any eligible votes exist)
      - summary metrics (global counts)
    """
    score_G = [0.0 for _ in range(num_agents)]
    matchups_G = [0.0 for _ in range(num_agents)]
    total_votes = 0
    total_malformed = 0

    for author, agent_a, op, agent_b in _iter_comparisons_with_authors(agent_responses_rounds):
        if not (0 <= agent_a < num_agents and 0 <= agent_b < num_agents):
            total_malformed += 1
            continue
        if agent_a == agent_b:
            total_malformed += 1
            continue
        if op not in {">", "="}:
            total_malformed += 1
            continue

        total_votes += 1

        for i in range(num_agents):
            if author == i:
                continue
            if i != agent_a and i != agent_b:
                continue
            matchups_G[i] += 1.0
            if op == ">":
                if i == agent_a:
                    score_G[i] += 1.0
                elif i == agent_b:
                    score_G[i] -= 1.0
            elif op == "=":
                pass

    rewards_G = [(s / n if n > 0 else 0.0) for s, n in zip(score_G, matchups_G, strict=True)]
    metrics = {
        "pairwise_total_votes": float(total_votes),
        "pairwise_malformed": float(total_malformed),
        "pairwise_any_votes": 1.0 if total_votes > 0 else 0.0,
    }
    return rewards_G, metrics


__all__ = [
    "compute_pairwise_win_rates",
    "compute_pairwise_win_minus_loss",
]
