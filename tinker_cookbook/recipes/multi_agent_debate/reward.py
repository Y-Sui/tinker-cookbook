from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .prompts import ParsedResponse


def compute_pairwise_win_rates(
    agent_responses_rounds: list[list["ParsedResponse"]], num_agents: int
) -> tuple[list[float], dict[str, float]]:
    """
    Compute per-agent win rates from all parsed pairwise comparisons across all rounds.

    We treat each comparison line as a vote:
    - "Agent A > Agent B" gives A 1.0 win, B 0.0 win
    - "Agent A = Agent B" gives A 0.5 win, B 0.5 win

    Returns:
      - rewards_G: list[float] of length num_agents, each in [0, 1] (if any votes exist)
      - summary metrics
    """
    wins_G = [0.0 for _ in range(num_agents)]
    votes_G = [0.0 for _ in range(num_agents)]
    total_votes = 0
    total_malformed = 0

    for round_responses in agent_responses_rounds:
        for resp in round_responses:
            for agent_a, op, agent_b in resp.comparisons:
                if not (0 <= agent_a < num_agents and 0 <= agent_b < num_agents):
                    total_malformed += 1
                    continue
                if agent_a == agent_b:
                    total_malformed += 1
                    continue
                if op == ">":
                    wins_G[agent_a] += 1.0
                    votes_G[agent_a] += 1.0
                    votes_G[agent_b] += 1.0
                    total_votes += 1
                elif op == "=":
                    wins_G[agent_a] += 0.5
                    wins_G[agent_b] += 0.5
                    votes_G[agent_a] += 1.0
                    votes_G[agent_b] += 1.0
                    total_votes += 1
                else:
                    total_malformed += 1

    rewards_G = [
        (wins / votes if votes > 0 else 0.0) for wins, votes in zip(wins_G, votes_G, strict=True)
    ]
    metrics = {
        "pairwise_total_votes": float(total_votes),
        "pairwise_malformed": float(total_malformed),
        "pairwise_any_votes": 1.0 if total_votes > 0 else 0.0,
    }
    return rewards_G, metrics

