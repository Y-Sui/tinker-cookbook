from tinker_cookbook.recipes.multi_agent_debate.reward import (
    compute_pairwise_win_minus_loss,
    compute_pairwise_win_rates,
)
from tinker_cookbook.recipes.multi_agent_debate.prompts import ParsedResponse


def _resp(comparisons: list[tuple[int, str, int]], *, author_id: int = -1) -> ParsedResponse:
    return ParsedResponse(
        thinking="",
        solution="s",
        evaluation="e",
        consensus_reached=False,
        consensus_reason="r",
        comparisons=comparisons,
        raw_response="",
        author_id=author_id,
    )


def test_compute_pairwise_win_rates_basic():
    responses_rounds = [
        [
            _resp([(0, ">", 1), (0, "=", 2)]),
            _resp([(2, ">", 0), (1, ">", 2)]),
            _resp([]),
        ]
    ]
    rewards_G, metrics = compute_pairwise_win_rates(responses_rounds, num_agents=3)
    assert rewards_G == [0.5, 0.5, 0.5]
    assert metrics["pairwise_total_votes"] == 4.0
    assert metrics["pairwise_malformed"] == 0.0


def test_compute_pairwise_win_rates_ignores_malformed():
    responses_rounds = [[_resp([(0, ">", 0), (9, ">", 1), (0, "?", 1), (0, "=", 1)])]]
    rewards_G, metrics = compute_pairwise_win_rates(responses_rounds, num_agents=2)
    # Only "0 = 1" counts: both agents get 0.5 wins over 1 vote.
    assert rewards_G == [0.5, 0.5]
    assert metrics["pairwise_total_votes"] == 1.0
    assert metrics["pairwise_malformed"] == 3.0


def test_compute_pairwise_win_minus_loss_basic():
    responses_rounds = [[_resp([(0, ">", 1), (0, "=", 2), (2, ">", 0)])]]
    rewards_G, metrics = compute_pairwise_win_minus_loss(responses_rounds, num_agents=3)
    # Matchups:
    # 0>1: 0:+1,1:-1
    # 0=2: no change
    # 2>0: 2:+1,0:-1
    # So win-minus-loss: [0, -1, +1], matchups: [3,1,2] => [0, -1, 0.5]
    assert rewards_G == [0.0, -1.0, 0.5]
    assert metrics["pairwise_total_votes"] == 3.0
    assert metrics["pairwise_malformed"] == 0.0


def test_compute_pairwise_win_rates_excludes_self_votes():
    # Agent 0 votes "0>1"; Agent 1 votes "1>0".
    # Each agent's reward should be computed using only the other agent's vote.
    responses_rounds = [
        [
            _resp([(0, ">", 1)], author_id=0),
            _resp([(1, ">", 0)], author_id=1),
        ]
    ]
    rewards_G, metrics = compute_pairwise_win_rates(responses_rounds, num_agents=2)
    assert rewards_G == [0.0, 0.0]
    assert metrics["pairwise_total_votes"] == 2.0
    assert metrics["pairwise_malformed"] == 0.0


def test_compute_pairwise_win_minus_loss_excludes_self_votes():
    responses_rounds = [
        [
            _resp([(0, ">", 1)], author_id=0),
            _resp([(1, ">", 0)], author_id=1),
        ]
    ]
    rewards_G, metrics = compute_pairwise_win_minus_loss(responses_rounds, num_agents=2)
    assert rewards_G == [-1.0, -1.0]
    assert metrics["pairwise_total_votes"] == 2.0
    assert metrics["pairwise_malformed"] == 0.0
