"""Tests for Bradley-Terry scoring."""

import numpy as np
import pytest

from tinker_cookbook.recipes.cant.bradley_terry import (
    compute_bradley_terry_scores,
    pairwise_comparisons_to_winner_pairs,
    compute_scores_from_rankings,
)


def test_bradley_terry_basic():
    """Test basic Bradley-Terry scoring."""
    # Agent 0 beats everyone
    comparisons = [(0, 1), (0, 2), (0, 3)]
    scores = compute_bradley_terry_scores(comparisons, num_agents=4)

    assert len(scores) == 4
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    assert scores[0] > scores[3]
    # Scores should be in [0, 1]
    assert all(0 <= s <= 1 for s in scores)


def test_bradley_terry_empty():
    """Test Bradley-Terry with no comparisons."""
    scores = compute_bradley_terry_scores([], num_agents=4)

    assert len(scores) == 4
    # Should return uniform 0.5 for all agents
    assert all(abs(s - 0.5) < 0.01 for s in scores)


def test_bradley_terry_circular():
    """Test Bradley-Terry with circular preferences (A > B > C > A)."""
    # This tests that BT can handle inconsistent preferences
    comparisons = [(0, 1), (1, 2), (2, 0)]
    scores = compute_bradley_terry_scores(comparisons, num_agents=3)

    assert len(scores) == 3
    # Scores should be close to each other due to circular preference
    assert abs(scores[0] - scores[1]) < 0.2
    assert abs(scores[1] - scores[2]) < 0.2


def test_pairwise_comparisons_to_winner_pairs():
    """Test conversion from ranking format to winner pairs."""
    rankings = [
        (0, '>', 1),
        (2, '<', 0),
        (1, '=', 3),  # Tie - should be skipped
    ]

    pairs = pairwise_comparisons_to_winner_pairs(rankings)

    assert len(pairs) == 2  # Tie excluded
    assert (0, 1) in pairs  # Agent 0 > Agent 1
    assert (0, 2) in pairs  # Agent 2 < Agent 0 → Agent 0 > Agent 2


def test_compute_scores_from_rankings():
    """Test end-to-end scoring from rankings dict."""
    rankings_by_agent = {
        0: [(0, '>', 1), (0, '>', 2)],  # Agent 0 thinks they're best
        1: [(0, '>', 1), (0, '>', 2)],  # Agent 1 agrees 0 is best
        2: [(0, '>', 1), (0, '>', 2)],  # Agent 2 agrees 0 is best
    }

    scores = compute_scores_from_rankings(rankings_by_agent, num_agents=3)

    assert len(scores) == 3
    # Agent 0 should have highest score (everyone voted for them)
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_bradley_terry_invalid_indices():
    """Test that invalid indices raise ValueError."""
    with pytest.raises(ValueError):
        compute_bradley_terry_scores([(0, 5)], num_agents=4)

    with pytest.raises(ValueError):
        compute_bradley_terry_scores([(-1, 2)], num_agents=4)
