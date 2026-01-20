"""Tests for reward computation."""

import numpy as np

from tinker_cookbook.recipes.cant.rewards import (
    compute_persuasion_rewards,
    compute_solution_rewards,
    compute_consensus_rewards,
    compute_acceptance_rewards,
)


def test_persuasion_rewards():
    """Test persuasion reward computation."""
    # Agent 0 critiqued agents 1 and 2
    critiques = {0: [1, 2]}

    # Initial scores
    v_t0 = np.array([0.5, 0.7, 0.6, 0.4])

    # Final scores: agents 1 and 2 dropped
    v_final = np.array([0.5, 0.4, 0.5, 0.4])

    r_disc = compute_persuasion_rewards(critiques, v_t0, v_final, beta=5.0)

    # Agent 0 should get reward for successful critiques
    assert 0 in r_disc
    assert r_disc[0] > 0

    # Delta for agent 1: 0.7 - 0.4 = 0.3 → reward = 5.0 * 0.3 = 1.5
    # Delta for agent 2: 0.6 - 0.5 = 0.1 → reward = 5.0 * 0.1 = 0.5
    # Total: 1.5 + 0.5 = 2.0
    assert abs(r_disc[0] - 2.0) < 0.01


def test_persuasion_rewards_no_effect():
    """Test persuasion reward when critique has no effect."""
    critiques = {0: [1]}

    # Scores unchanged
    v_t0 = np.array([0.5, 0.7])
    v_final = np.array([0.5, 0.7])

    r_disc = compute_persuasion_rewards(critiques, v_t0, v_final, beta=5.0)

    # No reward since score didn't drop
    assert r_disc.get(0, 0.0) == 0.0


def test_persuasion_rewards_backfire():
    """Test persuasion reward when critique backfires (target improved)."""
    critiques = {0: [1]}

    # Score increased instead of decreased
    v_t0 = np.array([0.5, 0.4])
    v_final = np.array([0.5, 0.6])

    r_disc = compute_persuasion_rewards(critiques, v_t0, v_final, beta=5.0)

    # No reward (ReLU clips negative values)
    assert r_disc.get(0, 0.0) == 0.0


def test_solution_rewards():
    """Test solution quality reward (Z-score normalization)."""
    v_final = np.array([0.8, 0.5, 0.2, 0.5])

    r_sol = compute_solution_rewards(v_final)

    assert len(r_sol) == 4
    # Mean should be ~0 after Z-score normalization
    assert abs(np.mean(r_sol)) < 0.01
    # Agent 0 (highest score) should have positive reward
    assert r_sol[0] > 0
    # Agent 2 (lowest score) should have negative reward
    assert r_sol[2] < 0


def test_solution_rewards_uniform():
    """Test solution rewards when all scores are identical."""
    v_final = np.array([0.5, 0.5, 0.5, 0.5])

    r_sol = compute_solution_rewards(v_final)

    # All zeros when no variance
    assert all(abs(r) < 0.01 for r in r_sol)


def test_consensus_rewards():
    """Test consensus reward computation."""
    final_rankings = {
        0: [(0, '>', 1), (0, '>', 2)],
        1: [(1, '>', 0), (1, '>', 2)],  # Disagrees on (0,1)
        2: [(2, '>', 0), (2, '>', 1)],  # Disagrees on both
    }

    r_meta = compute_consensus_rewards(final_rankings)

    # All agents vote but with different opinions
    # Each pair should have a majority vote
    assert len(r_meta) == 3
    # Rewards should be in [-1, 1] range
    for reward in r_meta.values():
        assert -1.0 <= reward <= 1.0


def test_consensus_rewards_perfect():
    """Test consensus when everyone agrees."""
    final_rankings = {
        0: [(0, '>', 1)],
        1: [(0, '>', 1)],
        2: [(0, '>', 1)],
    }

    r_meta = compute_consensus_rewards(final_rankings)

    # Everyone agrees → perfect consensus → reward = 1.0
    assert all(abs(r - 1.0) < 0.01 for r in r_meta.values())


def test_acceptance_rewards():
    """Test acceptance reward for improvement."""
    v_t0 = np.array([0.4, 0.6, 0.5, 0.7])
    v_final = np.array([0.6, 0.5, 0.5, 0.8])  # Agents 0 and 3 improved

    r_accept = compute_acceptance_rewards(v_t0, v_final, bonus=0.5)

    assert len(r_accept) == 4
    # Agent 0 improved → gets bonus
    assert r_accept[0] == 0.5
    # Agent 1 got worse → no bonus
    assert r_accept[1] == 0.0
    # Agent 2 stayed same → no bonus
    assert r_accept[2] == 0.0
    # Agent 3 improved → gets bonus
    assert r_accept[3] == 0.5
