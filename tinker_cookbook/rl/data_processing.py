"""
Data processing functions for RL training.

Contains functions for computing advantages, converting trajectories to training data,
and assembling training batches.
"""

import logging
from typing import List

import tinker
import torch
from tinker import TensorData

from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup
from tinker_cookbook.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets,
)
from tinker_cookbook.utils.misc_utils import all_same, safezip

logger = logging.getLogger(__name__)


def compute_advantages(trajectory_groups_P: List[TrajectoryGroup]) -> List[torch.Tensor]:
    """Compute advantages for each trajectory, centered within groups."""
    advantages_P: list[torch.Tensor] = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        # Center advantages within the group
        advantages_G = rewards_G - rewards_G.mean()
        advantages_P.append(advantages_G)

    return advantages_P


def compute_stepwise_advantages(
    trajectory_groups_P: List[TrajectoryGroup],
) -> List[List[List[float]]]:
    """Compute per-step advantages for each trajectory, centered within groups.

    Unlike compute_advantages() which collapses all step rewards into a single
    trajectory-level advantage, this function preserves per-step granularity.
    This enables proper credit assignment in multi-step environments like
    multi-agent debate, where different steps may have different rewards.

    Returns:
        advantages_P_G_S: A nested list where advantages_P_G_S[p][g][s] is the
        advantage for problem p, trajectory (agent) g, step s.
        Each step's advantage is the step's reward, centered by subtracting
        the mean reward across all steps of all trajectories in the group.

    Example:
        For a group with 3 agents, each with 3 steps:
        - Agent 0 step rewards: [0, +1, 0]
        - Agent 1 step rewards: [-1, -1, 0]
        - Agent 2 step rewards: [-1, +0.5, +1]

        Mean across all 9 steps = -0.167

        Centered advantages:
        - Agent 0: [+0.167, +1.167, +0.167]
        - Agent 1: [-0.833, -0.833, +0.167]
        - Agent 2: [-0.833, +0.667, +1.167]
    """
    advantages_P_G_S: List[List[List[float]]] = []

    for traj_group in trajectory_groups_P:
        # Collect all per-step rewards across all trajectories in this group
        all_step_rewards: List[List[float]] = []
        for traj in traj_group.trajectories_G:
            step_rewards = [t.reward for t in traj.transitions]
            all_step_rewards.append(step_rewards)

        # Compute mean across all steps of all trajectories for centering
        all_rewards_flat = [r for step_rewards in all_step_rewards for r in step_rewards]
        if all_rewards_flat:
            mean_reward = sum(all_rewards_flat) / len(all_rewards_flat)
        else:
            mean_reward = 0.0

        # Center each step's reward
        group_advantages: List[List[float]] = []
        for step_rewards in all_step_rewards:
            centered = [r - mean_reward for r in step_rewards]
            group_advantages.append(centered)

        advantages_P_G_S.append(group_advantages)

    return advantages_P_G_S


FlatObElem = int | tinker.ModelInputChunk
FlatOb = list[FlatObElem]


def _is_prefix(seq1: FlatOb, seq2: FlatOb) -> bool:
    """
    Check if seq1 is a prefix of seq2.
    """
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def _flat_ob_token_len(flat_ob: FlatOb) -> int:
    out = 0
    for elem in flat_ob:
        if isinstance(elem, int):
            out += 1
        else:
            out += elem.length
    return out


def _flat_ob_to_model_input(flat_ob: FlatOb) -> tinker.ModelInput:
    out: list[tinker.ModelInputChunk] = []
    current_text_chunk: list[int] = []

    def flush_text_chunk():
        if current_text_chunk:
            out.append(tinker.EncodedTextChunk(tokens=current_text_chunk))
            current_text_chunk.clear()

    for elem in flat_ob:
        if isinstance(elem, int):
            current_text_chunk.append(elem)
        else:
            flush_text_chunk()
            out.append(elem)
    flush_text_chunk()
    return tinker.ModelInput(chunks=out)


def _flatten_chunks(chunks: list[tinker.ModelInputChunk]) -> FlatOb:
    out: FlatOb = []
    for chunk in chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            out.extend(chunk.tokens)
        else:
            out.append(chunk)
    return out


def trajectory_to_data(traj: Trajectory, traj_advantage: float) -> list[tinker.Datum]:
    """
    Return one or more Datum objects corresponding to the trajectory.
    If the sequence grows by appending, i.e., each successive observation contains
    the previous observation+action as a prefix, then we can return a single Datum.
    However, if we get a sequence that's not an extension of the previous sequence,
    then that results in a new Datum.

    For example, let O1 denote a chunk of observation tokens, and let A1 denote an action.

    Then let's say ob_ac_pairs is as follows.

    (O1, A1)
    (O1+A1+O2, A2)
    (O3, A3)

    Then we will merge the first two observation-action pairs into a single Datum,
    and the last observation-action pair into a separate Datum.
    """

    class SequenceAccumulator:
        full_sequence: list[FlatObElem] = []
        sampled_logprobs: list[float] = []
        advantages: list[float] = []
        mask: list[float] = []

        @classmethod
        def clear(cls):
            cls.full_sequence = []
            cls.sampled_logprobs = []
            cls.advantages = []
            cls.mask = []

    def make_datum_from_state():
        all_tokens_T = _flat_ob_to_model_input(SequenceAccumulator.full_sequence)
        input_tokens_T, target_tokens_T = create_rightshifted_model_input_and_leftshifted_targets(
            list(all_tokens_T.chunks)
        )
        sampled_logprobs_T = SequenceAccumulator.sampled_logprobs[1:]
        advantages_T = SequenceAccumulator.advantages[1:]
        mask_T = SequenceAccumulator.mask[1:]
        assert (
            input_tokens_T.length
            == len(target_tokens_T)
            == len(sampled_logprobs_T)
            == len(advantages_T)
            == len(mask_T)
        )
        return tinker.Datum(
            model_input=input_tokens_T,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens_T)),
                "logprobs": TensorData.from_torch(torch.tensor(sampled_logprobs_T)),
                "advantages": TensorData.from_torch(torch.tensor(advantages_T)),
                "mask": TensorData.from_torch(torch.tensor(mask_T)),
            },
        )

    data: list[tinker.Datum] = []
    for transition in traj.transitions:
        ob = transition.ob
        ob_flat = _flatten_chunks(ob.chunks)
        ac_with_logprobs = transition.ac
        if len(SequenceAccumulator.full_sequence) == 0:
            delta_ob_flat = ob_flat
        elif _is_prefix(SequenceAccumulator.full_sequence, ob_flat):
            delta_ob_flat = ob_flat[len(SequenceAccumulator.full_sequence) :]
        else:
            data.append(make_datum_from_state())
            SequenceAccumulator.clear()
            delta_ob_flat = ob_flat
        delta_ob_len = _flat_ob_token_len(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(ac_with_logprobs.tokens)
        SequenceAccumulator.sampled_logprobs.extend(
            [0.0] * delta_ob_len + ac_with_logprobs.logprobs
        )
        SequenceAccumulator.advantages.extend(
            [0] * delta_ob_len + [traj_advantage] * len(ac_with_logprobs.tokens)
        )
        SequenceAccumulator.mask.extend([0.0] * delta_ob_len + [1.0] * len(ac_with_logprobs.tokens))

    if SequenceAccumulator.full_sequence:
        data.append(make_datum_from_state())

    return data


def trajectory_to_data_stepwise(
    traj: Trajectory, step_advantages: List[float]
) -> list[tinker.Datum]:
    """
    Return one or more Datum objects corresponding to the trajectory, with
    per-step advantages.

    Unlike trajectory_to_data() which applies the same advantage to all tokens,
    this function assigns each step's tokens their specific step advantage.
    This enables proper credit assignment in multi-step environments.

    Args:
        traj: Trajectory with multiple transitions (steps).
        step_advantages: Per-step advantages, one value per transition in traj.
            Each step's action tokens will be assigned that step's advantage.

    Example:
        If step_advantages = [-0.5, +1.0, +0.2] for a 3-step trajectory:
        - Step 0's 100 action tokens get advantage -0.5
        - Step 1's 150 action tokens get advantage +1.0
        - Step 2's 120 action tokens get advantage +0.2
    """
    assert len(step_advantages) == len(traj.transitions), (
        f"step_advantages length ({len(step_advantages)}) must match "
        f"number of transitions ({len(traj.transitions)})"
    )

    class SequenceAccumulator:
        full_sequence: list[FlatObElem] = []
        sampled_logprobs: list[float] = []
        advantages: list[float] = []
        mask: list[float] = []

        @classmethod
        def clear(cls):
            cls.full_sequence = []
            cls.sampled_logprobs = []
            cls.advantages = []
            cls.mask = []

    def make_datum_from_state():
        all_tokens_T = _flat_ob_to_model_input(SequenceAccumulator.full_sequence)
        input_tokens_T, target_tokens_T = create_rightshifted_model_input_and_leftshifted_targets(
            list(all_tokens_T.chunks)
        )
        sampled_logprobs_T = SequenceAccumulator.sampled_logprobs[1:]
        advantages_T = SequenceAccumulator.advantages[1:]
        mask_T = SequenceAccumulator.mask[1:]
        assert (
            input_tokens_T.length
            == len(target_tokens_T)
            == len(sampled_logprobs_T)
            == len(advantages_T)
            == len(mask_T)
        )
        return tinker.Datum(
            model_input=input_tokens_T,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens_T)),
                "logprobs": TensorData.from_torch(torch.tensor(sampled_logprobs_T)),
                "advantages": TensorData.from_torch(torch.tensor(advantages_T)),
                "mask": TensorData.from_torch(torch.tensor(mask_T)),
            },
        )

    data: list[tinker.Datum] = []
    for step_idx, transition in enumerate(traj.transitions):
        ob = transition.ob
        ob_flat = _flatten_chunks(ob.chunks)
        ac_with_logprobs = transition.ac
        step_adv = step_advantages[step_idx]  # Use this step's specific advantage
        if len(SequenceAccumulator.full_sequence) == 0:
            delta_ob_flat = ob_flat
        elif _is_prefix(SequenceAccumulator.full_sequence, ob_flat):
            delta_ob_flat = ob_flat[len(SequenceAccumulator.full_sequence) :]
        else:
            data.append(make_datum_from_state())
            SequenceAccumulator.clear()
            delta_ob_flat = ob_flat
        delta_ob_len = _flat_ob_token_len(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(ac_with_logprobs.tokens)
        SequenceAccumulator.sampled_logprobs.extend(
            [0.0] * delta_ob_len + ac_with_logprobs.logprobs
        )
        SequenceAccumulator.advantages.extend(
            [0] * delta_ob_len + [step_adv] * len(ac_with_logprobs.tokens)
        )
        SequenceAccumulator.mask.extend([0.0] * delta_ob_len + [1.0] * len(ac_with_logprobs.tokens))

    if SequenceAccumulator.full_sequence:
        data.append(make_datum_from_state())

    return data


def assemble_training_data(
    trajectory_groups_P: List[TrajectoryGroup],
    advantages_P: List[torch.Tensor],
) -> tuple[List[tinker.Datum], List[dict[str, int]]]:
    """Convert trajectories to training data format."""
    data_D: list[tinker.Datum] = []
    metadata_D: list[dict[str, int]] = []

    for i_group, (traj_group, advantages_G) in enumerate(
        safezip(trajectory_groups_P, advantages_P)
    ):
        for i_traj, (traj, traj_advantage) in enumerate(
            safezip(traj_group.trajectories_G, advantages_G)
        ):
            # Build the full sequence from the trajectory
            new_data = trajectory_to_data(traj, float(traj_advantage))
            data_D.extend(new_data)
            metadata_D.extend([dict(group_idx=i_group, traj_idx=i_traj) for _ in new_data])

    return data_D, metadata_D


def assemble_training_data_stepwise(
    trajectory_groups_P: List[TrajectoryGroup],
    advantages_P_G_S: List[List[List[float]]],
) -> tuple[List[tinker.Datum], List[dict[str, int]]]:
    """Convert trajectories to training data format with per-step advantages.

    Unlike assemble_training_data() which uses a single advantage per trajectory,
    this function uses per-step advantages for proper credit assignment.

    Args:
        trajectory_groups_P: List of trajectory groups, one per problem.
        advantages_P_G_S: Per-step advantages where advantages_P_G_S[p][g][s] is
            the advantage for problem p, trajectory g, step s.

    Returns:
        Tuple of (data_D, metadata_D) where data_D is a list of Datum objects
        and metadata_D contains group/trajectory indices for each datum.
    """
    data_D: list[tinker.Datum] = []
    metadata_D: list[dict[str, int]] = []

    for i_group, (traj_group, advantages_G_S) in enumerate(
        safezip(trajectory_groups_P, advantages_P_G_S)
    ):
        for i_traj, (traj, step_advantages) in enumerate(
            safezip(traj_group.trajectories_G, advantages_G_S)
        ):
            # Build the full sequence from the trajectory with per-step advantages
            new_data = trajectory_to_data_stepwise(traj, step_advantages)
            data_D.extend(new_data)
            metadata_D.extend([dict(group_idx=i_group, traj_idx=i_traj) for _ in new_data])

    return data_D, metadata_D


def remove_constant_reward_groups(
    trajectory_groups_P: List[TrajectoryGroup],
) -> List[TrajectoryGroup]:
    new_groups: list[TrajectoryGroup] = []
    for group in trajectory_groups_P:
        if not all_same(group.get_total_rewards()):
            new_groups.append(group)
    if not new_groups:
        logger.warning("All rewards are uniform. There will be no gradient")
        return trajectory_groups_P[0:1]  # return singleton list in case empty
        # list will cause problems
    return new_groups


# =============================================================================
# Dual Advantage System (v2 Rewards)
# =============================================================================
# These functions support the v2 reward system for multi-agent debate, where
# generator tokens (<solution>/<evaluation>) and judge tokens (<comparison>)
# receive separate advantages computed from different reward signals.


def has_dual_rewards(trajectory_groups_P: List[TrajectoryGroup]) -> bool:
    """Check if trajectory groups have v2 dual rewards (gen_rewards + judge_rewards).

    Returns True if at least one trajectory has both gen_rewards and judge_rewards
    attributes set by the environment's _compute_rewards_v2() method.
    """
    for traj_group in trajectory_groups_P:
        for traj in traj_group.trajectories_G:
            if hasattr(traj, "gen_rewards") and hasattr(traj, "judge_rewards"):
                return True
    return False


def normalize_rewards_separately(
    trajectory_groups_P: List[TrajectoryGroup],
) -> tuple[List[List[List[float]]], List[List[List[float]]]]:
    """Compute centered advantages for gen_rewards and judge_rewards separately.

    For the two-stream reward system, generator and judge rewards have different meanings:
    - gen_rewards: Win rate for <solution>/<evaluation> tokens, measures peer approval
    - judge_rewards: Consensus alignment for <comparison> tokens, measures judgment accuracy

    Each reward type is centered independently WITHIN EACH GROUP by subtracting the
    group mean. This ensures advantages reflect relative performance among agents
    debating the SAME problem, which aligns with the debate structure and prevents
    problem difficulty from confounding the learning signal.

    This matches the behavior of compute_stepwise_advantages() but applied separately
    to each reward stream.

    Args:
        trajectory_groups_P: List of trajectory groups with gen_rewards/judge_rewards
            attached to each trajectory.

    Returns:
        Tuple of:
        - gen_advantages_P_G_S: Centered generator advantages [problem][traj][step]
        - judge_advantages_P_G_S: Centered judge advantages [problem][traj][step]
    """
    gen_advantages_P_G_S: List[List[List[float]]] = []
    judge_advantages_P_G_S: List[List[List[float]]] = []

    for traj_group in trajectory_groups_P:
        # Collect all rewards within this group only
        all_gen_rewards: List[float] = []
        all_judge_rewards: List[float] = []

        for traj in traj_group.trajectories_G:
            gen_rewards = getattr(traj, "gen_rewards", [])
            judge_rewards = getattr(traj, "judge_rewards", [])
            all_gen_rewards.extend(gen_rewards)
            all_judge_rewards.extend(judge_rewards)

        # Compute group-level means for centering
        gen_mean = sum(all_gen_rewards) / len(all_gen_rewards) if all_gen_rewards else 0.0
        judge_mean = sum(all_judge_rewards) / len(all_judge_rewards) if all_judge_rewards else 0.0

        # Build centered advantages within this group
        gen_group: List[List[float]] = []
        judge_group: List[List[float]] = []

        for traj in traj_group.trajectories_G:
            gen_rewards = getattr(traj, "gen_rewards", [])
            judge_rewards = getattr(traj, "judge_rewards", [])

            # Center by subtracting group mean (consistent with compute_stepwise_advantages)
            gen_adv = [r - gen_mean for r in gen_rewards]
            judge_adv = [r - judge_mean for r in judge_rewards]

            gen_group.append(gen_adv)
            judge_group.append(judge_adv)

        gen_advantages_P_G_S.append(gen_group)
        judge_advantages_P_G_S.append(judge_group)

    return gen_advantages_P_G_S, judge_advantages_P_G_S


def trajectory_to_data_dual_advantage(
    traj: Trajectory,
    gen_step_advantages: List[float],
    judge_step_advantages: List[float],
    lambda_gen: float = 1.0,
    lambda_judge: float = 1.0,
) -> list[tinker.Datum]:
    """Convert trajectory to training data with dual advantages for v2 rewards.

    This function assigns different advantages to different token types:
    - Generator tokens (most action tokens): Use gen_step_advantages
    - Judge tokens (<comparison> content): Use judge_step_advantages

    The final advantage for each token is:
        advantage = lambda_gen * gen_advantage  (for generator tokens)
        advantage = lambda_judge * judge_advantage  (for judge tokens)

    Since we don't have token-level type information here, we use a simplified
    approach: combine advantages with their respective lambda weights.

    For the v2 system, the reward computation in the environment already
    separates rewards by token type conceptually. Here we apply the weighted
    combination at training time.

    Args:
        traj: Trajectory with multiple transitions.
        gen_step_advantages: Per-step normalized generator advantages.
        judge_step_advantages: Per-step normalized judge advantages.
        lambda_gen: Weight for generator loss (default 1.0).
        lambda_judge: Weight for judge loss (default 1.0).

    Returns:
        List of Datum objects for training.
    """
    assert len(gen_step_advantages) == len(traj.transitions), (
        f"gen_step_advantages length ({len(gen_step_advantages)}) must match "
        f"number of transitions ({len(traj.transitions)})"
    )
    assert len(judge_step_advantages) == len(traj.transitions), (
        f"judge_step_advantages length ({len(judge_step_advantages)}) must match "
        f"number of transitions ({len(traj.transitions)})"
    )

    # For the v2 system, we combine gen and judge advantages at the step level.
    # The combined advantage represents the total learning signal for that step.
    #
    # Note: A more sophisticated implementation could do token-level assignment
    # by parsing the action tokens to identify <comparison> vs other tags.
    # For now, we use a weighted combination that captures both signals.
    combined_step_advantages = [
        lambda_gen * gen_adv + lambda_judge * judge_adv
        for gen_adv, judge_adv in safezip(gen_step_advantages, judge_step_advantages)
    ]

    # Use the existing stepwise function with combined advantages
    return trajectory_to_data_stepwise(traj, combined_step_advantages)


def assemble_training_data_dual_advantage(
    trajectory_groups_P: List[TrajectoryGroup],
    gen_advantages_P_G_S: List[List[List[float]]],
    judge_advantages_P_G_S: List[List[List[float]]],
    lambda_gen: float = 1.0,
    lambda_judge: float = 1.0,
) -> tuple[List[tinker.Datum], List[dict[str, int]]]:
    """Assemble training data with dual advantages for v2 reward system.

    Args:
        trajectory_groups_P: List of trajectory groups.
        gen_advantages_P_G_S: Normalized generator advantages [problem][traj][step].
        judge_advantages_P_G_S: Normalized judge advantages [problem][traj][step].
        lambda_gen: Weight for generator loss.
        lambda_judge: Weight for judge loss.

    Returns:
        Tuple of (data_D, metadata_D) for training.
    """
    data_D: list[tinker.Datum] = []
    metadata_D: list[dict[str, int]] = []

    for i_group, (traj_group, gen_advs_G_S, judge_advs_G_S) in enumerate(
        safezip(trajectory_groups_P, gen_advantages_P_G_S, judge_advantages_P_G_S)
    ):
        for i_traj, (traj, gen_step_advs, judge_step_advs) in enumerate(
            safezip(traj_group.trajectories_G, gen_advs_G_S, judge_advs_G_S)
        ):
            new_data = trajectory_to_data_dual_advantage(
                traj,
                gen_step_advs,
                judge_step_advs,
                lambda_gen=lambda_gen,
                lambda_judge=lambda_judge,
            )
            data_D.extend(new_data)
            metadata_D.extend([dict(group_idx=i_group, traj_idx=i_traj) for _ in new_data])

    return data_D, metadata_D
