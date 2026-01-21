"""
CANT environment for verifiable tasks (math problems, coding challenges, etc.).

This variant is similar to the base CANT environment but can optionally use ground truth
answers for additional reward signals or evaluation.
"""

from dataclasses import dataclass, field
from typing import Sequence

from tinker_cookbook.rl.types import Env, Trajectory

from tinker_cookbook.recipes.cant.env import CANTEnv, CANTEnvGroupBuilder
from tinker_cookbook.recipes.cant.coordinator import CANTCoordinator
from tinker_cookbook.recipes.cant.metrics import compute_verifiable_metrics


@dataclass
class VerifiableCANTEnv(CANTEnv):
    """
    CANT environment with support for verifiable ground truth.

    This is essentially the same as CANTEnv, but the coordinator
    has access to the ground truth answer for potential future use.
    """

    # Inherits all functionality from CANTEnv
    # The coordinator already has an 'answer' field that can store ground truth
    pass


@dataclass
class VerifiableCANTEnvGroupBuilder(CANTEnvGroupBuilder):
    """
    Builder for verifiable CANT environments.

    This variant can optionally incorporate correctness checking or ground truth
    comparisons. For now, it follows pure peer evaluation (no correctness rewards).

    Future extensions could add:
    - Bonus rewards for solutions matching ground truth
    - Correctness-weighted Bradley-Terry scoring
    - Hybrid peer + accuracy evaluation
    """

    # Additional configuration for verifiable tasks
    use_ground_truth_bonus: bool = field(default=False, kw_only=True)
    ground_truth_bonus_weight: float = field(default=0.0, kw_only=True)

    async def make_envs(self) -> Sequence[Env]:
        """
        Create a group of verifiable CANT environments.

        Args:
            problem_state: Dict containing 'question' and 'answer' (ground truth)

        Returns:
            List of VerifiableCANTEnv instances
        """
        question = self.problem_state["question"]
        answer = self.problem_state.get("answer")

        if answer is None:
            raise ValueError("Verifiable environment requires 'answer' field in problem_state")

        # Create shared coordinator with ground truth
        coordinator = CANTCoordinator(
            question=question,
            num_agents=self.num_agents,
            answer=answer,
        )

        # Create one environment per agent
        envs = []
        for agent_id in range(self.num_agents):
            env = VerifiableCANTEnv(
                agent_id=agent_id,
                coordinator=coordinator,
                renderer=self.renderer,
                persona=self.persona,
                max_response_tokens=self.max_response_tokens,
            )
            envs.append(env)

        return envs

    async def compute_group_rewards(
        self,
        trajectory_group: Sequence[Trajectory],
        env_group: Sequence[Env],
    ) -> list[tuple[float, dict]]:
        """
        Compute rewards for verifiable tasks.

        Currently follows pure peer evaluation (same as base CANT).
        Ground truth is available in coordinator.answer for future use.

        Args:
            trajectory_group: Trajectories for all agents
            env_group: Environment instances

        Returns:
            List of (final_reward, metrics) tuples
        """
        # Use base implementation (pure peer evaluation)
        metrics_list = await super().compute_group_rewards(trajectory_group, env_group)

        coordinator = env_group[0].coordinator
        answer = coordinator.answer
        if answer is not None:
            dataset_name = self.problem_state.get("dataset_name")
            per_agent_metrics = compute_verifiable_metrics(
                coordinator=coordinator,
                answer=answer,
                num_agents=self.num_agents,
                dataset_name=dataset_name,
            )
            if len(per_agent_metrics) == len(metrics_list):
                updated_metrics_list: list[tuple[float, dict]] = []
                for (final_reward, metrics), agent_metrics in zip(
                    metrics_list, per_agent_metrics, strict=True
                ):
                    merged = dict(metrics)
                    merged.update(agent_metrics)
                    updated_metrics_list.append((final_reward, merged))
                metrics_list = updated_metrics_list

        # Optionally add ground truth bonus (not implemented yet)
        if self.use_ground_truth_bonus and self.ground_truth_bonus_weight > 0:
            for _final_reward, metrics in metrics_list:
                metrics["ground_truth_bonus"] = 0.0  # Placeholder for future implementation
            # TODO: Implement correctness checking and bonus assignment
        return metrics_list
