"""
Baseline environments for CANT using ground-truth rewards.
"""

from dataclasses import dataclass
from typing import Literal, Sequence

from tinker import types

from tinker_cookbook.recipes.cant.coordinator import CANTCoordinator
from tinker_cookbook.recipes.cant.env import CANTEnv, CANTEnvGroupBuilder, STOP_CONDITION
from tinker_cookbook.recipes.cant.metrics import compute_verifiable_metrics
from tinker_cookbook.recipes.cant.prompts import get_default_agent_personas
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.rl.types import Action, Env, StepResult, Trajectory


@dataclass
class SingleRoundCANTEnv(CANTEnv):
    """Single-round variant that only generates the initial solution."""

    async def step(self, action: Action) -> StepResult:
        if self._episode_done:
            return self._get_done_step()

        action_message = self.renderer.parse_response(action)[0]
        action_content = get_text_content(action_message)
        self.coordinator.add_round1_response(self.agent_id, action_content)

        self._step_count += 1
        self._episode_done = True

        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=types.ModelInput.empty(),
            next_stop_condition=STOP_CONDITION,
            metrics={
                "round": 0.0,
                "step": float(self._step_count),
            },
        )


@dataclass
class ThreeRoundCANTEnv(CANTEnv):
    """Three-round variant that stops after revision (round 2)."""

    async def step(self, action: Action) -> StepResult:
        if self._episode_done:
            return self._get_done_step()

        current_round = self.coordinator.get_current_round()
        action_message = self.renderer.parse_response(action)[0]
        action_content = get_text_content(action_message)

        if current_round == 0:
            self.coordinator.add_round1_response(self.agent_id, action_content)
        elif current_round == 1:
            self.coordinator.add_round2_response(self.agent_id, action_content)
        elif current_round == 2:
            self.coordinator.add_round3_response(self.agent_id, action_content)
        else:
            return self._get_done_step()

        self._step_count += 1

        if current_round == 2:
            self._episode_done = True
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=types.ModelInput.empty(),
                next_stop_condition=STOP_CONDITION,
                metrics={
                    "round": float(current_round),
                    "step": float(self._step_count),
                },
            )

        if self.coordinator.can_advance_round():
            self.coordinator.advance_round()

        next_obs = await self._get_observation()
        return StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=next_obs,
            next_stop_condition=STOP_CONDITION,
            metrics={
                "round": float(current_round),
                "step": float(self._step_count),
            },
        )


@dataclass
class BaselineVerifiableCANTEnvGroupBuilder(CANTEnvGroupBuilder):
    """Baseline builder that assigns rewards from ground-truth correctness."""

    ground_truth_mode: Literal["single_turn", "multi_round"] = "multi_round"

    async def make_envs(self) -> Sequence[Env]:
        question = self.problem_state["question"]
        answer = self.problem_state.get("answer")

        if answer is None:
            raise ValueError("Verifiable environment requires 'answer' field in problem_state")

        coordinator = CANTCoordinator(
            question=question,
            num_agents=self.num_agents,
            answer=answer,
        )

        env_cls = ThreeRoundCANTEnv
        if self.ground_truth_mode == "single_turn":
            if self.num_agents != 1:
                raise ValueError("ground_truth_mode='single_turn' requires num_agents=1")
            env_cls = SingleRoundCANTEnv

        personas = self.personas or get_default_agent_personas()
        envs = []
        for agent_id in range(self.num_agents):
            env = env_cls(
                agent_id=agent_id,
                coordinator=coordinator,
                renderer=self.renderer,
                persona=personas[agent_id % len(personas)],
                max_response_tokens=self.max_response_tokens,
            )
            envs.append(env)

        return envs

    async def compute_group_rewards(
        self,
        trajectory_group: Sequence[Trajectory],
        env_group: Sequence[Env],
    ) -> list[tuple[float, dict]]:
        coordinator = env_group[0].coordinator
        answer = coordinator.answer
        if answer is None:
            raise ValueError("Ground-truth rewards require answer in problem_state")

        dataset_name = self.problem_state.get("dataset_name")
        if self.ground_truth_mode == "single_turn":
            original_revised = coordinator.revised_solutions
            coordinator.revised_solutions = coordinator.initial_solutions
            try:
                per_agent_metrics = compute_verifiable_metrics(
                    coordinator=coordinator,
                    answer=answer,
                    num_agents=self.num_agents,
                    dataset_name=dataset_name,
                )
            finally:
                coordinator.revised_solutions = original_revised
        else:
            per_agent_metrics = compute_verifiable_metrics(
                coordinator=coordinator,
                answer=answer,
                num_agents=self.num_agents,
                dataset_name=dataset_name,
            )
        dataset_label = dataset_name or "unknown"

        rewards_with_metrics: list[tuple[float, dict]] = []
        for metrics in per_agent_metrics:
            correct_key = f"{dataset_label}/correct"
            correct = float(metrics.get(correct_key, 0.0))
            reward_value = 1.0 if correct > 0.5 else 0.0
            rewards_with_metrics.append((reward_value, metrics))

        return rewards_with_metrics
