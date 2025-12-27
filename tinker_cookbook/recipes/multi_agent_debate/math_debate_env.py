"""Math-focused multi-agent debate environment.

This environment combines:
1. Multi-agent debate structure (agents collaborate and critique each other)
2. Math problem-solving (with verifiable \\boxed{} answers)
3. Flexible reward: debate-based rewards for training, correctness for evaluation

Training mode:
- Uses self-play with debate rewards (agents learn to collaborate)
- Datasets: MATH, GSM8K, Polaris

Evaluation mode:
- Uses fixed opponent policies
- Uses correctness metrics (format + accuracy)
- Datasets: AIME 2024, AIME 2025, MATH-500
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import chz
import tinker
from tinker import types

from tinker_cookbook.completers import StopCondition, TinkerMessageCompleter
from tinker_cookbook.renderers import Message, Renderer, ensure_text, get_renderer
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)

from .env import MultiAgentCoordinator
from .prompts import ParsedResponse, parse_agent_response
from .reward import compute_pairwise_win_minus_loss
from .math_debate_prompts import (
    MATH_DEBATE_SYSTEM_PROMPT,
    MATH_DEBATE_USER_PROMPT_FIRST_TURN,
    MATH_DEBATE_USER_PROMPT_LATER_TURN,
    format_math_debate_history,
)
from .math_debate_datasets import MathProblem, load_math_dataset


# Stop when we see the closing tag
_STOP_CONDITION_TEXT = ["</comparison>"]


def _get_debate_stop_condition(renderer: Renderer) -> StopCondition:
    renderer_stop = renderer.get_stop_sequences()
    if not renderer_stop:
        return _STOP_CONDITION_TEXT
    if isinstance(renderer_stop[0], int):
        return renderer_stop
    return list(dict.fromkeys([*renderer_stop, *_STOP_CONDITION_TEXT]))


def _get_summarizer_stop_condition(renderer: Renderer) -> StopCondition:
    renderer_stop = renderer.get_stop_sequences()
    return renderer_stop or []


def safe_grade(
    given_answer: str,
    ground_truth: str,
    grader: Literal["sympy", "math_verify"] = "sympy",
    timeout: float = 1.0,
) -> bool:
    """Grade a math answer with timeout protection."""
    if grader == "sympy":
        grader_func = grade_answer
    elif grader == "math_verify":
        grader_func = grade_answer_math_verify
    else:
        raise ValueError(f"Invalid grader: {grader}")

    out = run_with_timeout_signal(
        grader_func,
        args=(given_answer, ground_truth),
        timeout_seconds=int(math.ceil(timeout)),
    )
    if out is None:
        return False
    return out


def _extract_boxed_from_solution(solution: str) -> tuple[bool, str | None]:
    """Extract boxed answer from solution.

    Returns:
        (has_boxed_format, extracted_answer)
    """
    try:
        return True, extract_boxed(solution)
    except ValueError:
        return False, None


@dataclass
class MathDebateEnv(Env):
    """Environment for one agent in a math-focused multi-agent debate."""

    agent_id: int
    coordinator: MultiAgentCoordinator
    renderer: Renderer
    self_play: bool = True
    opponent_policies: list[TinkerMessageCompleter] | None = None
    history_turns: int = 2
    summarize_history: bool = False
    summarize_model: str | None = "Qwen/Qwen3-4B-Instruct-2507"
    model_name: str | None = None

    def __post_init__(self) -> None:
        if self.self_play:
            if self.opponent_policies is not None:
                raise ValueError("self_play=True requires opponent_policies=None")
        else:
            if self.opponent_policies is None:
                raise ValueError("Need opponent_policies for non-self-play")
            if len(self.opponent_policies) != self.coordinator.state.num_agents - 1:
                raise ValueError("Need N-1 opponent policies for non-self-play")

    @property
    def stop_condition(self) -> StopCondition:
        return _get_debate_stop_condition(self.renderer)

    def get_system_prompt(self) -> str:
        return MATH_DEBATE_SYSTEM_PROMPT.format(agent_id=self.agent_id)

    async def _summarize(self, history: str) -> str:
        """Summarize conversation history if it gets too long."""
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(
            base_model=self.summarize_model or self.model_name
        )
        summarizer_policy = TinkerMessageCompleter(
            sampling_client=sampling_client,
            renderer=self.renderer,
            max_tokens=892,
            stop_condition=_get_summarizer_stop_condition(self.renderer),
        )
        messages: list[Message] = [
            {
                "role": "system",
                "content": (
                    "You summarize multi-agent math problem-solving discussions.\n"
                    "Preserve:\n"
                    "- The problem statement\n"
                    "- Each agent's key solution approach\n"
                    "- Any identified errors or corrections\n"
                    "- Different proposed answers\n"
                    "Output plain text only."
                ),
            },
            {"role": "user", "content": history},
        ]
        resp = await summarizer_policy(messages)
        return ensure_text(resp["content"]).strip()

    async def get_conversation_context(self) -> str:
        """Get the formatted conversation history."""
        if not self.coordinator.state.agent_responses:
            return ""

        question = self.coordinator.state.question
        turns = list(self.coordinator.state.agent_responses)

        if not self.summarize_history:
            return format_math_debate_history(turns, num_recent_turns=len(turns))

        if self.history_turns < 0:
            return format_math_debate_history(turns, num_recent_turns=len(turns))
        elif self.history_turns == 0:
            older = turns
            raw_recent = []
        else:
            raw_recent = turns[-self.history_turns :] if len(turns) > self.history_turns else turns
            older = turns[: -self.history_turns] if len(turns) > self.history_turns else []

        summary = ""
        if older:
            older_text = f"Problem: {question}\n\n{format_math_debate_history(older, len(older))}"
            summary = await self._summarize(older_text)

        parts: list[str] = []
        if summary:
            parts.append("--- Summary of earlier discussion ---\n" + summary)
        if raw_recent:
            parts.append("\n--- Recent solutions ---\n" + format_math_debate_history(raw_recent, len(raw_recent)))

        return "\n".join(parts).rstrip()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        if self.agent_id != 0:
            await self.wait_for_turn()
        return await self.get_observation(), self.stop_condition

    async def wait_for_turn(self) -> None:
        if not self.coordinator.done:
            if self.self_play:
                await self.coordinator.wait_for_turn(self.agent_id)
            else:
                await self.run_opponent_steps()

    async def run_opponent_steps(self) -> None:
        """Run opponent policies (for evaluation mode)."""
        assert not self.self_play and self.opponent_policies is not None

        while not self.coordinator.done and self.coordinator.current_agent_id != self.agent_id:
            opponent_agent_id = self.coordinator.current_agent_id
            policy_idx = (
                opponent_agent_id if opponent_agent_id < self.agent_id else opponent_agent_id - 1
            )
            opponent_policy = self.opponent_policies[policy_idx]
            observation_str = await self.get_observation_string()

            messages: list[Message] = [
                {
                    "role": "system",
                    "content": MATH_DEBATE_SYSTEM_PROMPT.format(agent_id=opponent_agent_id),
                },
                {"role": "user", "content": observation_str},
            ]
            opponent_response = await opponent_policy(messages)
            opponent_content = ensure_text(opponent_response["content"])

            try:
                await self.coordinator.submit_response(
                    opponent_agent_id,
                    opponent_content,
                    observation=observation_str,
                )
            except ValueError:
                await self.coordinator.abort()
                return

    async def get_observation_string(self) -> str:
        """Get the current observation as a string."""
        turn_idx = self.coordinator.state.current_turn
        cycle_idx = self.coordinator.state.get_current_cycle()
        max_cycles = self.coordinator.state.max_turns // self.coordinator.state.num_agents
        question = self.coordinator.state.question

        if turn_idx == 0:
            # First turn - no history yet
            return MATH_DEBATE_USER_PROMPT_FIRST_TURN.format(
                question=question,
                max_turns=self.coordinator.state.max_turns,
                cycle=cycle_idx + 1,
                max_cycles=max_cycles,
            )

        # Later turns - include history
        history = await self.get_conversation_context()
        return MATH_DEBATE_USER_PROMPT_LATER_TURN.format(
            question=question,
            history=history,
            turn=turn_idx + 1,
            max_turns=self.coordinator.state.max_turns,
            cycle=cycle_idx + 1,
            max_cycles=max_cycles,
        )

    async def get_observation(self) -> types.ModelInput:
        observation_str = await self.get_observation_string()
        messages: list[Message] = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": observation_str},
        ]
        return self.renderer.build_generation_prompt(messages)

    async def step(self, action: Action) -> StepResult:
        if self.coordinator.done:
            return self.get_done_step()

        observation_str = await self.get_observation_string()

        action_message: Message = self.renderer.parse_response(action)[0]
        action_content = ensure_text(action_message["content"])

        try:
            await self.coordinator.submit_response(
                self.agent_id,
                action_content,
                observation=observation_str,
            )
        except ValueError:
            await self.coordinator.abort()
            return StepResult(
                reward=-1.0,
                episode_done=True,
                next_observation=types.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={"parse_error": 1.0},
            )

        await self.wait_for_turn()
        return StepResult(
            reward=0.0,
            episode_done=self.coordinator.done,
            next_observation=await self.get_observation(),
            next_stop_condition=self.stop_condition,
            metrics={"cycle": float(self.coordinator.state.get_current_cycle())},
        )

    def get_done_step(self) -> StepResult:
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=types.ModelInput.empty(),
            next_stop_condition=_STOP_CONDITION_TEXT,
            metrics={},
        )


def _latest_response_by_author(
    agent_responses: list[ParsedResponse], *, author_id: int
) -> ParsedResponse | None:
    """Get the most recent response from a specific agent."""
    for resp in reversed(agent_responses):
        if resp.author_id == author_id:
            return resp
    return None


@dataclass
class MathDebateEnvGroupBuilder(EnvGroupBuilder):
    """Builder for math debate environment groups."""

    problems: list[MathProblem]
    problem_index: int
    renderer: Renderer
    num_agents: int
    max_rounds: int
    self_play: bool
    history_turns: int = 2
    summarize_history: bool = False
    summarize_model: str | None = "Qwen/Qwen3-4B-Instruct-2507"
    log_full_transcript: bool = False
    opponent_policies_all: list[TinkerMessageCompleter] | None = None
    model_name: str | None = None

    # Reward configuration
    reward_mode: Literal["debate", "correctness"] = "debate"
    format_coef: float = 0.1
    grader: Literal["sympy", "math_verify"] = "sympy"
    grade_timeout: float = 1.0

    async def make_envs(self) -> Sequence[Env]:
        problem = self.problems[self.problem_index % len(self.problems)]
        question = problem.problem
        if not question.rstrip().endswith("\\boxed{} format."):
            question = question.rstrip() + " Write your answer in \\boxed{} format."

        max_turns = self.num_agents * self.max_rounds

        if self.self_play:
            coordinator = MultiAgentCoordinator(
                question=question, num_agents=self.num_agents, max_turns=max_turns
            )
            return [
                MathDebateEnv(
                    agent_id=i,
                    coordinator=coordinator,
                    renderer=self.renderer,
                    self_play=True,
                    opponent_policies=None,
                    history_turns=self.history_turns,
                    summarize_history=self.summarize_history,
                    summarize_model=self.summarize_model,
                    model_name=self.model_name,
                )
                for i in range(self.num_agents)
            ]

        if self.opponent_policies_all is None:
            raise ValueError("opponent_policies_all is required for non-self-play evaluation")

        envs: list[Env] = []
        for i in range(self.num_agents):
            coordinator = MultiAgentCoordinator(
                question=question, num_agents=self.num_agents, max_turns=max_turns
            )
            opponent_policies_for_i = [
                p for j, p in enumerate(self.opponent_policies_all) if j != i
            ]
            envs.append(
                MathDebateEnv(
                    agent_id=i,
                    coordinator=coordinator,
                    renderer=self.renderer,
                    self_play=False,
                    opponent_policies=opponent_policies_for_i,
                    history_turns=self.history_turns,
                    summarize_history=self.summarize_history,
                    summarize_model=self.summarize_model,
                    model_name=self.model_name,
                )
            )
        return envs

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """Compute rewards for all agents in the group.

        For training (reward_mode='debate'):
            Uses debate-based rewards (win-loss from comparisons)

        For evaluation (reward_mode='correctness'):
            Uses correctness-based rewards (format + accuracy)
        """
        assert env_group, "empty env_group"

        if self.reward_mode == "correctness":
            return await self._compute_correctness_rewards(env_group)
        else:
            return await self._compute_debate_rewards(env_group)

    async def _compute_correctness_rewards(
        self, env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """Compute correctness-based rewards (for evaluation)."""
        problem = self.problems[self.problem_index % len(self.problems)]
        ground_truth = problem.answer

        def compute_one_agent_reward(
            *,
            coordinator: MultiAgentCoordinator,
            agent_id: int,
        ) -> tuple[float, Metrics]:
            latest = _latest_response_by_author(coordinator.state.agent_responses, author_id=agent_id)
            if latest is None:
                reward = -self.format_coef
                return (
                    reward,
                    {
                        "agent_id": float(agent_id),
                        "format": 0.0,
                        "correct": 0.0,
                        "correctness_reward": reward,
                        "missing_response": 1.0,
                    },
                )

            has_box, boxed = _extract_boxed_from_solution(latest.solution)
            correct = False
            if boxed is not None:
                correct = safe_grade(
                    boxed,
                    ground_truth,
                    grader=self.grader,
                    timeout=self.grade_timeout,
                )
            correct_f = 1.0 if correct else 0.0
            format_f = 1.0 if has_box else 0.0
            reward = self.format_coef * (format_f - 1.0) + correct_f
            return (
                reward,
                {
                    "agent_id": float(agent_id),
                    "format": format_f,
                    "correct": correct_f,
                    "correctness_reward": reward,
                    "missing_response": 0.0,
                },
            )

        if self.self_play:
            env0 = env_group[0]
            assert isinstance(env0, MathDebateEnv)
            coordinator = env0.coordinator

            if self.log_full_transcript:
                self._log_transcript(coordinator, "Correctness Evaluation")

            rewards_and_metrics: list[tuple[float, Metrics]] = []
            for agent_id in range(self.num_agents):
                reward, metrics = compute_one_agent_reward(coordinator=coordinator, agent_id=agent_id)
                with logtree.scope_header(f"Correctness Reward (Agent {agent_id})"):
                    logtree.log_text(f"Ground truth: {ground_truth}")
                    logtree.log_text(
                        f"Format ok: {metrics['format']}, Correct: {metrics['correct']}, Reward: {reward}"
                    )
                rewards_and_metrics.append((reward, metrics))
            return rewards_and_metrics

        rewards_and_metrics: list[tuple[float, Metrics]] = []
        for env in env_group:
            assert isinstance(env, MathDebateEnv)
            coordinator = env.coordinator
            if self.log_full_transcript:
                self._log_transcript(coordinator, f"Agent {env.agent_id} Transcript")

            reward, metrics = compute_one_agent_reward(coordinator=coordinator, agent_id=env.agent_id)
            rewards_and_metrics.append((reward, metrics))

        return rewards_and_metrics

    async def _compute_debate_rewards(
        self, env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """Compute debate-based rewards (for training)."""
        env0 = env_group[0]
        assert isinstance(env0, MathDebateEnv)
        coordinator = env0.coordinator

        if self.log_full_transcript:
            self._log_transcript(coordinator, "Training Debate")

        # Compute debate rewards from comparisons
        rewards_and_metrics = compute_pairwise_win_minus_loss(
            coordinator.state.agent_responses,
            num_agents=self.num_agents,
        )

        # Log rewards
        for agent_id, (reward, metrics) in enumerate(rewards_and_metrics):
            with logtree.scope_header(f"Debate Reward (Agent {agent_id})"):
                logtree.log_text(f"Win-Loss: {metrics.get('win_minus_loss', 0.0)}, Reward: {reward}")

        return rewards_and_metrics

    def _log_transcript(self, coordinator: MultiAgentCoordinator, title: str) -> None:
        """Log the full debate transcript."""
        with logtree.scope_header(title):
            logtree.log_text(f"Question: {coordinator.state.question}")
            turns = coordinator.state.agent_responses
            if not turns:
                logtree.log_text("(No responses captured)")
            for turn_idx, response in enumerate(turns, start=1):
                with logtree.scope_header(f"Turn {turn_idx} (Agent {response.author_id})"):
                    if response.observation:
                        with logtree.scope_details("Observation"):
                            logtree.log_text(response.observation)
                    with logtree.scope_details("Solution"):
                        logtree.log_text(response.solution)
                    with logtree.scope_details("Evaluation"):
                        logtree.log_text(response.evaluation)
                    if response.comparison_text:
                        with logtree.scope_details("Comparison"):
                            logtree.log_text(response.comparison_text)
