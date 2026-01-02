"""Verifiable (math-style) multi-agent debate environment for Tinker RL.

This mirrors the multi-agent debate coordinator/turn-taking logic, but uses a
math-style verifiable objective:
- Each agent must produce a final answer in \\boxed{...} format in <solution>.
- At episode end, each agent is rewarded by correctness (with a small format penalty).

This module is intentionally separate from the non-verifiable debate env to keep the
implementations isolated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import chz
from tinker import types

from tinker_cookbook.completers import StopCondition
from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)
from tinker_cookbook.renderers import Renderer, get_renderer
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

from .base_env import BaseMultiAgentDebateEnv, BaseMultiAgentEnvGroupBuilder
from .coordinator import MultiAgentCoordinator
from .loaders import load_math_problems_from_jsonl
from .prompts import VERIFIABLE_AGENT_SYSTEM_PROMPT, ParsedResponse
from .utils import (
    STOP_CONDITION,
    get_debate_stop_condition,
    log_debate_evaluation_final_solutions,
    log_debate_transcript,
    log_direct_evaluation,
)


def safe_grade(
    given_answer: str,
    ground_truth: str,
    grader: Literal["sympy", "math_verify"] = "sympy",
    timeout: float = 1.0,
) -> bool:
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


@dataclass(frozen=True)
class VerifiableMathProblem:
    problem: str
    answer: str
    dataset_name: str = "math"


@dataclass
class DirectMathEvaluationEnv(Env):
    """Simple single-turn environment for direct math problem evaluation."""

    problem: VerifiableMathProblem
    renderer: Renderer

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        messages = [
            {
                "role": "system",
                "content": "Solve the following math problem. Write your final answer in \\boxed{} format. No need to do extended reasoning.",
            },
            {
                "role": "user",
                "content": self.problem.problem,
            },
        ]
        return (
            self.renderer.build_generation_prompt(messages),
            get_debate_stop_condition(self.renderer),
        )

    async def step(self, action: Action) -> StepResult:
        # Episode ends immediately after first response
        return StepResult(
            reward=0.0,  # Reward computed in compute_group_rewards
            episode_done=True,
            next_observation=types.ModelInput.empty(),
            next_stop_condition=STOP_CONDITION,
            metrics={},
        )


@dataclass
class VerifiableMultiAgentDebateEnv(BaseMultiAgentDebateEnv):
    """Environment for one agent in a verifiable multi-agent debate."""

    @property
    def stop_condition(self) -> StopCondition:
        return get_debate_stop_condition(self.renderer)

    def get_system_prompt(self) -> str:
        return VERIFIABLE_AGENT_SYSTEM_PROMPT.format(agent_id=self.agent_id)

    def _format_turns(self, turns: list[ParsedResponse]) -> str:
        if not turns:
            return ""
        lines: list[str] = ["--- HISTORY OF PREVIOUS TURNS ---"]
        for turn_idx, response in enumerate(turns, start=1):
            lines.append(f"== Turn {turn_idx} (Agent {response.author_id}) ==")
            lines.append(f"Agent {response.author_id}'s Solution:")
            lines.append(response.solution.rstrip())
            lines.append(f"Agent {response.author_id}'s Evaluation:")
            lines.append(response.evaluation.rstrip())
            if response.comparison_text:
                lines.append(f"Agent {response.author_id}'s Comparison:")
                lines.append(response.comparison_text.rstrip())
            lines.append("")
        lines.append("--- END OF HISTORY ---\n")
        return "\n".join(lines).rstrip()

    async def get_conversation_context(self) -> str:
        """Format the conversation context for this agent.

        Shows recent history_turns. If summarize_history is True, summarizes them.
        Otherwise, keeps them verbatim.
        """
        if not self.coordinator.state.agent_responses:
            return ""

        question = self.coordinator.state.question
        turns = list(self.coordinator.state.agent_responses)

        # Get recent turns based on history_turns setting
        if self.history_turns < 0:
            # Show all turns
            recent_turns = turns
        elif self.history_turns == 0:
            # Show no turns
            recent_turns = []
        else:
            # Show last N turns
            recent_turns = (
                turns[-self.history_turns :] if len(turns) > self.history_turns else turns
            )

        history_text = f"USER QUERY: {question}\n{self._format_turns(recent_turns)}".rstrip()

        return await self._summarize(history_text) if self.summarize_history else history_text

    async def get_observation_string(self) -> str:
        history = await self.get_conversation_context()
        turn_idx = self.coordinator.state.current_turn
        # First turn prompt
        if turn_idx == 0:
            return (
                f"USER QUERY: {self.coordinator.state.question}\n"
                f"Agent {self.coordinator.state.current_agent_id}, please proceed with your response.\n\n"
                'Set <evaluation> to "N/A" and <comparison> to "N/A" as this is the first turn.\n'
                "Noted that your <solution> must include your final answer in \\boxed{...} format;"
            )
        # Regular turn prompt
        return (
            f"{history}\n\n Agent {self.coordinator.state.current_agent_id}, it is your turn.\n"
            "Please continue the debate by providing your solution, evaluation, and comparison."
            "Noted that your <solution> must include your final answer in \\boxed{...} format; and the do not include Agent {agent_id} in your comparisons."
        )


def _latest_response_by_author(
    agent_responses: list[ParsedResponse], *, author_id: int
) -> ParsedResponse | None:
    for resp in reversed(agent_responses):
        if resp.author_id == author_id:
            return resp
    return None


def _extract_boxed_from_solution(solution: str) -> tuple[bool, str | None]:
    try:
        return True, extract_boxed(solution)
    except ValueError:
        return False, None


@dataclass
class VerifiableMultiAgentEnvGroupBuilder(BaseMultiAgentEnvGroupBuilder):
    problems: list[VerifiableMathProblem]
    problem_index: int
    max_rounds: int
    self_play: bool
    history_turns: int = 2
    summarize_history: bool = False
    summarize_model: str | None = "Qwen/Qwen3-4B-Instruct-2507"
    log_full_transcript: bool = False
    model_name: str | None = None
    grader: Literal["sympy", "math_verify"] = "sympy"
    grade_timeout: float = 1.0
    eval_mode: Literal["direct", "debate", "both"] = "debate"
    is_training: bool = True  # training or evaluation mode

    async def make_envs(self) -> Sequence[Env]:
        problem = self.problems[self.problem_index % len(self.problems)]

        # Direct evaluation mode: single-turn problem solving
        if self.eval_mode == "direct" and not self.is_training:
            return [
                DirectMathEvaluationEnv(
                    problem=problem,
                    renderer=self.renderer,
                )
            ]

        # Debate mode (training or evaluation): multi-turn self-play debate
        max_turns = self.num_agents * self.max_rounds

        coordinator = MultiAgentCoordinator(
            question=problem.problem, num_agents=self.num_agents, max_turns=max_turns
        )
        return [
            VerifiableMultiAgentDebateEnv(
                agent_id=i,
                coordinator=coordinator,
                renderer=self.renderer,
                self_play=True,
                history_turns=self.history_turns,
                summarize_history=self.summarize_history,
                summarize_model=self.summarize_model,
                model_name=self.model_name,
            )
            for i in range(self.num_agents)
        ]

    def _compute_eval_correctness_metrics(
        self, solution_text: str, ground_truth: str, agent_id: int
    ) -> Metrics:
        """Compute correctness metrics for evaluation (no rewards)."""
        # Check if solution is valid (not incomplete or parse error)
        is_valid_format = not solution_text.startswith(
            "[INCOMPLETE]"
        ) and not solution_text.startswith("[PARSE_ERROR")

        has_box, boxed = _extract_boxed_from_solution(solution_text)
        correct = False
        if boxed is not None:
            correct = safe_grade(
                boxed, ground_truth, grader=self.grader, timeout=self.grade_timeout
            )

        return {
            "format": 1.0 if (is_valid_format and has_box) else 0.0,
            "correct": 1.0 if correct else 0.0,
        }

    def _compute_training_rewards(
        self,
        trajectory_group: list[Trajectory],
        env_group: Sequence[Env],
        problem: VerifiableMathProblem,
    ) -> list[tuple[float, Metrics]]:
        """Compute rewards and metrics for training mode."""
        env0 = env_group[0]
        assert isinstance(env0, VerifiableMultiAgentDebateEnv)
        coordinator = env0.coordinator

        if self.log_full_transcript:
            log_debate_transcript(coordinator)

        # Populate step-wise rewards in trajectories
        stepwise_metrics = self._populate_stepwise_rewards(trajectory_group, env_group)

        # Compute accuracy metrics for monitoring (NOT used as rewards)
        dataset_name = problem.dataset_name
        accuracy_metrics = []

        for agent_id in range(self.num_agents):
            # Get ALL responses from this agent to compute format accuracy
            agent_responses = [
                resp for resp in coordinator.state.agent_responses if resp.author_id == agent_id
            ]

            # Format: fraction of responses that have valid ParsedResponse
            if agent_responses:
                valid_responses = sum(
                    1
                    for resp in agent_responses
                    if not resp.solution.startswith("[INCOMPLETE]")
                    and not resp.solution.startswith("[PARSE_ERROR")
                )
                format_fraction = valid_responses / len(agent_responses)
            else:
                format_fraction = 0.0

            # Correctness: check the latest response only
            latest = _latest_response_by_author(
                coordinator.state.agent_responses, author_id=agent_id
            )
            correct = False
            if latest is not None:
                has_box, boxed = _extract_boxed_from_solution(latest.solution)
                if boxed is not None:
                    correct = safe_grade(
                        boxed, problem.answer, grader=self.grader, timeout=self.grade_timeout
                    )

            accuracy_metrics.append(
                {
                    "train_format": format_fraction,
                    "train_correct": 1.0 if correct else 0.0,
                }
            )

        # Compute pass@k: did any agent get it correct?
        any_correct = any(m["train_correct"] > 0.5 for m in accuracy_metrics)

        # Return metrics for each agent
        return [
            (
                0.0,  # No final reward (all rewards are already in trajectory steps)
                {
                    **stepwise_metrics,
                    **accuracy_metrics[agent_id],
                    f"train_{dataset_name}/format": accuracy_metrics[agent_id]["train_format"],
                    f"train_{dataset_name}/correct": accuracy_metrics[agent_id]["train_correct"],
                    f"train_{dataset_name}/pass@{self.num_agents}": 1.0 if any_correct else 0.0,
                },
            )
            for agent_id in range(self.num_agents)
        ]

    def _compute_direct_eval(
        self, trajectory_group: list[Trajectory], problem: VerifiableMathProblem
    ) -> list[tuple[float, Metrics]]:
        """Compute metrics for direct evaluation mode (single-turn)."""
        trajectory = trajectory_group[0]
        if not trajectory.transitions:
            return [(0.0, {"format": 0.0, "correct": 0.0})]

        action_tokens = trajectory.transitions[0].ac.tokens
        response_text = self.renderer.parse_response(action_tokens)[0]["content"]
        metrics = self._compute_eval_correctness_metrics(response_text, problem.answer, agent_id=0)

        # Log direct evaluation details
        if self.log_full_transcript:
            _, boxed = _extract_boxed_from_solution(response_text)
            parsed_solution = boxed if boxed is not None else "(No boxed answer found)"
            log_direct_evaluation(
                problem=problem.problem,
                response_text=response_text,
                parsed_solution=parsed_solution,
                metrics=metrics,
            )

        return [(0.0, metrics)]

    def _compute_debate_eval(
        self,
        trajectory_group: list[Trajectory],
        env_group: Sequence[Env],
        problem: VerifiableMathProblem,
    ) -> list[tuple[float, Metrics]]:
        """Compute metrics for debate evaluation mode (multi-turn)."""
        env0 = env_group[0]
        assert isinstance(env0, VerifiableMultiAgentDebateEnv)
        coordinator = env0.coordinator

        if self.log_full_transcript:
            log_debate_transcript(coordinator)

        results: list[tuple[float, Metrics]] = []
        agent_solutions_for_logging: list[tuple[int, str | None, str, dict[str, float]]] = []

        for agent_id in range(self.num_agents):
            latest = _latest_response_by_author(
                coordinator.state.agent_responses, author_id=agent_id
            )

            if latest is None:
                metrics = {"format": 0.0, "correct": 0.0}
                if self.log_full_transcript:
                    agent_solutions_for_logging.append((agent_id, None, "(No response)", metrics))
            else:
                metrics = self._compute_eval_correctness_metrics(
                    latest.solution, problem.answer, agent_id
                )
                if self.log_full_transcript:
                    _, boxed = _extract_boxed_from_solution(latest.solution)
                    parsed_answer = boxed if boxed is not None else "(No boxed answer found)"
                    agent_solutions_for_logging.append(
                        (agent_id, latest.solution, parsed_answer, metrics)
                    )

            results.append((0.0, metrics))

        if self.log_full_transcript:
            log_debate_evaluation_final_solutions(agent_solutions_for_logging)

        return results

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        assert env_group, "empty env_group"
        problem = self.problems[self.problem_index % len(self.problems)]

        # Training mode: Use pairwise step-wise rewards + track accuracy metrics
        if self.is_training:
            return self._compute_training_rewards(trajectory_group, env_group, problem)

        # Direct evaluation: single-turn correctness metrics
        if self.eval_mode == "direct":
            return self._compute_direct_eval(trajectory_group, problem)

        # Debate evaluation: multi-turn self-play, check all agents' final answers
        if self.eval_mode == "debate":
            return self._compute_debate_eval(trajectory_group, env_group, problem)

        raise ValueError(f"Invalid eval_mode: {self.eval_mode}")


class VerifiableMathDebateDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        problems: list[VerifiableMathProblem],
        num_agents: int,
        renderer: Renderer,
        self_play: bool,
        history_turns: int,
        summarize_history: bool,
        summarize_model: str | None,
        log_full_transcript: bool,
        max_rounds: int,
        num_datapoints: int,
        model_name: str,
        grader: Literal["sympy", "math_verify"] = "sympy",
        grade_timeout: float = 1.0,
        eval_mode: Literal["direct", "debate", "both"] = "debate",
        is_training: bool = True,
    ):
        self.batch_size = batch_size
        self.problems = problems
        self.num_agents = num_agents
        self.renderer = renderer
        self.self_play = self_play
        self.history_turns = history_turns
        self.summarize_history = summarize_history
        self.summarize_model = summarize_model
        self.log_full_transcript = log_full_transcript
        self.max_rounds = max_rounds
        self.num_datapoints = num_datapoints
        self.model_name = model_name
        self.grader = grader
        self.grade_timeout = grade_timeout
        self.eval_mode = eval_mode
        self.is_training = is_training

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, self.num_datapoints)
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            VerifiableMultiAgentEnvGroupBuilder(
                problems=self.problems,
                problem_index=problem_index,
                renderer=self.renderer,
                num_agents=self.num_agents,
                self_play=self.self_play,
                history_turns=self.history_turns,
                summarize_history=self.summarize_history,
                summarize_model=self.summarize_model,
                log_full_transcript=self.log_full_transcript,
                max_rounds=self.max_rounds,
                model_name=self.model_name,
                grader=self.grader,
                grade_timeout=self.grade_timeout,
                eval_mode=self.eval_mode,
                is_training=self.is_training,
            )
            for problem_index in range(batch_start, batch_end)
        ]

    def __len__(self) -> int:
        return (self.num_datapoints + self.batch_size - 1) // self.batch_size


@chz.chz
class VerifiableMathDebateDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    num_train_datapoints: int
    epoch: int = 1
    num_agents: int = 3
    max_rounds: int = 3
    history_rounds: int = 2
    summarize_history: bool = False
    summarize_model: str | None = None
    log_full_transcript: bool = False
    model_name: str
    renderer_name: str
    dataset_path: str = "tinker_cookbook/example_data/verifiable_math_problems.jsonl"
    problem_field: str = "problem"
    answer_field: str = "answer"
    max_questions: int = -1  # No limit by default
    grader: Literal["sympy", "math_verify"] = "sympy"
    grade_timeout: float = 2.0  # Increased timeout for safety

    async def __call__(
        self,
    ) -> tuple[VerifiableMathDebateDataset, VerifiableMathDebateDataset | None]:
        """Build training dataset for online TTL.

        Loads ALL problems from dataset_path and creates a training dataset that samples
        num_train_datapoints per epoch. The total number of training datapoints is
        num_train_datapoints * epoch. For online test-time learning, evaluation is
        handled separately by a custom evaluator that uses the same problem set.
        """
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))

        # Load ALL problems from dataset (no train/test split for online TTL)
        all_problems = load_math_problems_from_jsonl(
            path=self.dataset_path,
            problem_field=self.problem_field,
            answer_field=self.answer_field,
            max_count=self.max_questions,
        )

        # Training dataset: samples num_train_datapoints from all problems
        total_train_datapoints = self.num_train_datapoints * self.epoch
        train_dataset = VerifiableMathDebateDataset(
            batch_size=self.batch_size,
            problems=all_problems,
            num_agents=self.num_agents,
            renderer=renderer,
            self_play=True,
            history_turns=self.history_rounds,
            summarize_history=self.summarize_history,
            summarize_model=self.summarize_model,
            log_full_transcript=self.log_full_transcript,
            max_rounds=self.max_rounds,
            num_datapoints=total_train_datapoints,
            model_name=self.model_name,
            grader=self.grader,
            grade_timeout=self.grade_timeout,
            eval_mode="debate",  # Training always uses debate mode
            is_training=True,  # Training dataset computes step-wise rewards
        )

        # No separate test dataset - evaluation handled by custom evaluator
        return train_dataset, None
