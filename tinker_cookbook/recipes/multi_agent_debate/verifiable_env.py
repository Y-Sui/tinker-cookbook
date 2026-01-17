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
from .prompts import VERIFIABLE_AGENT_SYSTEM_PROMPT, ParsedResponse, format_persona_intro
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
        persona_intro = format_persona_intro(self.agent_id)
        return VERIFIABLE_AGENT_SYSTEM_PROMPT.format(
            agent_id=self.agent_id,
            persona_intro=persona_intro,
        )

    def _format_turns(self, turns: list[ParsedResponse], start_offset: int = 0) -> str:
        """Format history with blind review (comparisons are hidden).

        Blind review ensures agents form independent judgments by hiding
        previous <comparison> tags. This prevents bandwagon effects and
        produces more meaningful consensus for reward computation.

        Only <solution> and <evaluation> are shown in history.
        """
        if not turns:
            return ""
        lines: list[str] = ["--- HISTORY OF PREVIOUS TURNS ---"]
        for idx, response in enumerate(turns):
            # Add offset to preserve original turn numbering when history is truncated
            display_turn_idx = start_offset + idx + 1
            agent_label = f"Agent {response.author_id}"
            lines.append(f"== Turn {display_turn_idx} ({agent_label}) ==")
            lines.append(f"{agent_label}'s Solution:")
            lines.append(response.solution.rstrip())
            lines.append(f"{agent_label}'s Evaluation:")
            lines.append(response.evaluation.rstrip())
            # BLIND REVIEW: Comparisons are intentionally hidden from history
            # so that each agent forms independent judgments
            lines.append("")
        lines.append("--- END OF HISTORY ---\n")
        return "\n".join(lines).rstrip()

    async def get_conversation_context(self) -> str:
        """Format the conversation context for this agent.

        Shows the last round. If summarize_history is True, summarizes it.
        Otherwise, keeps them verbatim.
        """
        if not self.coordinator.state.agent_responses:
            return ""

        question = self.coordinator.state.question
        turns = list(self.coordinator.state.agent_responses)
        num_agents = self.coordinator.state.num_agents
        history_turns = num_agents

        # Get recent turns based on history_turns setting
        total_turns = len(turns)
        start_offset = max(0, total_turns - history_turns)
        recent_turns = turns[-history_turns:] if len(turns) > history_turns else turns

        history_text = (
            f"Query: {question}\n{self._format_turns(recent_turns, start_offset)}".rstrip()
        )

        return await self._summarize(history_text) if self.summarize_history else history_text

    async def get_observation_string(self) -> str:
        history = await self.get_conversation_context()
        current_cycle = self.coordinator.state.get_current_cycle()

        # With parallel generation, count OTHER agents from committed responses only.
        # In cycle N, all agents see responses from cycles 0..N-1.
        # All other (num_agents - 1) agents have responded in previous cycles if cycle > 0.
        num_agents = self.coordinator.state.num_agents
        num_other_agents = num_agents - 1 if current_cycle > 0 else 0

        # Build guidance for evaluation and comparison based on what's available
        if num_other_agents == 0:
            eval_guidance = 'Set <evaluation> to "N/A" (no other agents have responded yet).'
            comp_guidance = 'Set <comparison> to "N/A" (no other agents to compare).'
        elif num_other_agents == 1:
            eval_guidance = "Evaluate the other agent's solution and reasoning."
            comp_guidance = 'Set <comparison> to "N/A" (need at least 2 other agents to compare).'
        else:
            eval_guidance = "Evaluate each other agent's solution and reasoning."
            comp_guidance = (
                f"Compare pairs of other agents (do not include yourself, Agent {self.agent_id})."
            )

        # First cycle prompt (no history available)
        if current_cycle == 0:
            return (
                f"Query: {self.coordinator.state.question}\n"
                f"Agent {self.agent_id}, please proceed with your response.\n\n"
                f"{eval_guidance}\n{comp_guidance}\n"
                "Your <solution> must include your final answer in \\boxed{{...}} format."
            )

        # Regular turn prompt (with history from previous cycles)
        return (
            f"{history}\n\nAgent {self.agent_id}, it is your turn.\n"
            f"{eval_guidance}\n{comp_guidance}\n"
            "Your <solution> must include your final answer in \\boxed{{...}} format."
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
    log_full_transcript: bool = False
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

        # Create shared summarizer once for all agents in this group
        shared_summarizer = self._create_shared_summarizer()

        coordinator = MultiAgentCoordinator(
            question=problem.problem, num_agents=self.num_agents, max_turns=max_turns
        )
        return [
            VerifiableMultiAgentDebateEnv(
                agent_id=i,
                coordinator=coordinator,
                renderer=self.renderer,
                self_play=True,
                summarize_history=self.summarize_history,
                _summarizer_policy=shared_summarizer,
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
        """Compute rewards and metrics for training mode.

        Uses v2 reward system:
        - gen_rewards: Soft vote ratio for <solution>/<evaluation> tokens
        - judge_rewards: Consensus alignment for <comparison> tokens

        These are stored in trajectory_group[agent_id].gen_rewards and
        trajectory_group[agent_id].judge_rewards for later processing.
        """
        env0 = env_group[0]
        assert isinstance(env0, VerifiableMultiAgentDebateEnv)
        coordinator = env0.coordinator

        # Use v2 reward system (soft vote ratio + consensus alignment)
        gen_rewards, judge_rewards, v2_metrics = self._compute_rewards_v2(
            trajectory_group, env_group
        )

        # Store rewards in trajectory group for later data assembly
        # We attach these as attributes so the training loop can access them
        # Use object.__setattr__ to bypass frozen dataclass restriction
        for agent_id in range(self.num_agents):
            object.__setattr__(
                trajectory_group[agent_id], "gen_rewards", gen_rewards.get(agent_id, [])
            )
            object.__setattr__(
                trajectory_group[agent_id], "judge_rewards", judge_rewards.get(agent_id, [])
            )

        # Compute reward statistics for metrics logging
        import numpy as np

        all_gen_rewards = [
            r for agent_id in range(self.num_agents) for r in gen_rewards.get(agent_id, [])
        ]
        all_judge_rewards = [
            r for agent_id in range(self.num_agents) for r in judge_rewards.get(agent_id, [])
        ]

        if all_gen_rewards:
            v2_metrics["reward/gen/mean"] = float(np.mean(all_gen_rewards))
            v2_metrics["reward/gen/std"] = float(np.std(all_gen_rewards))
        if all_judge_rewards:
            v2_metrics["reward/judge/mean"] = float(np.mean(all_judge_rewards))
            v2_metrics["reward/judge/std"] = float(np.std(all_judge_rewards))

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

        # Compute multi-agent aggregation metrics:
        # - pass@k: Optimistic - at least one agent got it correct (best-of-k)
        # - avg@k: Mean accuracy across all k agents (average quality)
        # - cons@k: Consensus - majority of agents got it correct (voting)
        any_correct = any(m["train_correct"] > 0.5 for m in accuracy_metrics)
        avg_correct = sum(m["train_correct"] for m in accuracy_metrics) / len(accuracy_metrics)
        num_correct = sum(1 for m in accuracy_metrics if m["train_correct"] > 0.5)
        consensus_correct = num_correct > len(accuracy_metrics) / 2  # More than half correct

        if self.log_full_transcript:
            per_agent_metrics = [
                {
                    "train_format": accuracy_metrics[agent_id]["train_format"],
                    "train_correct": accuracy_metrics[agent_id]["train_correct"],
                }
                for agent_id in range(self.num_agents)
            ]
            log_debate_transcript(coordinator, metrics_by_agent=per_agent_metrics)

        # Return metrics for each agent
        return [
            (
                0.0,  # No final reward (all rewards are already in trajectory steps)
                {
                    **v2_metrics,  # v2 reward system metrics
                    f"train_{dataset_name}/format": accuracy_metrics[agent_id]["train_format"],
                    f"train_{dataset_name}/correct": accuracy_metrics[agent_id]["train_correct"],
                    f"train_{dataset_name}/pass@{self.num_agents}": 1.0 if any_correct else 0.0,
                    f"train_{dataset_name}/avg@{self.num_agents}": avg_correct,
                    f"train_{dataset_name}/cons@{self.num_agents}": 1.0
                    if consensus_correct
                    else 0.0,
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
            per_agent_metrics = [metrics for _, metrics in results]
            log_debate_transcript(coordinator, metrics_by_agent=per_agent_metrics)
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
        enable_format_penalty: bool = True,
    ):
        self.batch_size = batch_size
        self.problems = problems
        self.num_agents = num_agents
        self.renderer = renderer
        self.self_play = self_play
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
        self.enable_format_penalty = enable_format_penalty

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
                summarize_history=self.summarize_history,
                summarize_model=self.summarize_model,
                log_full_transcript=self.log_full_transcript,
                max_rounds=self.max_rounds,
                model_name=self.model_name,
                grader=self.grader,
                grade_timeout=self.grade_timeout,
                eval_mode=self.eval_mode,
                is_training=self.is_training,
                enable_format_penalty=self.enable_format_penalty,
            )
            for problem_index in range(batch_start, batch_end)
        ]

    def __len__(self) -> int:
        return (self.num_datapoints + self.batch_size - 1) // self.batch_size


@chz.chz
class VerifiableMathDebateDatasetBuilder(RLDatasetBuilder):
    """Builder for verifiable math debate datasets."""

    # Required fields (no defaults) must come first
    batch_size: int
    num_train_datapoints: int
    model_name: str
    renderer_name: str
    dataset_path: str
    problem_field: str
    answer_field: str

    # Optional fields (with defaults)
    epoch: int = 1
    num_agents: int = 3
    max_rounds: int = 3
    summarize_history: bool = False
    summarize_model: str | None = None
    log_full_transcript: bool = False
    grader: Literal["sympy", "math_verify"] = "sympy"
    max_questions: int = -1  # No limit by default
    grade_timeout: float = 2.0  # Increased timeout for safety
    enable_format_penalty: bool = True
    enable_sequence_extension: bool = True  # Preserve thinking in history for O(T) compute

    async def __call__(
        self,
    ) -> tuple[VerifiableMathDebateDataset, VerifiableMathDebateDataset | None]:
        """Build training dataset for online TTL.

        Loads ALL problems from dataset_path and creates a training dataset that samples
        num_train_datapoints per epoch. The total number of training datapoints is
        num_train_datapoints * epoch. For online test-time learning, evaluation is
        handled separately by a custom evaluator that uses the same problem set.
        """
        # Create renderer with sequence extension if requested
        tokenizer = get_tokenizer(self.model_name)
        if self.enable_sequence_extension and self.renderer_name == "qwen3_disable_thinking":
            # Enable sequence extension by preserving thinking in history
            from tinker_cookbook.renderers.qwen3 import Qwen3DisableThinkingRenderer

            renderer = Qwen3DisableThinkingRenderer(tokenizer, strip_thinking_from_history=False)
        else:
            renderer = get_renderer(self.renderer_name, tokenizer)

        # Load ALL problems from dataset (no train/test split for online TTL)
        all_problems = load_math_problems_from_jsonl(
            path=self.dataset_path,
            problem_field=self.problem_field,
            answer_field=self.answer_field,
            max_count=self.max_questions,
        )

        # Training dataset: samples num_train_datapoints from all problems
        if self.num_train_datapoints > len(all_problems) or self.num_train_datapoints < 0:
            total_train_datapoints = len(all_problems) * self.epoch
        else:
            total_train_datapoints = self.num_train_datapoints * self.epoch
        train_dataset = VerifiableMathDebateDataset(
            batch_size=self.batch_size,
            problems=all_problems,
            num_agents=self.num_agents,
            renderer=renderer,
            self_play=True,
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
            enable_format_penalty=self.enable_format_penalty,
        )

        # No separate test dataset - evaluation handled by custom evaluator
        return train_dataset, None
