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
import random
from dataclasses import dataclass
from typing import Literal, Sequence

import chz
import tinker
from tinker import types

from tinker_cookbook.completers import StopCondition, TinkerMessageCompleter
from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)
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

from .env import MultiAgentCoordinator
from .prompts import VERIFIABLE_AGENT_SYSTEM_PROMPT, ParsedResponse
from .utils import (
    STOP_CONDITION,
    get_debate_stop_condition,
    get_step_idx_before_turn,
    get_summarizer_stop_condition,
    log_debate_transcript,
)

QUESTION_SUFFIX = " Write your answer in \\boxed{} format."


def _append_question_suffix(problem: str) -> str:
    text = problem.rstrip()
    if text.endswith(QUESTION_SUFFIX.rstrip()):
        return text
    return text + QUESTION_SUFFIX


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
        question = _append_question_suffix(self.problem.problem)
        messages = [
            {
                "role": "system",
                "content": "Solve the following math problem. Write your final answer in \\boxed{} format.",
            },
            {"role": "user", "content": question},
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
class VerifiableMultiAgentDebateEnv(Env):
    """Environment for one agent in a verifiable multi-agent debate."""

    agent_id: int
    coordinator: MultiAgentCoordinator
    renderer: Renderer
    self_play: bool = True
    history_turns: int = 2
    summarize_history: bool = False
    summarize_model: str | None = "Qwen/Qwen3-4B-Instruct-2507"
    model_name: str | None = None

    def __post_init__(self) -> None:
        assert self.self_play, "Only self-play mode is supported"

    @property
    def stop_condition(self) -> StopCondition:
        return get_debate_stop_condition(self.renderer)

    def get_system_prompt(self) -> str:
        return VERIFIABLE_AGENT_SYSTEM_PROMPT.format(agent_id=self.agent_id)

    def _format_turns(self, turns: list[ParsedResponse]) -> str:
        if not turns:
            return ""
        lines: list[str] = []
        for turn_idx, response in enumerate(turns, start=1):
            lines.append(f"--- Turn {turn_idx} (Agent {response.author_id}) ---")
            lines.append("Solution:")
            lines.append(response.solution.rstrip())
            lines.append("Evaluation:")
            lines.append(response.evaluation.rstrip())
            if response.comparison_text:
                lines.append("Comparison:")
                lines.append(response.comparison_text.rstrip())
            lines.append("")
        return "\n".join(lines).rstrip()

    async def _summarize(self, history: str) -> str:
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(
            base_model=self.summarize_model or self.model_name
        )
        summarizer_policy = TinkerMessageCompleter(
            sampling_client=sampling_client,
            renderer=self.renderer,
            max_tokens=892,
            stop_condition=get_summarizer_stop_condition(self.renderer),
        )
        messages: list[Message] = [
            {
                "role": "system",
                "content": (
                    "You summarize multi-agent debate transcripts.\n"
                    "Write a concise, information-dense summary that preserves:\n"
                    "- The user question\n"
                    "- Each agent's key solution ideas\n"
                    "- Each agent's critiques/evaluations of others\n"
                    "- Any explicit comparisons\n"
                    "Do not add new information. Output plain text only."
                ),
            },
            {"role": "user", "content": history},
        ]
        resp = await summarizer_policy(messages)
        return ensure_text(resp["content"]).strip()

    async def get_conversation_context(self) -> str:
        if not self.coordinator.state.agent_responses:
            return ""

        question = self.coordinator.state.question
        turns = list(self.coordinator.state.agent_responses)

        if not self.summarize_history:
            return f"Question: {question}\n\n{self._format_turns(turns)}".rstrip()

        if self.history_turns < 0:
            raw_recent = turns
            older: list[ParsedResponse] = []
        elif self.history_turns == 0:
            raw_recent = []
            older = turns
        else:
            raw_recent = turns[-self.history_turns :] if len(turns) > self.history_turns else turns
            older = turns[: -self.history_turns] if len(turns) > self.history_turns else []

        summary = ""
        if older:
            older_text = f"Question: {question}\n\n{self._format_turns(older)}"
            summary = await self._summarize(older_text)

        recent_text = self._format_turns(raw_recent)
        parts: list[str] = [f"Question: {question}"]
        if summary:
            parts.append("\n--- Summary of earlier turns ---\n" + summary)
        if recent_text:
            parts.append("\n--- Recent turns (verbatim) ---\n" + recent_text)
        return "\n".join(parts).rstrip()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        if self.agent_id != 0:
            await self.wait_for_turn()
        return await self.get_observation(), self.stop_condition

    async def wait_for_turn(self) -> None:
        if not self.coordinator.done:
            await self.coordinator.wait_for_turn(self.agent_id)

    async def get_observation_string(self) -> str:
        history = await self.get_conversation_context()
        turn_idx = self.coordinator.state.current_turn
        cycle_idx = self.coordinator.state.get_current_cycle()
        max_cycles = self.coordinator.state.max_turns // self.coordinator.state.num_agents

        if turn_idx == 0:
            return (
                f"Question: {self.coordinator.state.question}\n\n"
                f"Cycle {cycle_idx + 1} of {max_cycles}, Turn {turn_idx + 1}.\n"
                "First completion: propose your solution and include a final \\boxed{...} answer.\n"
                'Set <evaluation> to "N/A" and <comparison> to "N/A".'
            )

        return (
            f"{history}\n\n"
            f"Cycle {cycle_idx + 1} of {max_cycles}, Turn {turn_idx + 1}.\n"
            "Write a solution to the math problem. Include a final \\boxed{...} answer in <solution>.\n"
            "In <evaluation>, evaluate the most recent visible prior completions (up to 2 turns).\n"
            "In <comparison>, compare only OTHER agents visible in the history (never include yourself). If fewer than two "
            'other completions are visible, write "N/A".'
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
            next_stop_condition=STOP_CONDITION,
            metrics={},
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
class VerifiableMultiAgentEnvGroupBuilder(EnvGroupBuilder):
    problems: list[VerifiableMathProblem]
    problem_index: int
    renderer: Renderer
    num_agents: int
    max_rounds: int
    self_play: bool
    history_turns: int = 2
    summarize_history: bool = False
    summarize_model: str | None = "Qwen/Qwen3-4B-Instruct-2507"
    log_full_transcript: bool = False
    model_name: str | None = None
    grader: Literal["sympy", "math_verify"] = "sympy"
    format_coef: float = 0.1
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
        question = _append_question_suffix(problem.problem)
        max_turns = self.num_agents * self.max_rounds

        coordinator = MultiAgentCoordinator(
            question=question, num_agents=self.num_agents, max_turns=max_turns
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

    def _populate_stepwise_rewards(
        self,
        trajectory_group: list[Trajectory],
        env_group: Sequence[Env],
    ) -> dict[str, float]:
        """
        Modify trajectories in-place to assign step-wise rewards based on comparisons.

        Returns summary metrics.
        """
        env0 = env_group[0]
        assert isinstance(env0, VerifiableMultiAgentDebateEnv)
        coordinator = env0.coordinator

        # Process each response in order
        for turn_idx, response in enumerate(coordinator.state.agent_responses):
            author_id = response.author_id

            # Process each comparison from this response
            for agent_a, op, agent_b in response.comparisons:
                # Leave-one-out: skip if author is involved
                if author_id == agent_a or author_id == agent_b:
                    continue

                # Validate agent IDs
                if not (0 <= agent_a < self.num_agents and 0 <= agent_b < self.num_agents):
                    continue
                if agent_a == agent_b:
                    continue
                if op not in {">", "="}:
                    continue

                # Find step indices for agent_a and agent_b
                # (most recent step before this turn)
                agent_a_step_idx = get_step_idx_before_turn(agent_a, turn_idx, self.num_agents)
                agent_b_step_idx = get_step_idx_before_turn(agent_b, turn_idx, self.num_agents)

                # Skip if either agent hasn't acted yet
                if agent_a_step_idx < 0 or agent_b_step_idx < 0:
                    continue

                # Assign rewards (mutate trajectories in-place)
                if op == ">":
                    trajectory_group[agent_a].transitions[agent_a_step_idx].reward += 1.0
                    trajectory_group[agent_b].transitions[agent_b_step_idx].reward -= 1.0
                elif op == "=":
                    # Equal comparison: no reward change (or could use ±0.5)
                    pass

        # Compute summary metrics
        total_comparisons_used = 0
        total_rewards_assigned = 0
        for trajectory in trajectory_group:
            for transition in trajectory.transitions:
                if transition.reward != 0.0:
                    total_rewards_assigned += 1
                    total_comparisons_used += abs(transition.reward)

        return {
            "stepwise_comparisons_used": total_comparisons_used,
            "stepwise_rewards_assigned": total_rewards_assigned,
        }

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        assert env_group, "empty env_group"

        problem = self.problems[self.problem_index % len(self.problems)]
        ground_truth = problem.answer

        # Training mode: Use pairwise step-wise rewards
        if self.is_training:
            env0 = env_group[0]
            assert isinstance(env0, VerifiableMultiAgentDebateEnv)
            coordinator = env0.coordinator

            if self.log_full_transcript:
                log_debate_transcript(coordinator)

            # Populate step-wise rewards in trajectories
            stepwise_metrics = self._populate_stepwise_rewards(trajectory_group, env_group)

            return [
                (
                    0.0,  # No final reward (all rewards are in steps)
                    {
                        "agent_id": agent_id,
                        **stepwise_metrics,
                    },
                )
                for agent_id in range(self.num_agents)
            ]

        # Evaluation modes: no rewards, only metrics
        # Helper function for correctness metrics (no rewards)
        def compute_correctness_metrics(solution_text: str, agent_id: int) -> Metrics:
            has_box, boxed = _extract_boxed_from_solution(solution_text)
            correct = False
            if boxed is not None:
                correct = safe_grade(
                    boxed, ground_truth, grader=self.grader, timeout=self.grade_timeout
                )

            return {
                "agent_id": float(agent_id),
                "format": 1.0 if has_box else 0.0,
                "correct": 1.0 if correct else 0.0,
                "eval_mode": self.eval_mode,
            }

        # Evaluation modes: no rewards, only metrics
        # Direct evaluation: single-turn correctness metrics
        if self.eval_mode == "direct" and not self.is_training:
            trajectory = trajectory_group[0]
            if not trajectory.transitions:
                return [
                    (
                        0.0,
                        {
                            "agent_id": 0.0,
                            "format": 0.0,
                            "correct": 0.0,
                            "missing_response": 1.0,
                            "eval_mode": self.eval_mode,
                        },
                    )
                ]

            action_tokens = trajectory.transitions[0].ac.tokens
            response_text = self.renderer.parse_response(action_tokens)[0]["content"]

            metrics = compute_correctness_metrics(response_text, agent_id=0)
            return [(0.0, metrics)]  # No reward, only metrics

        # Debate evaluation: multi-turn self-play, check all agents' final answers
        if self.eval_mode == "debate" and not self.is_training:
            env0 = env_group[0]
            assert isinstance(env0, VerifiableMultiAgentDebateEnv)
            coordinator = env0.coordinator

            if self.log_full_transcript:
                log_debate_transcript(coordinator)

            results: list[tuple[float, Metrics]] = []
            for agent_id in range(self.num_agents):
                latest = _latest_response_by_author(
                    coordinator.state.agent_responses, author_id=agent_id
                )

                if latest is None:
                    metrics = {
                        "agent_id": float(agent_id),
                        "format": 0.0,
                        "correct": 0.0,
                        "missing_response": 1.0,
                        "eval_mode": self.eval_mode,
                    }
                else:
                    metrics = compute_correctness_metrics(latest.solution, agent_id)

                results.append((0.0, metrics))  # No reward, only metrics

            return results

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
        opponent_policies: list[TinkerMessageCompleter] | None = None,
        grader: Literal["sympy", "math_verify"] = "sympy",
        format_coef: float = 0.1,
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
        self.opponent_policies = opponent_policies
        self.grader = grader
        self.format_coef = format_coef
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
                format_coef=self.format_coef,
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
    num_test_datapoints: int
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
    dataset_name_field: str | None = "dataset_name"
    max_questions: int = 1000
    test_question_frac: float = 0.1
    opponent_model_name: str | None = None
    grader: Literal["sympy", "math_verify"] = "sympy"
    format_coef: float = 0.1
    grade_timeout: float = 1.0
    eval_mode: Literal["direct", "debate", "both"] = "debate"

    def _load_problems_from_file(self) -> list[VerifiableMathProblem]:
        import json

        problems: list[VerifiableMathProblem] = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= self.max_questions:
                    break
                data = json.loads(line)
                if self.problem_field not in data or self.answer_field not in data:
                    raise ValueError(
                        f"Each JSONL row must include '{self.problem_field}' and '{self.answer_field}'. "
                        f"Got keys={list(data.keys())}"
                    )
                dataset_name = "math"
                if self.dataset_name_field is not None and self.dataset_name_field in data:
                    dataset_name = str(data[self.dataset_name_field])
                problems.append(
                    VerifiableMathProblem(
                        problem=str(data[self.problem_field]),
                        answer=str(data[self.answer_field]),
                        dataset_name=dataset_name,
                    )
                )
        if not problems:
            raise ValueError(f"No problems loaded from {self.dataset_path}")
        return problems

    def _split_problems(
        self, problems: list[VerifiableMathProblem]
    ) -> tuple[list[VerifiableMathProblem], list[VerifiableMathProblem]]:
        if self.num_test_datapoints <= 0 or self.test_question_frac <= 0:
            return problems, []
        if len(problems) < 2:
            return problems, []

        rng = random.Random(42)
        shuffled = list(problems)
        rng.shuffle(shuffled)
        test_n = int(round(len(shuffled) * self.test_question_frac))
        test_n = max(1, min(test_n, len(shuffled) - 1))
        return shuffled[test_n:], shuffled[:test_n]

    async def __call__(
        self,
    ) -> tuple[VerifiableMathDebateDataset, VerifiableMathDebateDataset | None]:
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        problems = self._load_problems_from_file()
        train_problems, test_problems = self._split_problems(problems)

        train_dataset = VerifiableMathDebateDataset(
            batch_size=self.batch_size,
            problems=train_problems,
            num_agents=self.num_agents,
            renderer=renderer,
            self_play=True,
            history_turns=self.history_rounds,
            summarize_history=self.summarize_history,
            summarize_model=self.summarize_model,
            log_full_transcript=self.log_full_transcript,
            max_rounds=self.max_rounds,
            num_datapoints=self.num_train_datapoints,
            model_name=self.model_name,
            grader=self.grader,
            format_coef=self.format_coef,
            grade_timeout=self.grade_timeout,
            eval_mode="debate",  # Training always uses debate mode
            is_training=True,  # Training dataset computes step-wise rewards
        )

        test_dataset: VerifiableMathDebateDataset | None
        if self.num_test_datapoints <= 0 or not test_problems:
            test_dataset = None
        else:
            test_dataset = VerifiableMathDebateDataset(
                batch_size=min(self.num_test_datapoints, self.batch_size),
                problems=test_problems,
                num_agents=self.num_agents,
                renderer=renderer,
                self_play=True,  # Evaluation also uses self-play (no fixed opponents)
                history_turns=self.history_rounds,
                summarize_history=self.summarize_history,
                summarize_model=self.summarize_model,
                log_full_transcript=self.log_full_transcript,
                max_rounds=self.max_rounds,
                num_datapoints=self.num_test_datapoints,
                model_name=self.model_name,
                grader=self.grader,
                format_coef=self.format_coef,
                grade_timeout=self.grade_timeout,
                eval_mode=self.eval_mode,  # Uses builder's eval_mode (configurable)
                is_training=False,  # Test dataset only computes metrics (no rewards)
            )

        return train_dataset, test_dataset
