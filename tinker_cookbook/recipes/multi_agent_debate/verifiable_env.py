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

from .prompts import ParsedResponse
from .verifiable_env import MultiAgentCoordinator


VERIFIABLE_AGENT_SYSTEM_PROMPT = """You are Agent {agent_id} participating in a multi-agent self-play debate to solve a verifiable math problem.

Your objectives:
- Propose or refine a correct solution to the problem.
- In <solution>, include a final answer written in \\boxed{{...}} format.
- Evaluate other agents’ recent contributions (solution + critique quality), and compare other agents’ outputs.

OUTPUT FORMAT (use exact XML tags, in this order):

<solution>
Your detailed solution. You MUST include a final answer in \\boxed{{...}} format.
</solution>

<evaluation>
Evaluate other agents’ recent work. If there are no prior completions visible, write "N/A".
</evaluation>

<comparison>
Compare only OTHER agents visible in the history (never include yourself). If fewer than two other completions are visible, write "N/A".
</comparison>

Key reminders:
- Use EXACTLY these three XML tags, in strict order.
- Do NOT compare your own work in <comparison>.
"""


QUESTION_SUFFIX = " Write your answer in \\boxed{} format."

# Stop when we see the closing tag of the last required field.
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
class VerifiableMultiAgentDebateEnv(Env):
    """Environment for one agent in a verifiable multi-agent debate."""

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
            stop_condition=_get_summarizer_stop_condition(self.renderer),
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
            if self.self_play:
                await self.coordinator.wait_for_turn(self.agent_id)
            else:
                await self.run_opponent_steps()

    async def run_opponent_steps(self) -> None:
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
                    "content": VERIFIABLE_AGENT_SYSTEM_PROMPT.format(agent_id=opponent_agent_id),
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
            next_stop_condition=_STOP_CONDITION_TEXT,
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
    opponent_policies_all: list[TinkerMessageCompleter] | None = None
    model_name: str | None = None
    format_coef: float = 0.1
    grader: Literal["sympy", "math_verify"] = "sympy"
    grade_timeout: float = 1.0

    async def make_envs(self) -> Sequence[Env]:
        problem = self.problems[self.problem_index % len(self.problems)]
        question = _append_question_suffix(problem.problem)
        max_turns = self.num_agents * self.max_rounds

        if self.self_play:
            coordinator = MultiAgentCoordinator(
                question=question, num_agents=self.num_agents, max_turns=max_turns
            )
            return [
                VerifiableMultiAgentDebateEnv(
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
                VerifiableMultiAgentDebateEnv(
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
        assert env_group, "empty env_group"

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
                        "verifiable_reward": reward,
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
                    "verifiable_reward": reward,
                    "missing_response": 0.0,
                },
            )

        if self.self_play:
            env0 = env_group[0]
            assert isinstance(env0, VerifiableMultiAgentDebateEnv)
            coordinator = env0.coordinator

            if self.log_full_transcript:
                with logtree.scope_header("Debate Transcript"):
                    logtree.log_text(f"Question: {coordinator.state.question}")
                    turns = coordinator.state.agent_responses
                    if not turns:
                        logtree.log_text("(No responses captured)")
                    for turn_idx, response in enumerate(turns, start=1):
                        with logtree.scope_header(f"Turn {turn_idx}"):
                            with logtree.scope_header(f"Agent {response.author_id}"):
                                if response.observation:
                                    with logtree.scope_details("Observation (context)"):
                                        logtree.log_text(response.observation)
                                with logtree.scope_details("Solution"):
                                    logtree.log_text(response.solution)
                                with logtree.scope_details("Evaluation"):
                                    logtree.log_text(response.evaluation)
                                if response.comparison_text:
                                    with logtree.scope_details("Comparison"):
                                        logtree.log_text(response.comparison_text)
                                with logtree.scope_details("Raw response"):
                                    logtree.log_text(response.raw_response)

            rewards_and_metrics: list[tuple[float, Metrics]] = []
            for agent_id in range(self.num_agents):
                reward, metrics = compute_one_agent_reward(coordinator=coordinator, agent_id=agent_id)
                with logtree.scope_header(f"Verifiable Reward (Agent {agent_id})"):
                    logtree.log_text(f"Ground truth: {ground_truth}")
                    logtree.log_text(
                        f"Format ok: {metrics['format']}, Correct: {metrics['correct']}, Reward: {reward}"
                    )
                rewards_and_metrics.append((reward, metrics))
            return rewards_and_metrics

        rewards_and_metrics: list[tuple[float, Metrics]] = []
        for env in env_group:
            assert isinstance(env, VerifiableMultiAgentDebateEnv)
            coordinator = env.coordinator
            if self.log_full_transcript:
                with logtree.scope_header(f"Debate Transcript (controlled Agent {env.agent_id})"):
                    logtree.log_text(f"Question: {coordinator.state.question}")
                    turns = coordinator.state.agent_responses
                    if not turns:
                        logtree.log_text("(No responses captured)")
                    for turn_idx, response in enumerate(turns, start=1):
                        with logtree.scope_header(f"Turn {turn_idx}"):
                            with logtree.scope_header(f"Agent {response.author_id}"):
                                if response.observation:
                                    with logtree.scope_details("Observation (context)"):
                                        logtree.log_text(response.observation)
                                with logtree.scope_details("Solution"):
                                    logtree.log_text(response.solution)
                                with logtree.scope_details("Evaluation"):
                                    logtree.log_text(response.evaluation)
                                if response.comparison_text:
                                    with logtree.scope_details("Comparison"):
                                        logtree.log_text(response.comparison_text)
                                with logtree.scope_details("Raw response"):
                                    logtree.log_text(response.raw_response)

            reward, metrics = compute_one_agent_reward(coordinator=coordinator, agent_id=env.agent_id)
            rewards_and_metrics.append((reward, metrics))

        return rewards_and_metrics


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
        format_coef: float = 0.1,
        grader: Literal["sympy", "math_verify"] = "sympy",
        grade_timeout: float = 1.0,
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
        self.format_coef = format_coef
        self.grader = grader
        self.grade_timeout = grade_timeout

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
                opponent_policies_all=self.opponent_policies,
                format_coef=self.format_coef,
                grader=self.grader,
                grade_timeout=self.grade_timeout,
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
    format_coef: float = 0.1
    grader: Literal["sympy", "math_verify"] = "sympy"
    grade_timeout: float = 1.0

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

    def _construct_fixed_opponent_policies(self, renderer: Renderer) -> list[TinkerMessageCompleter]:
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(
            base_model=self.opponent_model_name or self.model_name
        )
        return [
            TinkerMessageCompleter(
                sampling_client=sampling_client,
                renderer=renderer,
                max_tokens=2048,
                stop_condition=_get_debate_stop_condition(renderer),
            )
            for _ in range(self.num_agents)
        ]

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
            opponent_policies=None,
            format_coef=self.format_coef,
            grader=self.grader,
            grade_timeout=self.grade_timeout,
        )

        test_dataset: VerifiableMathDebateDataset | None
        if self.num_test_datapoints <= 0 or not test_problems:
            test_dataset = None
        else:
            opponent_policies_all = self._construct_fixed_opponent_policies(renderer)
            test_dataset = VerifiableMathDebateDataset(
                batch_size=min(self.num_test_datapoints, self.batch_size),
                problems=test_problems,
                num_agents=self.num_agents,
                renderer=renderer,
                self_play=False,
                history_turns=self.history_rounds,
                summarize_history=self.summarize_history,
                summarize_model=self.summarize_model,
                log_full_transcript=self.log_full_transcript,
                max_rounds=self.max_rounds,
                num_datapoints=self.num_test_datapoints,
                model_name=self.model_name,
                opponent_policies=opponent_policies_all,
                format_coef=self.format_coef,
                grader=self.grader,
                grade_timeout=self.grade_timeout,
            )

        return train_dataset, test_dataset
