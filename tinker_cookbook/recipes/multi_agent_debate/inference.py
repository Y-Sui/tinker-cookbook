"""Self-play multi-agent debate using OpenRouter (no Tinker training).

This script exercises the same:
- prompts (`AGENT_SYSTEM_PROMPT` / `VERIFIABLE_AGENT_SYSTEM_PROMPT`)
- turn-taking (`MultiAgentCoordinator`)
- response parsing (`parse_agent_response` via `submit_response`)
- reward shaping (pairwise comparisons -> step-wise rewards)

as the current multi-agent debate implementation, but with a pure OpenRouter chat
model as the policy.

Differences from training:
- No weight updates / optimizer steps.
- No token-level rollouts; we operate on message strings.
- Rewards are still computed using the same comparison-based logic.

Run (module form):
  export OPENROUTER_API_KEY=...
  python -m tinker_cookbook.recipes.multi_agent_debate.openrouter_selfplay \\
    env=\"verifiable\" \\
    dataset_path=\"tinker_cookbook/data/aime2025_sample.jsonl\" \\
    problem_field=\"query\" \\
    answer_field=\"answer\" \\
    policy_model=\"openai/gpt-4o-mini\" \\
    num_agents=3 \\
    max_rounds=3 \\
    history_rounds=2 \\
    max_questions=1
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal

from dotenv import load_dotenv

load_dotenv()


import chz

from tinker_cookbook.completers import OpenRouterMessageCompleter
from tinker_cookbook.renderers import Message, get_text_content

from .coordinator import MultiAgentCoordinator
from .env import MultiAgentDebateEnv
from .loaders import load_math_problems_from_jsonl, load_questions_from_jsonl
from .utils import STOP_CONDITION, get_step_idx_before_turn
from .verifiable_env import VerifiableMathProblem, VerifiableMultiAgentDebateEnv

PolicyFn = Callable[[list[Message]], Awaitable[Message]]


@dataclass
class _Transition:
    reward: float = 0.0


@dataclass
class _Trajectory:
    transitions: list[_Transition]


def _populate_stepwise_rewards(
    *,
    trajectory_group: list[_Trajectory],
    env_group: list[object],
    num_agents: int,
) -> dict[str, float]:
    """Same logic as `BaseMultiAgentEnvGroupBuilder._populate_stepwise_rewards`."""
    if not env_group:
        raise ValueError("empty env_group")

    env0 = env_group[0]
    if not hasattr(env0, "coordinator"):
        raise ValueError(f"Environment {type(env0)} does not have a coordinator")
    coordinator = env0.coordinator

    for turn_idx, response in enumerate(coordinator.state.agent_responses):
        author_id = response.author_id

        for agent_a, op, agent_b in response.comparisons:
            if not (0 <= agent_a < num_agents and 0 <= agent_b < num_agents):
                continue
            if agent_a == agent_b:
                continue
            if op not in {">", "<"}:
                continue

            agent_a_step_idx = get_step_idx_before_turn(agent_a, turn_idx, num_agents)
            agent_b_step_idx = get_step_idx_before_turn(agent_b, turn_idx, num_agents)
            if agent_a_step_idx < 0 or agent_b_step_idx < 0:
                continue

            # Bounds guard (can happen if an agent produced no usable step for some reason).
            if agent_a_step_idx >= len(trajectory_group[agent_a].transitions):
                continue
            if agent_b_step_idx >= len(trajectory_group[agent_b].transitions):
                continue

            if op == ">":
                trajectory_group[agent_a].transitions[agent_a_step_idx].reward += 1.0
                trajectory_group[agent_b].transitions[agent_b_step_idx].reward -= 1.0
            else:
                trajectory_group[agent_a].transitions[agent_a_step_idx].reward -= 1.0
                trajectory_group[agent_b].transitions[agent_b_step_idx].reward += 1.0

    total_comparisons_used = 0.0
    for trajectory in trajectory_group:
        for transition in trajectory.transitions:
            if transition.reward != 0.0:
                total_comparisons_used += abs(transition.reward)

    return {"stepwise_comparisons_used": total_comparisons_used}


@chz.chz
class SelfPlayConfig:
    env: Literal["verifiable", "non-verifiable"] = "verifiable"

    # OpenRouter policy model (NOT the Tinker base model name)
    policy_model: str = "openai/gpt-4o-mini"
    max_tokens: int = 8196
    temperature: float = 0.7

    # Debate config
    num_agents: int = 3
    max_rounds: int = 3
    history_rounds: int = 2

    # Optional summarization of history (also via OpenRouter)
    summarize_history: bool = False
    summarize_model: str = "openai/gpt-4o-mini"
    summarize_max_tokens: int = 1024
    summarize_temperature: float = 0.2

    # Dataset (verifiable)
    dataset_path: str = "tinker_cookbook/data/aime2025_sample.jsonl"
    problem_field: str = "query"
    answer_field: str = "answer"

    # Dataset (non-verifiable)
    non_verifiable_dataset_path: str = "tinker_cookbook/data/longwriter_6k_sample.jsonl"
    non_verifiable_problem_field: str = "query"

    # How many questions to run
    max_questions: int = 1

    # Logging
    log_full_transcript: bool = True
    stream_agent_responses: bool = True  # Print each agent response as it arrives.
    stream_turn_context: bool = True  # Print the exact observation/context given to the agent.
    output_dir: str | None = None  # If set, writes transcript files here.


class _OpenRouterPolicy:
    """OpenRouter chat policy with explicit `stop` support."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        stop: list[str],
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._stop = stop

    async def __call__(self, messages: list[Message]) -> Message:
        resp = await self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stop=self._stop,
        )
        return {"role": "assistant", "content": resp.choices[0].message.content or ""}


def _format_transcript(
    coordinator: MultiAgentCoordinator,
    *,
    env_kind: Literal["verifiable", "non-verifiable"],
    num_agents: int,
) -> str:
    """Format the full conversation transcript (including raw + parsed fields)."""
    lines: list[str] = []
    lines.append(f"Question: {coordinator.state.question}")
    lines.append(f"num_agents={num_agents} max_turns={coordinator.state.max_turns} env={env_kind}")
    lines.append("")
    for turn_idx, resp in enumerate(coordinator.state.agent_responses, start=1):
        lines.append(f"== Turn {turn_idx} (Agent {resp.author_id}) ==")
        if resp.observation:
            lines.append("-- Observation --")
            lines.append(resp.observation.rstrip())
        if resp.thinking:
            lines.append("-- Thinking (stripped) --")
            lines.append(resp.thinking.rstrip())
        lines.append("-- Parsed <solution> --")
        lines.append(resp.solution.rstrip())
        lines.append("-- Parsed <evaluation> --")
        lines.append(resp.evaluation.rstrip())
        lines.append("-- Parsed <comparison> (raw text) --")
        lines.append(resp.comparison_text.rstrip())
        lines.append("-- Parsed comparisons (tuples) --")
        lines.append(str(resp.comparisons))
        lines.append("-- Raw response --")
        lines.append(resp.raw_response.rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


async def _run_one_debate(
    *,
    question: str,
    num_agents: int,
    max_rounds: int,
    history_rounds: int,
    policy: PolicyFn,
    summarize_history: bool,
    summarizer: OpenRouterMessageCompleter | None,
    env_kind: Literal["verifiable", "non-verifiable"],
    stream_agent_responses: bool,
    stream_turn_context: bool,
) -> tuple[list[_Trajectory], dict[str, float], MultiAgentCoordinator]:
    coordinator = MultiAgentCoordinator(
        question=question, num_agents=num_agents, max_turns=num_agents * max_rounds
    )

    # We reuse the existing env objects to format system prompts and observation strings,
    # but we do not use renderers/tokens in this script.
    env_group: list[object] = []
    for agent_id in range(num_agents):
        if env_kind == "verifiable":
            env_group.append(
                VerifiableMultiAgentDebateEnv(
                    agent_id=agent_id,
                    coordinator=coordinator,
                    renderer=None,  # not used here (OpenRouter chat messages)
                    self_play=True,
                    history_turns=history_rounds,
                    summarize_history=summarize_history,
                    _summarizer_policy=summarizer,
                    model_name=None,
                )
            )
        else:
            env_group.append(
                MultiAgentDebateEnv(
                    agent_id=agent_id,
                    coordinator=coordinator,
                    renderer=None,  # not used here (OpenRouter chat messages)
                    self_play=True,
                    history_turns=history_rounds,
                    summarize_history=summarize_history,
                    _summarizer_policy=summarizer,
                    model_name=None,
                )
            )

    trajectories: list[_Trajectory] = [_Trajectory(transitions=[]) for _ in range(num_agents)]

    async def run_agent(agent_id: int) -> None:
        env = env_group[agent_id]
        while True:
            await coordinator.wait_for_turn(agent_id)
            if coordinator.done:
                return

            system_prompt = env.get_system_prompt()  # type: ignore[attr-defined]
            observation = await env.get_observation_string()  # type: ignore[attr-defined]

            if stream_turn_context:
                turn_num = coordinator.state.current_turn + 1
                print("-" * 80, flush=True)
                print(f"Turn {turn_num} (Agent {agent_id}) observation/context:", flush=True)
                print(observation.rstrip(), flush=True)
                print("-" * 80, flush=True)

            messages: list[Message] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": observation},
            ]
            resp = await policy(messages)
            resp_text = get_text_content(resp)

            parsed = await coordinator.submit_response(agent_id, resp_text, observation=observation)
            trajectories[agent_id].transitions.append(_Transition(reward=0.0))

            if stream_agent_responses:
                turn_num = len(coordinator.state.agent_responses)
                print("-" * 80, flush=True)
                print(f"Turn {turn_num} (Agent {agent_id}) raw response:", flush=True)
                print(parsed.raw_response.rstrip(), flush=True)
                print("-" * 80, flush=True)

            if coordinator.done:
                return

    await asyncio.gather(*[run_agent(i) for i in range(num_agents)])

    stepwise_metrics = _populate_stepwise_rewards(
        trajectory_group=trajectories,
        env_group=env_group,
        num_agents=num_agents,
    )
    return trajectories, stepwise_metrics, coordinator


async def main_async() -> None:
    cfg = chz.entrypoint(SelfPlayConfig)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    # Main policy model (agents). Use explicit stop, matching `STOP_CONDITION`.
    base_policy = _OpenRouterPolicy(
        api_key=api_key,
        model_name=cfg.policy_model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        stop=list(STOP_CONDITION),
    )

    # Summarizer model (optional)
    summarizer: OpenRouterMessageCompleter | None = None
    if cfg.summarize_history:
        summarizer = OpenRouterMessageCompleter(
            api_key=api_key,
            model_name=cfg.summarize_model,
            max_tokens=cfg.summarize_max_tokens,
            temperature=cfg.summarize_temperature,
        )

    async def policy(messages: list[Message]) -> Message:
        return await base_policy(messages)

    # Load questions
    questions: list[str]
    if cfg.env == "verifiable":
        problems: list[VerifiableMathProblem] = load_math_problems_from_jsonl(
            path=cfg.dataset_path,
            problem_field=cfg.problem_field,
            answer_field=cfg.answer_field,
            max_count=cfg.max_questions,
        )
        questions = [p.problem for p in problems]
    else:
        questions = load_questions_from_jsonl(
            path=cfg.non_verifiable_dataset_path,
            field=cfg.non_verifiable_problem_field,
            max_count=cfg.max_questions,
        )

    for qi, question in enumerate(questions):
        print("=" * 80)
        print(f"Question {qi}")
        print(question)

        trajectories, stepwise_metrics, coordinator = await _run_one_debate(
            question=question,
            num_agents=cfg.num_agents,
            max_rounds=cfg.max_rounds,
            history_rounds=cfg.history_rounds,
            policy=policy,
            summarize_history=cfg.summarize_history,
            summarizer=summarizer,
            env_kind=cfg.env,
            stream_agent_responses=cfg.stream_agent_responses,
            stream_turn_context=cfg.stream_turn_context,
        )

        print("Stepwise metrics:", stepwise_metrics)
        for agent_id, traj in enumerate(trajectories):
            total = sum(t.reward for t in traj.transitions)
            print(f"Agent {agent_id}: steps={len(traj.transitions)} total_reward={total}")

        if cfg.log_full_transcript or cfg.output_dir is not None:
            transcript = _format_transcript(
                coordinator,
                env_kind=cfg.env,
                num_agents=cfg.num_agents,
            )
            if cfg.log_full_transcript:
                print("-" * 80)
                print(transcript)

            out_dir = Path(cfg.output_dir).expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = f".openrouter_selfplay_q{qi:03d}.txt"
            out_path.write_text(transcript, encoding="utf-8")
            print(f"Wrote transcript: {out_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
