"""Task-specific metrics for CANT."""

import asyncio
import json
import logging
import math
import os
import re
from typing import Literal

from tinker_cookbook.recipes.cant.coordinator import CANTCoordinator
from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)
from tinker_cookbook.completers import OpenRouterMessageCompleter
from tinker_cookbook.renderers import Message

logger = logging.getLogger(__name__)


def _extract_gsm8k_final_answer(text: str) -> str:
    lines = text.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("####"):
            content = s[4:].strip()
            if content.startswith(":"):
                content = content[1:].strip()
            content = content.replace(",", "").strip()
            return content
    matches = re.findall(r"####\\s*(.+)", text)
    if matches:
        return matches[-1].strip()
    raise ValueError("No GSM8K final answer found")


def _safe_grade(
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


def _extract_answer(solution: str, dataset_name: str) -> tuple[bool, str | None]:
    if dataset_name == "gsm8k":
        try:
            return True, _extract_gsm8k_final_answer(solution)
        except ValueError:
            pass
    try:
        return True, extract_boxed(solution)
    except ValueError:
        return False, None


def compute_verifiable_metrics(
    coordinator: CANTCoordinator,
    answer: str,
    num_agents: int,
    dataset_name: str | None = None,
    grader: Literal["sympy", "math_verify"] = "sympy",
    grade_timeout: float = 1.0,
) -> list[dict[str, float]]:
    """Compute correctness metrics for verifiable CANT tasks.

    Metrics are based on each agent's revised solution (Round 3).
    """
    dataset_label = dataset_name or "unknown"
    per_agent: list[dict[str, float]] = []
    for agent_id in range(num_agents):
        solution = coordinator.revised_solutions.get(agent_id, "")
        has_answer, extracted = _extract_answer(solution, dataset_label)
        correct = False
        if extracted is not None:
            correct = _safe_grade(extracted, answer, grader=grader, timeout=grade_timeout)
        per_agent.append(
            {
                "format": 1.0 if has_answer else 0.0,
                "correct": 1.0 if correct else 0.0,
            }
        )

    any_correct = any(m["correct"] > 0.5 for m in per_agent)
    metrics = []
    for m in per_agent:
        metrics.append(
            {
                f"{dataset_label}/format": m["format"],
                f"{dataset_label}/correct": m["correct"],
                f"{dataset_label}/pass@{num_agents}": 1.0 if any_correct else 0.0,
            }
        )
    return metrics


def _parse_judge_score(text: str) -> float | None:
    try:
        data = json.loads(text)
        score = float(data.get("score"))
        return max(0.0, min(1.0, score))
    except Exception:
        pass

    match = re.search(r"\"score\"\\s*:\\s*([0-9]*\\.?[0-9]+)", text)
    if match:
        try:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        except ValueError:
            return None
    return None


async def compute_non_verifiable_metrics(
    coordinator: CANTCoordinator,
    num_agents: int,
    dataset_name: str | None = None,
    judge_model: str = "openai/gpt-4o-mini",
    api_key: str | None = None,
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> list[dict[str, float]]:
    """Compute judge-based metrics for non-verifiable CANT tasks."""
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Required for LLM judge metrics."
        )

    completer = OpenRouterMessageCompleter(
        api_key=api_key,
        model_name=judge_model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    dataset_label = dataset_name or "unknown"
    question = coordinator.question

    async def judge_answer(answer: str, agent_id: int) -> float:
        prompt: list[Message] = [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator. Score the answer to the user question "
                    "from 0.0 to 1.0. Consider correctness, completeness, relevance, and "
                    "clarity. Return JSON only: {\"score\": <float>}."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nAnswer:\n{answer}",
            },
        ]
        response = await completer(prompt)
        content = response.get("content") or ""
        score = _parse_judge_score(content)
        if score is None:
            logger.warning("Failed to parse judge score for agent %s: %s", agent_id, content)
            return 0.0
        return score

    tasks = []
    for agent_id in range(num_agents):
        answer = coordinator.revised_solutions.get(agent_id, "")
        tasks.append(judge_answer(answer, agent_id))
    scores = await asyncio.gather(*tasks)

    metrics = []
    for score in scores:
        metrics.append({f"{dataset_label}/judge_score": score})
    return metrics
