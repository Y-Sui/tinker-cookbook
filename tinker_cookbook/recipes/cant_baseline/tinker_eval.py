"""
Evaluate Qwen3-8B (Tinker sampling) on verifiable math datasets.
"""

import asyncio
import math
from collections import defaultdict
from typing import Literal

import chz
import tinker
from dotenv import load_dotenv

from tinker_cookbook import model_info
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.recipes.cant import loader
from tinker_cookbook.recipes.math_rl import math_env
from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)
from tinker_cookbook.renderers import Message, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

load_dotenv(override=True)


@chz.chz
class TinkerEvalConfig:
    """Configuration for Tinker evaluation."""

    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str | None = "qwen3_disable_thinking"
    max_tokens: int = 1024
    datasets: str = "aime2024,aime2025,math500"
    split: Literal["train", "test"] = "test"
    max_questions: int = -1
    max_concurrency: int = 4
    grader: Literal["sympy", "math_verify"] = "sympy"
    grade_timeout: float = 1.0


def _build_prompt(question: str) -> list[Message]:
    return [
        {
            "role": "system",
            "content": (
                "Solve the problem and provide only the final answer. "
                "Do not include reasoning. Use LaTeX \\boxed{...}."
            ),
        },
        {"role": "user", "content": question},
    ]


def _safe_grade(
    given_answer: str,
    ground_truth: str,
    grader: Literal["sympy", "math_verify"],
    timeout: float,
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
    return bool(out)


def _extract_answer(solution: str, dataset_name: str) -> tuple[bool, str | None]:
    if dataset_name == "gsm8k":
        try:
            return True, math_env.extract_gsm8k_final_answer(solution)
        except ValueError:
            return False, None
    try:
        return True, extract_boxed(solution)
    except ValueError:
        return False, None


async def _solve_one(
    completer: TinkerMessageCompleter,
    question: str,
    semaphore: asyncio.Semaphore,
) -> str:
    async with semaphore:
        response = await completer(_build_prompt(question))
    return response.get("content") or ""


async def _evaluate_problem(
    completer: TinkerMessageCompleter,
    problem: dict,
    semaphore: asyncio.Semaphore,
    grader: Literal["sympy", "math_verify"],
    grade_timeout: float,
) -> tuple[str, dict[str, float]]:
    question = problem["question"]
    answer = problem["answer"]
    dataset_name = problem.get("dataset_name") or "unknown"

    output = await _solve_one(completer, question, semaphore)

    has_answer, extracted = _extract_answer(output, dataset_name)
    correct = False
    if extracted is not None:
        correct = _safe_grade(extracted, answer, grader=grader, timeout=grade_timeout)

    metrics = {
        f"{dataset_name}/format": 1.0 if has_answer else 0.0,
        f"{dataset_name}/correct": 1.0 if correct else 0.0,
        f"{dataset_name}/pass@1": 1.0 if correct else 0.0,
    }
    return dataset_name, metrics


def _accumulate_metrics(
    totals: dict[str, dict[str, float]],
    dataset_name: str,
    metrics: dict[str, float],
) -> None:
    if dataset_name not in totals:
        totals[dataset_name] = defaultdict(float)
        totals[dataset_name]["count"] = 0.0
    totals[dataset_name]["count"] += 1.0
    for key, value in metrics.items():
        if key.endswith("/format"):
            totals[dataset_name]["format"] += float(value)
        elif key.endswith("/correct"):
            totals[dataset_name]["correct"] += float(value)
        elif key.endswith("/pass@1"):
            totals[dataset_name]["pass"] += float(value)


def _print_summary(totals: dict[str, dict[str, float]]) -> None:
    overall = defaultdict(float)
    overall_count = 0.0

    for dataset_name, stats in totals.items():
        count = stats.get("count", 0.0)
        if count <= 0:
            continue
        overall_count += count
        overall["format"] += stats.get("format", 0.0)
        overall["correct"] += stats.get("correct", 0.0)
        overall["pass"] += stats.get("pass", 0.0)

        print(f"\nDataset: {dataset_name}")
        print(f"  Count: {int(count)}")
        print(f"  Format: {stats.get('format', 0.0) / count:.4f}")
        print(f"  Correct: {stats.get('correct', 0.0) / count:.4f}")
        print(f"  Pass@1: {stats.get('pass', 0.0) / count:.4f}")

    if overall_count > 0:
        print("\nOverall:")
        print(f"  Count: {int(overall_count)}")
        print(f"  Format: {overall.get('format', 0.0) / overall_count:.4f}")
        print(f"  Correct: {overall.get('correct', 0.0) / overall_count:.4f}")
        print(f"  Pass@1: {overall.get('pass', 0.0) / overall_count:.4f}")


async def main() -> None:
    config = chz.entrypoint(TinkerEvalConfig)

    datasets = loader.parse_dataset_list(config.datasets)
    if not datasets:
        raise ValueError("datasets must be set to at least one dataset name")

    problems = loader.load_verifiable_dataset_states(
        datasets,
        split=config.split,
        max_count=config.max_questions,
    )
    print(f"Loaded {len(problems)} problems for {config.datasets} ({config.split}).", flush=True)
    if not problems:
        return

    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    tokenizer = get_tokenizer(config.model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    service_client = tinker.ServiceClient()
    completer = TinkerMessageCompleter(
        sampling_client=service_client.create_sampling_client(base_model=config.model_name),
        renderer=renderer,
        max_tokens=config.max_tokens,
        stop_condition=renderer.get_stop_sequences(),
    )

    semaphore = asyncio.Semaphore(max(1, config.max_concurrency))
    totals: dict[str, dict[str, float]] = {}
    total = len(problems)
    completed = 0
    log_every = max(1, total // 10)

    tasks = [
        _evaluate_problem(
            completer,
            problem,
            semaphore,
            grader=config.grader,
            grade_timeout=config.grade_timeout,
        )
        for problem in problems
    ]
    for task in asyncio.as_completed(tasks):
        dataset_name, metrics = await task
        _accumulate_metrics(totals, dataset_name, metrics)
        completed += 1
        if completed % log_every == 0 or completed == total:
            print(f"Progress: {completed}/{total}", flush=True)

    _print_summary(totals)


if __name__ == "__main__":
    asyncio.run(main())
