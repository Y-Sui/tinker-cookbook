"""File loading utilities for multi-agent debate datasets."""

import logging
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

if TYPE_CHECKING:
    from .verifiable_env import VerifiableMathProblem

T = TypeVar("T")
logger = logging.getLogger(__name__)


def load_questions_from_jsonl(
    path: str,
    field: str,
    max_count: int = 1000,
) -> list[str]:
    """Load questions from JSONL file.

    Args:
        path: Path to JSONL file
        field: Field name containing the question text
        max_count: Maximum number of questions to load

    Returns:
        List of question strings

    Raises:
        ValueError: If no questions are loaded or field is missing
    """
    questions: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_count and max_count > 0:
                break
            data = json.loads(line)
            if field not in data:
                raise ValueError(
                    f"Each JSONL row must include '{field}'. Got keys={list(data.keys())}"
                )
            questions.append(str(data[field]))
    if not questions:
        raise ValueError(f"No questions loaded from {path}")
    return questions


def load_math_problems_from_jsonl(
    path: str,
    problem_field: str,
    answer_field: str,
    max_count: int = 1000,
) -> list["VerifiableMathProblem"]:
    """Load verifiable math problems from JSONL file.

    Args:
        path: Path to JSONL file
        problem_field: Field name containing the problem text
        answer_field: Field name containing the answer
        max_count: Maximum number of problems to load

    Returns:
        List of VerifiableMathProblem instances

    Raises:
        ValueError: If no problems are loaded or required fields are missing
    """
    # Import here to avoid circular dependency
    from .verifiable_env import VerifiableMathProblem

    problems: list[VerifiableMathProblem] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_count and max_count > 0:
                break
            data = json.loads(line)
            if problem_field not in data or answer_field not in data:
                raise ValueError(
                    f"Each JSONL row must include '{problem_field}' and '{answer_field}'. "
                    f"Got keys={list(data.keys())}"
                )
            # Infer dataset_name from filename
            # (e.g., "aime2024_sample.jsonl" -> "aime2024")
            dataset_name = Path(path).stem.split("_sample")[0]
            if not dataset_name:
                dataset_name = Path(path).stem  # Use full stem if no _sample
            problems.append(
                VerifiableMathProblem(
                    problem=str(data[problem_field]),
                    answer=str(data[answer_field]),
                    dataset_name=dataset_name,
                )
            )
    if not problems:
        raise ValueError(f"No problems loaded from {path}")
    return problems


def _limit_dataset_rows(ds: "object", max_count: int) -> "object":
    """Return a dataset limited to max_count rows when possible."""
    if max_count > 0 and hasattr(ds, "select"):
        try:
            return ds.select(range(min(max_count, len(ds))))
        except Exception:
            # Fall back to full dataset; caller will slice via iteration.
            return ds
    return ds


def load_math_problems_from_hf(
    dataset_name: Literal["math", "math500", "polaris", "deepmath", "gsm8k"],
    split: Literal["train", "test"] = "train",
    max_count: int = 1000,
) -> list["VerifiableMathProblem"]:
    """Load verifiable math problems from HuggingFace datasets.

    Supported datasets (aligned with math_rl):
    - math: Hendrycks MATH (train split excludes MATH-500 test problems)
    - math500: MATH-500 test set
    - polaris: Polaris-Dataset-53K
    - deepmath: DeepMath-103K
    - gsm8k: GSM8K (requires parsing the final answer line)
    """
    from datasets import load_dataset

    from .verifiable_env import VerifiableMathProblem
    from tinker_cookbook.recipes.math_rl import math_env
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

    if dataset_name == "math":
        if split == "test":
            ds = math_env._get_hendrycks_math_test()  # MATH-500
            dataset_label = "math500"
        else:
            ds = math_env._get_hendrycks_math_train()
            dataset_label = "math"
    elif dataset_name == "math500":
        ds = math_env._get_hendrycks_math_test()
        dataset_label = "math500"
    elif dataset_name == "polaris":
        ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")
        dataset_label = "polaris"
    elif dataset_name == "deepmath":
        ds = load_dataset("zwhe99/DeepMath-103K", split="train")
        dataset_label = "deepmath"
    elif dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", name="main", split=split)
        dataset_label = "gsm8k"
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    ds = _limit_dataset_rows(ds, max_count)

    problems: list[VerifiableMathProblem] = []
    for idx, row in enumerate(ds):
        if max_count > 0 and idx >= max_count:
            break
        try:
            if dataset_label in {"math", "math500"}:
                problem = row["problem"]
                solution = row.get("solution") or row.get("answer") or ""
                answer = extract_boxed(solution)
            elif dataset_label == "polaris":
                problem = row.get("problem", "")
                answer = row.get("answer", "")
            elif dataset_label == "deepmath":
                problem = row.get("question", "")
                answer = row.get("final_answer", "")
            elif dataset_label == "gsm8k":
                problem = row.get("question", "")
                answer = math_env.extract_gsm8k_final_answer(row.get("answer", ""))
            else:
                raise ValueError(f"Unhandled dataset label: {dataset_label}")
            if not (problem and answer):
                raise ValueError("Missing problem or answer")
        except Exception as exc:
            logger.warning("Skipping %s row %s due to parse error: %s", dataset_label, idx, exc)
            continue
        problems.append(
            VerifiableMathProblem(
                problem=str(problem),
                answer=str(answer),
                dataset_name=dataset_label,
            )
        )

    if not problems:
        raise ValueError(f"No problems loaded from HF dataset {dataset_name}:{split}")

    return problems
