"""File loading utilities for multi-agent debate datasets."""

import json
import random
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .verifiable_env import VerifiableMathProblem

T = TypeVar("T")


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
            if i >= max_count:
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
    dataset_name_field: str | None = None,
    max_count: int = 1000,
) -> list["VerifiableMathProblem"]:
    """Load verifiable math problems from JSONL file.

    Args:
        path: Path to JSONL file
        problem_field: Field name containing the problem text
        answer_field: Field name containing the answer
        dataset_name_field: Optional field name for dataset identifier
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
            dataset_name = "math" if dataset_name_field is None else dataset_name_field
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


def split_items(
    items: list[T],
    test_frac: float,
    num_test_items: int = 0,
    seed: int = 42,
) -> tuple[list[T], list[T]]:
    """Split items into disjoint train/test sets.

    Args:
        items: List of items to split
        test_frac: Fraction of items to use for testing
        num_test_items: Number of test items (if 0, no split)
        seed: Random seed for shuffling

    Returns:
        Tuple of (train_items, test_items)
    """
    if num_test_items <= 0 or test_frac <= 0:
        return items, []
    if len(items) < 2:
        return items, []

    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    test_n = int(round(len(shuffled) * test_frac))
    test_n = max(1, min(test_n, len(shuffled) - 1))
    return shuffled[test_n:], shuffled[:test_n]
