"""File loading utilities for multi-agent debate datasets."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from ..environments.verifiable import VerifiableMathProblem

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
    from ..environments.verifiable import VerifiableMathProblem

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
