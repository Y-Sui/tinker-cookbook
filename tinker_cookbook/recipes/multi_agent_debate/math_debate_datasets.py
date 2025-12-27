"""Dataset loaders for math multi-agent debate.

Supports:
- Training datasets: MATH, GSM8K, Polaris (via HuggingFace)
- Evaluation datasets: AIME 2024, AIME 2025, MATH-500 (via JSONL files or HuggingFace)
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import chz
from datasets import Dataset, load_dataset

from tinker_cookbook.recipes.math_rl.math_env import (
    _get_hendrycks_math_test,
    _get_hendrycks_math_train,
    extract_boxed,
    extract_gsm8k_final_answer,
)
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed as extract_boxed_answer
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logger


@dataclass(frozen=True)
class MathProblem:
    """A math problem with ground truth answer."""

    problem: str
    answer: str
    dataset_name: str = "math"
    problem_id: str | None = None
    metadata: dict | None = None


def load_jsonl_math_problems(
    file_path: str | Path,
    problem_field: str = "problem",
    answer_field: str = "answer",
    dataset_name: str = "custom",
    max_problems: int | None = None,
) -> list[MathProblem]:
    """Load math problems from a JSONL file.

    Expected format per line:
    {"problem": "What is 2+2?", "answer": "4", ...}

    Args:
        file_path: Path to JSONL file
        problem_field: Field name for problem text
        answer_field: Field name for answer
        dataset_name: Name to assign to this dataset
        max_problems: Maximum number of problems to load

    Returns:
        List of MathProblem objects
    """
    problems = []
    path = Path(file_path)

    if not path.exists():
        logger.warning(f"Dataset file not found: {file_path}")
        return problems

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_problems is not None and i >= max_problems:
                break

            try:
                data = json.loads(line.strip())
                if problem_field not in data or answer_field not in data:
                    logger.warning(
                        f"Line {i+1} missing required fields '{problem_field}' or '{answer_field}'"
                    )
                    continue

                problems.append(
                    MathProblem(
                        problem=str(data[problem_field]),
                        answer=str(data[answer_field]),
                        dataset_name=dataset_name,
                        problem_id=data.get("id", str(i)),
                        metadata=data,
                    )
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {i+1}: {e}")
                continue

    logger.info(f"Loaded {len(problems)} problems from {file_path}")
    return problems


def load_aime_2024(file_path: str | Path = "data/aime_2024.jsonl") -> list[MathProblem]:
    """Load AIME 2024 problems."""
    return load_jsonl_math_problems(
        file_path,
        problem_field="problem",
        answer_field="answer",
        dataset_name="aime_2024",
    )


def load_aime_2025(file_path: str | Path = "data/aime_2025.jsonl") -> list[MathProblem]:
    """Load AIME 2025 problems."""
    return load_jsonl_math_problems(
        file_path,
        problem_field="problem",
        answer_field="answer",
        dataset_name="aime_2025",
    )


def load_math_500_eval() -> list[MathProblem]:
    """Load MATH-500 evaluation set from HuggingFace."""
    try:
        ds = _get_hendrycks_math_test()
        problems = []
        for i, item in enumerate(ds):
            try:
                answer = extract_boxed(item["solution"])  # type: ignore
                problems.append(
                    MathProblem(
                        problem=str(item["problem"]),  # type: ignore
                        answer=answer,
                        dataset_name="math_500",
                        problem_id=str(i),
                        metadata=dict(item),  # type: ignore
                    )
                )
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse MATH-500 problem {i}: {e}")
                continue

        logger.info(f"Loaded {len(problems)} problems from MATH-500")
        return problems
    except Exception as e:
        logger.error(f"Failed to load MATH-500: {e}")
        return []


def load_math_train() -> list[MathProblem]:
    """Load MATH training set from HuggingFace."""
    try:
        ds = _get_hendrycks_math_train()
        problems = []
        for i, item in enumerate(ds):
            try:
                answer = extract_boxed(item["solution"])  # type: ignore
                problems.append(
                    MathProblem(
                        problem=str(item["problem"]),  # type: ignore
                        answer=answer,
                        dataset_name="math",
                        problem_id=str(i),
                        metadata=dict(item),  # type: ignore
                    )
                )
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse MATH problem {i}: {e}")
                continue

        logger.info(f"Loaded {len(problems)} problems from MATH training")
        return problems
    except Exception as e:
        logger.error(f"Failed to load MATH training: {e}")
        return []


def load_gsm8k_train() -> list[MathProblem]:
    """Load GSM8K training set from HuggingFace."""
    try:
        ds = load_dataset("openai/gsm8k", name="main", split="train")
        problems = []
        for i, item in enumerate(ds):  # type: ignore
            try:
                answer = extract_gsm8k_final_answer(item["answer"])  # type: ignore
                problems.append(
                    MathProblem(
                        problem=str(item["question"]),  # type: ignore
                        answer=answer,
                        dataset_name="gsm8k",
                        problem_id=str(i),
                        metadata=dict(item),  # type: ignore
                    )
                )
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse GSM8K problem {i}: {e}")
                continue

        logger.info(f"Loaded {len(problems)} problems from GSM8K")
        return problems
    except Exception as e:
        logger.error(f"Failed to load GSM8K: {e}")
        return []


def load_polaris_train() -> list[MathProblem]:
    """Load Polaris training set from HuggingFace."""
    try:
        ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train")
        problems = []
        for i, item in enumerate(ds):  # type: ignore
            try:
                problem = item.get("problem", "")  # type: ignore
                answer = item.get("answer", "")  # type: ignore
                if not (problem and answer):
                    continue

                problems.append(
                    MathProblem(
                        problem=str(problem),
                        answer=str(answer),
                        dataset_name="polaris",
                        problem_id=str(i),
                        metadata=dict(item),  # type: ignore
                    )
                )
            except (ValueError, KeyError) as e:
                logger.warning(f"Failed to parse Polaris problem {i}: {e}")
                continue

        logger.info(f"Loaded {len(problems)} problems from Polaris")
        return problems
    except Exception as e:
        logger.error(f"Failed to load Polaris: {e}")
        return []


# Dataset registry for easy access
DATASET_LOADERS = {
    # Training datasets
    "math": load_math_train,
    "gsm8k": load_gsm8k_train,
    "polaris": load_polaris_train,

    # Evaluation datasets
    "math_500": load_math_500_eval,
    "aime_2024": lambda: load_aime_2024("data/aime_2024.jsonl"),
    "aime_2025": lambda: load_aime_2025("data/aime_2025.jsonl"),
}


def load_math_dataset(
    dataset_name: str,
    max_problems: int | None = None,
    shuffle_seed: int | None = None,
) -> list[MathProblem]:
    """Load a math dataset by name.

    Args:
        dataset_name: One of: math, gsm8k, polaris, math_500, aime_2024, aime_2025
        max_problems: Maximum number of problems to load
        shuffle_seed: If provided, shuffle problems with this seed

    Returns:
        List of MathProblem objects
    """
    if dataset_name not in DATASET_LOADERS:
        available = ", ".join(DATASET_LOADERS.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
        )

    problems = DATASET_LOADERS[dataset_name]()

    if shuffle_seed is not None:
        import random
        rng = random.Random(shuffle_seed)
        rng.shuffle(problems)

    if max_problems is not None and len(problems) > max_problems:
        problems = problems[:max_problems]

    return problems
