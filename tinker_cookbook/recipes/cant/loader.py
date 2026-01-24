"""Dataset loading utilities for CANT training."""

from pathlib import Path
from typing import Literal


def parse_dataset_list(value: str | None) -> list[str]:
    if value is None:
        return []
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


def _load_jsonl(path: str) -> list[dict]:
    import json

    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    if not items:
        raise ValueError(f"No records loaded from {path}")
    return items


def _slice_records(records: list[dict], max_count: int) -> list[dict]:
    if max_count > 0:
        return records[:max_count]
    return records


_NON_VERIFIABLE_DATASETS: dict[str, dict[str, str]] = {
    "longwriter_6k": {
        "path": "tinker_cookbook/data/longwriter_6k_sample.jsonl",
        "field": "query",
    },
    "write_bench": {
        "path": "tinker_cookbook/data/write_bench.jsonl",
        "field": "query",
    },
    "hellobench": {
        "path": "tinker_cookbook/data/hellobench_sample.jsonl",
        "field": "query",
    },
}

_VERIFIABLE_DATASETS = {
    "math",
    "math500",
    "polaris",
    "deepmath",
    "gsm8k",
    "aime2024",
    "aime2025",
    "deepmath",
    "gpqa",
}


def load_non_verifiable_dataset_states(name: str, max_count: int) -> list[dict]:
    dataset = _NON_VERIFIABLE_DATASETS.get(name)
    if dataset is None:
        raise ValueError(
            f"Unknown non-verifiable dataset: {name}. Supported: {sorted(_NON_VERIFIABLE_DATASETS)}"
        )
    problems = _slice_records(_load_jsonl(dataset["path"]), max_count)
    return [
        {
            "question": p[dataset["field"]],
            "dataset_name": name,
        }
        for p in problems
    ]


def _limit_dataset_rows(ds: "object", max_count: int) -> "object":
    if max_count > 0 and hasattr(ds, "select"):
        try:
            return ds.select(range(min(max_count, len(ds))))
        except Exception:
            return ds
    return ds


def _load_math_problems_from_jsonl(
    path: str,
    problem_field: str,
    answer_field: str,
    dataset_name: str,
    max_count: int,
) -> list[dict]:
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_count and max_count > 0:
                break
            data = _load_jsonl_line(line)
            if problem_field in data:
                problem = data[problem_field]
            elif "query" in data:
                problem = data["query"]
            else:
                raise ValueError(
                    f"Each JSONL row must include '{problem_field}' (or 'query'). "
                    f"Got keys={list(data.keys())}"
                )
            if answer_field in data:
                answer = data[answer_field]
            elif "answer" in data:
                answer = data["answer"]
            elif "final_answer" in data:
                answer = data["final_answer"]
            elif "solution" in data:
                answer = data["solution"]
            else:
                raise ValueError(
                    f"Each JSONL row must include '{answer_field}' (or answer fields). "
                    f"Got keys={list(data.keys())}"
                )
            problems.append(
                {
                    "question": str(problem),
                    "answer": str(answer),
                    "dataset_name": dataset_name,
                }
            )
    if not problems:
        raise ValueError(f"No problems loaded from {path}")
    return problems


def _load_jsonl_line(line: str) -> dict:
    import json

    line = line.strip()
    if not line:
        raise ValueError("Empty JSONL line")
    return json.loads(line)


def load_math_problems_from_hf(
    dataset_name: Literal[
        "math", "math500", "polaris", "deepmath", "gsm8k", "aime2024", "aime2025", "deepmath"
    ],
    split: Literal["train", "test"] = "train",
    max_count: int = 1000,
) -> list[dict]:
    """Load verifiable math problems from HuggingFace datasets or local JSONL."""
    if dataset_name in {"aime2024", "aime2025"}:
        local_path = Path(__file__).resolve().parents[2] / "data" / f"{dataset_name}_sample.jsonl"
        if not local_path.exists():
            raise ValueError(f"Missing local dataset file: {local_path}")
        return _load_math_problems_from_jsonl(
            path=str(local_path),
            problem_field="problem",
            answer_field="answer",
            dataset_name=dataset_name,
            max_count=max_count,
        )

    if dataset_name == "deepmath":
        local_path = (
            Path(__file__).resolve().parents[2] / "data" / "deepmath_1410_level_9_10_sample.jsonl"
        )
        if not local_path.exists():
            raise ValueError(f"Missing local dataset file: {local_path}")
        return _load_math_problems_from_jsonl(
            path=str(local_path),
            problem_field="problem",
            answer_field="answer",
            dataset_name=dataset_name,
            max_count=max_count,
        )

    if dataset_name == "gpqa":
        local_path = Path(__file__).resolve().parents[2] / "data" / "gpqa_diamond_sample.jsonl"

        if not local_path.exists():
            raise ValueError(f"Missing local dataset file: {local_path}")
        return _load_math_problems_from_jsonl(
            path=str(local_path),
            problem_field="problem",
            answer_field="answer",
            dataset_name=dataset_name,
            max_count=max_count,
        )

    from datasets import load_dataset

    from tinker_cookbook.recipes.math_rl import math_env
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

    if dataset_name == "math":
        if split == "test":
            ds = math_env._get_hendrycks_math_test()
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

    problems: list[dict] = []
    for idx, row in enumerate(ds):
        if max_count > 0 and idx >= max_count:
            break
        try:
            if dataset_name in {"math", "math500"}:
                problem = row["problem"]
                solution = row.get("solution") or row.get("answer") or ""
                answer = extract_boxed(solution)
            elif dataset_name == "polaris":
                problem = row.get("problem", "")
                answer = row.get("answer", "")
            elif dataset_name == "deepmath":
                problem = row.get("question", "")
                answer = row.get("final_answer", "")
            elif dataset_name == "gsm8k":
                problem = row.get("question", "")
                answer = math_env.extract_gsm8k_final_answer(row.get("answer", ""))
            else:
                raise ValueError(f"Unhandled dataset_name: {dataset_name}")
            if not (problem and answer):
                raise ValueError("Missing problem or answer")
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "Skipping %s row %s due to parse error: %s", dataset_name, idx, exc
            )
            continue
        problems.append(
            {
                "question": str(problem),
                "answer": str(answer),
                "dataset_name": dataset_label,
            }
        )

    if not problems:
        raise ValueError(f"No problems loaded from dataset {dataset_name}:{split}")

    return problems


def load_verifiable_dataset_states(
    datasets: list[str],
    split: Literal["train", "test"],
    max_count: int,
) -> list[dict]:
    if not datasets:
        return []
    problems = []
    for name in datasets:
        if name not in _VERIFIABLE_DATASETS:
            raise ValueError(
                f"Unknown verifiable dataset: {name}. Supported: {sorted(_VERIFIABLE_DATASETS)}"
            )
        problems.extend(load_math_problems_from_hf(name, split=split, max_count=max_count))
    return problems
