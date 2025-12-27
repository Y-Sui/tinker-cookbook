"""Example evaluation script for math multi-agent debate.

This script demonstrates how to evaluate a trained model on various
math benchmarks using multi-agent debate with correctness metrics.

Usage:
    python eval_math_debate.py --checkpoint path/to/model --dataset aime_2024

Configuration:
    Edit the parameters in the evaluation functions to customize.
"""

import argparse
import asyncio
from pathlib import Path

from tinker_cookbook.recipes.multi_agent_debate.math_debate_dataset import (
    AIME2024EvalBuilder,
    AIME2025EvalBuilder,
    MATH500EvalBuilder,
    MathDebateDatasetBuilder,
)
from tinker_cookbook.rl.train import train_rl
from tinker_cookbook.utils import logger


async def eval_on_aime_2024(model_path: str):
    """Evaluate on AIME 2024 problems."""
    logger.info(f"Evaluating {model_path} on AIME 2024...")

    dataset_builder = AIME2024EvalBuilder(
        batch_size=1,  # Evaluate one problem at a time
        num_test_datapoints=30,  # All 30 AIME problems
        num_agents=3,
        max_rounds=4,  # Longer rounds for hard problems
        history_rounds=3,
        summarize_history=True,
        model_name=model_path,
        renderer_name="qwen3",  # Adjust based on your model
        eval_reward_mode="correctness",
        log_full_transcript=True,  # See full debate for analysis
        format_coef=0.1,
        grader="sympy",
    )

    await train_rl(
        dataset_builder=dataset_builder,
        base_model=model_path,
        n_batches=0,  # No training, eval only
        eval_every=1,
        output_dir=f"outputs/eval_aime_2024_{Path(model_path).name}",
    )

    logger.info("AIME 2024 evaluation completed!")


async def eval_on_aime_2025(model_path: str):
    """Evaluate on AIME 2025 problems."""
    logger.info(f"Evaluating {model_path} on AIME 2025...")

    dataset_builder = AIME2025EvalBuilder(
        batch_size=1,
        num_test_datapoints=30,
        num_agents=3,
        max_rounds=4,
        history_rounds=3,
        summarize_history=True,
        model_name=model_path,
        renderer_name="qwen3",
        eval_reward_mode="correctness",
        log_full_transcript=True,
        format_coef=0.1,
        grader="sympy",
    )

    await train_rl(
        dataset_builder=dataset_builder,
        base_model=model_path,
        n_batches=0,
        eval_every=1,
        output_dir=f"outputs/eval_aime_2025_{Path(model_path).name}",
    )

    logger.info("AIME 2025 evaluation completed!")


async def eval_on_math_500(model_path: str):
    """Evaluate on MATH-500 benchmark."""
    logger.info(f"Evaluating {model_path} on MATH-500...")

    dataset_builder = MATH500EvalBuilder(
        batch_size=1,
        num_test_datapoints=500,
        num_agents=3,
        max_rounds=3,
        history_rounds=2,
        model_name=model_path,
        renderer_name="qwen3",
        eval_reward_mode="correctness",
        log_full_transcript=False,  # Disable for large eval
        format_coef=0.1,
        grader="sympy",
    )

    await train_rl(
        dataset_builder=dataset_builder,
        base_model=model_path,
        n_batches=0,
        eval_every=1,
        output_dir=f"outputs/eval_math_500_{Path(model_path).name}",
    )

    logger.info("MATH-500 evaluation completed!")


async def eval_single_agent_baseline(model_path: str, dataset: str = "aime_2024"):
    """Evaluate single agent (no debate) as baseline."""
    logger.info(f"Evaluating {model_path} single agent on {dataset}...")

    if dataset == "aime_2024":
        builder_class = AIME2024EvalBuilder
        num_test = 30
    elif dataset == "aime_2025":
        builder_class = AIME2025EvalBuilder
        num_test = 30
    elif dataset == "math_500":
        builder_class = MATH500EvalBuilder
        num_test = 500
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    dataset_builder = builder_class(
        batch_size=1,
        num_test_datapoints=num_test,
        num_agents=1,  # Single agent (no debate)
        max_rounds=1,
        model_name=model_path,
        renderer_name="qwen3",
        eval_reward_mode="correctness",
        log_full_transcript=True,
    )

    await train_rl(
        dataset_builder=dataset_builder,
        base_model=model_path,
        n_batches=0,
        eval_every=1,
        output_dir=f"outputs/eval_{dataset}_single_{Path(model_path).name}",
    )

    logger.info(f"Single agent evaluation on {dataset} completed!")


async def eval_with_different_num_agents(model_path: str, dataset: str = "aime_2024"):
    """Compare performance with different numbers of agents."""
    logger.info(f"Evaluating {model_path} with varying num_agents on {dataset}...")

    if dataset == "aime_2024":
        builder_class = AIME2024EvalBuilder
        num_test = 30
    elif dataset == "math_500":
        builder_class = MATH500EvalBuilder
        num_test = 100  # Use subset for faster comparison
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    for num_agents in [1, 2, 3, 5]:
        logger.info(f"Testing with {num_agents} agents...")

        dataset_builder = builder_class(
            batch_size=1,
            num_test_datapoints=num_test,
            num_agents=num_agents,
            max_rounds=3,
            model_name=model_path,
            renderer_name="qwen3",
            eval_reward_mode="correctness",
            log_full_transcript=False,
        )

        await train_rl(
            dataset_builder=dataset_builder,
            base_model=model_path,
            n_batches=0,
            eval_every=1,
            output_dir=f"outputs/eval_{dataset}_agents{num_agents}_{Path(model_path).name}",
        )

    logger.info("Multi-agent comparison completed!")


async def eval_all_benchmarks(model_path: str):
    """Run evaluation on all available benchmarks."""
    logger.info(f"Running comprehensive evaluation for {model_path}...")

    # Evaluate on all datasets
    await eval_on_aime_2024(model_path)
    await eval_on_aime_2025(model_path)
    await eval_on_math_500(model_path)

    logger.info("All evaluations completed!")


async def main():
    """Run evaluation experiments."""
    parser = argparse.ArgumentParser(description="Evaluate math debate model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Qwen/Qwen3-8B-Instruct",
        help="Path to model checkpoint or HuggingFace model name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["aime_2024", "aime_2025", "math_500", "all"],
        default="aime_2024",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run single-agent baseline (no debate)",
    )
    parser.add_argument(
        "--compare-agents",
        action="store_true",
        help="Compare performance with different numbers of agents",
    )

    args = parser.parse_args()

    if args.baseline:
        await eval_single_agent_baseline(args.checkpoint, args.dataset)
    elif args.compare_agents:
        await eval_with_different_num_agents(args.checkpoint, args.dataset)
    elif args.dataset == "all":
        await eval_all_benchmarks(args.checkpoint)
    elif args.dataset == "aime_2024":
        await eval_on_aime_2024(args.checkpoint)
    elif args.dataset == "aime_2025":
        await eval_on_aime_2025(args.checkpoint)
    elif args.dataset == "math_500":
        await eval_on_math_500(args.checkpoint)


if __name__ == "__main__":
    asyncio.run(main())
