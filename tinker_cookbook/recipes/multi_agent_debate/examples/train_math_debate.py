"""Example training script for math multi-agent debate.

This script demonstrates how to train a model using multi-agent debate
on math problems with debate-based rewards.

Usage:
    python train_math_debate.py

Configuration:
    Edit the parameters in main() to customize training.
"""

import asyncio
from pathlib import Path

from tinker_cookbook.recipes.multi_agent_debate.math_debate_dataset import (
    MathDebateDatasetBuilder,
    MathTrainingBuilder,
    GSM8KTrainingBuilder,
)
from tinker_cookbook.rl.train import train_rl
from tinker_cookbook.utils import logger


async def train_on_math_with_debate():
    """Train on MATH dataset using debate rewards."""
    logger.info("Starting training on MATH with debate rewards...")

    dataset_builder = MathTrainingBuilder(
        # Data configuration
        batch_size=4,
        num_train_datapoints=100,
        num_test_datapoints=20,
        max_problems=1000,  # Limit to first 1000 problems
        shuffle_seed=42,

        # Debate configuration
        num_agents=3,
        max_rounds=3,  # 3 rounds × 3 agents = 9 total turns
        history_rounds=2,  # Show last 2 rounds in context
        summarize_history=False,  # Disable for short debates
        log_full_transcript=True,  # Enable to see debate transcripts

        # Model configuration
        model_name="Qwen/Qwen3-8B-Instruct",
        renderer_name="qwen3",

        # Reward configuration
        train_reward_mode="debate",  # Learn from peer evaluations
        eval_reward_mode="correctness",  # Test accuracy
        format_coef=0.1,
        grader="sympy",
    )

    # Train
    await train_rl(
        dataset_builder=dataset_builder,
        base_model="Qwen/Qwen3-8B-Instruct",
        n_batches=25,  # 25 batches × 4 problems = 100 training steps
        lora_rank=64,
        lr=5e-5,
        loss_fn="forward_backward",
        eval_every=5,  # Evaluate every 5 batches
        save_every=10,  # Save checkpoint every 10 batches
        output_dir="outputs/math_debate_training",
    )

    logger.info("Training completed!")


async def train_on_gsm8k_with_debate():
    """Train on GSM8K dataset using debate rewards."""
    logger.info("Starting training on GSM8K with debate rewards...")

    dataset_builder = GSM8KTrainingBuilder(
        batch_size=8,
        num_train_datapoints=200,
        num_test_datapoints=50,
        num_agents=3,
        max_rounds=2,  # Shorter rounds for easier problems
        model_name="Qwen/Qwen3-8B-Instruct",
        renderer_name="qwen3",
        train_reward_mode="debate",
        log_full_transcript=False,  # Disable to save space
    )

    await train_rl(
        dataset_builder=dataset_builder,
        base_model="Qwen/Qwen3-8B-Instruct",
        n_batches=25,
        lora_rank=64,
        lr=5e-5,
        loss_fn="forward_backward",
        output_dir="outputs/gsm8k_debate_training",
    )

    logger.info("Training completed!")


async def train_with_correctness_baseline():
    """Train using correctness rewards (baseline without debate)."""
    logger.info("Starting baseline training with correctness rewards...")

    dataset_builder = MathDebateDatasetBuilder(
        dataset_name="math",
        batch_size=4,
        num_train_datapoints=100,
        num_test_datapoints=20,
        num_agents=1,  # Single agent (no debate)
        max_rounds=1,
        model_name="Qwen/Qwen3-8B-Instruct",
        renderer_name="qwen3",
        train_reward_mode="correctness",  # Direct supervision
        eval_reward_mode="correctness",
    )

    await train_rl(
        dataset_builder=dataset_builder,
        base_model="Qwen/Qwen3-8B-Instruct",
        n_batches=25,
        lora_rank=64,
        lr=5e-5,
        loss_fn="forward_backward",
        output_dir="outputs/math_correctness_baseline",
    )

    logger.info("Baseline training completed!")


async def train_multi_round_debate():
    """Train with longer debates (5 rounds) for complex problems."""
    logger.info("Starting multi-round debate training...")

    dataset_builder = MathTrainingBuilder(
        batch_size=2,  # Smaller batches for longer context
        num_train_datapoints=50,
        num_test_datapoints=10,
        num_agents=4,  # More agents for diverse perspectives
        max_rounds=5,  # Longer debate: 5 × 4 = 20 turns
        history_rounds=3,  # Show more history
        summarize_history=True,  # Summarize old turns to manage context
        summarize_model="Qwen/Qwen3-4B-Instruct-2507",
        model_name="Qwen/Qwen3-8B-Instruct",
        renderer_name="qwen3",
        train_reward_mode="debate",
        log_full_transcript=True,
    )

    await train_rl(
        dataset_builder=dataset_builder,
        base_model="Qwen/Qwen3-8B-Instruct",
        n_batches=25,
        lora_rank=64,
        lr=5e-5,
        loss_fn="forward_backward",
        output_dir="outputs/math_multi_round_debate",
    )

    logger.info("Multi-round debate training completed!")


async def main():
    """Run training experiments."""
    # Choose which training to run
    # Uncomment the one you want to run:

    # 1. Standard debate training on MATH
    await train_on_math_with_debate()

    # 2. Debate training on GSM8K (easier problems)
    # await train_on_gsm8k_with_debate()

    # 3. Baseline training with correctness rewards (no debate)
    # await train_with_correctness_baseline()

    # 4. Advanced: Multi-round debate with summarization
    # await train_multi_round_debate()


if __name__ == "__main__":
    asyncio.run(main())
