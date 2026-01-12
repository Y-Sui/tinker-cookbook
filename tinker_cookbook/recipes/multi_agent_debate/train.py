"""Training script for multi-agent debate RL with online test-time learning (TTL).

This script supports two debate environments:
1. Verifiable (math): Multi-agent debate on math problems with ground-truth answers
2. Non-verifiable: Multi-agent debate on open-ended questions

For verifiable env, it implements online TTL by:
- Training on sampled batches from the dataset
- Evaluating on ALL problems periodically (dual-mode: direct + debate)
- Tracking rich per-dataset metrics and pass@k
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import chz
from dotenv import load_dotenv

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.multi_agent_debate.env import MultiAgentDebateDatasetBuilder
from tinker_cookbook.recipes.multi_agent_debate.evaluator import MultiAgentDebateEvaluator
from tinker_cookbook.recipes.multi_agent_debate.verifiable_env import (
    VerifiableMathDebateDatasetBuilder,
)
from tinker_cookbook.rl import train

load_dotenv(override=True)


@chz.chz
class CLIConfig:
    """CLI configuration for multi-agent debate training with online TTL."""

    # ============================================================================
    # Model Configuration
    # ============================================================================
    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str | None = "qwen3_disable_thinking"
    max_tokens: int = 8196

    # ============================================================================
    # Environment Configuration
    # ============================================================================
    env: str = "verifiable"  # Options: verifiable, non-verifiable
    num_agents: int = 3  # Number of agents in debate
    max_rounds: int = 3  # Maximum debate rounds
    history_rounds: int = 2  # Recent turns in context (-1 = entire history)

    # ============================================================================
    # Training Configuration
    # ============================================================================
    batch_size: int = 16  # Problems per training batch
    num_train_datapoints: int = 1024  # Training samples per epoch
    epoch: int = 1  # Number of times to cycle through the dataset
    learning_rate: float = 3e-5
    use_cosine_lr_schedule: bool = False  # Use cosine LR decay (base_lr → 0)
    eval_every: int = 50  # Evaluate every N batches (larger for TTL)
    save_every: int = 100  # Save checkpoint every N batches
    max_parallel_evals: int = 64  # Max concurrent evaluations (0=unlimited)

    # ============================================================================
    # History Summarization (optional) - Uses OpenRouter API
    # ============================================================================
    summarize_history: bool = False  # Summarize old debate history
    summarize_model: str | None = "openai/gpt-4o-mini"  # OpenRouter model for summarization

    # ============================================================================
    # Logging Configuration
    # ============================================================================
    num_groups_to_log: int = 4  # Groups to log per batch (0 = disable)
    log_full_transcript: bool = False  # Include full debate transcripts
    eval_num_groups_to_log: int = 2  # Groups to log during evaluation
    log_path: str | None = None  # Custom log path (auto-generated if None)

    # ============================================================================
    # Dataset Configuration: Non-Verifiable (for env="non-verifiable")
    # ============================================================================
    non_verifiable_dataset_path: str = "tinker_cookbook/data/longwriter_6k_sample.jsonl"
    non_verifiable_problem_field: str = "query"

    # ============================================================================
    # Dataset Configuration: Verifiable Math (for env="verifiable")
    # ============================================================================
    verifiable_dataset_path: str = "tinker_cookbook/data/aime2024_sample.jsonl"
    verifiable_problem_field: str = "problem"
    verifiable_answer_field: str = "answer"
    verifiable_grader: Literal["sympy", "math_verify"] = "sympy"
    max_questions: int = -1  # Max problems to load (-1 = all)

    # ============================================================================
    # Reward System Configuration
    # ============================================================================
    enable_format_penalty: bool = True  # Penalize missing/invalid comparisons

    # ============================================================================
    # Weights & Biases Configuration
    # ============================================================================
    wandb_project: str | None = None  # W&B project (or use WANDB_PROJECT env var)
    wandb_name: str | None = None  # W&B run name (auto-generated if None)


def _build_verifiable_dataset_builder(
    cli_config: CLIConfig, model_name: str, renderer_name: str
) -> VerifiableMathDebateDatasetBuilder:
    """Build dataset builder for verifiable (math) environment."""
    return VerifiableMathDebateDatasetBuilder(
        batch_size=cli_config.batch_size,
        num_train_datapoints=cli_config.num_train_datapoints,
        epoch=cli_config.epoch,
        num_agents=cli_config.num_agents,
        max_rounds=cli_config.max_rounds,
        history_rounds=cli_config.history_rounds,
        summarize_history=cli_config.summarize_history,
        summarize_model=cli_config.summarize_model,
        log_full_transcript=cli_config.log_full_transcript,
        model_name=model_name,
        renderer_name=renderer_name,
        dataset_path=cli_config.verifiable_dataset_path,
        problem_field=cli_config.verifiable_problem_field,
        answer_field=cli_config.verifiable_answer_field,
        grader=cli_config.verifiable_grader,
        max_questions=cli_config.max_questions,
        enable_format_penalty=cli_config.enable_format_penalty,
    )


def _build_non_verifiable_dataset_builder(
    cli_config: CLIConfig, model_name: str, renderer_name: str
) -> MultiAgentDebateDatasetBuilder:
    """Build dataset builder for non-verifiable environment."""
    return MultiAgentDebateDatasetBuilder(
        batch_size=cli_config.batch_size,
        num_train_datapoints=cli_config.num_train_datapoints,
        epoch=cli_config.epoch,
        num_test_datapoints=64,  # Keep test dataset for non-verifiable
        num_agents=cli_config.num_agents,
        max_rounds=cli_config.max_rounds,
        history_rounds=cli_config.history_rounds,
        summarize_history=cli_config.summarize_history,
        summarize_model=cli_config.summarize_model,
        log_full_transcript=cli_config.log_full_transcript,
        model_name=model_name,
        renderer_name=renderer_name,
        dataset_path=cli_config.non_verifiable_dataset_path,
        problem_field=cli_config.non_verifiable_problem_field,
        max_questions=cli_config.max_questions,
        enable_format_penalty=cli_config.enable_format_penalty,
    )


def build_config(cli_config: CLIConfig) -> train.Config:
    """Build the training configuration from CLI config.

    Args:
        cli_config: CLI configuration with all hyperparameters

    Returns:
        Training configuration ready for rl.train.main()
    """
    # Model setup
    model_name = cli_config.model_name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(model_name)
    dataset_name = Path(cli_config.verifiable_dataset_path).stem.split("_sample")[0]

    # Generate run name and paths
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{renderer_name}-{dataset_name}-{cli_config.batch_size}groups-{cli_config.epoch}epochs-{cli_config.max_tokens}tokens-{timestamp}"

    # W&B configuration (support env vars)
    wandb_project = cli_config.wandb_project or os.environ.get("WANDB_PROJECT")
    wandb_name = cli_config.wandb_name or os.environ.get("WANDB_NAME") or run_name
    log_path = cli_config.log_path or f"~/tinker/multi-agent-debate/{wandb_name}"

    # Build dataset based on environment type
    if cli_config.env == "verifiable":
        dataset_builder = _build_verifiable_dataset_builder(cli_config, model_name, renderer_name)
    elif cli_config.env == "non-verifiable":
        dataset_builder = _build_non_verifiable_dataset_builder(
            cli_config, model_name, renderer_name
        )
    else:
        raise ValueError(f"Invalid env: {cli_config.env}. Must be 'verifiable' or 'non-verifiable'")

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        use_cosine_lr_schedule=cli_config.use_cosine_lr_schedule,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        num_groups_to_log=cli_config.num_groups_to_log,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        use_stepwise_advantages=True,
    )


def _create_verifiable_evaluator(cli_config: CLIConfig, train_dataset) -> MultiAgentDebateEvaluator:
    """Create dual-mode evaluator for verifiable environment.

    This evaluator runs both direct (single-turn) and debate (multi-turn) evaluation
    on ALL problems in the dataset every eval_every steps.
    """
    return MultiAgentDebateEvaluator(
        problems=train_dataset.problems,
        renderer=train_dataset.renderer,
        num_agents=cli_config.num_agents,
        max_rounds=cli_config.max_rounds,
        history_rounds=cli_config.history_rounds,
        summarize_history=cli_config.summarize_history,
        summarize_model=cli_config.summarize_model,
        log_full_transcript=cli_config.log_full_transcript,
        model_name=cli_config.model_name,
        grader=cli_config.verifiable_grader,
        max_tokens=cli_config.max_tokens,
        num_groups_to_log=cli_config.eval_num_groups_to_log,
        max_parallel_evals=cli_config.max_parallel_evals,
    )


async def main_async():
    """Main entry point for training with custom evaluator.

    For verifiable env:

    - Uses online TTL: same dataset for training and evaluation
    - Creates dual-mode evaluator (direct + debate) for comprehensive metrics

    For non-verifiable env:
    - Standard RL with separate train/test datasets
    """
    # Parse CLI configuration
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)

    # For verifiable env, add custom dual-mode evaluator for online TTL
    if cli_config.env == "verifiable":
        train_dataset, _ = await config.dataset_builder()
        evaluator = _create_verifiable_evaluator(cli_config, train_dataset)
        config = chz.replace(config, evaluator_builders=[lambda: evaluator])

    # Check log directory and start training
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    await train.main(config)


def main():
    """Entry point wrapper."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
