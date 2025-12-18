"""Training script for multi-agent conversation preference RL.

Usage:
    python -m tinker_cookbook.recipes.conversation_preference_rl.train \
        model_name=meta-llama/Llama-3.2-1B \
        num_agents=3 \
        max_rounds=5 \
        batch_size=12 \
        train_prompts_path=/path/to/prompts.jsonl \
        log_path=/tmp/conv-pref-rl/run1
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import chz
from dotenv import load_dotenv

from tinker_cookbook import cli_utils, hyperparam_utils, model_info
from tinker_cookbook.recipes.conversation_preference_rl.env import (
    MultiAgentConversationDatasetBuilder,
)
from tinker_cookbook.rl import train

load_dotenv(override=True)


@chz.chz
class CLIConfig:
    """CLI configuration for multi-agent conversation preference RL training."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.2-1B"
    renderer_name: str | None = None  # Auto-detected if None

    # Multi-agent conversation params
    num_agents: int = 3
    max_rounds: int = 5

    # Training params
    batch_size: int = 128  # Must be divisible by num_agents
    learning_rate: float | None = None  # Auto-computed if None
    max_tokens: int = 512

    # Dataset sources (pick one or provide both)
    train_prompts: list[str] = chz.field(default_factory=list)  # Direct list
    train_prompts_path: str | None = None  # Path to JSONL file with {"query": "..."}
    test_prompts_path: str | None = None

    # HuggingFace dataset loading (alternative to prompts_path)
    hf_dataset_name: str | None = None
    hf_dataset_name_config: str | None = None
    hf_prompt_column: str = "query"
    hf_train_split: str = "train"
    hf_test_split: str = "test"
    max_train_samples: int | None = None
    max_test_samples: int = 100

    # Training schedule
    num_substeps: int = 1
    max_training_datapoints: int = 131072  # Default training length

    # Logging
    eval_every: int = 5
    save_every: int = 20
    wandb_project: str | None = None
    wandb_name: str | None = None
    log_path: str | None = None
    behavior_if_exists: str = "ask"  # or "delete", "resume", "raise"


def build_config(cli_config: CLIConfig) -> train.Config:
    """Build training config from CLI config."""
    model_name = cli_config.model_name
    renderer_name = cli_config.renderer_name
    if renderer_name is None:
        renderer_name = model_info.get_recommended_renderer_name(model_name)

    # Auto-compute learning rate if not specified
    learning_rate = cli_config.learning_rate
    if learning_rate is None:
        learning_rate = hyperparam_utils.get_lr(model_name)

    # Generate run name
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_short_name = model_name.split("/")[-1]
    run_name = (
        f"{model_short_name}-"
        f"multiagent-{cli_config.num_agents}agents-"
        f"{cli_config.max_rounds}rounds-"
        f"batch{cli_config.batch_size}-"
        f"{date_and_time}"
    )

    log_path = cli_config.log_path or f"~/experiments/conversation-preference-rl/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    # Load prompts from files if specified
    train_prompts = cli_config.train_prompts.copy()
    test_prompts: list[str] = []

    if cli_config.train_prompts_path:
        train_prompts_path = Path(cli_config.train_prompts_path)
        if not train_prompts_path.exists():
            raise FileNotFoundError(f"train_prompts_path not found: {train_prompts_path}")

        with open(train_prompts_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Support both {"query": "..."} and {"prompt": "..."}
                    query = data.get("query") or data.get("prompt") or data.get("text")
                    if query:
                        train_prompts.append(query)

    if cli_config.test_prompts_path:
        test_prompts_path = Path(cli_config.test_prompts_path)
        if test_prompts_path.exists():
            with open(test_prompts_path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        query = data.get("query") or data.get("prompt") or data.get("text")
                        if query:
                            test_prompts.append(query)

    # Validate that we have prompts from some source
    if not train_prompts and not cli_config.hf_dataset_name:
        raise ValueError(
            "Must provide either train_prompts, train_prompts_path, or hf_dataset_name"
        )

    # Build dataset
    dataset_builder = MultiAgentConversationDatasetBuilder(
        batch_size=cli_config.batch_size,
        num_agents=cli_config.num_agents,
        max_rounds=cli_config.max_rounds,
        model_name=model_name,
        renderer_name=renderer_name,
        max_tokens=cli_config.max_tokens,
        train_prompts=train_prompts,
        test_prompts=test_prompts,
        hf_dataset_name=cli_config.hf_dataset_name,
        hf_dataset_name_config=cli_config.hf_dataset_name_config,
        hf_prompt_column=cli_config.hf_prompt_column,
        hf_train_split=cli_config.hf_train_split,
        hf_test_split=cli_config.hf_test_split,
        max_train_samples=cli_config.max_train_samples,
        max_test_samples=cli_config.max_test_samples,
    )

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=learning_rate,
        max_tokens=cli_config.max_tokens,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
    )


def main():
    """Main entry point for training."""
    cli_config = chz.entrypoint(CLIConfig)

    # Validate batch_size divisible by num_agents
    if cli_config.batch_size % cli_config.num_agents != 0:
        raise ValueError(
            f"batch_size ({cli_config.batch_size}) must be divisible by "
            f"num_agents ({cli_config.num_agents})"
        )

    config = build_config(cli_config)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists=cli_config.behavior_if_exists)

    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
