"""Training script for multi-agent debate RL."""

import asyncio
import os
from datetime import datetime
from typing import Literal

import chz
from dotenv import load_dotenv

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.multi_agent_debate.env import MultiAgentDebateDatasetBuilder
from tinker_cookbook.recipes.multi_agent_debate.verifiable_env import (
    VerifiableMathDebateDatasetBuilder,
)
from tinker_cookbook.rl import train

load_dotenv(override=True)


@chz.chz
class CLIConfig:
    """CLI configuration for multi-agent debate training."""

    model_name: str = "Qwen/Qwen3-8B"
    env: str = "non-verifiable"  # Options: verifiable, non-verifiable
    renderer_name: str | None = None
    num_agents: int = 3
    max_rounds: int = 3
    batch_size: int = 16
    num_train_datapoints: int = 1024
    num_test_datapoints: int = 64
    learning_rate: float = 3e-5
    max_tokens: int = 8196
    eval_every: int = 10
    save_every: int = 20
    history_rounds: int = 2  # -1 = entire history
    summarize_history: bool = False
    summarize_model: str | None = "Qwen/Qwen3-4B-Instruct-2507"
    num_groups_to_log: int = 4  # 0 disables logtree; >=batch_size logs all groups
    log_full_transcript: bool = False  # include full per-group transcript in logtree
    # Question splitting for eval. Set test_question_frac=0 to disable the split.
    test_question_frac: float = 0.1

    # Non-verifiable prompt source (local JSONL by default; avoids network).
    non_verifiable_dataset_path: str = "tinker_cookbook/data/longwriter_6k_sample.jsonl"
    non_verifiable_dataset_field: str = "query"

    # Verifiable (math-style) prompt source (local JSONL by default; avoids network).
    verifiable_dataset_path: str = "tinker_cookbook/data/aime2024_sample.jsonl"
    verifiable_problem_field: str = "problem"
    verifiable_answer_field: str = "answer"
    verifiable_grader: Literal["sympy", "math_verify"] = "sympy"
    verifiable_eval_mode: Literal["direct", "debate", "both"] = "debate"

    max_questions: int = 1000
    wandb_project: str | None = None
    wandb_name: str | None = None
    log_path: str | None = None


def build_config(cli_config: CLIConfig) -> train.Config:
    """Build the training configuration from CLI config."""
    model_name = cli_config.model_name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"{model_name}-debate-{cli_config.num_agents}agents-"
        f"{cli_config.batch_size}groups-{cli_config.learning_rate}lr-{date_and_time}"
    )

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"~/tinker/multi-agent-debate/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    # Enable W&B when WANDB_PROJECT is set, even if the user doesn't pass wandb_project explicitly.
    wandb_project = cli_config.wandb_project or os.environ.get("WANDB_PROJECT")
    if wandb_name is None:
        wandb_name = os.environ.get("WANDB_NAME") or run_name

    if cli_config.env == "verifiable":
        dataset_builder = VerifiableMathDebateDatasetBuilder(
            batch_size=cli_config.batch_size,
            num_train_datapoints=cli_config.num_train_datapoints,
            num_test_datapoints=cli_config.num_test_datapoints,
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
            test_question_frac=cli_config.test_question_frac,
            grader=cli_config.verifiable_grader,
            eval_mode=cli_config.verifiable_eval_mode,
        )

    elif cli_config.env == "non-verifiable":
        dataset_builder = MultiAgentDebateDatasetBuilder(
            batch_size=cli_config.batch_size,
            num_train_datapoints=cli_config.num_train_datapoints,
            num_test_datapoints=cli_config.num_test_datapoints,
            num_agents=cli_config.num_agents,
            max_rounds=cli_config.max_rounds,
            history_rounds=cli_config.history_rounds,
            summarize_history=cli_config.summarize_history,
            summarize_model=cli_config.summarize_model,
            log_full_transcript=cli_config.log_full_transcript,
            model_name=model_name,
            renderer_name=renderer_name,
            dataset_path=cli_config.non_verifiable_dataset_path,
            dataset_field=cli_config.non_verifiable_dataset_field,
            test_question_frac=cli_config.test_question_frac,
            max_questions=cli_config.max_questions,
        )

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        num_groups_to_log=cli_config.num_groups_to_log,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )


def main():
    """Main entry point for training."""
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
