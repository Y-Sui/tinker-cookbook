"""Training script for multi-agent debate RL."""

import asyncio
from datetime import datetime

import chz
from dotenv import load_dotenv

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.multi_agent_debate.env import MultiAgentDebateDatasetBuilder
from tinker_cookbook.rl import train

load_dotenv(override=True)


@chz.chz
class CLIConfig:
    """CLI configuration for multi-agent debate training."""

    model_name: str = "meta-llama/Llama-3.2-1B"
    renderer_name: str | None = None
    num_agents: int = 3
    batch_size: int = 64
    num_train_datapoints: int = 1024
    num_test_datapoints: int = 64
    learning_rate: float = 3e-5
    max_tokens: int = 512
    eval_every: int = 10
    save_every: int = 20
    hf_dataset_name: str = "lighteval/mmlu"
    hf_dataset_subset: str | None = "all"  # e.g., "all" or "abstract_algebra"
    hf_dataset_split: str = "test"  # HF split name (e.g., "train", "test", "validation")
    hf_dataset_question_field: str = "question"
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
    # Backward compat: if user passed a subject via hf_dataset_split, treat it as the config
    hf_subset = cli_config.hf_dataset_subset
    hf_split = cli_config.hf_dataset_split

    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{model_name}-debate-{cli_config.num_agents}agents-{cli_config.batch_size}batch-{cli_config.learning_rate}lr-{date_and_time}"

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/multi-agent-debate/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    dataset_builder = MultiAgentDebateDatasetBuilder(
        batch_size=cli_config.batch_size,
        num_train_datapoints=cli_config.num_train_datapoints,
        num_test_datapoints=cli_config.num_test_datapoints,
        num_agents=cli_config.num_agents,
        model_name=model_name,
        renderer_name=renderer_name,
        hf_dataset_name=cli_config.hf_dataset_name,
        hf_dataset_subset=hf_subset,
        hf_dataset_split=hf_split,
        hf_dataset_question_field=cli_config.hf_dataset_question_field,
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
        wandb_project=cli_config.wandb_project,
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
