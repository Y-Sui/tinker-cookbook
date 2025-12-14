import asyncio
import logging
import os

import chz

from tinker_cookbook import model_info
from tinker_cookbook.preference.comparison_policy_evaluator import ComparisonEvaluator
from tinker_cookbook.preference.prompted_preference import PromptedPreferenceModelBuilder
from tinker_cookbook.recipes.preference.datasets import HHHComparisonBuilder
from tinker_cookbook.rl import preference_envs, train

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    base_model: str = "meta-llama/Llama-3.2-1B"
    short_name: str = "llama3b"
    policy_checkpoint_path: str | None = None  # Path to initial policy checkpoint
    wandb_project: str | None = None
    wandb_name: str | None = "rl-prompted-rm"
    lora_rank: int = 64
    batch_size: int = 256

    # Model to use for prompted rewards
    preference_model_name: str = "x-ai/grok-4.1-fast"
    # API provider: "openrouter", "anthropic", or "openai"
    preference_api_provider: str = "openrouter"
    # Custom prompt for preference model (optional)
    preference_prompt: str | None = None

    rl_learning_rate: float = 1e-5
    rl_max_tokens: int = 1024
    rl_group_size: int = 4

    save_every: int = 100
    eval_every: int = 20

    # Logtree configuration - number of groups to log per iteration (0 = disable)
    num_groups_to_log: int = 4


async def train_rl(
    log_path: str,
    policy_checkpoint_path: str | None,
    base_model: str,
    preference_model_name: str,
    preference_api_provider: str,
    preference_api_key: str | None,
    preference_prompt: str | None,
    wandb_project: str | None,
    wandb_name: str | None,
    lora_rank: int,
    group_size: int,
    batch_size: int,
    learning_rate: float,
    max_tokens: int,
    save_every: int,
    eval_every: int,
    num_groups_to_log: int = 4,
):
    """Train policy using RL with prompts from Anthropic HHH data and a prompted reward model."""
    # Use HHH comparison builder for prompts
    comparison_builder = HHHComparisonBuilder()
    renderer_name = model_info.get_recommended_renderer_name(base_model)

    # Create prompted preference model builder
    preference_model_builder = PromptedPreferenceModelBuilder(
        model_name=preference_model_name,
        api_provider=preference_api_provider,
        api_key=preference_api_key,
        preference_prompt=preference_prompt,
    )

    rl_dataset_builder = preference_envs.PairwisePreferenceRLDatasetBuilder(
        comparison_builder=comparison_builder,
        policy_renderer_name=renderer_name,
        policy_model_name=base_model,
        preference_model_builder=preference_model_builder,
        batch_size=batch_size,
        group_size=group_size,
        tournament_pattern=preference_envs.TournamentPattern.ALL_PAIRS_BOTH_WAYS,
    )

    def get_evaluator_builder() -> ComparisonEvaluator:
        comparison_builder_eval = HHHComparisonBuilder(test_size=256)
        _, test_dataset = comparison_builder_eval.get_train_and_test_datasets()
        assert test_dataset is not None
        test_labeled_comparisons = [
            comparison_builder_eval.example_to_labeled_comparison(example)  # type: ignore
            for example in test_dataset
        ]
        test_comparisons = [lc.comparison for lc in test_labeled_comparisons if lc is not None]
        return ComparisonEvaluator(
            preference_model_builder=preference_model_builder,
            comparisons=test_comparisons,
            renderer_name=renderer_name,
            model_name_for_tokenizer=base_model,
        )

    config = train.Config(
        model_name=base_model,
        dataset_builder=rl_dataset_builder,
        load_checkpoint_path=policy_checkpoint_path,
        learning_rate=learning_rate,
        max_tokens=max_tokens,
        log_path=log_path,
        evaluator_builders=[get_evaluator_builder],
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        lora_rank=lora_rank,
        save_every=save_every,
        eval_every=eval_every,
        num_groups_to_log=num_groups_to_log,
    )
    await train.main(config)


def cli_main(cli_config: CLIConfig):
    log_path_root = os.path.expanduser(f"~/experiments/rl-prompted-rm-{cli_config.short_name}")
    rl_log_path = os.path.join(log_path_root, "rl")

    asyncio.run(
        train_rl(
            rl_log_path,
            cli_config.policy_checkpoint_path,
            cli_config.base_model,
            cli_config.preference_model_name,
            cli_config.preference_api_provider,
            cli_config.preference_api_key,
            cli_config.preference_prompt,
            cli_config.wandb_project,
            cli_config.wandb_name,
            cli_config.lora_rank,
            cli_config.rl_group_size,
            cli_config.batch_size,
            cli_config.rl_learning_rate,
            cli_config.rl_max_tokens,
            cli_config.save_every,
            cli_config.eval_every,
            cli_config.num_groups_to_log,
        )
    )


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_main(cli_config)
