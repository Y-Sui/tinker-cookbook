import asyncio
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.multiplayer_rl.marketplace.env import MarketplaceDatasetBuilder
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str | None = None
    market_rule: str = "first_response"
    num_services: int = 2
    turn_budget: int = 6
    top_k: int = 2
    batch_size: int = 8
    num_markets_train: int = 64
    num_markets_test: int = 8
    learning_rate: float = 3e-5
    max_tokens: int = 96
    eval_every: int = 5
    save_every: int = 20
    wandb_project: str | None = None
    wandb_name: str | None = None
    log_path: str | None = None
    base_url: str | None = None
    service_base_model: str = "meta-llama/Llama-3.1-8B-Instruct"


def build_config(cli_config: CLIConfig) -> train.Config:
    model_name = cli_config.model_name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"{model_name}-market-{cli_config.market_rule}-{cli_config.batch_size}batch-"
        f"{cli_config.learning_rate}lr-{date_and_time}"
    )
    log_path = cli_config.log_path or f"/tmp/tinker-examples/marketplace/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    dataset_builder = MarketplaceDatasetBuilder(
        model_name=model_name,
        renderer_name=renderer_name,
        batch_size=cli_config.batch_size,
        num_markets_train=cli_config.num_markets_train,
        num_markets_test=cli_config.num_markets_test,
        num_services=cli_config.num_services,
        turn_budget=cli_config.turn_budget,
        market_rule=cli_config.market_rule,  # type: ignore[arg-type]
        top_k=cli_config.top_k,
        base_url=cli_config.base_url,
        service_base_model=cli_config.service_base_model,
    )

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        base_url=cli_config.base_url,
        save_every=cli_config.save_every,
    )


def main():
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
