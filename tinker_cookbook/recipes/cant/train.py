"""
Training script for CANT (Critique and Revision) framework.

Supports both verifiable and non-verifiable tasks with four-round protocol:
1. Proposal: Generate initial solutions
2. Critique: Provide blind rankings and targeted critiques
3. Revision: Revise solutions
4. Final Verdict: Rank revised solutions
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Literal

import chz
from dotenv import load_dotenv

from tinker_cookbook import model_info
from tinker_cookbook.recipes.cant import loader
from tinker_cookbook.recipes.cant.env import CANTEnvGroupBuilder
from tinker_cookbook.recipes.cant.verifiable_env import VerifiableCANTEnvGroupBuilder
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

load_dotenv(override=True)


@chz.chz
class CANTConfig:
    """Configuration for CANT training."""

    # ============================================================================
    # Model Configuration
    # ============================================================================
    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str | None = "qwen3_disable_thinking"
    max_tokens: int = 8196

    # ============================================================================
    # Environment Configuration
    # ============================================================================
    env_type: Literal["verifiable", "non-verifiable"] = "verifiable"
    num_agents: int = 4  # Number of agents in CANT protocol
    # ============================================================================
    # Training Configuration
    # ============================================================================
    batch_size: int = 16  # Problems per training batch
    num_train_datapoints: int = 1024  # Training samples per epoch
    epoch: int = 1  # Number of epochs
    learning_rate: float = 3e-5
    use_cosine_lr_schedule: bool = False
    eval_every: int = 0  # 0 = disable evaluation
    num_test_datapoints: int = -1  # <0 = use all test datapoints
    save_every: int = 20  # 0 = disable checkpointing

    # ============================================================================
    # CANT Reward Hyperparameters
    # ============================================================================
    beta_disc: float = 2.0  # Persuasion reward scaling factor
    weight_disc: float = 2.0  # Weight for persuasion component
    weight_sol: float = 1.0  # Weight for solution quality component
    weight_meta: float = 1.0  # Weight for consensus component
    # ============================================================================
    # Dataset Configuration
    # ============================================================================
    train_datasets: str | None = "aime2024"  # Comma-separated dataset names
    test_datasets: str | None = None  # Comma-separated dataset names (defaults to train)
    max_questions: int = -1  # Max problems to load (-1 = all)
    max_test_questions: int = -1  # Max test problems to load (-1 = all)

    # ============================================================================
    # Logging Configuration
    # ============================================================================
    num_groups_to_log: int = 4  # Groups to log per batch
    log_path: str | None = None  # Custom log path (auto-generated if None)

    # ============================================================================
    # Weights & Biases Configuration
    # ============================================================================
    wandb_project: str | None = None
    wandb_name: str | None = None


class CANTDataset(RLDataset):
    """Dataset wrapper that instantiates CANT env groups per problem."""

    def __init__(
        self,
        batch_size: int,
        problem_states: list[dict],
        num_datapoints: int,
        env_group_builder_cls: type[CANTEnvGroupBuilder],
        env_group_builder_kwargs: dict[str, object],
    ):
        self.batch_size = batch_size
        self.problem_states = problem_states
        self.num_datapoints = num_datapoints
        self.env_group_builder_cls = env_group_builder_cls
        self.env_group_builder_kwargs = env_group_builder_kwargs

    def get_batch(self, index: int) -> list[CANTEnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, self.num_datapoints)
        assert batch_start < batch_end, "Incorrect batch size"

        return [
            self.env_group_builder_cls(
                problem_state=self.problem_states[i % len(self.problem_states)],
                **self.env_group_builder_kwargs,
            )
            for i in range(batch_start, batch_end)
        ]

    def __len__(self) -> int:
        return (self.num_datapoints + self.batch_size - 1) // self.batch_size


@chz.chz
class CANTDatasetBuilder(RLDatasetBuilder):
    """Builds a CANT dataset from problem states and env config."""

    batch_size: int
    num_train_datapoints: int
    epoch: int
    problem_states: list[dict]
    num_test_datapoints: int = -1
    test_problem_states: list[dict] | None = None
    env_group_builder_cls: type[CANTEnvGroupBuilder]
    model_name: str
    renderer_name: str
    num_agents: int
    beta_disc: float
    weight_disc: float
    weight_sol: float
    weight_meta: float
    weight_accept: float

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        if not self.problem_states:
            raise ValueError("No problem states available for CANT dataset")

        tokenizer = get_tokenizer(self.model_name)
        renderer = get_renderer(self.renderer_name, tokenizer)
        env_group_builder_kwargs: dict[str, object] = {
            "num_agents": self.num_agents,
            "renderer": renderer,
            "model_name": self.model_name,
            "beta_disc": self.beta_disc,
            "weight_disc": self.weight_disc,
            "weight_sol": self.weight_sol,
            "weight_meta": self.weight_meta,
            "weight_accept": self.weight_accept,
        }
        if self.num_train_datapoints < 0 or self.num_train_datapoints > len(self.problem_states):
            num_datapoints = len(self.problem_states) * self.epoch
        else:
            num_datapoints = self.num_train_datapoints * self.epoch

        train_dataset = CANTDataset(
            batch_size=self.batch_size,
            problem_states=self.problem_states,
            num_datapoints=num_datapoints,
            env_group_builder_cls=self.env_group_builder_cls,
            env_group_builder_kwargs=env_group_builder_kwargs,
        )
        if self.num_test_datapoints == 0:
            return train_dataset, None

        test_states = self.test_problem_states or self.problem_states
        if self.num_test_datapoints < 0:
            test_size = len(test_states)
        else:
            test_size = min(self.num_test_datapoints, len(test_states))
        test_batch_size = min(self.batch_size, test_size)
        test_dataset = CANTDataset(
            batch_size=test_batch_size,
            problem_states=test_states[:test_size],
            num_datapoints=test_size,
            env_group_builder_cls=self.env_group_builder_cls,
            env_group_builder_kwargs=env_group_builder_kwargs,
        )
        return train_dataset, test_dataset


def build_dataset_builder(
    config: CANTConfig, model_name: str, renderer_name: str
) -> RLDatasetBuilder:
    """
    Build appropriate dataset builder based on environment type.

    Args:
        config: Training configuration
        model_name: Model identifier
        renderer_name: Renderer identifier

    Returns:
        RLDatasetBuilder instance
    """
    train_datasets = loader.parse_dataset_list(config.train_datasets)
    if not train_datasets:
        raise ValueError("train_datasets must be set to at least one dataset name")
    test_datasets = loader.parse_dataset_list(config.test_datasets) or train_datasets

    if config.env_type == "verifiable":
        problem_states = loader.load_verifiable_dataset_states(
            train_datasets,
            split="train",
            max_count=config.max_questions,
        )
        test_problem_states = loader.load_verifiable_dataset_states(
            test_datasets,
            split="test",
            max_count=config.max_test_questions,
        )

        env_group_builder_cls = VerifiableCANTEnvGroupBuilder

    else:  # non-verifiable
        problem_states = []
        for name in train_datasets:
            problem_states.extend(
                loader.load_non_verifiable_dataset_states(name, config.max_questions)
            )
        test_problem_states = []
        for name in test_datasets:
            test_problem_states.extend(
                loader.load_non_verifiable_dataset_states(name, config.max_test_questions)
            )

        env_group_builder_cls = CANTEnvGroupBuilder

    dataset_builder = CANTDatasetBuilder(
        batch_size=config.batch_size,
        num_train_datapoints=config.num_train_datapoints,
        epoch=config.epoch,
        problem_states=problem_states,
        num_test_datapoints=config.num_test_datapoints,
        test_problem_states=test_problem_states,
        env_group_builder_cls=env_group_builder_cls,
        model_name=model_name,
        renderer_name=renderer_name,
        num_agents=config.num_agents,
        beta_disc=config.beta_disc,
        weight_disc=config.weight_disc,
        weight_sol=config.weight_sol,
        weight_meta=config.weight_meta,
        weight_accept=config.weight_accept,
    )

    return dataset_builder


def get_run_name(config: CANTConfig, model_name: str) -> str:
    """Generate descriptive run name for logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1]
    env_short = "v" if config.env_type == "verifiable" else "nv"
    return f"cant_{env_short}_{model_short}_a{config.num_agents}_{timestamp}"


async def main():
    """Main training entry point."""
    config = chz.entrypoint(CANTConfig)

    # Resolve model/renderer names
    model_name = config.model_name
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(model_name)

    # Build dataset
    dataset_builder = build_dataset_builder(config, model_name, renderer_name)

    # Generate run name
    run_name = config.wandb_name or get_run_name(config, model_name)

    # Determine log path
    if config.log_path:
        log_path = Path(config.log_path)
        log_path.mkdir(parents=True, exist_ok=True)
        log_path = str(log_path)
    else:
        log_path = f"~/tinker/multi-agent-debate/{run_name}"

    print("Starting CANT training:")
    print(f"  Model: {model_name}")
    print(f"  Renderer: {renderer_name}")
    print(f"  Environment: {config.env_type}")
    print(f"  Num agents: {config.num_agents}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Training datapoints: {config.num_train_datapoints}")
    print(f"  Epochs: {config.epoch}")
    print(f"  Learning rate: {config.learning_rate}")
    train_datasets = loader.parse_dataset_list(config.train_datasets)
    test_datasets = loader.parse_dataset_list(config.test_datasets) or train_datasets
    print(f"  Train datasets: {', '.join(train_datasets)}")
    print(f"  Test datasets: {', '.join(test_datasets)}")
    print(
        f"  Reward weights: disc={config.weight_disc}, sol={config.weight_sol}, "
        f"meta={config.weight_meta}, accept={config.weight_accept}"
    )
    print(f"  Beta (persuasion): {config.beta_disc}")
    print(f"  Log path: {log_path}")
    print()

    # Build training config
    train_config = train.Config(
        model_name=model_name,
        dataset_builder=dataset_builder,
        learning_rate=config.learning_rate,
        use_cosine_lr_schedule=config.use_cosine_lr_schedule,
        max_tokens=config.max_tokens,
        use_stepwise_advantages=True,  # CANT uses stepwise advantages
        num_groups_to_log=config.num_groups_to_log,
        wandb_project=config.wandb_project,
        wandb_name=run_name,
        log_path=log_path,
        eval_every=config.eval_every,
        save_every=config.save_every,
    )

    # Start training
    await train.main(train_config)


if __name__ == "__main__":
    asyncio.run(main())
