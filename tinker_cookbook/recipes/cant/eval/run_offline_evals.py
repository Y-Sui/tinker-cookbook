"""
Offline evaluation script for CANT checkpoints.

Supports both CANT protocol and baseline evaluation modes.

Usage (CANT protocol):
    python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
        model_path=tinker://your-model \
        task_names=inspect_evals/gsm8k,inspect_evals/math \
        use_cant_protocol=true \
        num_agents=4

Usage (Baseline comparison):
    python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
        model_path=tinker://your-model \
        task_names=inspect_evals/gsm8k,inspect_evals/math \
        use_cant_protocol=false

Configurable Parameters:
    CANT-specific:
        task_names           Comma-separated Inspect AI tasks
        use_cant_protocol    true/false for CANT vs baseline
        num_agents           Number of agents (default: 4)
        model_path           Checkpoint path (optional)
    
    Sampling parameters (inherited from InspectEvaluatorBuilder):
        temperature          Sampling temperature (default: 1.0)
        max_tokens           Max tokens per generation (default: 1000)
        top_p                Nucleus sampling (default: 1.0)
        top_k                Top-k sampling (default: -1)
        limit                Max samples per task (default: None = all)
        seed                 Random seed (default: None)
        num_choices          Number of completions (default: 1)
    
    Other:
        model_name           Model name (required)
        renderer_name        Renderer name (required)
        log_dir              Log directory (default: ~/inspect-logs)
        max_connections      Max concurrent requests (default: 512)
        verbose              Verbose logging (default: False)

Example with custom parameters:
    python -m tinker_cookbook.recipes.cant.eval.run_offline_evals \
        model_name=Qwen/Qwen3-8B \
        renderer_name=qwen3 \
        task_names=inspect_evals/gsm8k \
        use_cant_protocol=true \
        num_agents=4 \
        temperature=0.7 \
        max_tokens=4096 \
        top_p=0.95 \
        limit=100 \
        seed=42
"""

import asyncio
import logging

import chz
import tinker
from dotenv import load_dotenv

from tinker_cookbook.recipes.cant.eval.evaluators import CANTEvaluatorBuilder

load_dotenv(override=True)

logger = logging.getLogger(__name__)


@chz.chz
class Config(CANTEvaluatorBuilder):
    """Configuration for offline CANT evaluation."""

    model_path: str | None = None


async def main(config: Config):
    """Run offline evaluation on a CANT checkpoint."""

    logging.basicConfig(level=logging.INFO)

    # Create service client
    service_client = tinker.ServiceClient()

    # Resolve model_name from model_path if needed
    if config.model_path and not config.model_name:
        rest_client = service_client.create_rest_client()
        training_run = await rest_client.get_training_run_by_tinker_path_async(config.model_path)
        config = chz.replace(config, model_name=training_run.base_model)
        logger.info(f"Resolved base model: {config.model_name}")

    # Create sampling client
    logger.info(f"Creating sampling client for: {config.model_path or config.model_name}")
    sampling_client = service_client.create_sampling_client(
        model_path=config.model_path, base_model=config.model_name
    )

    # Build evaluator
    mode = "CANT Protocol" if config.use_cant_protocol else "Baseline"
    logger.info(f"Running evaluation in {mode} mode")
    logger.info(f"Tasks: {', '.join(config.task_names)}")
    if config.use_cant_protocol:
        logger.info(f"Number of agents: {config.num_agents}")

    # Build evaluator (exclude model_path which is not part of CANTEvaluatorBuilder)
    config_dict = chz.asdict(config)
    config_dict.pop("model_path", None)  # Remove model_path before passing to builder
    evaluator_builder = CANTEvaluatorBuilder(**config_dict)
    evaluator = evaluator_builder()

    # Run evaluation
    metrics = await evaluator(sampling_client)

    # Print results
    print("\n" + "=" * 80)
    print(f"CANT Evaluation Results ({mode})")
    print("=" * 80)
    for metric_name, metric_value in sorted(metrics.items()):
        print(f"  {metric_name}: {metric_value:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
