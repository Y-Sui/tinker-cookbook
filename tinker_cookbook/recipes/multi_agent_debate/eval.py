"""Evaluation-only script for multi-agent debate (no training/TTL)."""

import asyncio
from typing import Literal

import chz
import tinker
from dotenv import load_dotenv

from tinker_cookbook.recipes.multi_agent_debate.evaluator import MultiAgentDebateEvaluator
from tinker_cookbook.recipes.multi_agent_debate.loaders import load_math_problems_from_jsonl

load_dotenv(override=True)


@chz.chz
class EvalConfig:
    """Configuration for evaluation-only run."""

    # ============================================================================
    # Model Configuration
    # ============================================================================
    checkpoint_path: str  # Required: path to checkpoint (e.g., "tinker://...")
    model_name: str = "Qwen/Qwen3-8B"  # Model name for renderer
    renderer_name: str = "qwen3_disable_thinking"
    max_tokens: int = 8196

    # ============================================================================
    # Debate Configuration
    # ============================================================================
    num_agents: int = 3  # Number of agents in debate
    max_rounds: int = 3  # Maximum debate rounds

    # ============================================================================
    # Dataset Configuration
    # ============================================================================
    dataset_path: str = "tinker_cookbook/data/aime2024_sample.jsonl"
    problem_field: str = "query"
    answer_field: str = "answer"
    max_questions: int = -1  # -1 = load all problems
    grader: Literal["sympy", "math_verify"] = "sympy"

    # ============================================================================
    # History Summarization (optional) - Uses OpenRouter API
    # ============================================================================
    summarize_history: bool = False
    summarize_model: str | None = "openai/gpt-4o-mini"

    # ============================================================================
    # Logging Configuration
    # ============================================================================
    log_full_transcript: bool = False  # Set to True for detailed logs (slower)
    num_groups_to_log: int = 0  # Set to >0 for transcript logging (slower)
    max_parallel_evals: int = 16  # Max concurrent evaluations (0=unlimited, 16-64=recommended)


async def main_async():
    """Run evaluation only (no training)."""
    config = chz.entrypoint(EvalConfig)

    # Load problems
    print(f"Loading problems from {config.dataset_path}...")
    problems = load_math_problems_from_jsonl(
        path=config.dataset_path,
        problem_field=config.problem_field,
        answer_field=config.answer_field,
        max_count=config.max_questions,
    )
    print(f"Loaded {len(problems)} problems")

    # Get renderer
    from tinker_cookbook.renderers import get_renderer

    renderer = get_renderer(config.renderer_name, config.model_name)
    print(f"Using model: {config.model_name}")
    print(f"Using renderer: {config.renderer_name}")

    # Create sampling client from checkpoint
    print(f"Loading checkpoint from {config.checkpoint_path}...")
    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_from_state_async(
        config.checkpoint_path
    )
    print("Checkpoint loaded successfully")

    # Create evaluator
    evaluator = MultiAgentDebateEvaluator(
        problems=problems,
        renderer=renderer,
        num_agents=config.num_agents,
        max_rounds=config.max_rounds,
        summarize_history=config.summarize_history,
        summarize_model=config.summarize_model,
        log_full_transcript=config.log_full_transcript,
        model_name=config.model_name,
        grader=config.grader,
        max_tokens=config.max_tokens,
        num_groups_to_log=config.num_groups_to_log,
        max_parallel_evals=config.max_parallel_evals,
    )

    # Run evaluation
    print("\n" + "=" * 80)
    print("Starting Evaluation")
    print("=" * 80)
    metrics = await evaluator(sampling_client)

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)

    # Direct mode results
    print("\nDirect Mode (Single-Turn):")
    for key, value in sorted(metrics.items()):
        if key.startswith("eval/direct/"):
            metric_name = key.replace("eval/direct/", "")
            print(f"  {metric_name:40s} = {value:.4f}")

    # Debate mode results
    print("\nDebate Mode (Multi-Turn):")
    for key, value in sorted(metrics.items()):
        if key.startswith("eval/debate/"):
            metric_name = key.replace("eval/debate/", "")
            print(f"  {metric_name:40s} = {value:.4f}")

    print("\n" + "=" * 80)


def main():
    """Entry point wrapper."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
