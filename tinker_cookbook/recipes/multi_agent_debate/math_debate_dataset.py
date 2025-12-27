"""Unified dataset builder for math multi-agent debate.

Supports both training and evaluation modes:

Training datasets:
- math: Hendrycks MATH (12k problems)
- gsm8k: GSM8K (7.5k problems)
- polaris: Polaris (53k problems)

Evaluation datasets:
- math_500: MATH-500 standard test set
- aime_2024: AIME 2024 problems
- aime_2025: AIME 2025 problems

Usage:
    # Training
    builder = MathDebateDatasetBuilder(
        dataset_name="math",
        batch_size=4,
        num_train_datapoints=100,
        num_test_datapoints=20,
        num_agents=3,
        max_rounds=3,
        model_name="Qwen/Qwen3-8B-Instruct",
        renderer_name="qwen3",
        reward_mode="debate",  # Use debate rewards for training
    )

    # Evaluation
    builder = MathDebateDatasetBuilder(
        dataset_name="aime_2024",
        batch_size=1,
        num_train_datapoints=0,
        num_test_datapoints=30,
        num_agents=3,
        max_rounds=3,
        model_name="Qwen/Qwen3-8B-Instruct",
        renderer_name="qwen3",
        reward_mode="correctness",  # Use correctness for evaluation
    )
"""

import random
from typing import Literal, Sequence

import chz
import tinker

from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .math_debate_datasets import MathProblem, load_math_dataset
from .math_debate_env import MathDebateEnvGroupBuilder
from .math_debate_prompts import MATH_DEBATE_SYSTEM_PROMPT


def _get_debate_stop_condition_for_dataset(renderer_name: str):
    """Get stop condition for debate environment."""
    from .math_debate_env import _get_debate_stop_condition
    tokenizer = get_tokenizer(renderer_name)
    renderer = get_renderer(renderer_name, tokenizer)
    return _get_debate_stop_condition(renderer)


class MathDebateDataset(RLDataset):
    """RL dataset for math multi-agent debate."""

    def __init__(
        self,
        batch_size: int,
        problems: list[MathProblem],
        num_agents: int,
        renderer_name: str,
        model_name: str,
        self_play: bool,
        history_turns: int,
        summarize_history: bool,
        summarize_model: str | None,
        log_full_transcript: bool,
        max_rounds: int,
        num_datapoints: int,
        reward_mode: Literal["debate", "correctness"],
        opponent_policies: list[TinkerMessageCompleter] | None = None,
        format_coef: float = 0.1,
        grader: Literal["sympy", "math_verify"] = "sympy",
        grade_timeout: float = 1.0,
    ):
        self.batch_size = batch_size
        self.problems = problems
        self.num_agents = num_agents
        self.renderer_name = renderer_name
        self.model_name = model_name
        self.self_play = self_play
        self.history_turns = history_turns
        self.summarize_history = summarize_history
        self.summarize_model = summarize_model
        self.log_full_transcript = log_full_transcript
        self.max_rounds = max_rounds
        self.num_datapoints = num_datapoints
        self.reward_mode = reward_mode
        self.opponent_policies = opponent_policies
        self.format_coef = format_coef
        self.grader = grader
        self.grade_timeout = grade_timeout

        # Cache renderer
        tokenizer = get_tokenizer(self.model_name)
        self._renderer = get_renderer(self.renderer_name, tokenizer)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, self.num_datapoints)
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            MathDebateEnvGroupBuilder(
                problems=self.problems,
                problem_index=problem_index,
                renderer=self._renderer,
                num_agents=self.num_agents,
                self_play=self.self_play,
                history_turns=self.history_turns,
                summarize_history=self.summarize_history,
                summarize_model=self.summarize_model,
                log_full_transcript=self.log_full_transcript,
                max_rounds=self.max_rounds,
                model_name=self.model_name,
                opponent_policies_all=self.opponent_policies,
                reward_mode=self.reward_mode,
                format_coef=self.format_coef,
                grader=self.grader,
                grade_timeout=self.grade_timeout,
            )
            for problem_index in range(batch_start, batch_end)
        ]

    def __len__(self) -> int:
        return (self.num_datapoints + self.batch_size - 1) // self.batch_size


@chz.chz
class MathDebateDatasetBuilder(RLDatasetBuilder):
    """Builder for math multi-agent debate datasets.

    Configuration:
        dataset_name: One of: math, gsm8k, polaris (training) or math_500, aime_2024, aime_2025 (eval)
        batch_size: Number of problem groups per batch
        num_train_datapoints: Number of training problems to use
        num_test_datapoints: Number of evaluation problems to use
        num_agents: Number of agents in debate (default: 3)
        max_rounds: Number of debate rounds (default: 3)
        history_rounds: Number of recent rounds to show in context (default: 2)
        summarize_history: Whether to summarize old history (default: False)
        summarize_model: Model for summarization (default: Qwen3-4B)
        log_full_transcript: Whether to log full debate transcripts (default: False)
        model_name: Base model for agents
        renderer_name: Renderer to use (should match model family)
        opponent_model_name: Model for opponent policies in eval (default: same as model_name)
        train_reward_mode: Reward mode for training ("debate" or "correctness")
        eval_reward_mode: Reward mode for evaluation ("debate" or "correctness")
        format_coef: Penalty coefficient for missing \\boxed{} format (default: 0.1)
        grader: Grading method ("sympy" or "math_verify", default: "sympy")
        grade_timeout: Timeout for grading in seconds (default: 1.0)
        max_problems: Maximum problems to load from dataset (default: None = all)
        shuffle_seed: Seed for shuffling problems (default: 42)
    """

    # Dataset configuration
    dataset_name: str = "math"
    batch_size: int = 4
    num_train_datapoints: int = 100
    num_test_datapoints: int = 20

    # Debate configuration
    num_agents: int = 3
    max_rounds: int = 3
    history_rounds: int = 2
    summarize_history: bool = False
    summarize_model: str | None = "Qwen/Qwen3-4B-Instruct-2507"
    log_full_transcript: bool = False

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B-Instruct"
    renderer_name: str = "qwen3"
    opponent_model_name: str | None = None

    # Reward configuration
    train_reward_mode: Literal["debate", "correctness"] = "debate"
    eval_reward_mode: Literal["debate", "correctness"] = "correctness"
    format_coef: float = 0.1
    grader: Literal["sympy", "math_verify"] = "sympy"
    grade_timeout: float = 1.0

    # Data configuration
    max_problems: int | None = None
    shuffle_seed: int = 42

    def _load_and_split_problems(self) -> tuple[list[MathProblem], list[MathProblem]]:
        """Load problems and split into train/test."""
        all_problems = load_math_dataset(
            self.dataset_name,
            max_problems=self.max_problems,
            shuffle_seed=self.shuffle_seed,
        )

        if not all_problems:
            raise ValueError(f"No problems loaded from dataset '{self.dataset_name}'")

        # For evaluation-only datasets, use all for test
        eval_only_datasets = ["math_500", "aime_2024", "aime_2025"]
        if self.dataset_name in eval_only_datasets:
            return [], all_problems

        # For training datasets, split
        if self.num_test_datapoints <= 0:
            return all_problems, []

        # Use a fixed split
        rng = random.Random(self.shuffle_seed)
        shuffled = list(all_problems)
        rng.shuffle(shuffled)

        # Take last N for test to avoid overlap
        test_size = min(self.num_test_datapoints, len(shuffled) // 10)  # At most 10% for test
        test_size = max(1, test_size)

        return shuffled[test_size:], shuffled[:test_size]

    def _construct_opponent_policies(self) -> list[TinkerMessageCompleter]:
        """Construct fixed opponent policies for evaluation."""
        from .math_debate_env import _get_debate_stop_condition

        tokenizer = get_tokenizer(self.model_name)
        renderer = get_renderer(self.renderer_name, tokenizer)

        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(
            base_model=self.opponent_model_name or self.model_name
        )
        return [
            TinkerMessageCompleter(
                sampling_client=sampling_client,
                renderer=renderer,
                max_tokens=2048,
                stop_condition=_get_debate_stop_condition(renderer),
            )
            for _ in range(self.num_agents)
        ]

    async def __call__(
        self,
    ) -> tuple[MathDebateDataset | None, MathDebateDataset | None]:
        """Build train and test datasets.

        Returns:
            (train_dataset, test_dataset)
            Either can be None if num_datapoints is 0
        """
        train_problems, test_problems = self._load_and_split_problems()

        # Build training dataset
        train_dataset: MathDebateDataset | None = None
        if self.num_train_datapoints > 0 and train_problems:
            train_dataset = MathDebateDataset(
                batch_size=self.batch_size,
                problems=train_problems,
                num_agents=self.num_agents,
                renderer_name=self.renderer_name,
                model_name=self.model_name,
                self_play=True,
                history_turns=self.history_rounds,
                summarize_history=self.summarize_history,
                summarize_model=self.summarize_model,
                log_full_transcript=self.log_full_transcript,
                max_rounds=self.max_rounds,
                num_datapoints=self.num_train_datapoints,
                reward_mode=self.train_reward_mode,
                opponent_policies=None,
                format_coef=self.format_coef,
                grader=self.grader,
                grade_timeout=self.grade_timeout,
            )

        # Build test/evaluation dataset
        test_dataset: MathDebateDataset | None = None
        if self.num_test_datapoints > 0 and test_problems:
            opponent_policies_all = self._construct_opponent_policies()
            test_dataset = MathDebateDataset(
                batch_size=min(self.num_test_datapoints, self.batch_size),
                problems=test_problems,
                num_agents=self.num_agents,
                renderer_name=self.renderer_name,
                model_name=self.model_name,
                self_play=False,
                history_turns=self.history_rounds,
                summarize_history=self.summarize_history,
                summarize_model=self.summarize_model,
                log_full_transcript=self.log_full_transcript,
                max_rounds=self.max_rounds,
                num_datapoints=self.num_test_datapoints,
                reward_mode=self.eval_reward_mode,
                opponent_policies=opponent_policies_all,
                format_coef=self.format_coef,
                grader=self.grader,
                grade_timeout=self.grade_timeout,
            )

        return train_dataset, test_dataset


# Convenience builders for specific datasets
@chz.chz
class MathTrainingBuilder(MathDebateDatasetBuilder):
    """Pre-configured for MATH training."""
    dataset_name: str = "math"
    train_reward_mode: Literal["debate", "correctness"] = "debate"
    eval_reward_mode: Literal["debate", "correctness"] = "correctness"


@chz.chz
class GSM8KTrainingBuilder(MathDebateDatasetBuilder):
    """Pre-configured for GSM8K training."""
    dataset_name: str = "gsm8k"
    train_reward_mode: Literal["debate", "correctness"] = "debate"
    eval_reward_mode: Literal["debate", "correctness"] = "correctness"


@chz.chz
class PolarisTrainingBuilder(MathDebateDatasetBuilder):
    """Pre-configured for Polaris training."""
    dataset_name: str = "polaris"
    train_reward_mode: Literal["debate", "correctness"] = "debate"
    eval_reward_mode: Literal["debate", "correctness"] = "correctness"


@chz.chz
class MATH500EvalBuilder(MathDebateDatasetBuilder):
    """Pre-configured for MATH-500 evaluation."""
    dataset_name: str = "math_500"
    num_train_datapoints: int = 0
    num_test_datapoints: int = 500
    eval_reward_mode: Literal["debate", "correctness"] = "correctness"


@chz.chz
class AIME2024EvalBuilder(MathDebateDatasetBuilder):
    """Pre-configured for AIME 2024 evaluation."""
    dataset_name: str = "aime_2024"
    num_train_datapoints: int = 0
    num_test_datapoints: int = 30
    eval_reward_mode: Literal["debate", "correctness"] = "correctness"


@chz.chz
class AIME2025EvalBuilder(MathDebateDatasetBuilder):
    """Pre-configured for AIME 2025 evaluation."""
    dataset_name: str = "aime_2025"
    num_train_datapoints: int = 0
    num_test_datapoints: int = 30
    eval_reward_mode: Literal["debate", "correctness"] = "correctness"
