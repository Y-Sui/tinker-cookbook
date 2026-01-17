"""Multi-agent debate for reinforcement learning.

This recipe implements multi-agent debate with pairwise comparison rewards.
Supports both verifiable (math) and non-verifiable (open-ended) tasks.

Quick Start:
    Training (verifiable math):
        python -m tinker_cookbook.recipes.multi_agent_debate.train \\
            env="verifiable" \\
            dataset_path="tinker_cookbook/data/aime2024_sample.jsonl" \\
            num_agents=3 \\
            max_rounds=3 \\
            batch_size=16

    Evaluation only:
        python -m tinker_cookbook.recipes.multi_agent_debate.eval \\
            checkpoint_path="tinker://..." \\
            dataset_path="tinker_cookbook/data/aime2024_sample.jsonl" \\
            num_agents=3

Architecture:
    See doc/algorithm_flowchart.md and doc/math_formulation.md for detailed
    documentation of the debate protocol and reward system.

Reward System V2:
    - Generator rewards (<solution> and <evaluation> tokens):
      Soft vote ratio providing dense gradient signal in range [-1, +1]

    - Judge rewards (<comparison> tokens):
      Consensus alignment rewarding accurate peer evaluation

Key Features:
    - Parallel generation: All agents generate simultaneously within each round
    - Agent personas: 5 distinct reasoning styles for diversity
    - Online test-time learning: Train and evaluate on the same dataset
    - Sequence extension: O(T) compute scaling via KV-cache preservation
"""

# Core components
from .coordinator import MultiAgentCoordinator
from .prompts import (
    AGENT_PERSONAS,
    AGENT_SYSTEM_PROMPT,
    SUMMARIZER_SYSTEM_PROMPT,
    VERIFIABLE_AGENT_SYSTEM_PROMPT,
    ParsedResponse,
    format_persona_intro,
    get_agent_persona,
    get_agent_temperature,
    parse_agent_response,
)

# Environments - Non-verifiable
from .env import (
    MultiAgentDebateDataset,
    MultiAgentDebateDatasetBuilder,
    MultiAgentDebateEnv,
    MultiAgentEnvGroupBuilder,
)

# Environments - Verifiable (Math)
from .verifiable_env import (
    DirectMathEvaluationEnv,
    VerifiableMathDebateDataset,
    VerifiableMathDebateDatasetBuilder,
    VerifiableMathProblem,
    VerifiableMultiAgentDebateEnv,
    VerifiableMultiAgentEnvGroupBuilder,
)

# Evaluation
from .evaluator import MultiAgentDebateEvaluator

# Data loading
from .loaders import load_math_problems_from_jsonl, load_questions_from_jsonl

# Utilities
from .utils import (
    get_debate_stop_condition,
    get_step_idx_before_turn,
    log_debate_evaluation_final_solutions,
    log_debate_transcript,
    log_direct_evaluation,
)

__all__ = [
    # Core coordination
    "MultiAgentCoordinator",
    # Prompts and parsing
    "AGENT_PERSONAS",
    "AGENT_SYSTEM_PROMPT",
    "SUMMARIZER_SYSTEM_PROMPT",
    "VERIFIABLE_AGENT_SYSTEM_PROMPT",
    "ParsedResponse",
    "parse_agent_response",
    "format_persona_intro",
    "get_agent_persona",
    "get_agent_temperature",
    # Non-verifiable debate environments
    "MultiAgentDebateEnv",
    "MultiAgentEnvGroupBuilder",
    "MultiAgentDebateDataset",
    "MultiAgentDebateDatasetBuilder",
    # Verifiable (math) debate environments
    "VerifiableMultiAgentDebateEnv",
    "VerifiableMultiAgentEnvGroupBuilder",
    "VerifiableMathDebateDataset",
    "VerifiableMathDebateDatasetBuilder",
    "VerifiableMathProblem",
    "DirectMathEvaluationEnv",
    # Evaluation
    "MultiAgentDebateEvaluator",
    # Data loading
    "load_questions_from_jsonl",
    "load_math_problems_from_jsonl",
    # Utilities
    "get_step_idx_before_turn",
    "get_debate_stop_condition",
    "log_debate_transcript",
    "log_direct_evaluation",
    "log_debate_evaluation_final_solutions",
]
