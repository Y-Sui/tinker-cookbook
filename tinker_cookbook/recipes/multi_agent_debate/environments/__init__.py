"""Debate environment implementations."""

from .debate import (
    MultiAgentDebateDataset,
    MultiAgentDebateDatasetBuilder,
    MultiAgentDebateEnv,
    MultiAgentEnvGroupBuilder,
)
from .verifiable import (
    DirectMathEvaluationEnv,
    VerifiableMathDebateDataset,
    VerifiableMathDebateDatasetBuilder,
    VerifiableMathProblem,
    VerifiableMultiAgentDebateEnv,
    VerifiableMultiAgentEnvGroupBuilder,
)

__all__ = [
    # Non-verifiable debate
    "MultiAgentDebateEnv",
    "MultiAgentEnvGroupBuilder",
    "MultiAgentDebateDataset",
    "MultiAgentDebateDatasetBuilder",
    # Verifiable (math) debate
    "VerifiableMultiAgentDebateEnv",
    "VerifiableMultiAgentEnvGroupBuilder",
    "VerifiableMathDebateDataset",
    "VerifiableMathDebateDatasetBuilder",
    "VerifiableMathProblem",
    "DirectMathEvaluationEnv",
]
