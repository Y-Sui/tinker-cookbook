"""
Multi-agent meta-evaluation preference models.

This package provides a preference model implementation where multiple agents
take turns evaluating comparisons, critiquing each other's judgments, and
tracking judgment evolution over time.

Key components:
- MultiAgentPreferenceModel: Preference model using agent pool with meta-evaluation
- MultiAgentPreferenceModelBuilder: Builder for creating multi-agent preference models
- MetaEvaluationAgent: Base class for agents that perform meta-evaluation
- AgentPool: Manages multiple agents with round-robin selection
- ComparisonHistoryStore: Stores per-comparison evaluation histories
"""

from tinker_cookbook.preference.meta_evaluation.agents import (
    AgentPool,
    MetaEvaluationAgent,
)
from tinker_cookbook.preference.meta_evaluation.history_store import (
    ComparisonHistoryStore,
)
from tinker_cookbook.preference.meta_evaluation.multi_agent_preference import (
    MultiAgentPreferenceModel,
    MultiAgentPreferenceModelBuilder,
)
from tinker_cookbook.preference.meta_evaluation.types import (
    AgentPersona,
    ComparisonHistory,
    MetaEvaluation,
)

__all__ = [
    "MetaEvaluationAgent",
    "AgentPool",
    "ComparisonHistoryStore",
    "MultiAgentPreferenceModel",
    "MultiAgentPreferenceModelBuilder",
    "AgentPersona",
    "ComparisonHistory",
    "MetaEvaluation",
]
