"""
Multi-agent preference model with meta-evaluation capabilities.

This module provides:
- MultiAgentPreferenceModel: Preference model using agent pool with meta-evaluation
- MultiAgentPreferenceModelBuilder: Builder for creating multi-agent preference models
"""

import logging
import os
from typing import Any

import chz

from tinker_cookbook.preference.meta_evaluation.agents import (
    AgentPool,
    AnthropicAgent,
    MetaEvaluationAgent,
    OpenAIAgent,
    OpenRouterAgent,
)
from tinker_cookbook.preference.meta_evaluation.history_store import (
    ComparisonHistoryStore,
)
from tinker_cookbook.preference.meta_evaluation.types import (
    AgentPersona,
    MetaEvaluation,
)
from tinker_cookbook.preference.types import (
    Comparison,
    PreferenceModel,
    PreferenceModelBuilder,
)

logger = logging.getLogger(__name__)


class MultiAgentPreferenceModel(PreferenceModel):
    """
    Preference model using multiple agents with meta-evaluation.

    Each comparison is evaluated by an agent from the pool (round-robin).
    The agent sees previous evaluations and provides meta-critique.
    """

    def __init__(
        self,
        agent_pool: AgentPool,
        history_store: ComparisonHistoryStore,
        max_rounds_per_comparison: int = 3,
        consensus_threshold: float = 0.9,
    ):
        """
        Initialize the multi-agent preference model.

        Args:
            agent_pool: Pool of agents for round-robin selection
            history_store: Store for comparison evaluation histories
            max_rounds_per_comparison: Maximum evaluations per comparison
            consensus_threshold: Confidence threshold for early consensus
        """
        self.agent_pool = agent_pool
        self.history_store = history_store
        self.max_rounds_per_comparison = max_rounds_per_comparison
        self.consensus_threshold = consensus_threshold

    async def __call__(self, comparison: Comparison) -> float:
        """
        Evaluate comparison using multi-agent meta-evaluation.

        Returns:
            1.0: B strongly preferred
            0.0: Tie
            -1.0: A strongly preferred
        """
        # Get history for this comparison
        history = self.history_store.get_history(comparison)

        # Check if we should do more rounds
        num_evaluations = len(history.evaluations)

        if num_evaluations >= self.max_rounds_per_comparison:
            # Use most recent evaluation
            return self._evaluation_to_score(history.evaluations[-1])

        # Check for early consensus
        if num_evaluations > 0:
            last_eval = history.evaluations[-1]
            if (
                last_eval.consensus_reached
                and last_eval.confidence >= self.consensus_threshold
            ):
                logger.info(
                    f"Consensus reached for comparison after {num_evaluations} rounds"
                )
                return self._evaluation_to_score(last_eval)

        # Get next agent and perform evaluation
        agent, round_id = self.agent_pool.get_agent_for_comparison(
            history.comparison_signature
        )

        try:
            evaluation = await agent.evaluate(
                comparison=comparison,
                history=history,
                round_id=round_id,
            )

            # Store evaluation
            self.history_store.add_evaluation(comparison, evaluation)

            # Log meta-evaluation details
            logger.info(
                f"Round {round_id} ({agent.persona.agent_id}): "
                f"{evaluation.comparison_result} "
                f"(confidence: {evaluation.confidence:.2f})"
            )
            if evaluation.critique_of_previous_judge:
                logger.debug(
                    f"Meta-critique: {evaluation.critique_of_previous_judge[:100]}..."
                )

            return self._evaluation_to_score(evaluation)

        except Exception as e:
            logger.warning(f"Error in meta-evaluation: {e}", exc_info=True)
            return 0.0  # Default to tie on error

    def _evaluation_to_score(self, evaluation: MetaEvaluation) -> float:
        """Convert meta-evaluation to preference score."""
        base_score = {
            "A": -1.0,
            "B": 1.0,
            "Tie": 0.0,
        }[evaluation.comparison_result]

        # Weight by confidence
        return base_score * evaluation.confidence


@chz.chz
class MultiAgentPreferenceModelBuilder(PreferenceModelBuilder):
    """
    Builder for multi-agent preference model.

    Args:
        model_name: LLM model to use (e.g., "anthropic/claude-sonnet-4.5")
        api_provider: "openrouter", "anthropic", or "openai"
        api_key: API key (defaults to env variable)
        num_agents: Number of agents in pool (default: 3)
        agent_personas: List of persona names (default: ["innovator", "critic", "synthesizer"])
        max_rounds_per_comparison: Max evaluations per comparison (default: 3)
        consensus_threshold: Confidence threshold for early stopping (default: 0.9)
        log_dir: Directory to store comparison histories (optional)
    """

    model_name: str = "anthropic/claude-sonnet-4.5"
    api_provider: str = "openrouter"
    api_key: str | None = None
    num_agents: int = 3
    agent_personas: list[str] = chz.field(
        default_factory=lambda: ["innovator", "critic", "synthesizer"]
    )
    max_rounds_per_comparison: int = 3
    consensus_threshold: float = 0.9
    log_dir: str | None = None

    def __call__(self) -> PreferenceModel:
        """Build and return a MultiAgentPreferenceModel."""
        # Get API key
        api_key = self._get_api_key()

        # Create API client
        api_client = self._create_api_client(api_key)

        # Create agent personas
        personas = self._create_personas()

        # Create agents
        agents = [self._create_agent(persona, api_client) for persona in personas]

        # Create agent pool
        agent_pool = AgentPool(agents)

        # Create history store
        history_store = ComparisonHistoryStore(log_dir=self.log_dir)

        return MultiAgentPreferenceModel(
            agent_pool=agent_pool,
            history_store=history_store,
            max_rounds_per_comparison=self.max_rounds_per_comparison,
            consensus_threshold=self.consensus_threshold,
        )

    def _get_api_key(self) -> str:
        """Get API key from config or environment."""
        api_key = self.api_key
        if api_key is None:
            env_var = f"{self.api_provider.upper()}_API_KEY"
            api_key = os.environ.get(env_var)

        if api_key is None:
            raise ValueError(
                f"API key not found. Set {env_var} or pass api_key parameter"
            )

        return api_key

    def _create_api_client(self, api_key: str) -> Any:
        """Create appropriate API client."""
        if self.api_provider == "openrouter":
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai"
                )

            return openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        elif self.api_provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install anthropic"
                )

            return anthropic.AsyncAnthropic(api_key=api_key)
        elif self.api_provider == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai"
                )

            return openai.AsyncOpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unknown API provider: {self.api_provider}")

    def _create_personas(self) -> list[AgentPersona]:
        """Create agent personas with system prompts."""
        persona_prompts = {
            "innovator": """You are an INNOVATOR agent focused on creativity and novel solutions.
When evaluating completions, prioritize:
- Novel approaches and creative thinking
- Practical applicability of ideas
- Clear communication of concepts

When critiquing previous judges, assess whether they considered innovative aspects.""",
            "critic": """You are a CRITIC agent focused on rigorous analysis and finding flaws.
When evaluating completions, prioritize:
- Logical consistency and correctness
- Potential edge cases or errors
- Robustness of reasoning

When critiquing previous judges, assess the thoroughness of their analysis.""",
            "synthesizer": """You are a SYNTHESIZER agent focused on integration and balance.
When evaluating completions, prioritize:
- Balance between different criteria
- Holistic quality of the response
- Alignment with user intent

When critiquing previous judges, assess whether they considered all relevant dimensions.""",
        }

        personas = []
        for i, persona_name in enumerate(self.agent_personas[: self.num_agents]):
            agent_id = f"agent_{chr(ord('a') + i)}"  # agent_a, agent_b, agent_c
            personas.append(
                AgentPersona(
                    agent_id=agent_id,
                    persona_name=persona_name,
                    system_prompt=persona_prompts.get(
                        persona_name,
                        f"You are a preference evaluation agent with persona: {persona_name}",
                    ),
                )
            )

        return personas

    def _create_agent(
        self,
        persona: AgentPersona,
        api_client: Any,
    ) -> MetaEvaluationAgent:
        """Create agent with specific API implementation."""
        if self.api_provider == "openrouter":
            return OpenRouterAgent(
                persona=persona,
                api_client=api_client,
                model_name=self.model_name,
            )
        elif self.api_provider == "anthropic":
            return AnthropicAgent(
                persona=persona,
                api_client=api_client,
                model_name=self.model_name,
            )
        elif self.api_provider == "openai":
            return OpenAIAgent(
                persona=persona,
                api_client=api_client,
                model_name=self.model_name,
            )
        else:
            raise ValueError(f"Unknown API provider: {self.api_provider}")
