"""
Agent logic for multi-agent meta-evaluation.

This module provides:
- MetaEvaluationAgent: Base class for agents that perform meta-evaluation
- API-specific subclasses: OpenRouterAgent, AnthropicAgent, OpenAIAgent
- AgentPool: Manages multiple agents with round-robin selection
"""

import logging
from typing import Any

from tinker_cookbook.preference.meta_evaluation.types import (
    AgentPersona,
    ComparisonHistory,
    MetaEvaluation,
)
from tinker_cookbook.preference.types import Comparison

logger = logging.getLogger(__name__)


class MetaEvaluationAgent:
    """Single agent with persona that performs meta-evaluation."""

    def __init__(
        self,
        persona: AgentPersona,
        api_client: Any,
        model_name: str,
    ):
        """
        Initialize the agent.

        Args:
            persona: Agent persona with system prompt
            api_client: API client (OpenAI/Anthropic/OpenRouter compatible)
            model_name: Model name to use for evaluation
        """
        self.persona = persona
        self.api_client = api_client
        self.model_name = model_name

    def _format_prompt(
        self,
        comparison: Comparison,
        history: ComparisonHistory,
    ) -> str:
        """Format comparison + history for LLM.

        Args:
            comparison: The comparison to evaluate
            history: Previous evaluations for this comparison

        Returns:
            Formatted prompt string
        """
        prompt = f"""{self.persona.system_prompt}

You are evaluating two completions (A and B) for the following conversation:

==== Conversation Prompt ====
{self._messages_to_text(comparison.prompt_conversation)}

==== Completion A ====
{self._messages_to_text(comparison.completion_A)}

==== Completion B ====
{self._messages_to_text(comparison.completion_B)}

{history.to_context_str()}

Based on the above, provide your meta-evaluation following this structure:
1. If there were previous evaluations, critique the most recent judge's reasoning
2. Analyze both solutions A and B
3. Decide which is better (A, B, or Tie)
4. Provide confidence (0.0 to 1.0)
5. Indicate if consensus has been reached
"""
        return prompt

    def _messages_to_text(self, messages: list) -> str:
        """Convert messages to text."""
        parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    async def evaluate(
        self,
        comparison: Comparison,
        history: ComparisonHistory,
        round_id: int,
    ) -> MetaEvaluation:
        """
        Generate meta-evaluation.

        Args:
            comparison: The comparison to evaluate
            history: Previous evaluations
            round_id: Current round number

        Returns:
            MetaEvaluation with judgment and meta-critique
        """
        prompt = self._format_prompt(comparison, history)

        # Call LLM with structured output (implementation depends on API)
        response = await self._get_structured_response(prompt)

        # Set metadata
        response.round_id = round_id
        response.agent_id = self.persona.agent_id

        return response

    async def _get_structured_response(self, prompt: str) -> MetaEvaluation:
        """Get structured response from API. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_structured_response")


class OpenRouterAgent(MetaEvaluationAgent):
    """Agent using OpenRouter API with structured generation."""

    async def _get_structured_response(self, prompt: str) -> MetaEvaluation:
        """Call OpenRouter API using beta.chat.completions.parse for structured output."""
        response = await self.api_client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format=MetaEvaluation,
            extra_body={
                "max_tokens": 1000,
            },
        )
        # The response is already parsed into MetaEvaluation
        return response.choices[0].message.parsed


class AnthropicAgent(MetaEvaluationAgent):
    """Agent using Anthropic's Claude API."""

    async def _get_structured_response(self, prompt: str) -> MetaEvaluation:
        """Call Anthropic API and parse JSON response."""
        import json

        # Add instruction for JSON output
        full_prompt = (
            prompt
            + "\n\nPlease respond with a valid JSON object with these fields:\n"
            + '- "thinking": your reasoning process\n'
            + '- "comparison_result": "A", "B", or "Tie"\n'
            + '- "confidence": a number between 0.0 and 1.0\n'
            + '- "critique_of_previous_judge": critique of previous judge (or null if first round)\n'
            + '- "critique_of_solutions": analysis of both solutions\n'
            + '- "consensus_reached": true or false\n'
        )

        message = await self.api_client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            temperature=0.0,
            messages=[{"role": "user", "content": full_prompt}],
        )
        response_text = message.content[0].text
        return self._parse_json_response(response_text)

    def _parse_json_response(self, response_text: str) -> MetaEvaluation:
        """Parse JSON response into MetaEvaluation."""
        import json

        response_text = response_text.strip()

        try:
            # Handle markdown code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            else:
                json_str = response_text

            data = json.loads(json_str)
            return MetaEvaluation.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
            # Fallback parsing
            return self._fallback_parse(response_text)

    def _fallback_parse(self, response_text: str) -> MetaEvaluation:
        """Fallback parsing when JSON parsing fails."""
        response_upper = response_text.upper()

        if "COMPLETION B" in response_upper or response_upper.strip().startswith("B"):
            result = "B"
        elif "COMPLETION A" in response_upper or response_upper.strip().startswith(
            "A"
        ):
            result = "A"
        elif "TIE" in response_upper:
            result = "Tie"
        else:
            logger.warning(
                f"Could not determine preference from: {response_text[:100]}"
            )
            result = "Tie"

        return MetaEvaluation(
            round_id=0,  # Will be set by caller
            agent_id="",  # Will be set by caller
            comparison_result=result,  # type: ignore
            thinking=f"Fallback parsing from unstructured response: {response_text[:200]}",
            confidence=0.5,
            critique_of_solutions="Fallback: Could not parse structured response",
            consensus_reached=False,
        )


class OpenAIAgent(MetaEvaluationAgent):
    """Agent using OpenAI's API with structured output parsing."""

    async def _get_structured_response(self, prompt: str) -> MetaEvaluation:
        """Call OpenAI API using beta.chat.completions.parse for structured output."""
        response = await self.api_client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format=MetaEvaluation,
            max_tokens=1000,
            temperature=0.0,
        )
        # The response is already parsed into MetaEvaluation
        return response.choices[0].message.parsed


class AgentPool:
    """Manages multiple agents with round-robin selection."""

    def __init__(self, agents: list[MetaEvaluationAgent]):
        """
        Initialize the agent pool.

        Args:
            agents: List of agents to rotate through
        """
        self.agents = agents
        self.num_agents = len(agents)
        self._round_counter = 0

    def get_agent_for_comparison(
        self, comparison_signature: str
    ) -> tuple[MetaEvaluationAgent, int]:
        """
        Get agent for this comparison using round-robin.

        Args:
            comparison_signature: Signature of the comparison (not currently used,
                                 but included for potential future routing logic)

        Returns:
            (agent, round_id): The selected agent and current round number
        """
        # Round-robin based on total evaluations performed
        self._round_counter += 1
        agent_idx = (self._round_counter - 1) % self.num_agents

        return self.agents[agent_idx], self._round_counter
