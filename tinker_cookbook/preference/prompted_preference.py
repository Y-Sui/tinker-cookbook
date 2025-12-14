"""
Prompted preference model that uses external API models (like Claude, GPT) for preference evaluation.

This module provides a preference model implementation that calls external LLM APIs
to evaluate which completion is preferred in a comparison.
"""

import json
import logging
import os
from typing import Any, Literal

import chz
from pydantic import BaseModel, Field

from tinker_cookbook.preference.types import (
    Comparison,
    PreferenceModel,
    PreferenceModelBuilder,
)
from tinker_cookbook.renderers import Message

logger = logging.getLogger(__name__)


class PreferenceEvaluation(BaseModel):
    """Structured output for preference evaluation."""

    thinking: str = Field(description="Explanation and reasoning for the preference decision")
    comparison_result: Literal["A", "B", "Tie"] = Field(
        description="Which completion is better: A, B, or Tie"
    )


class PromptedPreferenceModel(PreferenceModel):
    """
    Preference model that uses an external API to judge preferences.

    This implementation calls an external LLM (like Claude or GPT) with a prompt
    asking it to compare two completions and return which one is preferred.
    """

    def __init__(
        self,
        api_client: Any,
        model_name: str,
        preference_prompt: str | None = None,
    ):
        self.api_client = api_client
        self.model_name = model_name
        self.preference_prompt = preference_prompt or self._default_preference_prompt()

    def _default_preference_prompt(self) -> str:
        return """You are a helpful, harmless, and honest AI assistant evaluating conversation responses.

Given a conversation prompt and two different completions (A and B), determine which completion is better.

Consider these criteria:
- Helpfulness: Does it answer the user's question or request?
- Harmlessness: Is it safe and avoids harmful content?
- Honesty: Is it truthful and doesn't hallucinate?

Analyze both completions carefully, provide your detailed reasoning, and indicate which completion is better."""

    def _format_comparison_for_api(self, comparison: Comparison) -> str:
        """Format a comparison into a text prompt for the API."""
        prompt_text = self._messages_to_text(comparison.prompt_conversation)
        completion_a_text = self._messages_to_text(comparison.completion_A)
        completion_b_text = self._messages_to_text(comparison.completion_B)

        return f"""{self.preference_prompt}

==== Conversation Prompt ====
{prompt_text}

==== Completion A ====
{completion_a_text}

==== Completion B ====
{completion_b_text}"""

    def _messages_to_text(self, messages: list[Message]) -> str:
        """Convert messages to readable text format."""
        parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    async def __call__(self, comparison: Comparison) -> float:
        """
        Evaluate preference using external API.

        Returns:
            1.0: B is strongly preferred
            0.0: Tie
            -1.0: A is strongly preferred
        """
        prompt = self._format_comparison_for_api(comparison)

        try:
            evaluation = await self._get_evaluation(prompt)

            # Log the thinking process for analysis
            logger.info(f"Preference: {evaluation.comparison_result}")
            logger.debug(f"Reasoning: {evaluation.thinking[:200]}...")

            # Convert to reward score
            return self._evaluation_to_score(evaluation)
        except Exception as e:
            logger.warning(f"Error calling preference API: {e}", exc_info=True)
            return 0.0  # Default to tie on error

    async def _get_evaluation(self, prompt: str) -> PreferenceEvaluation:
        """
        Get structured evaluation from API. Override in subclasses.
        """
        raise NotImplementedError("Subclasses must implement _get_evaluation")

    def _evaluation_to_score(self, evaluation: PreferenceEvaluation) -> float:
        """Convert evaluation to numeric score."""
        if evaluation.comparison_result == "A":
            return -1.0
        elif evaluation.comparison_result == "B":
            return 1.0
        else:  # Tie
            return 0.0


class OpenRouterPromptedPreferenceModel(PromptedPreferenceModel):
    """Preference model using OpenRouter API with structured generation."""

    async def _get_evaluation(self, prompt: str) -> PreferenceEvaluation:
        """Call OpenRouter API using beta.chat.completions.parse for structured output."""
        response = await self.api_client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format=PreferenceEvaluation,
            extra_body={
                "max_tokens": 500,
            },
        )
        # The response is already parsed into PreferenceEvaluation
        return response.choices[0].message.parsed


class AnthropicPromptedPreferenceModel(PromptedPreferenceModel):
    """Preference model using Anthropic's Claude API."""

    async def _get_evaluation(self, prompt: str) -> PreferenceEvaluation:
        """Call Anthropic API and parse JSON response."""
        message = await self.api_client.messages.create(
            model=self.model_name,
            max_tokens=500,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text
        return self._parse_json_response(response_text)

    def _parse_json_response(self, response_text: str) -> PreferenceEvaluation:
        """Parse JSON response into PreferenceEvaluation."""
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
            return PreferenceEvaluation.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
            # Fallback parsing
            return self._fallback_parse(response_text)

    def _fallback_parse(self, response_text: str) -> PreferenceEvaluation:
        """Fallback parsing when JSON parsing fails."""
        response_upper = response_text.upper()

        if "COMPLETION B" in response_upper or response_upper.strip().startswith("B"):
            result = "B"
        elif "COMPLETION A" in response_upper or response_upper.strip().startswith("A"):
            result = "A"
        elif "TIE" in response_upper:
            result = "Tie"
        else:
            logger.warning(f"Could not determine preference from: {response_text[:100]}")
            result = "Tie"

        return PreferenceEvaluation(
            comparison_result=result,  # type: ignore
            thinking=f"Fallback parsing from unstructured response: {response_text[:200]}",
        )


class OpenAIPromptedPreferenceModel(PromptedPreferenceModel):
    """Preference model using OpenAI's API with structured output parsing."""

    async def _get_evaluation(self, prompt: str) -> PreferenceEvaluation:
        """Call OpenAI API using beta.chat.completions.parse for structured output."""
        response = await self.api_client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format=PreferenceEvaluation,
            max_tokens=500,
            temperature=0.0,
        )
        # The response is already parsed into PreferenceEvaluation
        return response.choices[0].message.parsed


@chz.chz
class PromptedPreferenceModelBuilder(PreferenceModelBuilder):
    """
    Builder for prompted preference models.

    Args:
        model_name: Name of the model (e.g., "anthropic/claude-sonnet-4.5", "openai/gpt-4")
        api_provider: Which API to use ("openrouter", "anthropic", or "openai")
        api_key: API key for the service (defaults to environment variable)
        preference_prompt: Custom prompt for preference evaluation (optional)
    """

    model_name: str = "anthropic/claude-sonnet-4.5"
    api_provider: str = "openrouter"
    api_key: str | None = None
    preference_prompt: str | None = None

    def __call__(self) -> PreferenceModel:
        # Get API key from config or environment
        api_key = self.api_key
        if api_key is None:
            if self.api_provider == "openrouter":
                api_key = os.environ.get("OPENROUTER_API_KEY")
            elif self.api_provider == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif self.api_provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")

        if api_key is None:
            raise ValueError(
                f"API key not found. Set {self.api_provider.upper()}_API_KEY environment variable "
                f"or pass api_key parameter."
            )

        # Create appropriate API client and model
        if self.api_provider == "openrouter":
            try:
                import openai
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")

            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
            return OpenRouterPromptedPreferenceModel(
                api_client=client,
                model_name=self.model_name,
                preference_prompt=self.preference_prompt,
            )

        elif self.api_provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install anthropic"
                )

            client = anthropic.AsyncAnthropic(api_key=api_key)
            return AnthropicPromptedPreferenceModel(
                api_client=client,
                model_name=self.model_name,
                preference_prompt=self.preference_prompt,
            )

        elif self.api_provider == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")

            client = openai.AsyncOpenAI(api_key=api_key)
            return OpenAIPromptedPreferenceModel(
                api_client=client,
                model_name=self.model_name,
                preference_prompt=self.preference_prompt,
            )

        else:
            raise ValueError(
                f"Unknown API provider: {self.api_provider}. "
                f"Must be 'openrouter', 'anthropic', or 'openai'"
            )
