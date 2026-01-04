"""
Implementations that correspond to a model or policy that can be sampled from, but with different amounts of additional structure.

The TokenCompleter operates on tokens. This is the version used by RL algorithms, because RL algorithms work on Tokens. The MessageCompleter operates on messages, so it needs to be used with a renderer.

Evals and other code should use the appropriate interface.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import tinker

from tinker_cookbook import renderers

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# Interfaces

StopCondition: TypeAlias = list[str] | list[int]


@dataclass
class TokensWithLogprobs:
    tokens: list[int]
    maybe_logprobs: list[float] | None

    @property
    def logprobs(self) -> list[float]:
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs


class TokenCompleter:
    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        raise NotImplementedError


class MessageCompleter:
    # TODO maybe add n_samples to the interfaces?
    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        raise NotImplementedError


# Implementations


@dataclass
class TinkerTokenCompleter(TokenCompleter):
    """
    The most standard TokenCompleter, which uses a tinker.SamplingClient to sample actions.
    """

    sampling_client: tinker.SamplingClient
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        """Sample an action from the policy given an observation."""
        # Sample from the model
        sample_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            ),
        )

        # Extract tokens and logprobs from the first (and only) sample
        sampled_tokens = sample_result.sequences[0].tokens
        sampled_logprobs = sample_result.sequences[0].logprobs
        assert sampled_logprobs is not None

        return TokensWithLogprobs(tokens=sampled_tokens, maybe_logprobs=sampled_logprobs)


class TinkerMessageCompleter(MessageCompleter):
    """A completer that uses the actual model to generate responses."""

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        max_tokens: int,
        stop_condition: StopCondition | None = None,
    ):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.max_tokens = max_tokens
        if stop_condition is None:
            self.stop_condition = self.renderer.get_stop_sequences()
        else:
            self.stop_condition = stop_condition

    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        # Render the conversation for the model
        model_input = self.renderer.build_generation_prompt(messages)

        # Sample from the model
        response = await self.sampling_client.sample_async(
            model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=1.0,
                max_tokens=self.max_tokens,
                stop=self.stop_condition,
            ),
        )

        # Decode the response
        parsed_message, _success = self.renderer.parse_response(response.sequences[0].tokens)

        return {"role": "assistant", "content": parsed_message["content"]}


class OpenRouterMessageCompleter(MessageCompleter):
    """A completer that uses OpenRouter API to generate responses.

    This completer is useful for tasks like debate history summarization where using
    external APIs (e.g., gpt-4o-mini) can be more cost-effective than running
    inference on Tinker-hosted models.

    Requires OPENROUTER_API_KEY environment variable to be set.

    Example:
        >>> import os
        >>> api_key = os.getenv("OPENROUTER_API_KEY")
        >>> completer = OpenRouterMessageCompleter(
        ...     api_key=api_key,
        ...     model_name="openai/gpt-4o-mini",
        ...     max_tokens=1024,
        ... )
        >>> response = await completer([{"role": "user", "content": "Summarize this..."}])
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "openai/gpt-4o-mini",
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ):
        """Initialize OpenRouter message completer.

        Args:
            api_key: OpenRouter API key (get from https://openrouter.ai/keys)
            model_name: Model to use (format: "provider/model-name")
                Examples: "openai/gpt-4o-mini", "openai/gpt-4o", "anthropic/claude-3.5-sonnet"
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
        """
        from openai import AsyncOpenAI

        self.client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key, base_url="https://openrouter.ai/api/v1"
        )
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        """Generate a response using OpenRouter API.

        Args:
            messages: List of messages in the conversation (compatible with OpenAI format)

        Returns:
            Assistant message with generated content
        """
        # OpenRouter API is compatible with OpenAI's message format
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return {"role": "assistant", "content": response.choices[0].message.content or ""}
