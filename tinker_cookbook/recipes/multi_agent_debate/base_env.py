"""Base classes for multi-agent debate environments.

This module provides abstract base classes that eliminate code duplication between
verifiable and non-verifiable debate implementations.

Classes:
    BaseMultiAgentDebateEnv: Base environment with shared step/observation logic
    BaseMultiAgentEnvGroupBuilder: Base builder with shared reward computation

Design Pattern:
    Template Method - base classes provide shared logic, subclasses override
    specific methods like get_system_prompt() and get_observation_string().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from tinker import types

from tinker_cookbook.completers import MessageCompleter, StopCondition
from tinker_cookbook.renderers import Message, Renderer, get_text_content
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Observation,
    StepResult,
    Trajectory,
)

from .coordinator import MultiAgentCoordinator
from .prompts import SUMMARIZER_SYSTEM_PROMPT
from .utils import STOP_CONDITION, get_step_idx_before_turn

# Reward system constants
FORMAT_PENALTY = -0.5  # Penalty for missing/invalid comparisons
FORMAT_EXEMPT_TURNS = 2  # Turns 0-1 exempt from format checking


@dataclass
class BaseMultiAgentDebateEnv(Env, ABC):
    """Base environment for multi-agent debate with shared logic.

    This abstract base class provides common functionality for multi-agent debate
    environments, including turn coordination, observation building, and step execution.

    Subclasses must implement:
        - stop_condition: Return stop sequences for generation
        - get_system_prompt(): Return the system prompt for this agent
        - get_observation_string(): Return the observation string for the current turn

    Attributes:
        agent_id: Integer identifier for this agent (0-indexed)
        coordinator: Shared coordinator managing turn-taking
        renderer: Renderer for converting messages to/from token sequences
        self_play: Whether all agents use the same policy (must be True)
        history_turns: Number of recent turns to include in context
        summarize_history: Whether to summarize older conversation history
        _summarizer_policy: Optional pre-created summarizer completer (shared across envs)
        model_name: Name of the base model
    """

    agent_id: int
    coordinator: MultiAgentCoordinator
    renderer: Renderer
    self_play: bool = True
    history_turns: int = 2
    summarize_history: bool = False
    _summarizer_policy: MessageCompleter | None = None
    model_name: str | None = None

    def __post_init__(self) -> None:
        """Initialize the environment.

        Note: The summarizer policy should be passed in via _summarizer_policy
        rather than created here to avoid creating multiple API clients.

        Subclasses should call super().__post_init__() if they override this.
        """
        assert self.self_play, "Only self-play mode is supported"

        # Validate that if summarization is enabled, a summarizer was provided
        if self.summarize_history and self._summarizer_policy is None:
            raise ValueError(
                "summarize_history=True but no _summarizer_policy provided. "
                "The EnvGroupBuilder should create and pass a shared summarizer."
            )

    @property
    @abstractmethod
    def stop_condition(self) -> StopCondition:
        """Get stop sequences for this environment."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass

    @abstractmethod
    async def get_observation_string(self) -> str:
        """Get the observation string for the current turn."""
        pass

    def get_summarizer_system_prompt(self) -> str:
        """Get the system prompt for the summarizer.

        Subclasses can override this to customize summarization behavior.
        Default uses the standard summarizer prompt from prompts.py.
        """
        return SUMMARIZER_SYSTEM_PROMPT

    async def _summarize(self, history: str) -> str:
        """Summarize debate history using the pre-initialized summarizer policy.

        Args:
            history: The conversation history to summarize

        Returns:
            Summarized text

        Raises:
            RuntimeError: If summarizer is not initialized (summarize_history=False)
        """
        if self._summarizer_policy is None:
            raise RuntimeError("Summarizer not initialized. Set summarize_history=True to enable.")

        messages: list[Message] = [
            {"role": "system", "content": self.get_summarizer_system_prompt()},
            {"role": "user", "content": history},
        ]
        resp = await self._summarizer_policy(messages)
        return get_text_content(resp).strip()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Get the initial observation for this agent."""
        if self.agent_id != 0:
            await self.wait_for_turn()
        return await self.get_observation(), self.stop_condition

    async def wait_for_turn(self) -> None:
        """Wait until it's this agent's turn."""
        if not self.coordinator.done:
            await self.coordinator.wait_for_turn(self.agent_id)

    async def get_observation(self) -> types.ModelInput:
        """Get the current observation for this agent."""
        observation_str = await self.get_observation_string()
        messages: list[Message] = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": observation_str},
        ]
        return self.renderer.build_generation_prompt(messages)

    async def step(self, action: Action) -> StepResult:
        """Take a step in the environment."""
        if self.coordinator.done:
            return self.get_done_step()

        observation_str = await self.get_observation_string()

        # Parse and submit response
        action_message: Message = self.renderer.parse_response(action)[0]
        action_content = get_text_content(action_message)

        await self.coordinator.submit_response(
            self.agent_id,
            action_content,
            observation=observation_str,
        )

        # Wait for next turn
        await self.wait_for_turn()
        # Per-step reward is 0 by default; the main learning signal is computed as a final group
        # reward by the EnvGroupBuilder using all pairwise comparisons.
        return StepResult(
            reward=0.0,
            episode_done=self.coordinator.done,
            next_observation=await self.get_observation(),
            next_stop_condition=self.stop_condition,
            metrics={
                "cycle": float(self.coordinator.state.get_current_cycle()),
            },
        )

    def get_done_step(self) -> StepResult:
        """Get a step result for when the episode is done."""
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=types.ModelInput.empty(),
            next_stop_condition=STOP_CONDITION,
            metrics={},
        )


@dataclass
class BaseMultiAgentEnvGroupBuilder(EnvGroupBuilder, ABC):
    """Base builder for groups of multi-agent debate environments.

    This abstract base class provides shared reward computation logic based on
    pairwise comparisons extracted from agent responses. Subclasses implement
    environment creation and final reward computation.

    Key Method:
        _populate_stepwise_rewards(): Assigns rewards based on comparisons in
        agent responses using a leave-one-out approach (agents don't judge
        themselves). This method was previously duplicated byte-for-byte in
        both env.py and verifiable_env.py.

    Subclasses must implement:
        - make_envs(): Create environment instances for this group
        - compute_group_rewards(): Compute final rewards and metrics

    Attributes:
        num_agents: Number of agents in the debate
        renderer: Renderer for converting messages to/from tokens
        summarize_history: Whether to summarize older conversation history
        summarize_model: OpenRouter model to use for summarization (e.g., "openai/gpt-4o-mini")
        model_name: Name of the base model being trained
    """

    num_agents: int
    renderer: Renderer
    summarize_history: bool = field(default=False, kw_only=True)
    summarize_model: str | None = field(default="openai/gpt-4o-mini", kw_only=True)
    model_name: str | None = field(default=None, kw_only=True)

    # Reward system configuration
    enable_format_penalty: bool = field(default=True, kw_only=True)

    def _create_shared_summarizer(self) -> "MessageCompleter | None":
        """Create a single shared OpenRouter summarizer for all environments in this group.

        This method should be called once per group to create a summarizer that
        is shared across all agents, avoiding the creation of multiple API clients.

        Requires OPENROUTER_API_KEY environment variable to be set when summarize_history=True.

        Returns:
            OpenRouterMessageCompleter if summarize_history=True, None otherwise

        Raises:
            ValueError: If summarize_history=True but OPENROUTER_API_KEY is not set
        """
        if not self.summarize_history:
            return None

        import os

        from tinker_cookbook.completers import OpenRouterMessageCompleter

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Required for debate history summarization when summarize_history=True. "
                "Get your API key from https://openrouter.ai/keys and add to .env file."
            )

        # Default to gpt-4o-mini if no model specified
        model_name = self.summarize_model or "openai/gpt-4o-mini"

        return OpenRouterMessageCompleter(
            api_key=api_key,
            model_name=model_name,
            max_tokens=1024,
            temperature=1.0,
        )

    def _populate_stepwise_rewards(
        self,
        trajectory_group: list[Trajectory],
        env_group: Sequence[Env],
    ) -> dict[str, float]:
        """
        Modify trajectories in-place to assign step-wise rewards based on comparisons.

        Reward computation:
        - Each comparison provides ±1.0 reward assigned to the specific step being compared
        - Format penalties: -0.5 per missing comparison (if enable_format_penalty=True)
        - Advantage normalization: center rewards by subtracting mean across agents

        Returns summary metrics.
        """
        env0 = env_group[0]
        # Get coordinator - works for both MultiAgentDebateEnv and VerifiableMultiAgentDebateEnv
        if hasattr(env0, "coordinator"):
            coordinator = env0.coordinator
        else:
            raise ValueError(f"Environment {type(env0)} does not have a coordinator")

        # Initialize step-level reward accumulators
        # step_rewards[agent_id][step_idx] tracks rewards for that specific step
        step_rewards: list[list[float]] = []
        for agent_id in range(self.num_agents):
            num_steps = len(trajectory_group[agent_id].transitions)
            step_rewards.append([0.0 for _ in range(num_steps)])

        total_valid_comparisons = 0
        missing_comparisons = 0

        # Step 1: Assign comparison rewards directly to specific steps
        for turn_idx, response in enumerate(coordinator.state.agent_responses):
            # Process each comparison from this response
            for agent_a, op, agent_b in response.comparisons:
                # Validate agent IDs
                if not (0 <= agent_a < self.num_agents and 0 <= agent_b < self.num_agents):
                    continue
                if agent_a == agent_b:
                    continue
                if op not in {">", "<"}:
                    continue

                # Find step indices for agent_a and agent_b
                # (most recent step before this turn)
                agent_a_step_idx = get_step_idx_before_turn(agent_a, turn_idx, self.num_agents)
                agent_b_step_idx = get_step_idx_before_turn(agent_b, turn_idx, self.num_agents)

                # Skip if either agent hasn't acted yet
                if agent_a_step_idx < 0 or agent_b_step_idx < 0:
                    continue

                # Directly assign rewards to the specific steps being compared
                if op == ">":
                    step_rewards[agent_a][agent_a_step_idx] += 1.0
                    step_rewards[agent_b][agent_b_step_idx] -= 1.0
                elif op == "<":
                    step_rewards[agent_a][agent_a_step_idx] -= 1.0
                    step_rewards[agent_b][agent_b_step_idx] += 1.0

                total_valid_comparisons += 1

        # Step 2: Assign format penalties to the step that produced the response
        if self.enable_format_penalty:
            for turn_idx, response in enumerate(coordinator.state.agent_responses):
                # Skip exempt turns (0 and 1)
                if turn_idx < FORMAT_EXEMPT_TURNS:
                    continue

                author_id = response.author_id
                author_step_idx = turn_idx // self.num_agents

                # If there are fewer than two *other* agents who have acted at least once
                # before this turn, comparisons are not meaningful and the prompts allow "N/A".
                # In particular, for num_agents <= 2 this prevents penalizing every turn.
                num_other_agents_who_have_acted = 0
                for other_agent_id in range(self.num_agents):
                    if other_agent_id == author_id:
                        continue
                    if get_step_idx_before_turn(other_agent_id, turn_idx, self.num_agents) >= 0:
                        num_other_agents_who_have_acted += 1
                if num_other_agents_who_have_acted < 2:
                    continue

                # Check if this response has valid comparisons
                if len(response.comparisons) == 0:
                    step_rewards[author_id][author_step_idx] += FORMAT_PENALTY
                    missing_comparisons += 1

        # Step 3: Compute total reward for each agent (sum of all step rewards)
        agent_totals = [sum(step_rewards[agent_id]) for agent_id in range(self.num_agents)]

        # Step 4: Apply advantage normalization (center rewards by subtracting mean)
        # This ensures above-average agents get positive rewards, below-average get negative,
        # and maintains zero-sum property for multi-agent competition
        mean_reward = sum(agent_totals) / len(agent_totals) if agent_totals else 0.0

        # Distribute the centering offset uniformly across each agent's steps
        for agent_id in range(self.num_agents):
            num_steps = len(trajectory_group[agent_id].transitions)
            if num_steps == 0:
                continue

            # Uniform normalization offset per step
            normalization_offset = mean_reward / num_steps

            for step_idx in range(num_steps):
                step_rewards[agent_id][step_idx] -= normalization_offset

        # Step 5: Assign final rewards to trajectory transitions
        for agent_id in range(self.num_agents):
            trajectory = trajectory_group[agent_id]
            for step_idx, transition in enumerate(trajectory.transitions):
                transition.reward += step_rewards[agent_id][step_idx]

        # Compute centered agent totals for metrics
        centered_agent_totals = [total - mean_reward for total in agent_totals]

        # Step 6: Return summary metrics
        return {
            "stepwise_comparisons_used": total_valid_comparisons,
            "missing_comparisons": missing_comparisons,
            "mean_reward_before_centering": mean_reward,
            "max_reward_after_centering": (
                max(centered_agent_totals) if centered_agent_totals else 0.0
            ),
            "min_reward_after_centering": (
                min(centered_agent_totals) if centered_agent_totals else 0.0
            ),
        }
