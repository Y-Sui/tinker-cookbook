"""Base classes for multi-agent debate environments.

This module provides abstract base classes that eliminate code duplication between
verifiable and non-verifiable debate implementations.

Classes:
    BaseMultiAgentDebateEnv: Base environment with shared step/observation logic
    BaseMultiAgentEnvGroupBuilder: Base builder with shared reward computation
"""

from abc import ABC, abstractmethod
from collections import defaultdict
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
DEFAULT_LAMBDA_GEN = 1.0  # Weights for Generation
DEFAULT_LAMBDA_JUDGE = 1.0  # Weights for Judge


@dataclass
class Vote:
    """A single pairwise comparison vote for the reward system."""

    source_agent: int  # Who made the comparison
    source_turn: int  # Global turn index
    winner_agent: int  # Agent judged as better
    winner_step: int  # Step index of winner
    loser_agent: int  # Agent judged as worse
    loser_step: int  # Step index of loser


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
        summarize_history: Whether to summarize older conversation history
        _summarizer_policy: Optional pre-created summarizer completer (shared across envs)
        model_name: Name of the base model
    """

    agent_id: int
    coordinator: MultiAgentCoordinator
    renderer: Renderer
    self_play: bool = True
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
        """Get the initial observation for this agent.

        With parallel generation, all agents wait for the cycle to start,
        then proceed simultaneously.
        """
        await self.coordinator.wait_for_cycle_start(self.agent_id)
        return await self.get_observation(), self.stop_condition

    async def get_observation(self) -> types.ModelInput:
        """Get the current observation for this agent."""
        observation_str = await self.get_observation_string()
        messages: list[Message] = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": observation_str},
        ]
        return self.renderer.build_generation_prompt(messages)

    async def step(self, action: Action) -> StepResult:
        """Take a step in the environment.

        With parallel generation:
        1. All agents in a cycle submit their responses simultaneously
        2. They all wait for the cycle to complete (all agents submitted)
        3. Then they all get observations for the next cycle together
        """
        if self.coordinator.done:
            return self.get_done_step()

        observation_str = await self.get_observation_string()

        # Parse and submit response
        action_message: Message = self.renderer.parse_response(action)[0]
        action_content = get_text_content(action_message)

        # Submit response - this will wait for all agents to submit before returning
        await self.coordinator.submit_response(
            self.agent_id,
            action_content,
            observation=observation_str,
        )

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

    Key Methods:
        _compute_rewards(): Computes rewards using win rate + consensus alignment
            - gen_rewards: Win rate for <solution>/<evaluation> tokens
            - judge_rewards: Consensus alignment for <comparison> tokens

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

    # =========================================================================
    # Reward System: Win Rate + Consensus Alignment
    # =========================================================================

    def _collect_votes(self, coordinator: MultiAgentCoordinator) -> list[Vote]:
        """
        Parse all <comparison> tags from all rounds and build vote registry.

        Constraints:
        - Ignores comparisons where either agent hasn't acted yet

        Returns:
            List of Vote objects representing all valid pairwise comparisons
        """
        votes: list[Vote] = []

        for turn_idx, response in enumerate(coordinator.state.agent_responses):
            source_agent = response.author_id

            for agent_a, op, agent_b in response.comparisons:
                # Validate agent IDs
                if not (0 <= agent_a < self.num_agents and 0 <= agent_b < self.num_agents):
                    continue
                if agent_a == agent_b:
                    continue

                # Get step indices (most recent step before this turn)
                step_a = get_step_idx_before_turn(agent_a, turn_idx, self.num_agents)
                step_b = get_step_idx_before_turn(agent_b, turn_idx, self.num_agents)

                # Skip if either agent hasn't acted yet
                if step_a < 0 or step_b < 0:
                    continue

                # Determine winner/loser based on comparison operator
                if op == ">":
                    winner_agent, winner_step = agent_a, step_a
                    loser_agent, loser_step = agent_b, step_b
                elif op == "<":
                    winner_agent, winner_step = agent_b, step_b
                    loser_agent, loser_step = agent_a, step_a
                else:
                    continue

                votes.append(
                    Vote(
                        source_agent=source_agent,
                        source_turn=turn_idx,
                        winner_agent=winner_agent,
                        winner_step=winner_step,
                        loser_agent=loser_agent,
                        loser_step=loser_step,
                    )
                )

        return votes

    def _compute_generator_rewards(
        self,
        votes: list[Vote],
        num_steps_per_agent: dict[int, int],
    ) -> dict[int, list[float]]:
        """
        Compute win rate rewards for generator tokens (<solution> + <evaluation>).

        For each (agent, step), computes:
            R_gen = count(wins) / count(total_votes)

        Result is in range [0, 1], where:
        - 1.0 = won all comparisons
        - 0.5 = won half of comparisons
        - 0.0 = lost all comparisons

        Args:
            votes: List of Vote objects from _collect_votes()
            num_steps_per_agent: {agent_id: num_steps} from trajectories

        Returns:
            {agent_id: [reward_per_step]} for generator rewards
        """
        # Accumulate win counts and total vote counts per (agent, step)
        win_counts: dict[tuple[int, int], int] = defaultdict(int)
        total_counts: dict[tuple[int, int], int] = defaultdict(int)

        for vote in votes:
            # Winner gets +1 win
            key_w = (vote.winner_agent, vote.winner_step)
            win_counts[key_w] += 1
            total_counts[key_w] += 1

            # Loser gets +0 wins but +1 total vote
            key_l = (vote.loser_agent, vote.loser_step)
            total_counts[key_l] += 1

        # Build reward dict with win rate
        gen_rewards: dict[int, list[float]] = {}
        for agent_id in range(self.num_agents):
            num_steps = num_steps_per_agent.get(agent_id, 0)
            rewards = []
            for step in range(num_steps):
                key = (agent_id, step)
                if total_counts[key] > 0:
                    # Win rate: wins / total_votes
                    rewards.append(win_counts[key] / total_counts[key])
                else:
                    rewards.append(0.0)  # No votes = neutral
            gen_rewards[agent_id] = rewards

        return gen_rewards

    def _compute_judge_rewards(
        self,
        votes: list[Vote],
        coordinator: MultiAgentCoordinator,
        num_steps_per_agent: dict[int, int],
    ) -> tuple[dict[int, list[float]], int]:
        """
        Compute consensus-aligned rewards for judge tokens (<comparison>).

        Two-stage process:
        1. Build hard consensus for each pair across ALL rounds
        2. Reward judges for aligning with consensus (+1) or disagreeing (-1)

        For ties (equal votes), judge reward is 0 (no signal).

        Args:
            votes: List of Vote objects from _collect_votes()
            coordinator: Coordinator with agent responses
            num_steps_per_agent: {agent_id: num_steps} from trajectories

        Returns:
            Tuple of:
            - {agent_id: [reward_per_step]} for judge rewards
            - missing_comparisons: number of turns penalized for missing comparisons
        """
        # Step 1: Build consensus for each pair
        # Key: (min_agent, min_step, max_agent, max_step) - normalized ordering
        # Value: list of +1 (first agent won) or -1 (second agent won)
        pair_votes: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)

        for vote in votes:
            # Normalize key: smaller agent first for consistent ordering
            if (vote.winner_agent, vote.winner_step) < (vote.loser_agent, vote.loser_step):
                key = (vote.winner_agent, vote.winner_step, vote.loser_agent, vote.loser_step)
                pair_votes[key].append(+1)  # First agent won
            else:
                key = (vote.loser_agent, vote.loser_step, vote.winner_agent, vote.winner_step)
                pair_votes[key].append(-1)  # First agent lost (second won)

        # Compute consensus: majority wins, ties = None
        consensus: dict[tuple[int, int, int, int], int | None] = {}
        for pair_key, vote_list in pair_votes.items():
            total = sum(vote_list)
            if total > 0:
                consensus[pair_key] = +1  # First agent is better
            elif total < 0:
                consensus[pair_key] = -1  # Second agent is better
            else:
                consensus[pair_key] = None  # Tie - no signal

        # Step 2: Evaluate each judge's comparisons against consensus
        judge_reward_sums: dict[int, list[float]] = {
            agent_id: [0.0] * num_steps_per_agent.get(agent_id, 0)
            for agent_id in range(self.num_agents)
        }
        judge_counts: dict[int, list[int]] = {
            agent_id: [0] * num_steps_per_agent.get(agent_id, 0)
            for agent_id in range(self.num_agents)
        }

        for turn_idx, response in enumerate(coordinator.state.agent_responses):
            source_agent = response.author_id
            source_step = turn_idx // self.num_agents

            if source_step >= len(judge_reward_sums.get(source_agent, [])):
                continue

            for agent_a, op, agent_b in response.comparisons:
                # Skip invalid comparisons
                if agent_a == agent_b:
                    continue

                step_a = get_step_idx_before_turn(agent_a, turn_idx, self.num_agents)
                step_b = get_step_idx_before_turn(agent_b, turn_idx, self.num_agents)

                if step_a < 0 or step_b < 0:
                    continue

                # Build normalized key and determine judge's vote
                if (agent_a, step_a) < (agent_b, step_b):
                    key = (agent_a, step_a, agent_b, step_b)
                    judge_vote = +1 if op == ">" else -1
                else:
                    key = (agent_b, step_b, agent_a, step_a)
                    judge_vote = -1 if op == ">" else +1

                ground_truth = consensus.get(key)

                if ground_truth is None:
                    # Tie: R_judge = 0, don't count toward average
                    continue

                # Alignment reward: +1 if aligned, -1 if not
                if judge_vote == ground_truth:
                    judge_reward_sums[source_agent][source_step] += 1.0
                else:
                    judge_reward_sums[source_agent][source_step] -= 1.0
                judge_counts[source_agent][source_step] += 1

        missing_comparisons = 0
        if self.enable_format_penalty:
            for turn_idx, response in enumerate(coordinator.state.agent_responses):
                if turn_idx < FORMAT_EXEMPT_TURNS:
                    continue

                author_id = response.author_id
                author_step = turn_idx // self.num_agents
                if author_step >= len(judge_reward_sums.get(author_id, [])):
                    continue

                # Require at least two other agents to have acted for comparisons to be meaningful.
                num_other_agents_who_have_acted = 0
                for other_agent_id in range(self.num_agents):
                    if other_agent_id == author_id:
                        continue
                    if get_step_idx_before_turn(other_agent_id, turn_idx, self.num_agents) >= 0:
                        num_other_agents_who_have_acted += 1
                if num_other_agents_who_have_acted < 2:
                    continue

                if len(response.comparisons) == 0:
                    judge_reward_sums[author_id][author_step] += FORMAT_PENALTY
                    judge_counts[author_id][author_step] += 1
                    missing_comparisons += 1

        # Average rewards per step
        judge_rewards: dict[int, list[float]] = {}
        for agent_id in range(self.num_agents):
            rewards = []
            for step in range(len(judge_reward_sums.get(agent_id, []))):
                if judge_counts[agent_id][step] > 0:
                    rewards.append(judge_reward_sums[agent_id][step] / judge_counts[agent_id][step])
                else:
                    rewards.append(0.0)
            judge_rewards[agent_id] = rewards

        return judge_rewards, missing_comparisons

    def _compute_rewards(
        self,
        trajectory_group: list[Trajectory],
        env_group: Sequence[Env],
    ) -> tuple[dict[int, list[float]], dict[int, list[float]], dict[str, float]]:
        """
        Compute rewards using win rate + consensus alignment system.

        Two-reward-stream approach:
        - gen_rewards: For <solution> and <evaluation> tokens (win rate)
        - judge_rewards: For <comparison> tokens (consensus alignment)

        Args:
            trajectory_group: List of trajectories, one per agent
            env_group: List of environments, one per agent

        Returns:
            Tuple of:
            - gen_rewards: {agent_id: [reward_per_step]} for generator tokens
            - judge_rewards: {agent_id: [reward_per_step]} for judge tokens
            - metrics: Summary metrics dict
        """
        env0 = env_group[0]
        if hasattr(env0, "coordinator"):
            coordinator = env0.coordinator
        else:
            raise ValueError(f"Environment {type(env0)} does not have a coordinator")

        # Get number of steps per agent from trajectories
        num_steps_per_agent = {
            agent_id: len(trajectory_group[agent_id].transitions)
            for agent_id in range(self.num_agents)
        }

        # Collect all votes
        votes = self._collect_votes(coordinator)

        # Compute rewards
        gen_rewards = self._compute_generator_rewards(votes, num_steps_per_agent)
        judge_rewards, missing_comparisons = self._compute_judge_rewards(
            votes, coordinator, num_steps_per_agent
        )

        # Build metrics
        total_steps = sum(num_steps_per_agent.values())
        comparison_lines_total = 0
        comparison_lines_invalid = 0
        comparison_pairs_tie = 0
        for response in coordinator.state.agent_responses:
            comparison_lines_total += response.comparison_lines_total
            comparison_lines_invalid += response.comparison_lines_invalid
            comparison_pairs_tie += response.comparison_pairs_tie

        metrics = {
            "reward/total_votes": float(len(votes)),
            "reward/votes_per_step": float(len(votes)) / max(1, total_steps),
            "reward/missing_comparisons": float(missing_comparisons),
            "reward/comparison_lines_total": float(comparison_lines_total),
            "reward/comparison_lines_invalid": float(comparison_lines_invalid),
            "reward/comparison_pairs_tie": float(comparison_pairs_tie),
        }

        return gen_rewards, judge_rewards, metrics
