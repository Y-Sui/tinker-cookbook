"""
CANT environment for non-verifiable tasks (general Q&A, creative writing, etc.).
"""

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from tinker import types

from tinker_cookbook.completers import (
    OpenRouterMessageCompleter,
)
from tinker_cookbook.completers import (
    StopCondition as StopConditionType,
)
from tinker_cookbook.recipes.cant.coordinator import CANTCoordinator

logger = logging.getLogger(__name__)
from tinker_cookbook.recipes.cant.prompts import (
    get_default_agent_personas,
    get_round1_system_prompt,
    get_round2_system_prompt,
    get_round3_system_prompt,
    get_round4_system_prompt,
    get_user_message_round1,
    get_user_message_round2,
    get_user_message_round3,
    get_user_message_round4,
)
from tinker_cookbook.renderers import Message, Renderer, get_text_content
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Observation,
    StepResult,
    Trajectory,
)

# Stop condition for CANT (empty list means use default EOS)
STOP_CONDITION: StopConditionType = []


@dataclass
class CANTEnv(Env):
    """
    Environment for a single agent in CANT protocol.

    Each agent proceeds through 4 rounds:
    - Round 0: Generate initial solution
    - Round 1: Rank and critique other solutions
    - Round 2: Revise own solution
    - Round 3: Provide final ranking of revised solutions
    """

    agent_id: int
    coordinator: CANTCoordinator
    renderer: Renderer
    persona: str | None = None
    max_response_tokens: int = 2048
    openrouter_completer: OpenRouterMessageCompleter | None = None
    use_llm_summarization: bool = True

    # Track current step within the episode
    _step_count: int = field(default=0, init=False)
    _episode_done: bool = field(default=False, init=False)

    async def _buffer_initial_solutions(self) -> None:
        """Buffer initial solutions after Round 0."""
        from tinker_cookbook.recipes.cant.memory_buffer import buffer_solutions

        buffered = await buffer_solutions(
            self.coordinator.initial_solutions,
            self.openrouter_completer,
            use_llm_summarization=self.use_llm_summarization,
        )
        self.coordinator.set_buffered_initial_solutions(buffered)

    async def _buffer_critiques(self) -> None:
        """Buffer critiques after Round 1."""
        from tinker_cookbook.recipes.cant.memory_buffer import buffer_critiques

        buffered = await buffer_critiques(
            self.coordinator.critique_texts,
            self.openrouter_completer,
            use_llm_summarization=self.use_llm_summarization,
        )
        self.coordinator.set_buffered_critique_texts(buffered)

    async def _buffer_revised_solutions(self) -> None:
        """Buffer revised solutions after Round 2."""
        from tinker_cookbook.recipes.cant.memory_buffer import buffer_solutions

        buffered = await buffer_solutions(
            self.coordinator.revised_solutions,
            self.openrouter_completer,
            use_llm_summarization=self.use_llm_summarization,
        )
        self.coordinator.set_buffered_revised_solutions(buffered)

    async def initial_observation(self) -> tuple[Observation, StopConditionType]:
        """Get the initial observation (Round 0: Proposal)."""
        return await self._get_observation(), STOP_CONDITION

    async def _get_observation(self) -> types.ModelInput:
        """Build observation based on current round."""
        current_round = self.coordinator.get_current_round()

        # Select appropriate system prompt and user message
        if current_round == 0:
            system_prompt = get_round1_system_prompt(self.persona)
            user_message = get_user_message_round1(self.coordinator.question)

        elif current_round == 1:
            system_prompt = get_round2_system_prompt(self.persona)
            user_message = get_user_message_round2(
                self.coordinator.question,
                self.coordinator.get_initial_solutions(buffered=self.use_llm_summarization),
            )

        elif current_round == 2:
            system_prompt = get_round3_system_prompt(self.persona)
            user_message = get_user_message_round3(
                self.coordinator.question,
                self.coordinator.get_initial_solutions(buffered=self.use_llm_summarization),
                self.agent_id,
                self.coordinator.get_critiques_for_agent(
                    self.agent_id, buffered=self.use_llm_summarization
                ),
            )

        elif current_round == 3:
            system_prompt = get_round4_system_prompt(self.persona)
            user_message = get_user_message_round4(
                self.coordinator.question,
                self.coordinator.get_revised_solutions(buffered=self.use_llm_summarization),
            )

        else:
            # Episode is complete
            return types.ModelInput.empty()

        # Build messages and render
        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        return self.renderer.build_generation_prompt(messages)

    async def step(self, action: Action) -> StepResult:
        """
        Process agent's action (response) for the current round.

        Args:
            action: Agent's generated response

        Returns:
            StepResult with next observation and episode status
        """
        if self._episode_done:
            return self._get_done_step()

        current_round = self.coordinator.get_current_round()

        # Parse response
        action_message: Message = self.renderer.parse_response(action)[0]
        action_content = get_text_content(action_message)

        # Store response in coordinator
        if current_round == 0:
            self.coordinator.add_round1_response(self.agent_id, action_content)
        elif current_round == 1:
            self.coordinator.add_round2_response(self.agent_id, action_content)
        elif current_round == 2:
            self.coordinator.add_round3_response(self.agent_id, action_content)
        elif current_round == 3:
            self.coordinator.add_round4_response(self.agent_id, action_content)

        # Increment step count
        self._step_count += 1

        if self.coordinator.can_advance_round():
            if current_round == 0 and self.use_llm_summarization:
                await self._buffer_initial_solutions()
            elif current_round == 1 and self.use_llm_summarization:
                await self._buffer_critiques()
            elif current_round == 2 and self.use_llm_summarization:
                await self._buffer_revised_solutions()
            self.coordinator.advance_round()

        # Check if episode is complete (all rounds done)
        if self.coordinator.is_complete():
            self._episode_done = True

        # Get next observation
        next_obs = await self._get_observation()

        return StepResult(
            reward=0.0,  # Rewards computed at episode end by builder
            episode_done=self._episode_done,
            next_observation=next_obs,
            next_stop_condition=STOP_CONDITION,
            metrics={
                "round": float(current_round),
                "step": float(self._step_count),
            },
        )

    def _get_done_step(self) -> StepResult:
        """Return a terminal step result."""
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=types.ModelInput.empty(),
            next_stop_condition=STOP_CONDITION,
            metrics={},
        )


@dataclass
class CANTEnvGroupBuilder(EnvGroupBuilder):
    """
    Builder for creating groups of CANT environments (one per agent).

    Handles environment creation, reward computation, and advantage normalization.

    Key hyperparameters:
    - beta_disc: Scaling factor for persuasion rewards
    - weight_disc: Weight for persuasion component (critique effectiveness)
    - weight_sol: Weight for solution quality component
    - weight_meta: Weight for consensus component

    """

    num_agents: int
    renderer: Renderer
    problem_state: dict
    model_name: str | None = None

    # CANT-specific hyperparameters
    beta_disc: float = field(default=5.0, kw_only=True)
    weight_disc: float = field(default=2.0, kw_only=True)
    weight_sol: float = field(default=1.0, kw_only=True)
    weight_meta: float = field(default=1.0, kw_only=True)

    # Environment configuration
    personas: list[str] | None = field(default=None, kw_only=True)
    max_response_tokens: int = field(default=2048, kw_only=True)
    use_llm_summarization: bool = field(default=True, kw_only=True)

    def logging_tags(self) -> list[str]:
        dataset_name = self.problem_state.get("dataset_name")
        if dataset_name:
            return [str(dataset_name)]
        return []

    async def make_envs(self) -> Sequence[Env]:
        """
        Create a group of CANT environments for one problem.

        Args:
            problem_state: Dict containing 'question' and optionally 'answer'

        Returns:
            List of CANTEnv instances (one per agent)
        """
        import os

        # Create OpenRouter completer for memory buffering if LLM summarization enabled
        openrouter_completer = None
        if self.use_llm_summarization:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("use_llm_summarization=True requires OPENROUTER_API_KEY")

            from tinker_cookbook.recipes.cant.memory_buffer import (
                DEFAULT_MAX_SOLUTION_TOKENS,
                DEFAULT_SUMMARIZATION_MODEL,
            )

            openrouter_completer = OpenRouterMessageCompleter(
                api_key=api_key,
                model_name=DEFAULT_SUMMARIZATION_MODEL,
                max_tokens=DEFAULT_MAX_SOLUTION_TOKENS,
                temperature=0.8,
            )

        question = self.problem_state["question"]
        answer = self.problem_state.get("answer")

        # Create shared coordinator
        coordinator = CANTCoordinator(
            question=question,
            num_agents=self.num_agents,
            answer=answer,
        )

        # Create one environment per agent
        personas = self.personas or get_default_agent_personas()
        envs = []
        for agent_id in range(self.num_agents):
            persona = personas[agent_id % len(personas)]
            env = CANTEnv(
                agent_id=agent_id,
                coordinator=coordinator,
                renderer=self.renderer,
                persona=persona,
                max_response_tokens=self.max_response_tokens,
                openrouter_completer=openrouter_completer,
                use_llm_summarization=self.use_llm_summarization,
            )
            envs.append(env)

        return envs

    async def compute_group_rewards(
        self,
        trajectory_group: Sequence[Trajectory],
        env_group: Sequence[Env],
    ) -> list[tuple[float, dict]]:
        """
        Compute rewards for all agents after episode completion.

        This implements the CANT reward system with four components:
        1. r_disc: Persuasion rewards (did critiques lower targets' scores?)
        2. r_sol: Solution quality (Bradley-Terry scores, Z-normalized)
        3. r_meta: Consensus alignment (match majority opinion)
        4. r_meta: Consensus alignment (match majority opinion)

        Args:
            trajectory_group: Trajectories for all agents
            env_group: Environment instances for all agents

        Returns:
            List of (final_reward, metrics) tuples, one per trajectory
        """
        # Get coordinator from first env
        coordinator = env_group[0].coordinator

        # Compute all reward components
        coordinator.compute_rewards(
            beta_disc=self.beta_disc,
            weight_disc=self.weight_disc,
            weight_sol=self.weight_sol,
            weight_meta=self.weight_meta,
        )

        # Assign rewards to trajectory tokens
        for agent_id, trajectory in enumerate(trajectory_group):
            self._assign_rewards_to_trajectory(
                trajectory,
                agent_id,
                coordinator,
            )

        # Note: Advantages are normalized later in the training loop.

        # Collect metrics for logging
        metrics = self._collect_metrics(coordinator)

        return [(0.0, metrics) for _ in trajectory_group]

    def _assign_rewards_to_trajectory(
        self,
        trajectory: Trajectory,
        agent_id: int,
        coordinator: CANTCoordinator,
    ) -> None:
        """
        Assign rewards to trajectory steps based on CANT reward breakdown.

        Assign per-token rewards based on CANT segments.

        Args:
            trajectory: Trajectory to modify
            agent_id: ID of the agent
            coordinator: Coordinator with computed rewards
        """
        breakdown = coordinator.reward_breakdown[agent_id]
        sol_reward = breakdown.get("sol", 0.0)
        disc_reward = breakdown.get("disc", 0.0)
        meta_reward = breakdown.get("meta", 0.0)

        for step_idx, transition in enumerate(trajectory.transitions):
            if step_idx == 0:
                step_reward = sol_reward
                transition.reward = step_reward
                token_rewards = self._token_rewards_for_tags(
                    transition.ac.tokens,
                    [("initial_solution", step_reward)],
                    default_reward=0.0,
                )
                if token_rewards is None:
                    token_rewards = [step_reward] * len(transition.ac.tokens)
                transition.token_rewards = token_rewards
            elif step_idx == 1:
                step_reward = disc_reward
                transition.reward = step_reward
                token_rewards = self._token_rewards_for_tags(
                    transition.ac.tokens,
                    [("blind_ranking", 0.0), ("critique", disc_reward)],
                    default_reward=0.0,
                )
                if token_rewards is None:
                    token_rewards = [step_reward] * len(transition.ac.tokens)
                transition.token_rewards = token_rewards
            elif step_idx == 2:
                step_reward = sol_reward
                transition.reward = step_reward
                token_rewards = self._token_rewards_for_tags(
                    transition.ac.tokens,
                    [("revised_solution", sol_reward)],
                    default_reward=0.0,
                )
                if token_rewards is None:
                    token_rewards = [step_reward] * len(transition.ac.tokens)
                transition.token_rewards = token_rewards
            elif step_idx == 3:
                step_reward = meta_reward
                transition.reward = step_reward
                token_rewards = self._token_rewards_for_tags(
                    transition.ac.tokens,
                    [("final_ranking", meta_reward)],
                    default_reward=0.0,
                )
                if token_rewards is None:
                    token_rewards = [step_reward] * len(transition.ac.tokens)
                transition.token_rewards = token_rewards
            else:
                transition.reward = 0.0
                transition.token_rewards = [0.0] * len(transition.ac.tokens)

    def _token_rewards_for_tags(
        self,
        tokens: list[int],
        tag_rewards: list[tuple[str, float]],
        default_reward: float = 0.0,
    ) -> list[float] | None:
        offsets_text = self._get_token_offsets(tokens)
        if offsets_text is None:
            return None
        offsets, text = offsets_text
        rewards = [default_reward] * len(tokens)
        found_any = False

        for tag, reward in tag_rewards:
            span = self._find_tag_span(text, tag)
            if span is None:
                continue
            found_any = True
            start, end = span
            for i, (tok_start, tok_end) in enumerate(offsets):
                if tok_start < end and tok_end > start:
                    rewards[i] = reward

        if not found_any:
            return None
        return rewards

    def _get_token_offsets(self, tokens: list[int]) -> tuple[list[tuple[int, int]], str] | None:
        tokenizer = getattr(self.renderer, "tokenizer", None)
        if tokenizer is None:
            return None
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except Exception:
            return None
        offsets = encoded.get("offset_mapping")
        if offsets is None or len(offsets) != len(tokens):
            return None
        return list(offsets), text

    @staticmethod
    def _find_tag_span(text: str, tag: str) -> tuple[int, int] | None:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return None
        end = text.find(end_tag, start + len(start_tag))
        if end == -1:
            return None
        return start, end + len(end_tag)

    def _collect_metrics(self, coordinator: CANTCoordinator) -> dict:
        """
        Collect metrics for logging.

        Args:
            coordinator: Coordinator with computed rewards

        Returns:
            Dict of metrics
        """
        metrics = {}

        # Per-agent metrics
        for agent_id in range(self.num_agents):
            prefix = f"agent_{agent_id}"

            # Bradley-Terry scores
            bt_t0 = coordinator.bradley_terry_scores_t0.get(agent_id, 0.0)
            bt_final = coordinator.bradley_terry_scores_final.get(agent_id, 0.0)
            metrics[f"{prefix}/bt_score_t0"] = bt_t0
            metrics[f"{prefix}/bt_score_final"] = bt_final
            metrics[f"{prefix}/bt_score_delta"] = bt_final - bt_t0

            # Reward breakdown
            breakdown = coordinator.reward_breakdown.get(agent_id, {})
            for component, value in breakdown.items():
                metrics[f"{prefix}/reward_{component}"] = value

            # Total reward
            metrics[f"{prefix}/reward_total"] = coordinator.get_total_reward(agent_id)

            # Number of critiques given
            num_critiques = len(coordinator.critiques.get(agent_id, []))
            metrics[f"{prefix}/num_critiques_given"] = float(num_critiques)

        # Group-level metrics
        all_bt_t0 = [
            coordinator.bradley_terry_scores_t0.get(i, 0.0) for i in range(self.num_agents)
        ]
        all_bt_final = [
            coordinator.bradley_terry_scores_final.get(i, 0.0) for i in range(self.num_agents)
        ]

        metrics["group/bt_score_t0_mean"] = np.mean(all_bt_t0)
        metrics["group/bt_score_t0_std"] = np.std(all_bt_t0)
        metrics["group/bt_score_final_mean"] = np.mean(all_bt_final)
        metrics["group/bt_score_final_std"] = np.std(all_bt_final)

        # Total critiques in episode
        total_critiques = sum(len(targets) for targets in coordinator.critiques.values())
        metrics["group/total_critiques"] = float(total_critiques)

        return metrics
