"""Multi-agent conversation environment with self-comparison rewards.

This module implements a multi-agent conversation system where:
- N agents take turns in a conversation via a Coordinator pattern
- Each agent produces structured output (reflection, evaluation, solution)
- Policy itself acts as judge (LLM-as-judge) for self-comparison
- Tournament-style pairwise comparison determines rewards
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Literal, Sequence

import chz
import tinker
from pydantic import BaseModel, Field

from tinker_cookbook.renderers import Message, Renderer, get_renderer
from tinker_cookbook.rl.preference_envs import TournamentPattern, get_pairs
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================


class AgentResponse(BaseModel):
    """Structured output from an agent's turn."""

    reflection: str = Field(description="Agent's reasoning about the conversation and query")
    evaluation: str = Field(description="Critique and analysis of previous solutions")
    solution: str = Field(description="The agent's proposed solution to the problem")


class JudgmentResponse(BaseModel):
    """Structured output when policy acts as judge."""

    reasoning: str = Field(description="Detailed comparison reasoning")
    preference: Literal["A", "B", "Tie"] = Field(description="Which solution is better")


# ============================================================================
# Prompt Templates (Minimal placeholders - user will customize)
# ============================================================================

AGENT_SYSTEM_PROMPTS = {
    0: """You are Agent 0 - The Innovator.
Generate creative, novel solutions with unique perspectives.

Respond in JSON format:
{
  "reflection": "Your reasoning about the query and conversation history",
  "evaluation": "Your critique of previous solutions (or 'N/A' for first turn)",
  "solution": "Your creative solution to the query"
}""",
    1: """You are Agent 1 - The Critic.
Generate rigorous, analytically sound solutions. Find flaws and edge cases.

Respond in JSON format:
{
  "reflection": "Your analytical reasoning",
  "evaluation": "Your critique of previous solutions (or 'N/A' for first turn)",
  "solution": "Your rigorous solution addressing identified gaps"
}""",
    2: """You are Agent 2 - The Synthesizer.
Integrate the best ideas from all perspectives into a balanced solution.

Respond in JSON format:
{
  "reflection": "How you're synthesizing previous work",
  "evaluation": "Your analysis of previous solutions (or 'N/A' for first turn)",
  "solution": "Your balanced, integrated solution"
}""",
}

JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating two solutions to a problem.

Compare the solutions based on:
1. Correctness: Is the solution accurate and logically sound?
2. Completeness: Does it fully address the query?
3. Clarity: Is it well-explained and understandable?
4. Practical value: Would this solution be useful?

Respond in JSON format:
{
  "reasoning": "Detailed comparison of the two solutions",
  "preference": "A" or "B" or "Tie"
}

Be objective. If solutions are equally good, choose "Tie"."""


# ============================================================================
# Parsing Utilities (with multiple fallbacks)
# ============================================================================


def parse_agent_response(text: str) -> AgentResponse:
    """Parse AgentResponse from LLM output with multiple fallback strategies."""
    # Attempt 1: Direct Pydantic JSON parse
    try:
        return AgentResponse.model_validate_json(text)
    except Exception:
        pass

    # Attempt 2: Direct JSON parse
    try:
        data = json.loads(text)
        return AgentResponse.model_validate(data)
    except Exception:
        pass

    # Attempt 2: Extract from markdown code block
    if "```json" in text:
        match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return AgentResponse.model_validate(data)
            except Exception:
                pass

    # Attempt 3: Extract from plain code block
    if "```" in text:
        match = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return AgentResponse.model_validate(data)
            except Exception:
                pass

    # Final fallback: Unstructured
    return AgentResponse(
        reflection="(parsing failed)",
        evaluation="(parsing failed)",
        solution=text,  # Use entire text as solution
    )


def parse_judgment_response(text: str) -> JudgmentResponse:
    """Parse JudgmentResponse from LLM output with multiple fallback strategies."""
    # Attempt 1: Direct Pydantic JSON parse
    try:
        return JudgmentResponse.model_validate_json(text)
    except Exception:
        pass

    # Attempt 2: Direct JSON parse
    try:
        data = json.loads(text)
        return JudgmentResponse.model_validate(data)
    except Exception:
        pass

    # Attempt 2: Extract from markdown code block
    if "```json" in text:
        match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return JudgmentResponse.model_validate(data)
            except Exception:
                pass

    # Attempt 3: Extract from plain code block
    if "```" in text:
        match = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return JudgmentResponse.model_validate(data)
            except Exception:
                pass

    # Attempt 4: Simple text analysis to infer preference
    text_lower = text.lower()
    if "solution a" in text_lower and "better" in text_lower:
        preference = "A"
    elif "solution b" in text_lower and "better" in text_lower:
        preference = "B"
    elif "tie" in text_lower or "equal" in text_lower:
        preference = "Tie"
    else:
        # Default to Tie if can't determine
        preference = "Tie"

    return JudgmentResponse(reasoning=text, preference=preference)


# ============================================================================
# MultiAgentCoordinator - Manages turn-taking for N agents
# ============================================================================


class MultiAgentCoordinator:
    """Coordinates turn-taking for N agents in a conversation.

    Generalizes TwoPlayerCoordinator to N agents with round-robin turns.
    Uses asyncio.Condition for synchronization.
    """

    def __init__(self, num_agents: int, max_rounds: int, query: str):
        self.num_agents = num_agents
        self.max_rounds = max_rounds
        self.query = query

        # Synchronization
        self.condition = asyncio.Condition()
        self.current_turn = 0  # Which agent's turn (0 to num_agents-1)
        self.current_round = 1  # Which round (1 to max_rounds)
        self.conversation_history: list[AgentResponse] = []
        self.episode_done = False

    async def wait_for_turn(self, agent_id: int) -> None:
        """Block until it's this agent's turn or episode is done."""
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.current_turn == agent_id or self.episode_done
            )

    async def submit_response(self, agent_id: int, response: AgentResponse) -> None:
        """Submit agent's response, advance turn, and notify waiting agents."""
        async with self.condition:
            # Verify it's actually this agent's turn
            if not self.episode_done and self.current_turn != agent_id:
                raise ValueError(f"Not agent {agent_id}'s turn (current turn: {self.current_turn})")

            # Add response to history
            self.conversation_history.append(response)

            # Advance turn
            self.current_turn = (self.current_turn + 1) % self.num_agents

            # If wrapped back to 0, increment round
            if self.current_turn == 0:
                self.current_round += 1

            # Check termination
            if self.current_round > self.max_rounds:
                self.episode_done = True

            # Notify all waiting agents
            self.condition.notify_all()

    def get_context_for_agent(self, agent_id: int) -> str:
        """Build conversation context string for an agent's prompt."""
        if not self.conversation_history:
            return "(No previous conversation)"

        context = ""
        agent_names = ["Innovator", "Critic", "Synthesizer"]

        for idx, response in enumerate(self.conversation_history):
            resp_agent_id = idx % self.num_agents
            agent_name = (
                agent_names[resp_agent_id]
                if resp_agent_id < len(agent_names)
                else f"Agent {resp_agent_id}"
            )
            turn_num = idx + 1

            context += f"\n--- Turn {turn_num} by {agent_name} (Agent {resp_agent_id}) ---\n"
            context += f"Reflection: {response.reflection}\n"
            context += f"Evaluation: {response.evaluation}\n"
            context += f"Solution: {response.solution}\n"

        return context


# ============================================================================
# MultiAgentConversationEnv - Individual agent's environment
# ============================================================================


@dataclass
class MultiAgentConversationEnv(Env):
    """Environment representing one agent's perspective in a multi-agent conversation.

    Multiple envs share a single MultiAgentCoordinator for turn synchronization.
    """

    agent_id: int
    coordinator: MultiAgentCoordinator
    renderer: Renderer
    max_tokens: int = 512

    @property
    def stop_condition(self) -> list[str]:
        """Stop sequences for JSON output."""
        return ["\n}"]

    async def initial_observation(self) -> tuple[Observation, list[str]]:
        """Get initial observation. Wait for turn if not first agent."""
        # If not first agent, wait for turn
        if self.agent_id != 0:
            await self.coordinator.wait_for_turn(self.agent_id)

        # Build prompt
        prompt = self._build_agent_prompt()
        return self.renderer.build_generation_prompt(prompt), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        """Process agent's action and return next observation."""
        # Parse action as AgentResponse
        action_text = self.renderer.tokenizer.decode(action)
        response = parse_agent_response(action_text)

        # Submit to coordinator (advances turn)
        await self.coordinator.submit_response(self.agent_id, response)

        # Check if episode done
        if self.coordinator.episode_done:
            return StepResult(
                reward=0.0,  # Final rewards computed in compute_group_rewards
                episode_done=True,
                next_observation=tinker.types.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "agent_id": self.agent_id,
                    "turns_taken": len(self.coordinator.conversation_history)
                    // self.coordinator.num_agents,
                },
            )

        # Wait for next turn
        await self.coordinator.wait_for_turn(self.agent_id)

        # Build next observation
        next_prompt = self._build_agent_prompt()
        return StepResult(
            reward=0.0,  # Per-step reward is 0
            episode_done=False,
            next_observation=self.renderer.build_generation_prompt(next_prompt),
            next_stop_condition=self.stop_condition,
            metrics={"agent_id": self.agent_id},
        )

    def _build_agent_prompt(self) -> list[Message]:
        """Build prompt for agent's turn."""
        # System prompt with role
        system_prompt = AGENT_SYSTEM_PROMPTS.get(self.agent_id, AGENT_SYSTEM_PROMPTS[0])

        # User message with query + conversation history
        context = self.coordinator.get_context_for_agent(self.agent_id)
        user_content = f"""Query: {self.coordinator.query}

Conversation History:
{context}

--- Your Turn (Round {self.coordinator.current_round}) ---
Provide your response in the JSON format specified."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]


# ============================================================================
# PolicySelfComparisonJudge - Uses policy to compare solutions
# ============================================================================


class PolicySelfComparisonJudge:
    """Uses the policy model itself to judge which solution is better (LLM-as-judge)."""

    def __init__(self, sampling_client: tinker.SamplingClient, renderer: Renderer):
        self.sampling_client = sampling_client
        self.renderer = renderer

    async def compare(self, query: str, solution_A: str, solution_B: str) -> float:
        """Compare two solutions and return score.

        Returns:
            -1.0: A strongly preferred
             0.0: Tie
             1.0: B strongly preferred
        """
        # Build comparison prompt
        prompt = self._build_comparison_prompt(query, solution_A, solution_B)

        # Sample from policy (deterministic for judging)
        result = await self.sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=0.0,  # Deterministic
                max_tokens=500,
            ),
        )

        # Parse response
        response_text = self.renderer.tokenizer.decode(result.sequences[0].tokens)
        judgment = parse_judgment_response(response_text)

        # Convert to score
        return self._judgment_to_score(judgment.preference)

    def _build_comparison_prompt(
        self, query: str, solution_A: str, solution_B: str
    ) -> tinker.types.ModelInput:
        """Build prompt for judging comparison."""
        messages: list[Message] = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Query: {query}

Solution A:
{solution_A}

Solution B:
{solution_B}

Which solution is better? Provide your response in JSON format.""",
            },
        ]
        return self.renderer.build_generation_prompt(messages)

    def _judgment_to_score(self, preference: Literal["A", "B", "Tie"]) -> float:
        """Convert preference to score."""
        if preference == "A":
            return -1.0
        elif preference == "B":
            return 1.0
        else:  # Tie
            return 0.0


# ============================================================================
# MultiAgentConversationEnvGroupBuilder - Creates groups and computes rewards
# ============================================================================


@dataclass(frozen=True)
class MultiAgentConversationEnvGroupBuilder(EnvGroupBuilder):
    """Builder for a group of agents sharing one conversation.

    Creates N agents that share a MultiAgentCoordinator for turn-taking.
    After episode completes, runs tournament-style pairwise comparison to compute rewards.
    """

    query: str
    num_agents: int
    max_rounds: int
    renderer: Renderer
    self_comparison_judge: PolicySelfComparisonJudge
    max_tokens: int = 512
    tournament_pattern: TournamentPattern = TournamentPattern.ALL_PAIRS_BOTH_WAYS

    async def make_envs(self) -> Sequence[Env]:
        """Create N agents sharing one coordinator."""
        # Create one coordinator shared by all agents
        coordinator = MultiAgentCoordinator(
            num_agents=self.num_agents, max_rounds=self.max_rounds, query=self.query
        )

        # Create N envs, one per agent, sharing the coordinator
        return [
            MultiAgentConversationEnv(
                agent_id=i,
                coordinator=coordinator,
                renderer=self.renderer,
                max_tokens=self.max_tokens,
            )
            for i in range(self.num_agents)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """Compute rewards via tournament-style pairwise comparison.

        Algorithm (from preference_envs.py:196-213):
        1. Extract final solutions from each trajectory
        2. Create pairwise comparisons (ALL_PAIRS pattern)
        3. Call self_comparison_judge for each pair
        4. Accumulate win-loss scores
        5. Normalize by matchup count
        """
        # Extract final solutions from trajectories
        solutions = []
        for traj in trajectory_group:
            # Get coordinator from first env (all share same coordinator)
            env = env_group[0]
            assert isinstance(env, MultiAgentConversationEnv)
            coordinator = env.coordinator

            # Extract this agent's final solution from conversation history
            # Agent i's responses are at indices i, i+num_agents, i+2*num_agents, ...
            agent_id = trajectory_group.index(traj)
            agent_responses = [
                coordinator.conversation_history[idx]
                for idx in range(agent_id, len(coordinator.conversation_history), self.num_agents)
            ]

            if agent_responses:
                # Use the last response from this agent
                final_solution = agent_responses[-1].solution
            else:
                # Fallback if no responses (shouldn't happen)
                final_solution = "(no solution generated)"

            solutions.append(final_solution)

        # Create comparison pairs
        comparison_pairs = get_pairs(len(solutions), self.tournament_pattern)

        # Log comparison setup
        with logtree.scope_header("Tournament Comparison"):
            logtree.log_text(
                f"Got {len(solutions)} solutions from {self.num_agents} agents, "
                f"doing {len(comparison_pairs)} pairwise matchups."
            )

        # Execute all pairwise comparisons
        comparison_results = []
        for i, j in comparison_pairs:
            score = await self.self_comparison_judge.compare(
                query=self.query, solution_A=solutions[i], solution_B=solutions[j]
            )
            comparison_results.append(score)

            # Log individual comparison
            preference_str = "A wins" if score < -0.5 else "B wins" if score > 0.5 else "Tie"
            with logtree.scope_details(f"Matchup {i} vs {j}"):
                logtree.log_text(f"Result: {preference_str} (score: {score:.2f})")

        # Accumulate win-loss scores (same as preference_envs.py lines 196-213)
        win_minus_loss_list = [0.0 for _ in range(len(solutions))]
        matchup_count = [0 for _ in range(len(solutions))]

        for (i, j), j_reward in zip(comparison_pairs, comparison_results):
            win_minus_loss_list[j] += j_reward
            win_minus_loss_list[i] -= j_reward
            matchup_count[j] += 1
            matchup_count[i] += 1

        # Return normalized rewards
        rewards_and_metrics = []
        for agent_id, (win_loss, count) in enumerate(zip(win_minus_loss_list, matchup_count)):
            normalized_reward = win_loss / count if count > 0 else 0.0
            metrics = {
                "agent_id": agent_id,
                "win_minus_loss": normalized_reward,
                "matchups": count,
            }
            rewards_and_metrics.append((normalized_reward, metrics))

        return rewards_and_metrics

    def logging_tags(self) -> list[str]:
        return ["multi_agent_conv"]


# ============================================================================
# MultiAgentConversationDataset - Dataset of prompts
# ============================================================================


class MultiAgentConversationDataset(RLDataset):
    """Dataset that yields conversation prompts for multi-agent RL."""

    def __init__(
        self,
        prompts: list[str],
        num_agents: int,
        max_rounds: int,
        renderer: Renderer,
        batch_size: int,
        self_comparison_judge: PolicySelfComparisonJudge,
        max_tokens: int = 512,
    ):
        self.prompts = prompts
        self.num_agents = num_agents
        self.max_rounds = max_rounds
        self.renderer = renderer
        self.batch_size = batch_size
        self.judge = self_comparison_judge
        self.max_tokens = max_tokens

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Return batch of environment group builders.

        Each group builder handles ONE conversation (N agents).
        batch_size should be divisible by num_agents.
        """
        # Each conversation involves num_agents trajectories
        num_conversations = self.batch_size // self.num_agents
        start_idx = index * num_conversations
        end_idx = min(start_idx + num_conversations, len(self.prompts))

        return [
            MultiAgentConversationEnvGroupBuilder(
                query=self.prompts[i],
                num_agents=self.num_agents,
                max_rounds=self.max_rounds,
                renderer=self.renderer,
                self_comparison_judge=self.judge,
                max_tokens=self.max_tokens,
            )
            for i in range(start_idx, end_idx)
        ]

    def __len__(self) -> int:
        return (len(self.prompts) * self.num_agents) // self.batch_size


# ============================================================================
# MultiAgentConversationDatasetBuilder - Configurable builder
# ============================================================================


@chz.chz
class MultiAgentConversationDatasetBuilder(RLDatasetBuilder):
    """Builder for multi-agent conversation RL dataset.

    Supports loading prompts from:
    - Direct lists (train_prompts, test_prompts)
    - HuggingFace datasets (hf_dataset_name)
    - JSONL files (via external loading)
    """

    batch_size: int
    num_agents: int = 3
    max_rounds: int = 5
    model_name: str
    renderer_name: str
    max_tokens: int = 512

    # Prompt sources
    train_prompts: list[str] = field(default_factory=list)
    test_prompts: list[str] = field(default_factory=list)

    # HuggingFace dataset loading (optional)
    hf_dataset_name: str | None = None
    hf_dataset_name_config: str | None = None
    hf_prompt_column: str = "query"
    hf_train_split: str = "train"
    hf_test_split: str = "test"
    max_train_samples: int | None = None
    max_test_samples: int | None = None

    async def __call__(
        self,
    ) -> tuple[MultiAgentConversationDataset, MultiAgentConversationDataset | None]:
        """Build training and test datasets."""
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))

        # Load prompts from HF dataset if specified
        train_prompts = self.train_prompts
        test_prompts = self.test_prompts

        if self.hf_dataset_name:
            from datasets import load_dataset

            dataset = load_dataset(self.hf_dataset_name, self.hf_dataset_name_config or None)

            if self.hf_train_split in dataset:
                train_data = dataset[self.hf_train_split]
                train_prompts = [ex[self.hf_prompt_column] for ex in train_data]
                if self.max_train_samples:
                    train_prompts = train_prompts[: self.max_train_samples]

            if self.hf_test_split in dataset:
                test_data = dataset[self.hf_test_split]
                test_prompts = [ex[self.hf_prompt_column] for ex in test_data]
                if self.max_test_samples:
                    test_prompts = test_prompts[: self.max_test_samples]

        # Create judge (uses current policy)
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(base_model=self.model_name)
        judge = PolicySelfComparisonJudge(sampling_client=sampling_client, renderer=renderer)

        # Build datasets
        train_dataset = MultiAgentConversationDataset(
            prompts=train_prompts,
            num_agents=self.num_agents,
            max_rounds=self.max_rounds,
            renderer=renderer,
            batch_size=self.batch_size,
            self_comparison_judge=judge,
            max_tokens=self.max_tokens,
        )

        test_dataset = None
        if test_prompts:
            test_dataset = MultiAgentConversationDataset(
                prompts=test_prompts,
                num_agents=self.num_agents,
                max_rounds=self.max_rounds,
                renderer=renderer,
                batch_size=min(len(test_prompts) * self.num_agents, 32),
                self_comparison_judge=judge,
                max_tokens=self.max_tokens,
            )

        return train_dataset, test_dataset
