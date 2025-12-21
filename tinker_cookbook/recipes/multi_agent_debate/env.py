"""Multi-agent debate environment for tinker RL."""

import asyncio
from dataclasses import dataclass, field
from typing import Sequence

import chz
import tinker
from tinker import types

from tinker_cookbook.completers import StopCondition, TinkerMessageCompleter
from tinker_cookbook.renderers import Message, Renderer, ensure_text, get_renderer
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

from .prompts import (
    AGENT_SYSTEM_PROMPT,
    ParsedResponse,
    parse_agent_response,
)
from .reward import compute_pairwise_win_rates

# Stop when we see the closing tag of the last required field
STOP_CONDITION = ["</consensus_reason>"]
DEFAULT_MAX_ROUNDS = 3


def _get_debate_stop_condition(renderer: Renderer) -> StopCondition:
    """
    Pick stop sequences that are compatible with the active renderer.

    - Token-based renderers (e.g. `llama3`) return `list[int]` stop tokens; we must not mix in strings.
    - Text-based renderers (e.g. `role_colon`) return `list[str]` stop sequences; we can add debate-specific tags.
    """
    renderer_stop = renderer.get_stop_sequences()
    if not renderer_stop:
        return STOP_CONDITION
    if isinstance(renderer_stop[0], int):
        return renderer_stop
    # De-duplicate while preserving order
    return list(dict.fromkeys([*renderer_stop, *STOP_CONDITION]))


@dataclass
class ConversationState:
    """Shared state for a multi-agent conversation."""

    question: str
    num_agents: int
    max_rounds: int = DEFAULT_MAX_ROUNDS
    current_turn: int = 0  # Global turn counter
    current_agent_id: int = 0  # Which agent's turn it is
    agent_responses: list[list[ParsedResponse]] = field(default_factory=list)  # [turn][agent_id]
    done: bool = False
    consensus_reached: bool = False

    def get_current_round(self) -> int:
        """Get the current round number (each round = all agents take one turn)."""
        return self.current_turn // self.num_agents

    def get_completed_rounds(self) -> int:
        """Number of fully completed rounds so far."""
        if not self.agent_responses:
            return 0
        # All rounds except possibly the last are completed; the last is completed iff it has num_agents.
        completed = len(self.agent_responses) - 1
        if len(self.agent_responses[-1]) == self.num_agents:
            completed += 1
        return completed

    def advance_turn(self, response: ParsedResponse) -> None:
        """Advance to the next turn after an agent responds."""
        # Store response
        if self.current_turn % self.num_agents == 0:
            # Start of a new round
            self.agent_responses.append([])

        self.agent_responses[-1].append(response)

        # Move to next agent
        self.current_turn += 1
        self.current_agent_id = self.current_turn % self.num_agents

        # Check if we should end
        if self.current_turn % self.num_agents == 0:
            # Completed a full round, check consensus
            latest_round = self.agent_responses[-1]
            if all(resp.consensus_reached for resp in latest_round):
                self.consensus_reached = True
                self.done = True
            elif self.get_current_round() >= self.max_rounds:
                self.done = True


class MultiAgentCoordinator:
    """Coordinates a multi-agent debate conversation."""

    def __init__(self, question: str, num_agents: int, max_rounds: int = DEFAULT_MAX_ROUNDS):
        self.state = ConversationState(question=question, num_agents=num_agents, max_rounds=max_rounds)
        self.condition = asyncio.Condition()

    @property
    def done(self) -> bool:
        return self.state.done

    @property
    def current_agent_id(self) -> int:
        return self.state.current_agent_id

    async def wait_for_turn(self, agent_id: int) -> None:
        """Wait until it's this agent's turn."""
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.state.current_agent_id == agent_id or self.state.done
            )

    async def submit_response(self, agent_id: int, response: str) -> ParsedResponse:
        """Submit an agent's response and advance the conversation."""
        async with self.condition:
            if self.state.current_agent_id != agent_id:
                raise ValueError(
                    f"Not agent {agent_id}'s turn (current: {self.state.current_agent_id})"
                )

            # Parse the response
            parsed = parse_agent_response(response)

            # Advance state
            self.state.advance_turn(parsed)

            # Notify waiting agents
            self.condition.notify_all()

            return parsed

    async def abort(self) -> None:
        """End the episode early and release any waiting agents."""
        async with self.condition:
            self.state.done = True
            self.condition.notify_all()


@dataclass
class MultiAgentDebateEnv(Env):
    """Environment for one agent in a multi-agent debate."""

    agent_id: int
    coordinator: MultiAgentCoordinator
    renderer: Renderer
    self_play: bool = True
    opponent_policies: list[TinkerMessageCompleter] | None = None
    history_rounds: int = 2
    max_chars_per_field: int = 800

    def __post_init__(self):
        if self.self_play:
            assert self.opponent_policies is None, "self_play=True requires opponent_policies=None"
        else:
            assert (
                self.opponent_policies is not None
                and len(self.opponent_policies) == self.coordinator.state.num_agents - 1
            ), "Need N-1 opponent policies for non-self-play"

    @property
    def stop_condition(self) -> StopCondition:
        return _get_debate_stop_condition(self.renderer)

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return AGENT_SYSTEM_PROMPT.format(agent_id=self.agent_id)

    def get_conversation_history(self) -> str:
        """Format the conversation history for this agent."""
        if not self.coordinator.state.agent_responses:
            return ""

        def _clip(s: str) -> str:
            s = s.strip()
            if len(s) <= self.max_chars_per_field:
                return s
            return s[: self.max_chars_per_field].rstrip() + "…"

        lines = [f"Question: {self.coordinator.state.question}\n"]
        rounds = self.coordinator.state.agent_responses
        start_idx = max(0, len(rounds) - self.history_rounds)
        for round_idx, round_responses in enumerate(rounds[start_idx:], start=start_idx):
            lines.append(f"--- Round {round_idx + 1} ---")
            for agent_id, response in enumerate(round_responses):
                lines.append(f"\nAgent {agent_id}:")
                lines.append(f"Solution: {_clip(response.solution)}")
                lines.append(f"Evaluation: {_clip(response.evaluation)}")
                consensus_status = "YES" if response.consensus_reached else "NO"
                lines.append(f"Consensus: {consensus_status} - {_clip(response.consensus_reason)}")
            lines.append("")

        return "\n".join(lines)

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Get the initial observation for this agent."""
        if self.agent_id != 0:
            await self.wait_for_turn()
        return self.get_observation(), self.stop_condition

    async def wait_for_turn(self) -> None:
        """Wait until it's this agent's turn."""
        if not self.coordinator.done:
            if self.self_play:
                await self.coordinator.wait_for_turn(self.agent_id)
            else:
                # Run opponent policies until it's our turn
                await self.run_opponent_steps()

    async def run_opponent_steps(self) -> None:
        """Run opponent policies until it's this agent's turn."""
        assert not self.self_play and self.opponent_policies is not None

        while not self.coordinator.done and self.coordinator.current_agent_id != self.agent_id:
            opponent_agent_id = self.coordinator.current_agent_id
            # Map opponent agent ID to policy index (skip our own ID)
            policy_idx = (
                opponent_agent_id if opponent_agent_id < self.agent_id else opponent_agent_id - 1
            )

            opponent_policy = self.opponent_policies[policy_idx]
            observation_str = self.get_observation_string()

            # Get opponent response
            messages: list[Message] = [
                {
                    "role": "system",
                    "content": AGENT_SYSTEM_PROMPT.format(agent_id=opponent_agent_id),
                },
                {"role": "user", "content": observation_str},
            ]
            opponent_response = await opponent_policy(messages)
            opponent_content = ensure_text(opponent_response["content"])

            # Submit to coordinator
            try:
                await self.coordinator.submit_response(opponent_agent_id, opponent_content)
            except ValueError:
                await self.coordinator.abort()
                return

    def get_observation_string(self) -> str:
        """Get the observation as a string."""
        round_idx = self.coordinator.state.get_current_round()
        completed_rounds = self.coordinator.state.get_completed_rounds()
        history = self.get_conversation_history()

        if completed_rounds == 0:
            return (
                f"Question: {self.coordinator.state.question}\n\n"
                f"Round {round_idx + 1} of {self.coordinator.state.max_rounds}.\n"
                "First round: propose your solution. Set <evaluation> to N/A and <comparison> to N/A or empty."
            )

        return (
            f"{history}\n\n"
            f"Round {round_idx + 1} of {self.coordinator.state.max_rounds}.\n"
            "Use the most recently COMPLETED round as the basis for <evaluation> and <comparison>."
        )

    def get_observation(self) -> types.ModelInput:
        """Get the current observation for this agent."""
        observation_str = self.get_observation_string()
        messages: list[Message] = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": observation_str},
        ]
        return self.renderer.build_generation_prompt(messages)

    async def step(self, action: Action) -> StepResult:
        """Take a step in the environment."""
        if self.coordinator.done:
            return self.get_done_step()

        # Parse and submit response
        action_message: Message = self.renderer.parse_response(action)[0]
        action_content = ensure_text(action_message["content"])

        try:
            await self.coordinator.submit_response(self.agent_id, action_content)
        except ValueError:
            # Failed to parse or wrong turn: abort the episode to avoid deadlocking other agents.
            await self.coordinator.abort()
            return StepResult(
                reward=-1.0,  # Penalty for malformed response
                episode_done=True,
                next_observation=types.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={"parse_error": 1.0},
            )

        # Wait for next turn
        await self.wait_for_turn()
        # Per-step reward is 0 by default; the main learning signal is computed as a final group
        # reward by the EnvGroupBuilder using all pairwise comparisons.
        return StepResult(
            reward=0.0,
            episode_done=self.coordinator.done,
            next_observation=self.get_observation(),
            next_stop_condition=self.stop_condition,
            metrics={
                "consensus_reached": 1.0 if self.coordinator.state.consensus_reached else 0.0,
                "round": float(self.coordinator.state.get_current_round()),
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
class MultiAgentEnvGroupBuilder(EnvGroupBuilder):
    """Builder for groups of multi-agent debate environments."""

    questions: list[str]  # List of questions for the dataset
    question_index: int  # Which question to use for this group
    renderer: Renderer
    num_agents: int
    self_play: bool
    max_rounds: int = DEFAULT_MAX_ROUNDS
    opponent_policies: list[TinkerMessageCompleter] | None = None
    model_name: str | None = None
    base_url: str | None = None

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of environments for a multi-agent debate."""
        question = self.questions[self.question_index % len(self.questions)]

        if self.self_play:
            # All agents share the same coordinator
            coordinator = MultiAgentCoordinator(
                question=question, num_agents=self.num_agents, max_rounds=self.max_rounds
            )
            return [
                MultiAgentDebateEnv(
                    agent_id=i,
                    coordinator=coordinator,
                    renderer=self.renderer,
                    self_play=True,
                    opponent_policies=None,
                )
                for i in range(self.num_agents)
            ]
        else:
            # Each agent gets its own coordinator with opponent policies
            envs = []
            for i in range(self.num_agents):
                coordinator = MultiAgentCoordinator(
                    question=question, num_agents=self.num_agents, max_rounds=self.max_rounds
                )
                envs.append(
                    MultiAgentDebateEnv(
                        agent_id=i,
                        coordinator=coordinator,
                        renderer=self.renderer,
                        self_play=False,
                        opponent_policies=self.opponent_policies,
                    )
                )
            return envs

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        if not self.self_play:
            return [(0.0, {}) for _ in trajectory_group]

        assert env_group, "empty env_group"
        env0 = env_group[0]
        assert isinstance(env0, MultiAgentDebateEnv)
        coordinator = env0.coordinator

        rewards_G, summary_metrics = compute_pairwise_win_rates(
            coordinator.state.agent_responses, num_agents=self.num_agents
        )
        rewards_and_metrics = []
        for agent_id, reward in enumerate(rewards_G):
            rewards_and_metrics.append(
                (
                    reward,
                    {
                        "agent_id": agent_id,
                        "pairwise_win_rate": reward,
                        **summary_metrics,
                    },
                )
            )
        return rewards_and_metrics


class MultiAgentDebateDataset(RLDataset):
    """Dataset for multi-agent debate environments."""

    def __init__(
        self,
        batch_size: int,
        questions: list[str],
        num_agents: int,
        renderer: Renderer,
        self_play: bool,
        max_rounds: int,
        num_datapoints: int,
        model_name: str,
        base_url: str | None,
        opponent_policies: list[TinkerMessageCompleter] | None = None,
    ):
        self.batch_size = batch_size
        self.questions = questions
        self.num_agents = num_agents
        self.renderer = renderer
        self.self_play = self_play
        self.max_rounds = max_rounds
        self.num_datapoints = num_datapoints
        self.model_name = model_name
        self.base_url = base_url
        self.opponent_policies = opponent_policies

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of environment group builders."""
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, self.num_datapoints)
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            MultiAgentEnvGroupBuilder(
                questions=self.questions,
                question_index=question_index,
                renderer=self.renderer,
                num_agents=self.num_agents,
                self_play=self.self_play,
                max_rounds=self.max_rounds,
                model_name=self.model_name,
                base_url=self.base_url,
                opponent_policies=self.opponent_policies,
            )
            for question_index in range(batch_start, batch_end)
        ]

    def __len__(self) -> int:
        # Number of batches of environment groups.
        return (self.num_datapoints + self.batch_size - 1) // self.batch_size


@chz.chz
class MultiAgentDebateDatasetBuilder(RLDatasetBuilder):
    """Builder for multi-agent debate datasets."""

    batch_size: int
    num_train_datapoints: int
    num_test_datapoints: int
    num_agents: int = 3
    max_rounds: int = DEFAULT_MAX_ROUNDS
    base_url: str | None = None
    model_name: str
    renderer_name: str
    # Prompt source: local JSONL by default (no network).
    dataset_path: str = "tinker_cookbook/example_data/nonverifiable_queries.jsonl"
    dataset_field: str = "query"

    # Optional HF dataset loading (requires network).
    hf_dataset_name: str | None = None
    hf_dataset_subset: str | None = None
    hf_dataset_split: str = "train"
    hf_dataset_question_field: str = "question"
    max_questions: int = 1000

    def _load_questions_from_file(self) -> list[str]:
        import json

        questions: list[str] = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= self.max_questions:
                    break
                data = json.loads(line)
                if self.dataset_field not in data:
                    raise ValueError(
                        f"Each JSONL row must include '{self.dataset_field}'. Got keys={list(data.keys())}"
                    )
                questions.append(str(data[self.dataset_field]))
        if not questions:
            raise ValueError(f"No questions loaded from {self.dataset_path}")
        return questions

    def _load_questions_from_hf(self) -> list[str]:
        """Load questions from HuggingFace dataset."""
        from datasets import load_dataset

        print(
            f"Loading questions from HF dataset {self.hf_dataset_name}, "
            f"split {self.hf_dataset_split}, subset {self.hf_dataset_subset}..."
        )
        if self.hf_dataset_subset is not None:
            dataset = load_dataset(
                self.hf_dataset_name,
                self.hf_dataset_subset,
                split=self.hf_dataset_split,
            )
        else:
            dataset = load_dataset(
                self.hf_dataset_name,
                split=self.hf_dataset_split,
            )

        questions = []
        for i, example in enumerate(dataset):
            if i >= self.max_questions:
                break
            if self.hf_dataset_question_field in example:
                questions.append(str(example[self.hf_dataset_question_field]))

        if not questions:
            raise ValueError(
                f"No questions found in dataset {self.hf_dataset_name} "
                f"with field {self.hf_dataset_question_field}"
            )

        return questions

    def _construct_opponent_policies(self, renderer: Renderer) -> list[TinkerMessageCompleter]:
        """Create fixed opponent policies for testing (using base model)."""
        service_client = tinker.ServiceClient(base_url=self.base_url)
        sampling_client = service_client.create_sampling_client(base_model=self.model_name)

        # Create N-1 opponent policies (all using same base model)
        return [
            TinkerMessageCompleter(
                sampling_client=sampling_client,
                renderer=renderer,
                max_tokens=512,
                stop_condition=_get_debate_stop_condition(renderer),
            )
            for _ in range(self.num_agents - 1)
        ]

    async def __call__(
        self,
    ) -> tuple[MultiAgentDebateDataset, MultiAgentDebateDataset | None]:
        """Build the dataset for training and testing."""
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        if self.hf_dataset_name is not None:
            questions = self._load_questions_from_hf()
        else:
            questions = self._load_questions_from_file()

        # Training dataset (self-play)
        train_dataset = MultiAgentDebateDataset(
            batch_size=self.batch_size,
            questions=questions,
            num_agents=self.num_agents,
            renderer=renderer,
            self_play=True,
            max_rounds=self.max_rounds,
            num_datapoints=self.num_train_datapoints,
            model_name=self.model_name,
            base_url=self.base_url,
            opponent_policies=None,
        )

        # Test dataset (against fixed base model)
        test_dataset = MultiAgentDebateDataset(
            batch_size=min(self.num_test_datapoints, self.batch_size),
            questions=questions,
            num_agents=self.num_agents,
            renderer=renderer,
            # Keep the default test set self-play so the RL test evaluator reports the
            # same reward signal as training, just on held-out prompts.
            self_play=True,
            max_rounds=self.max_rounds,
            num_datapoints=self.num_test_datapoints,
            model_name=self.model_name,
            base_url=self.base_url,
            opponent_policies=None,
        )

        return train_dataset, test_dataset
