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
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .prompts import (
    AGENT_SYSTEM_PROMPT,
    ParsedResponse,
    parse_agent_response,
)

# Stop when we see the closing tag of the last required field
STOP_CONDITION = ["</consensus_reason>"]
MAX_TURNS_PER_AGENT = 10


@dataclass
class ConversationState:
    """Shared state for a multi-agent conversation."""

    question: str
    num_agents: int
    current_turn: int = 0  # Global turn counter
    current_agent_id: int = 0  # Which agent's turn it is
    agent_responses: list[list[ParsedResponse]] = field(default_factory=list)  # [turn][agent_id]
    done: bool = False
    consensus_reached: bool = False

    def get_current_round(self) -> int:
        """Get the current round number (each round = all agents take one turn)."""
        return self.current_turn // self.num_agents

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
            elif self.get_current_round() >= MAX_TURNS_PER_AGENT:
                # Hit max turns
                self.done = True


class MultiAgentCoordinator:
    """Coordinates a multi-agent debate conversation."""

    def __init__(self, question: str, num_agents: int):
        self.state = ConversationState(question=question, num_agents=num_agents)
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


@dataclass
class MultiAgentDebateEnv(Env):
    """Environment for one agent in a multi-agent debate."""

    agent_id: int
    coordinator: MultiAgentCoordinator
    renderer: Renderer
    self_play: bool
    opponent_policies: list[TinkerMessageCompleter] | None = None

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
        return STOP_CONDITION

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return AGENT_SYSTEM_PROMPT.format(agent_id=self.agent_id)

    def get_conversation_history(self) -> str:
        """Format the conversation history for this agent."""
        if not self.coordinator.state.agent_responses:
            return ""

        lines = [f"Question: {self.coordinator.state.question}\n"]
        for round_idx, round_responses in enumerate(self.coordinator.state.agent_responses):
            lines.append(f"--- Round {round_idx + 1} ---")
            for agent_id, response in enumerate(round_responses):
                lines.append(f"\nAgent {agent_id}:")
                lines.append(f"Solution: {response.solution}")
                lines.append(f"Evaluation: {response.evaluation}")
                consensus_status = "YES" if response.consensus_reached else "NO"
                lines.append(f"Consensus: {consensus_status} - {response.consensus_reason}")
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
            await self.coordinator.submit_response(opponent_agent_id, opponent_content)

    def get_observation_string(self) -> str:
        """Get the observation as a string."""
        history = self.get_conversation_history()
        if not history:
            return f"Question: {self.coordinator.state.question}\n\nYou are the first to respond. Provide your initial solution, evaluation (N/A for first turn), and consensus assessment."
        else:
            return f"{history}\n\nIt's your turn. Provide your solution, evaluation of others' recent contributions, and consensus assessment."

    def get_observation(self) -> types.ModelInput:
        """Get the current observation for this agent."""
        if self.coordinator.done:
            return types.ModelInput.empty()

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
            # Failed to parse or wrong turn
            return StepResult(
                reward=-1.0,  # Penalty for malformed response
                episode_done=True,
                next_observation=types.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={"parse_error": 1.0},
            )

        # Wait for next turn
        await self.wait_for_turn()

        # Compute reward
        # we are at the start of the next turn (or end of the episode), we should calculate the reward for the last turn
        # should we waited, other agents have now spoken and provided their <comparison> votes
        peer_reward = self.compute_reward_from_comparisons()

        outcome_reward = 0.0
        if self.coordinator.state.consensus_reached:
            outcome_reward = 1.0  # Reward for reaching consensus

        reward = peer_reward + outcome_reward

        return StepResult(
            reward=reward,
            episode_done=self.coordinator.done,
            next_observation=self.get_observation(),
            next_stop_condition=self.stop_condition,
            metrics={
                "consensus_reached": 1.0 if self.coordinator.state.consensus_reached else 0.0,
                "num_rounds": float(self.coordinator.state.get_current_round()),
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

    async def compute_reward_from_comparisons(self) -> float:
        """Compute reward based on pairwise comparisons of the last K turns.

        Returns:
            Win-rate for this agent (proportion of pairwise comparisons won)
        """
        responses = self.coordinator.state.agent_responses
        num_agents = self.coordinator.state.num_agents

        target_round_idx = -1
        for r_idx in range(len(responses) - 1, -1, -1):
            if len(responses[r_idx]) > self.agent_id:
                target_round_idx = r_idx
                break

        if target_round_idx < 0:
            return 0.0

        # We look for votes in the turns that followed my submission.
        # Who votes? Everyone else.
        # Where are their votes?
        # 1. Agents who spoke AFTER me in the SAME round (target_round_idx).
        # 2. Agents who spoke BEFORE me in the NEXT round (target_round_idx + 1).

        wins = 0
        total_comparisons = 0

        for other_id in range(num_agents):
            if other_id == self.agent_id:
                continue

            vote_source_response = None

            # Case 1: They spoke after me in the same round
            if other_id > self.agent_id:
                if len(responses[target_round_idx]) > other_id:
                    vote_source_response = responses[target_round_idx][other_id]

            # Case 2: They spoke before me in the next round
            # (Note: responses might not have the next round yet if we are the last agent and game ended,
            #  but usually step() waits until our NEXT turn, so next round exists partially)
            elif other_id < self.agent_id:
                if len(responses) > target_round_idx + 1:
                    if len(responses[target_round_idx + 1]) > other_id:
                        vote_source_response = responses[target_round_idx + 1][other_id]

            if vote_source_response:
                # Check their comparisons
                for agent_a, op, agent_b in vote_source_response.comparisons:
                    if agent_a == self.agent_id or agent_b == self.agent_id:
                        total_comparisons += 1

                        if op == ">":
                            if agent_a == self.agent_id:
                                wins += 1
                        elif op == "=":
                            # Tie
                            wins += 0.5

        if total_comparisons == 0:
            return 0.0

        return wins / total_comparisons


@dataclass
class MultiAgentEnvGroupBuilder(EnvGroupBuilder):
    """Builder for groups of multi-agent debate environments."""

    questions: list[str]  # List of questions for the dataset
    question_index: int  # Which question to use for this group
    renderer: Renderer
    num_agents: int
    self_play: bool
    opponent_policies: list[TinkerMessageCompleter] | None = None
    model_name: str | None = None
    base_url: str | None = None

    def set_comparison_sampling_client(self, sampling_client: tinker.SamplingClient) -> None:
        """Use the current policy's sampling client for pairwise comparisons."""
        self.comparison_sampling_client = sampling_client

    def _build_comparison_completer(self) -> TinkerMessageCompleter:
        sampling_client = self.comparison_sampling_client
        if sampling_client is None:
            # Fallback to a fresh client from the base model if no policy client was provided.
            service_client = tinker.ServiceClient(base_url=self.base_url)
            if self.model_name is None:
                raise ValueError("model_name must be set to create comparison sampling client")
            sampling_client = service_client.create_sampling_client(base_model=self.model_name)
            self.comparison_sampling_client = sampling_client
        return TinkerMessageCompleter(
            sampling_client=sampling_client,
            renderer=self.renderer,
            max_tokens=128,
            stop_condition=None,
        )

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of environments for a multi-agent debate."""
        question = self.questions[self.question_index % len(self.questions)]

        if self.self_play:
            # All agents share the same coordinator
            coordinator = MultiAgentCoordinator(question=question, num_agents=self.num_agents)
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
                coordinator = MultiAgentCoordinator(question=question, num_agents=self.num_agents)
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


class MultiAgentDebateDataset(RLDataset):
    """Dataset for multi-agent debate environments."""

    def __init__(
        self,
        batch_size: int,
        questions: list[str],
        num_agents: int,
        renderer: Renderer,
        self_play: bool,
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
        self.num_datapoints = num_datapoints
        self.model_name = model_name
        self.base_url = base_url
        self.opponent_policies = opponent_policies

        assert self.num_datapoints % self.num_agents == 0, (
            "num_datapoints must be divisible by num_agents"
        )

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of environment group builders."""
        builders = []
        for i in range(self.batch_size // self.num_agents):
            question_index = index * (self.batch_size // self.num_agents) + i
            if question_index * self.num_agents < self.num_datapoints:
                builders.append(
                    MultiAgentEnvGroupBuilder(
                        questions=self.questions,
                        question_index=question_index,
                        renderer=self.renderer,
                        num_agents=self.num_agents,
                        self_play=self.self_play,
                        model_name=self.model_name,
                        base_url=self.base_url,
                        opponent_policies=self.opponent_policies,
                    )
                )
        return builders

    def __len__(self) -> int:
        return self.num_datapoints // self.batch_size


@chz.chz
class MultiAgentDebateDatasetBuilder(RLDatasetBuilder):
    """Builder for multi-agent debate datasets."""

    batch_size: int
    num_train_datapoints: int
    num_test_datapoints: int
    num_agents: int = 3
    base_url: str | None = None
    model_name: str
    renderer_name: str
    hf_dataset_name: str = "lighteval/mmlu"  # Default dataset
    hf_dataset_subset: str | None = "all"  # e.g., "all" or "abstract_algebra"
    hf_dataset_split: str = "test"  # e.g., "train", "test", or "validation"
    hf_dataset_question_field: str = "question"  # Field name for questions
    max_questions: int = 1000  # Limit number of questions to load

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
                stop_condition=STOP_CONDITION,
            )
            for _ in range(self.num_agents - 1)
        ]

    async def __call__(
        self,
    ) -> tuple[MultiAgentDebateDataset, MultiAgentDebateDataset | None]:
        """Build the dataset for training and testing."""
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        questions = self._load_questions_from_hf()

        # Training dataset (self-play)
        train_dataset = MultiAgentDebateDataset(
            batch_size=self.batch_size,
            questions=questions,
            num_agents=self.num_agents,
            renderer=renderer,
            self_play=True,
            num_datapoints=self.num_train_datapoints,
            model_name=self.model_name,
            base_url=self.base_url,
            opponent_policies=None,
        )

        # Test dataset (against fixed base model)
        test_dataset = MultiAgentDebateDataset(
            batch_size=self.num_test_datapoints,
            questions=questions,
            num_agents=self.num_agents,
            renderer=renderer,
            self_play=False,
            num_datapoints=self.num_test_datapoints,
            model_name=self.model_name,
            base_url=self.base_url,
            opponent_policies=self._construct_opponent_policies(renderer),
        )

        return train_dataset, test_dataset
