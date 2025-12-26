"""Multi-agent debate environment for tinker RL."""

import asyncio
import random
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
from tinker_cookbook.utils import logtree

from .prompts import (
    AGENT_SYSTEM_PROMPT,
    ParsedResponse,
    parse_agent_response,
)
from .reward import (
    compute_pairwise_win_minus_loss,
    compute_pairwise_win_rates,
)

# Stop when we see the closing tag of the last required field
STOP_CONDITION = ["</comparison>"]


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


def _get_summarizer_stop_condition(renderer: Renderer) -> StopCondition:
    renderer_stop = renderer.get_stop_sequences()
    return renderer_stop or []


@dataclass
class ConversationState:
    """Shared state for a multi-agent conversation."""

    question: str
    num_agents: int
    max_turns: int
    current_turn: int = 0  # Global turn counter
    current_agent_id: int = 0  # Which agent's turn it is
    agent_responses: list[ParsedResponse] = field(default_factory=list)  # [turn]
    done: bool = False

    def get_current_cycle(self) -> int:
        """Get the current cycle index (each cycle = all agents take one turn)."""
        return self.current_turn // self.num_agents

    def advance_turn(self, response: ParsedResponse) -> None:
        """Advance to the next turn after an agent responds."""
        self.agent_responses.append(response)

        # Move to next agent
        self.current_turn += 1
        self.current_agent_id = self.current_turn % self.num_agents

        if self.current_turn >= self.max_turns:
            self.done = True


class MultiAgentCoordinator:
    """Coordinates a multi-agent debate conversation."""

    def __init__(self, question: str, num_agents: int, max_turns: int):
        self.state = ConversationState(
            question=question, num_agents=num_agents, max_turns=max_turns
        )
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

    async def submit_response(
        self,
        agent_id: int,
        response: str,
        *,
        observation: str = "",
    ) -> ParsedResponse:
        """Submit an agent's response and advance the conversation."""
        async with self.condition:
            if self.state.current_agent_id != agent_id:
                raise ValueError(
                    f"Not agent {agent_id}'s turn (current: {self.state.current_agent_id})"
                )

            # Parse the response
            parsed = parse_agent_response(
                response,
                author_id=agent_id,
                observation=observation,
            )

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
    history_turns: int = 2
    summarize_history: bool = False
    summarize_model: str | None = "Qwen/Qwen3-4B-Instruct-2507"
    model_name: str | None = None
    base_url: str | None = None

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

    def _format_turns(self, turns: list[ParsedResponse]) -> str:
        if not turns:
            return ""
        lines: list[str] = []
        for turn_idx, response in enumerate(turns, start=1):
            lines.append(f"--- Turn {turn_idx} (Agent {response.author_id}) ---")
            lines.append("Solution:")
            lines.append(response.solution.rstrip())
            lines.append("Evaluation:")
            lines.append(response.evaluation.rstrip())
            if response.comparison_text:
                lines.append("Comparison:")
                lines.append(response.comparison_text.rstrip())
            lines.append("")
        return "\n".join(lines).rstrip()

    async def _summarize(self, history: str) -> str:
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(
            base_model=self.summarize_model or self.model_name
        )
        # init the summarizer policy, sample from tinker message completer
        self._summarizer_policy = TinkerMessageCompleter(
            sampling_client=sampling_client,
            renderer=self.renderer,
            max_tokens=892,
            stop_condition=_get_summarizer_stop_condition(self.renderer),
        )
        messages: list[Message] = [
            {
                "role": "system",
                "content": (
                    "You summarize multi-agent debate transcripts.\n"
                    "Write a concise, information-dense summary that preserves:\n"
                    "- The user question\n"
                    "- Each agent's key solution ideas\n"
                    "- Each agent's critiques/evaluations of others (including meta-evaluation)\n"
                    "- Any explicit comparisons (e.g. Agent 1 > Agent 0)\n"
                    "Do not add new information. Output plain text only."
                ),
            },
            {"role": "user", "content": history},
        ]
        resp = await self._summarizer_policy(messages)
        return ensure_text(resp["content"]).strip()

    async def get_conversation_context(self) -> str:
        """Format the conversation context for this agent."""
        if not self.coordinator.state.agent_responses:
            return ""

        question = self.coordinator.state.question
        turns = list(self.coordinator.state.agent_responses)

        # using full history without summarization
        if not self.summarize_history:
            return f"Question: {question}\n\n{self._format_turns(turns)}".rstrip()

        # using summarization for older turns
        if self.history_turns < 0:
            raw_recent = turns
            older: list[ParsedResponse] = []
        elif self.history_turns == 0:
            raw_recent = []
            older = turns
        else:
            raw_recent = turns[-self.history_turns :] if len(turns) > self.history_turns else turns
            older = turns[: -self.history_turns] if len(turns) > self.history_turns else []

        summary = ""
        if older:
            older_text = f"Question: {question}\n\n{self._format_turns(older)}"
            summary = await self._summarize(older_text)

        recent_text = self._format_turns(raw_recent)
        parts: list[str] = [f"Question: {question}"]
        if summary:
            parts.append("\n--- Summary of earlier turns ---\n" + summary)
        if recent_text:
            parts.append("\n--- Recent turns (verbatim) ---\n" + recent_text)
        return "\n".join(parts).rstrip()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Get the initial observation for this agent."""
        if self.agent_id != 0:
            await self.wait_for_turn()
        return await self.get_observation(), self.stop_condition

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
            observation_str = await self.get_observation_string()

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
                await self.coordinator.submit_response(
                    opponent_agent_id,
                    opponent_content,
                    observation=observation_str,
                )
            except ValueError:
                await self.coordinator.abort()
                return

    async def get_observation_string(self) -> str:
        """Get the observation string for this agent."""
        history = await self.get_conversation_context()
        turn_idx = self.coordinator.state.current_turn
        cycle_idx = self.coordinator.state.get_current_cycle()
        max_cycles = self.coordinator.state.max_turns // self.coordinator.state.num_agents

        # intermission prompt for first turn
        if turn_idx == 0:
            return (
                f"Question: {self.coordinator.state.question}\n\n"
                f"Cycle {cycle_idx + 1} of {max_cycles}, Turn {turn_idx + 1}.\n"
                "First completion: propose your solution.\n"
                'Set <evaluation> to "N/A" and <comparison> to "N/A".'
            )

        # regular turn prompt, we use cycle/turn counting starting from 1 for user-friendliness
        # may help models better track progress
        return (
            f"{history}\n\n"
            f"Cycle {cycle_idx + 1} of {max_cycles}, Turn {turn_idx + 1}.\n"
            "Write a solution to the user query.\n"
            "In <evaluation>, evaluate the most recent visible prior completions (up to 2 turns), including their solutions "
            "and their evaluations/comparisons (meta-evaluate judge quality).\n"
            "In <comparison>, compare only OTHER agents visible in the history (never include yourself). If fewer than two "
            'other completions are visible, write "N/A".'
        )

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
        action_content = ensure_text(action_message["content"])

        try:
            await self.coordinator.submit_response(
                self.agent_id,
                action_content,
                observation=observation_str,
            )
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
class MultiAgentEnvGroupBuilder(EnvGroupBuilder):
    """Builder for groups of multi-agent debate environments."""

    questions: list[str]  # List of questions for the dataset
    question_index: int  # Which question to use for this group
    renderer: Renderer
    num_agents: int
    max_rounds: int
    self_play: bool
    reward_mode: str = "win_rate"  # "win_rate" | "win_minus_loss"
    history_turns: int = 2
    summarize_history: bool = False
    summarize_model: str | None = "Qwen/Qwen3-4B-Instruct-2507"
    log_full_transcript: bool = False
    # Optional fixed opponents for non-self-play evaluation. When provided, should have
    # length == num_agents, and we will pass per-env lists with the controlled agent removed.
    opponent_policies_all: list[TinkerMessageCompleter] | None = None
    model_name: str | None = None

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of environments for a multi-agent debate."""
        question = self.questions[self.question_index % len(self.questions)]
        max_turns = self.num_agents * self.max_rounds

        if self.self_play:
            # All agents share the same coordinator
            coordinator = MultiAgentCoordinator(
                question=question, num_agents=self.num_agents, max_turns=max_turns
            )
            return [
                MultiAgentDebateEnv(
                    agent_id=i,
                    coordinator=coordinator,
                    renderer=self.renderer,
                    self_play=True,
                    opponent_policies=None,
                    history_turns=self.history_turns,
                    summarize_history=self.summarize_history,
                    summarize_model=self.summarize_model,
                    model_name=self.model_name,
                )
                for i in range(self.num_agents)
            ]

        assert (
            self.opponent_policies_all is not None
            and len(self.opponent_policies_all) == self.num_agents
        ), "non-self-play requires opponent_policies_all with length == num_agents"

        # Role-averaged evaluation: each env is an independent debate where the *controlled* agent
        # is `agent_id=i`, and all other agents are played by fixed opponent policies.
        envs: list[MultiAgentDebateEnv] = []
        for i in range(self.num_agents):
            coordinator = MultiAgentCoordinator(
                question=question, num_agents=self.num_agents, max_turns=max_turns
            )
            opponent_policies_for_i = [
                pol for j, pol in enumerate(self.opponent_policies_all) if j != i
            ]
            envs.append(
                MultiAgentDebateEnv(
                    agent_id=i,
                    coordinator=coordinator,
                    renderer=self.renderer,
                    self_play=False,
                    opponent_policies=opponent_policies_for_i,
                    history_turns=self.history_turns,
                    summarize_history=self.summarize_history,
                    summarize_model=self.summarize_model,
                    model_name=self.model_name,
                )
            )
        return envs

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        assert env_group, "empty env_group"

        # Self-play: all envs share a coordinator, so compute once.
        if self.self_play:
            env0 = env_group[0]
            assert isinstance(env0, MultiAgentDebateEnv)
            coordinator = env0.coordinator

            if self.log_full_transcript:
                # If logtree logging is enabled by the outer RL harness, emit the full multi-agent transcript.
                # This is the easiest way to inspect the entire conversation across all turns/agents.
                with logtree.scope_header("Debate Transcript"):
                    logtree.log_text(f"Question: {coordinator.state.question}")
                    turns = coordinator.state.agent_responses
                    if not turns:
                        logtree.log_text("(No responses captured)")
                    for turn_idx, response in enumerate(turns, start=1):
                        with logtree.scope_header(f"Turn {turn_idx}"):
                            with logtree.scope_header(f"Agent {response.author_id}"):
                                with logtree.scope_details("System prompt"):
                                    logtree.log_text(
                                        AGENT_SYSTEM_PROMPT.format(agent_id=response.author_id)
                                    )
                                if response.observation:
                                    with logtree.scope_details("Observation (context)"):
                                        logtree.log_text(response.observation)
                                if response.solution:
                                    with logtree.scope_details("Solution"):
                                        logtree.log_text(response.solution)
                                if response.evaluation:
                                    with logtree.scope_details("Evaluation"):
                                        logtree.log_text(response.evaluation)
                                if response.comparison_text:
                                    with logtree.scope_details("Comparison"):
                                        logtree.log_text(response.comparison_text)
                                with logtree.scope_details("Raw response"):
                                    logtree.log_text(response.raw_response)

            # Leave-one-out rewards: agent i's reward excludes comparisons authored by i.
            if self.reward_mode == "win_rate":
                rewards_G, summary_metrics = compute_pairwise_win_rates(
                    coordinator.state.agent_responses, num_agents=self.num_agents
                )
            elif self.reward_mode == "win_minus_loss":
                rewards_G, summary_metrics = compute_pairwise_win_minus_loss(
                    coordinator.state.agent_responses, num_agents=self.num_agents
                )
            else:
                raise ValueError(f"Invalid reward_mode={self.reward_mode!r}")

            return [
                (
                    reward,
                    {
                        "agent_id": agent_id,
                        "pairwise_reward_is_win_rate": 1.0
                        if self.reward_mode == "win_rate"
                        else 0.0,
                        "pairwise_reward_is_win_minus_loss": 1.0
                        if self.reward_mode == "win_minus_loss"
                        else 0.0,
                        "pairwise_reward": reward,
                        **summary_metrics,
                    },
                )
                for agent_id, reward in enumerate(rewards_G)
            ]

        # Non-self-play: each env has its own coordinator (controlled agent vs fixed opponents).
        # Compute per-env rewards from that env's full transcript, and return the controlled agent's reward.
        rewards_and_metrics: list[tuple[float, Metrics]] = []
        for env in env_group:
            assert isinstance(env, MultiAgentDebateEnv)
            coordinator = env.coordinator
            if self.reward_mode == "win_rate":
                rewards_G, summary_metrics = compute_pairwise_win_rates(
                    coordinator.state.agent_responses, num_agents=self.num_agents
                )
                reward = rewards_G[env.agent_id]
            elif self.reward_mode == "win_minus_loss":
                rewards_G, summary_metrics = compute_pairwise_win_minus_loss(
                    coordinator.state.agent_responses, num_agents=self.num_agents
                )
                reward = rewards_G[env.agent_id]
            else:
                raise ValueError(f"Invalid reward_mode={self.reward_mode!r}")
            rewards_and_metrics.append(
                (
                    reward,
                    {
                        "agent_id": env.agent_id,
                        "pairwise_reward_is_win_rate": 1.0
                        if self.reward_mode == "win_rate"
                        else 0.0,
                        "pairwise_reward_is_win_minus_loss": 1.0
                        if self.reward_mode == "win_minus_loss"
                        else 0.0,
                        "pairwise_reward": reward,
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
        reward_mode: str,
        history_turns: int,
        summarize_history: bool,
        summarize_model: str | None,
        log_full_transcript: bool,
        max_rounds: int,
        num_datapoints: int,
        model_name: str,
        opponent_policies: list[TinkerMessageCompleter] | None = None,
    ):
        self.batch_size = batch_size
        self.questions = questions
        self.num_agents = num_agents
        self.renderer = renderer
        self.self_play = self_play
        self.reward_mode = reward_mode
        self.history_turns = history_turns
        self.summarize_history = summarize_history
        self.summarize_model = summarize_model
        self.log_full_transcript = log_full_transcript
        self.max_rounds = max_rounds
        self.num_datapoints = num_datapoints
        self.model_name = model_name
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
                reward_mode=self.reward_mode,
                history_turns=self.history_turns,
                summarize_history=self.summarize_history,
                summarize_model=self.summarize_model,
                log_full_transcript=self.log_full_transcript,
                max_rounds=self.max_rounds,
                model_name=self.model_name,
                opponent_policies_all=self.opponent_policies,
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
    max_rounds: int = 3
    reward_mode: str = "win_rate"  # "win_rate" | "win_minus_loss"
    history_rounds: int = 2  # number of turns of history to include
    summarize_history: bool = False
    summarize_model: str | None = None
    log_full_transcript: bool = False
    model_name: str
    renderer_name: str
    # Prompt source: local JSONL by default (no network).
    dataset_path: str = "tinker_cookbook/example_data/nonverifiable_queries.jsonl"
    dataset_field: str = "query"
    # If enabled, split loaded questions into disjoint train/test pools.
    test_question_frac: float = 0.1  # used only if num_test_datapoints > 0

    # Optional HF dataset loading (requires network).
    hf_dataset_name: str | None = None
    hf_dataset_subset: str | None = None
    hf_dataset_split: str = "train"
    hf_dataset_question_field: str = "question"
    max_questions: int = 1000
    # Optional: use a different fixed opponent base model for evaluation.
    opponent_model_name: str | None = None

    def load_questions(self) -> list[str]:
        if self.hf_dataset_name is not None:
            return self._load_questions_from_hf()
        return self._load_questions_from_file()

    def get_question_pools(self) -> tuple[list[str], list[str]]:
        questions = self.load_questions()
        return self._split_questions(questions)

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

    def _construct_fixed_opponent_policies(
        self, renderer: Renderer
    ) -> list[TinkerMessageCompleter]:
        """Create fixed opponent policies for evaluation (using a base model)."""
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(
            base_model=self.opponent_model_name or self.model_name
        )
        return [
            TinkerMessageCompleter(
                sampling_client=sampling_client,
                renderer=renderer,
                max_tokens=2048,
                stop_condition=_get_debate_stop_condition(renderer),
            )
            for _ in range(self.num_agents)
        ]

    def _split_questions(self, questions: list[str]) -> tuple[list[str], list[str]]:
        """
        Split questions into disjoint train/test pools.
        The test pool is determined by test_question_frac (capped to at least 1 if enabled).
        """
        if self.num_test_datapoints <= 0 or self.test_question_frac <= 0:
            return questions, []
        if len(questions) < 2:
            return questions, []

        rng = random.Random(42)
        shuffled = list(questions)
        rng.shuffle(shuffled)
        test_n = int(round(len(shuffled) * self.test_question_frac))
        test_n = max(1, min(test_n, len(shuffled) - 1))
        return shuffled[test_n:], shuffled[:test_n]

    async def __call__(
        self,
    ) -> tuple[MultiAgentDebateDataset, MultiAgentDebateDataset | None]:
        """Build the dataset for training and testing."""
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        train_questions, test_questions = self.get_question_pools()

        # Training dataset (self-play)
        train_dataset = MultiAgentDebateDataset(
            batch_size=self.batch_size,
            questions=train_questions,
            num_agents=self.num_agents,
            renderer=renderer,
            self_play=True,
            reward_mode=self.reward_mode,
            history_turns=self.history_rounds,
            summarize_history=self.summarize_history,
            summarize_model=self.summarize_model,
            log_full_transcript=self.log_full_transcript,
            max_rounds=self.max_rounds,
            num_datapoints=self.num_train_datapoints,
            model_name=self.model_name,
            opponent_policies=None,
        )

        # Test dataset (optional). If num_test_datapoints is 0, disable test set entirely.
        test_dataset: MultiAgentDebateDataset | None
        if self.num_test_datapoints <= 0 or not test_questions:
            test_dataset = None
        else:
            opponent_policies_all = self._construct_fixed_opponent_policies(renderer)
            test_dataset = MultiAgentDebateDataset(
                batch_size=min(self.num_test_datapoints, self.batch_size),
                questions=test_questions,
                num_agents=self.num_agents,
                renderer=renderer,
                self_play=False,
                reward_mode=self.reward_mode,
                history_turns=self.history_rounds,
                summarize_history=self.summarize_history,
                summarize_model=self.summarize_model,
                log_full_transcript=self.log_full_transcript,
                max_rounds=self.max_rounds,
                num_datapoints=self.num_test_datapoints,
                model_name=self.model_name,
                opponent_policies=opponent_policies_all,
            )

        return train_dataset, test_dataset
