"""Multi-agent debate environment for tinker RL."""

from dataclasses import dataclass
from typing import Sequence

import chz

from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .base_env import BaseMultiAgentDebateEnv, BaseMultiAgentEnvGroupBuilder
from .coordinator import MultiAgentCoordinator
from .loaders import load_questions_from_jsonl
from .prompts import AGENT_SYSTEM_PROMPT, ParsedResponse
from .utils import get_debate_stop_condition, log_debate_transcript


@dataclass
class MultiAgentDebateEnv(BaseMultiAgentDebateEnv):
    """Environment for one agent in a multi-agent debate."""

    @property
    def stop_condition(self) -> StopCondition:
        return get_debate_stop_condition(self.renderer)

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return AGENT_SYSTEM_PROMPT.format(agent_id=self.agent_id)

    def _format_turns(self, turns: list[ParsedResponse], num_turns: int) -> str:
        if not turns or num_turns == 0:
            return ""
        lines: list[str] = []
        total_turns = len(turns)
        num_agents = self.coordinator.state.num_agents
        # Respect history truncation but keep original turn numbering
        if num_turns < 0:
            start_idx = 0
        else:
            start_idx = max(0, total_turns - num_turns)
        for idx in range(start_idx, total_turns):
            response = turns[idx]
            display_turn_idx = idx + 1  # 1-based index aligned to the original conversation
            # Calculate round number (1-indexed) for disambiguation
            round_num = (display_turn_idx - 1) // num_agents + 1
            agent_label = f"Agent {response.author_id} (R{round_num})"

            lines.append(
                f"== Turn {display_turn_idx}/{self.coordinator.state.max_turns} ({agent_label}) ==\n"
            )
            lines.append(f"{agent_label}'s Solution:")
            lines.append(response.solution.rstrip())
            lines.append("\n")
            lines.append(f"{agent_label}'s Evaluation:")
            lines.append(response.evaluation.rstrip())
            lines.append("\n")
            if response.comparison_text:
                lines.append(f"{agent_label}'s Comparison:")
                lines.append(response.comparison_text.rstrip())
            lines.append("== End of Turn ==\n")
        return "\n".join(lines).rstrip()

    async def get_conversation_context(self) -> str:
        """Format the conversation context for this agent."""
        if not self.coordinator.state.agent_responses or self.history_turns == 0:
            return ""

        question = self.coordinator.state.question
        turns = list(self.coordinator.state.agent_responses)

        history_turns = (
            f"Question: {question}\n"
            f"Previous turns of conversation:\n"
            f"{self._format_turns(turns, self.history_turns)}".rstrip()
            + "\n"
        )

        return await self._summarize(history_turns) if self.summarize_history else history_turns

    def _get_context_header(self) -> str:
        """Generate a context header to disambiguate agent references across rounds."""
        turn_idx = self.coordinator.state.current_turn
        num_agents = self.coordinator.state.num_agents
        current_round = turn_idx // num_agents + 1
        agent_id = self.coordinator.state.current_agent_id

        if current_round <= 1:
            return ""

        # Build context header for rounds > 1
        return (
            f"[CONTEXT: You are Agent {agent_id}, Round {current_round}. "
            f"When you see 'Agent {agent_id} (R{current_round - 1})' in history, "
            f"that is a DIFFERENT agent from a previous round, not you. "
            f"Agents do NOT self-evaluate by design - this is correct behavior.]\n\n"
        )

    async def get_observation_string(self) -> str:
        """Get the observation string for this agent."""
        history = await self.get_conversation_context()
        turn_idx = self.coordinator.state.current_turn
        num_agents = self.coordinator.state.num_agents
        current_round = turn_idx // num_agents + 1
        agent_id = self.coordinator.state.current_agent_id

        # intermission prompt for first turn
        if turn_idx == 0:
            return (
                f"Question: {self.coordinator.state.question}\n\n"
                "First completion: propose your solution.\n"
                'Set <evaluation> to "N/A" and <comparison> to "N/A".'
            )
        else:
            # regular turn prompt with context header
            context_header = self._get_context_header()
            return (
                f"{context_header}{history}\n\nQuestion: {self.coordinator.state.question}\n\n"
                f"Agent {agent_id} (R{current_round}), please continue the debate by providing your solution, evaluation, and comparison. "
                f"Do not include Agent {agent_id} in your comparisons."
            )


@dataclass
class MultiAgentEnvGroupBuilder(BaseMultiAgentEnvGroupBuilder):
    """Builder for groups of multi-agent debate environments."""

    questions: list[str]  # List of questions for the dataset
    question_index: int  # Which question to use for this group
    max_rounds: int
    self_play: bool
    history_turns: int = 2
    log_full_transcript: bool = False

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of environments for a multi-agent debate."""
        # reuse the questions by cycling if num_train_datapoints > len(questions), we use max_questions to limit len(questions)
        question = self.questions[self.question_index % len(self.questions)]
        max_turns = self.num_agents * self.max_rounds

        # Create shared summarizer once for all agents in this group
        shared_summarizer = self._create_shared_summarizer()

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
                history_turns=self.history_turns,
                summarize_history=self.summarize_history,
                _summarizer_policy=shared_summarizer,
                model_name=self.model_name,
            )
            for i in range(self.num_agents)
        ]

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
                log_debate_transcript(coordinator)

            # Populate step-wise rewards in trajectories
            stepwise_metrics = self._populate_stepwise_rewards(trajectory_group, env_group)

            return [
                (
                    0.0,  # No final reward (all rewards are in steps)
                    {
                        "agent_id": agent_id,
                        **stepwise_metrics,
                    },
                )
                for agent_id in range(self.num_agents)
            ]


class MultiAgentDebateDataset(RLDataset):
    """Dataset for multi-agent debate environments."""

    def __init__(
        self,
        batch_size: int,
        questions: list[str],
        num_agents: int,
        renderer: Renderer,
        self_play: bool,
        history_turns: int,
        summarize_history: bool,
        summarize_model: str | None,
        log_full_transcript: bool,
        max_rounds: int,
        num_datapoints: int,
        model_name: str,
        enable_format_penalty: bool = True,
    ):
        self.batch_size = batch_size
        self.questions = questions
        self.num_agents = num_agents
        self.renderer = renderer
        self.self_play = self_play
        self.history_turns = history_turns
        self.summarize_history = summarize_history
        self.summarize_model = summarize_model
        self.log_full_transcript = log_full_transcript
        self.max_rounds = max_rounds
        self.num_datapoints = num_datapoints
        self.model_name = model_name
        self.enable_format_penalty = enable_format_penalty

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
                history_turns=self.history_turns,
                summarize_history=self.summarize_history,
                summarize_model=self.summarize_model,
                log_full_transcript=self.log_full_transcript,
                max_rounds=self.max_rounds,
                model_name=self.model_name,
                enable_format_penalty=self.enable_format_penalty,
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
    epoch: int = 1
    num_test_datapoints: int
    num_agents: int = 3
    max_rounds: int = 3
    history_rounds: int = 2  # number of turns of history to include
    summarize_history: bool = False
    summarize_model: str | None = None
    log_full_transcript: bool = False
    model_name: str
    renderer_name: str
    dataset_path: str = "tinker_cookbook/example_data/nonverifiable_queries.jsonl"
    problem_field: str = "query"
    max_questions: int = -1  # No limit by default
    enable_format_penalty: bool = True

    def load_questions(self) -> list[str]:
        return load_questions_from_jsonl(
            path=self.dataset_path,
            field=self.problem_field,
            max_count=self.max_questions,
        )

    async def __call__(
        self,
    ) -> tuple[MultiAgentDebateDataset, MultiAgentDebateDataset | None]:
        """Build the dataset for training and testing.

        Repeats the base dataset `epoch` times by scaling num_train_datapoints.
        """
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        train_questions = self.load_questions()
        if self.num_train_datapoints > len(train_questions) or self.num_train_datapoints < 0:
            total_train_datapoints = len(train_questions) * self.epoch
        else:
            total_train_datapoints = self.num_train_datapoints * self.epoch
        # Training dataset (self-play)
        train_dataset = MultiAgentDebateDataset(
            batch_size=self.batch_size,
            questions=train_questions,
            num_agents=self.num_agents,
            renderer=renderer,
            self_play=True,
            history_turns=self.history_rounds,
            summarize_history=self.summarize_history,
            summarize_model=self.summarize_model,
            log_full_transcript=self.log_full_transcript,
            max_rounds=self.max_rounds,
            num_datapoints=total_train_datapoints,
            model_name=self.model_name,
            enable_format_penalty=self.enable_format_penalty,
        )

        # # Test dataset (optional, also uses self-play). If num_test_datapoints is 0, disable test set entirely.
        # test_dataset: MultiAgentDebateDataset | None
        # if self.num_test_datapoints <= 0 or not test_questions:
        #     test_dataset = None
        # else:
        #     test_dataset = MultiAgentDebateDataset(
        #         batch_size=min(self.num_test_datapoints, self.batch_size),
        #         questions=test_questions,
        #         num_agents=self.num_agents,
        #         renderer=renderer,
        #         self_play=True,
        #         history_turns=self.history_rounds,
        #         summarize_history=self.summarize_history,
        #         summarize_model=self.summarize_model,
        #         log_full_transcript=self.log_full_transcript,
        #         max_rounds=self.max_rounds,
        #         num_datapoints=self.num_test_datapoints,
        #         model_name=self.model_name,
        #     )

        return train_dataset, None
