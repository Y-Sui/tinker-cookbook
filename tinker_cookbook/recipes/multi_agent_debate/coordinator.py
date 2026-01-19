"""Coordinator for multi-agent debate conversations.

This module provides shared coordination logic for multi-agent debate environments.
Both verifiable and non-verifiable debate implementations use these classes.

The coordinator supports **parallel generation within each round (cycle)**:
- All agents in a cycle generate their responses simultaneously
- They all see the same history (responses from previous cycles only)
- This ensures independent judgments and faster training
"""

import asyncio
from dataclasses import dataclass, field
from typing import TypeAlias

from .prompts import ParsedResponse, parse_agent_response

# Type aliases for better readability
AgentId: TypeAlias = int
TurnIndex: TypeAlias = int


@dataclass
class ConversationState:
    """Shared state for a multi-agent conversation.

    Supports parallel generation within each cycle:
    - current_cycle: Which round of responses we're in
    - cycle_responses: Responses being collected for the current cycle
    - agent_responses: All committed responses from completed cycles
    """

    question: str
    num_agents: int
    max_cycles: int  # Total number of cycles (rounds)
    current_cycle: int = 0  # Which cycle we're in (0-indexed)
    agent_responses: list[ParsedResponse] = field(default_factory=list)  # Committed responses
    done: bool = False

    # Parallel generation state for current cycle
    cycle_responses: dict[int, ParsedResponse] = field(default_factory=dict)  # agent_id -> response
    cycle_started: bool = False  # Whether current cycle has started

    @property
    def current_turn(self) -> int:
        """Total number of committed turns (for backward compatibility)."""
        return len(self.agent_responses)

    def get_current_cycle(self) -> int:
        """Get the current cycle index."""
        return self.current_cycle

    def start_cycle(self) -> None:
        """Start a new cycle - allows all agents to generate in parallel."""
        self.cycle_responses = {}
        self.cycle_started = True

    def submit_cycle_response(self, agent_id: int, response: ParsedResponse) -> bool:
        """Submit a response for the current cycle.

        Returns True if all agents have submitted and the cycle is complete.
        """
        self.cycle_responses[agent_id] = response
        return len(self.cycle_responses) == self.num_agents

    def commit_cycle(self) -> None:
        """Commit all responses from the current cycle and advance.

        Responses are committed in agent_id order for deterministic ordering.
        """
        # Commit responses in agent_id order
        for agent_id in range(self.num_agents):
            if agent_id in self.cycle_responses:
                self.agent_responses.append(self.cycle_responses[agent_id])

        # Advance to next cycle
        self.current_cycle += 1
        self.cycle_responses = {}
        self.cycle_started = False

        if self.current_cycle >= self.max_cycles:
            self.done = True


class MultiAgentCoordinator:
    """Coordinates a multi-agent debate conversation with parallel generation.

    Within each cycle (round), all agents generate their responses in parallel.
    They all see the same history from previous cycles, ensuring independent
    judgments. This is more efficient and produces better training signal for
    the v2 reward system.
    """

    def __init__(self, question: str, num_agents: int, max_turns: int):
        # max_turns is the total number of individual turns
        # Convert to max_cycles (each cycle = num_agents turns)
        max_cycles = (max_turns + num_agents - 1) // num_agents
        self.state = ConversationState(
            question=question, num_agents=num_agents, max_cycles=max_cycles
        )
        self.condition = asyncio.Condition()

    @property
    def done(self) -> bool:
        return self.state.done

    @property
    def current_agent_id(self) -> int:
        """For backward compatibility - not meaningful with parallel generation."""
        return 0

    async def wait_for_turn(self, agent_id: int) -> None:
        """Wait until this agent can generate (i.e., the cycle has started).

        With parallel generation, all agents in a cycle can proceed once
        the cycle starts. They wait at the end until all have submitted.
        """
        async with self.condition:
            # Wait for cycle to start or done
            await self.condition.wait_for(lambda: self.state.cycle_started or self.state.done)

    async def wait_for_cycle_start(self, agent_id: int) -> None:
        """Wait for the current cycle to start (called at beginning of step)."""
        async with self.condition:
            if self.state.done:
                return

            # First agent to arrive starts the cycle
            if not self.state.cycle_started:
                self.state.start_cycle()
                self.condition.notify_all()
            else:
                # Wait for cycle to be ready (in case of race condition)
                await self.condition.wait_for(lambda: self.state.cycle_started or self.state.done)

    async def submit_response(
        self,
        agent_id: int,
        response: str,
        *,
        observation: str = "",
    ) -> ParsedResponse:
        """Submit an agent's response for the current cycle.

        All agents submit in parallel. The last agent to submit triggers
        the cycle to commit and advance.
        """
        async with self.condition:
            if self.state.done:
                raise ValueError("Coordinator is done")

            # Parse the response
            parsed = parse_agent_response(
                response,
                author_id=agent_id,
                observation=observation,
            )

            # Submit to current cycle
            cycle_complete = self.state.submit_cycle_response(agent_id, parsed)

            if cycle_complete:
                # Last agent - commit the cycle and notify all
                self.state.commit_cycle()

                # Start next cycle if not done
                if not self.state.done:
                    self.state.start_cycle()

                self.condition.notify_all()
            else:
                # Wait for cycle to complete before returning
                current_cycle = self.state.current_cycle
                await self.condition.wait_for(
                    lambda: self.state.current_cycle > current_cycle or self.state.done
                )

            return parsed

    async def abort(self) -> None:
        """End the episode early and release any waiting agents."""
        async with self.condition:
            self.state.done = True
            self.condition.notify_all()
