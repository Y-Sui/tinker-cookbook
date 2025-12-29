"""Coordinator for multi-agent debate conversations.

This module provides shared coordination logic for multi-agent debate environments.
Both verifiable and non-verifiable debate implementations use these classes.

Type Aliases:
    AgentId: Integer identifier for an agent (0-indexed)
    TurnIndex: Integer index for a turn in the conversation
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
