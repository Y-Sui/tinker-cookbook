"""Core debate logic: coordination, prompts, and parsing."""

from .coordinator import MultiAgentCoordinator
from .prompts import (
    AGENT_PERSONAS,
    AGENT_SYSTEM_PROMPT,
    SUMMARIZER_SYSTEM_PROMPT,
    VERIFIABLE_AGENT_SYSTEM_PROMPT,
    ParsedResponse,
    format_persona_intro,
    get_agent_persona,
    get_agent_temperature,
    parse_agent_response,
)

__all__ = [
    "MultiAgentCoordinator",
    "AGENT_PERSONAS",
    "AGENT_SYSTEM_PROMPT",
    "SUMMARIZER_SYSTEM_PROMPT",
    "VERIFIABLE_AGENT_SYSTEM_PROMPT",
    "ParsedResponse",
    "parse_agent_response",
    "format_persona_intro",
    "get_agent_persona",
    "get_agent_temperature",
]
