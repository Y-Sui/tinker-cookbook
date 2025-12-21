"""Prompts and XML parsing utilities for multi-agent debate."""

import re
from dataclasses import dataclass

AGENT_SYSTEM_PROMPT = """You are Agent {agent_id} in a multi-agent collaborative discussion to solve open-ended questions.

Your goal: Provide high-quality reasoning, evaluate others constructively, and work toward consensus.

RESPONSE FORMAT (use exact XML tags):

<thinking>
Your internal reasoning process.
</thinking>

<solution>
Your proposed answer or contribution.
</solution>

<evaluation>
Assess other agents' recent contributions.
</evaluation>

<comparison>
Compare the solutions of other agents from the most recent round.
For every pair of agents (excluding yourself), determine who provided the better contribution.
Format: "Agent X > Agent Y" or "Agent X = Agent Y" (if tied).
Example:
Agent 1 > Agent 2
Agent 3 > Agent 1
</comparison>

<consensus>YES/NO</consensus>
<consensus_reason>...</consensus_reason>

IMPORTANT:
- Use EXACTLY these tags in this order
- For consensus, write ONLY "YES" or "NO" inside the tag
- Be honest about consensus - only say YES when truly satisfied with the answer
"""


@dataclass
class ParsedResponse:
    """Parsed agent response."""

    thinking: str
    solution: str
    evaluation: str
    consensus_reached: bool
    consensus_reason: str
    comparisons: list[tuple[int, int]]  # List of (winner_id, loser_id) tuples
    raw_response: str


def parse_agent_response(response: str) -> ParsedResponse:
    """Parse the XML-formatted agent response.

    Args:
        response: The agent's raw text response

    Returns:
        ParsedResponse with extracted fields

    Raises:
        ValueError if the response doesn't match expected format
    """
    # Extract thinking (optional - may not always be present)
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", response, re.DOTALL)
    thinking = thinking_match.group(1).strip() if thinking_match else ""

    # Extract solution (required)
    solution_match = re.search(r"<solution>(.*?)</solution>", response, re.DOTALL)
    if not solution_match:
        raise ValueError(f"Missing <solution> tag in response: {response[:200]}")
    solution = solution_match.group(1).strip()

    # Extract evaluation (required)
    evaluation_match = re.search(r"<evaluation>(.*?)</evaluation>", response, re.DOTALL)
    if not evaluation_match:
        raise ValueError(f"Missing <evaluation> tag in response: {response[:200]}")
    evaluation = evaluation_match.group(1).strip()

    # Extract consensus (required)
    consensus_match = re.search(
        r"<consensus>(.*?)</consensus>", response, re.DOTALL | re.IGNORECASE
    )
    if not consensus_match:
        raise ValueError(f"Missing <consensus> tag in response: {response[:200]}")
    consensus_text = consensus_match.group(1).strip().upper()

    comparisons = []
    comp_match = re.search(r"<comparison>(.*?)</comparison>", response, re.DOTALL | re.IGNORECASE)
    if comp_match:
        content = comp_match.group(1).strip()
        # Regex to find "Agent A > Agent B"
        pairs = re.findall(r"Agent\s+(\d+)\s*([>=])\s*Agent\s+(\d+)", content)
        for winner, loser in pairs:
            comparisons.append((int(winner), int(loser)))

    # Parse YES/NO
    if "YES" in consensus_text:
        consensus_reached = True
    elif "NO" in consensus_text:
        consensus_reached = False
    else:
        # Default to NO if unclear
        consensus_reached = False

    # Extract consensus reasoning (required)
    reason_match = re.search(r"<consensus_reason>(.*?)</consensus_reason>", response, re.DOTALL)
    if not reason_match:
        raise ValueError(f"Missing <consensus_reason> tag in response: {response[:200]}")
    consensus_reason = reason_match.group(1).strip()

    return ParsedResponse(
        thinking=thinking,
        solution=solution,
        evaluation=evaluation,
        consensus_reached=consensus_reached,
        consensus_reason=consensus_reason,
        comparisons=comparisons,
        raw_response=response,
    )


def parse_pairwise_comparison(response: str, agent_a_id: int, agent_b_id: int) -> str:
    """Parse the pairwise comparison response.

    Args:
        response: The model's comparison response
        agent_a_id: ID of first agent
        agent_b_id: ID of second agent

    Returns:
        One of: f"AGENT_{agent_a_id}_BETTER", f"AGENT_{agent_b_id}_BETTER", or "TIE"
    """
    response = response.strip().upper()

    # Look for the expected patterns
    if f"AGENT_{agent_a_id}_BETTER" in response:
        return f"AGENT_{agent_a_id}_BETTER"
    elif f"AGENT_{agent_b_id}_BETTER" in response:
        return f"AGENT_{agent_b_id}_BETTER"
    elif "TIE" in response:
        return "TIE"
    else:
        # Default to TIE if we can't parse
        return "TIE"


def format_conversation_history(
    agent_responses: list[list[ParsedResponse]], num_agents: int, k_turns: int
) -> str:
    """Format the last K turns of conversation history.

    Args:
        agent_responses: List of turns, where each turn is a list of ParsedResponse (one per agent)
        num_agents: Total number of agents
        k_turns: Number of recent turns to include

    Returns:
        Formatted string of conversation history
    """
    if len(agent_responses) == 0:
        return "(No previous turns yet)"

    # Get last k turns (or all if fewer than k)
    recent_turns = agent_responses[-k_turns:]

    lines = []
    for turn_idx, turn_responses in enumerate(recent_turns):
        actual_turn = len(agent_responses) - len(recent_turns) + turn_idx
        lines.append(f"--- Turn {actual_turn + 1} ---")
        for agent_id in range(num_agents):
            if agent_id < len(turn_responses):
                resp = turn_responses[agent_id]
                lines.append(f"\nAgent {agent_id} Solution:")
                lines.append(resp.solution)
                lines.append(f"\nAgent {agent_id} Evaluation:")
                lines.append(resp.evaluation)
        lines.append("")

    return "\n".join(lines)
