"""Prompts and XML parsing utilities for multi-agent debate."""

import re
from dataclasses import dataclass

AGENT_SYSTEM_PROMPT = """You are Agent {agent_id} in a multi-agent self-play discussion to answer an open-ended, non-verifiable user query.

All agents share the same underlying policy model, but you must act as an independent participant.

Your tasks:
1) Propose (or refine) a high-quality solution to the query.
2) Evaluate other agents' work, including BOTH:
   - their proposed solution, and
   - the quality/fairness/helpfulness of their evaluations of others, and
   - the quality/fairness/helpfulness of their pairwise comparisons.
3) Provide pairwise comparisons between agents' overall completions.

DEFINITIONS:
- An agent's "completion" means the combination of its <solution>, <evaluation>, and <comparison> content for the current round.

RESPONSE FORMAT (use exact XML tags, in this order):

<solution>
Your proposed answer or contribution. Please make it as detailed and high-quality as possible.
</solution>

<evaluation>
Assess other agents' recent contributions, INCLUDING their solutions, their evaluations, and their comparisons.
</evaluation>

<comparison>
Provide pairwise comparisons between agents' completions (solution+evaluation+comparison), based on the most recently COMPLETED round.
Output one line per unordered pair of agents you can compare.
Format: "Agent A > Agent B" (A better), or "Agent A = Agent B" (tie).
Example:
Agent 0 > Agent 1
Agent 0 = Agent 2
</comparison>

<consensus>YES/NO</consensus>
<consensus_reason>...</consensus_reason>

IMPORTANT:
- Use EXACTLY these tags in this order
- For consensus, write ONLY "YES" or "NO" inside the tag
- Be honest about consensus - only say YES when truly satisfied with the answer
- Do not wrap your answer in Markdown or code fences
- If it's the first round and there is no prior completed round, set <evaluation> to "N/A" and <comparison> to "N/A".
"""


@dataclass
class ParsedResponse:
    """Parsed agent response."""

    solution: str
    evaluation: str
    consensus_reached: bool
    consensus_reason: str
    comparisons: list[tuple[int, str, int]]  # List of (agent_a_id, op, agent_b_id) tuples
    raw_response: str
    comparison_text: str = ""
    author_id: int = -1
    observation: str = ""
    thinking: str = ""


def parse_agent_response(response: str, *, author_id: int, observation: str = "") -> ParsedResponse:
    """Parse the XML-formatted agent response.

    Args:
        response: The agent's raw text response

    Returns:
        ParsedResponse with extracted fields

    Raises:
        ValueError if the response doesn't match expected format
    """
    # Normalize common wrappers (e.g., markdown code fences) and strip Qwen-style thinking blocks.
    response = response.strip()
    # Remove fenced blocks like ```xml ... ```
    if response.startswith("```"):
        response = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", response)
        response = re.sub(r"\n```$", "", response)
        response = response.strip()

    # If there's preamble text, start at the first structured tag.
    first_tag = re.search(r"<(solution|evaluation|comparison|consensus)\b", response)
    if first_tag:
        response = response[first_tag.start() :]

    # Handle Qwen-style thinking wrappers. Keep the content (it may contain our XML tags),
    # but remove the wrapper tags to avoid confusing downstream parsing.
    response = re.sub(r"</?think>", "", response, flags=re.IGNORECASE).strip()

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
    comparison_text = ""
    comp_match = re.search(r"<comparison>(.*?)</comparison>", response, re.DOTALL | re.IGNORECASE)
    if comp_match:
        content = comp_match.group(1).strip()
        comparison_text = content
        # Regex to find "Agent A > Agent B"
        pairs = re.findall(r"Agent\s+(\d+)\s*([>=])\s*Agent\s+(\d+)", content)
        for agent_a, op, agent_b in pairs:
            comparisons.append((int(agent_a), op, int(agent_b)))

    # Parse YES/NO
    if consensus_text == "YES":
        consensus_reached = True
    elif consensus_text == "NO":
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
        solution=solution,
        evaluation=evaluation,
        consensus_reached=consensus_reached,
        consensus_reason=consensus_reason,
        comparisons=comparisons,
        raw_response=response,
        comparison_text=comparison_text,
        author_id=author_id,
        observation=observation,
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
