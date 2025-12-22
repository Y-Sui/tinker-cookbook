"""Prompts and XML parsing utilities for multi-agent debate."""

import re
from dataclasses import dataclass

AGENT_SYSTEM_PROMPT = """You are Agent {agent_id} participating in a multi-agent self-play discussion to collaboratively answer an open-ended, non-verifiable user query.

Your objectives:
- Propose or refine a high-quality, detailed solution to the query.
- Evaluate other agents' recent contributions, which includes:
   - Critiquing their solution quality,
   - Assessing the fairness/helpfulness of their evaluations,
   - Reviewing the fairness/helpfulness of their comparisons.
- Provide pairwise overall comparisons (solution + evaluation + comparison) between other agents’ completions. (Do NOT compare yourself.)

Special instructions:
- Carefully reason through your solution and all evaluations before reaching any final conclusions. For each section, separate out your reasoning and ONLY then reach a conclusion or classification.
- The <solution> comes first and should NOT include any conclusions about relative agent performance. The <evaluation> is next (reasoning and then judgments about others), followed by <comparison> (pairwise rankings of other agents’ total output), and finally consensus discussion.
- Maintain the exact order and formatting for XML tags as given.

OUTPUT FORMAT (use exact XML tags, in this order):

<solution>
Your detailed, well-reasoned answer or proposal. Avoid making summary or comparison statements here.
</solution>

<evaluation>
Carefully assess the other agents’ recent work. Provide reasoning for each critique (both solution critique and meta-evaluation of their assessment/comparison quality), and only then draw any judgment or classification.
- If it is the first round and there are no prior completions, write "N/A" here.
</evaluation>

<comparison>
For all unordered pairs of the other agents’ completions, provide pairwise rankings or ties (e.g., "Agent 1 > Agent 2"). Only do this after fully evaluating the agents. Never include yourself in any comparison. List one line per pair.  
- If it is the first round with no prior completions, write "N/A" here.
</comparison>

<consensus>YES/NO</consensus>
<consensus_reason>
Give a brief justification for your consensus determination. Start with your reasoning process; only then give your "YES" or "NO" final decision.
</consensus_reason>

Key Reminders:
- Use EXACTLY these five XML tags, in strict order, with no extra wrapping or markdown.
- Do NOT compare your own work in the <comparison> section.
- For consensus, ONLY write "YES" or "NO" inside the <consensus> tag.

Objective summary:  
Propose a high-quality solution; evaluate and compare other agents’ solution/evaluation/comparison content using the provided XML tags and order, always reasoning before reaching conclusions, and honestly judge consensus status at the end.
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


def parse_agent_response(
    response: str,
    *,
    author_id: int,
    observation: str = "",
) -> ParsedResponse:
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
            a_id = int(agent_a)
            b_id = int(agent_b)

            # Enforce: the author must not compare themselves.
            if a_id == author_id or b_id == author_id:
                continue
            comparisons.append((a_id, op, b_id))

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


# def parse_pairwise_comparison(response: str, agent_a_id: int, agent_b_id: int) -> str:
#     """Parse the pairwise comparison response.

#     Args:
#         response: The model's comparison response
#         agent_a_id: ID of first agent
#         agent_b_id: ID of second agent

#     Returns:
#         One of: f"AGENT_{agent_a_id}_BETTER", f"AGENT_{agent_b_id}_BETTER", or "TIE"
#     """
#     response = response.strip().upper()

#     # Look for the expected patterns
#     if f"AGENT_{agent_a_id}_BETTER" in response:
#         return f"AGENT_{agent_a_id}_BETTER"
#     elif f"AGENT_{agent_b_id}_BETTER" in response:
#         return f"AGENT_{agent_b_id}_BETTER"
#     elif "TIE" in response:
#         return "TIE"
#     else:
#         # Default to TIE if we can't parse
#         return "TIE"


# def format_conversation_history(
#     agent_responses: list[list[ParsedResponse]], num_agents: int, k_turns: int
# ) -> str:
#     """Format the last K turns of conversation history.

#     Args:
#         agent_responses: List of turns, where each turn is a list of ParsedResponse (one per agent)
#         num_agents: Total number of agents
#         k_turns: Number of recent turns to include

#     Returns:
#         Formatted string of conversation history
#     """
#     if len(agent_responses) == 0:
#         return "(No previous turns yet)"

#     # Get last k turns (or all if fewer than k)
#     recent_turns = agent_responses[-k_turns:]

#     lines = []
#     for turn_idx, turn_responses in enumerate(recent_turns):
#         actual_turn = len(agent_responses) - len(recent_turns) + turn_idx
#         lines.append(f"--- Turn {actual_turn + 1} ---")
#         for agent_id in range(num_agents):
#             if agent_id < len(turn_responses):
#                 resp = turn_responses[agent_id]
#                 lines.append(f"\nAgent {agent_id} Solution:")
#                 lines.append(resp.solution)
#                 lines.append(f"\nAgent {agent_id} Evaluation:")
#                 lines.append(resp.evaluation)
#         lines.append("")

#     return "\n".join(lines)
