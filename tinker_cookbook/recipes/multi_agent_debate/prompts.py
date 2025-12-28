"""Prompts and XML parsing utilities for multi-agent debate."""

import re
from dataclasses import dataclass

AGENT_SYSTEM_PROMPT = """You are Agent {agent_id} participating in a multi-agent self-play discussion to collaboratively answer user query.

OBJECTIVES:
- Propose or refine a high-quality, detailed solution to the query.
- Evaluate other agents' contributions across three dimensions:
   - Solution quality: Assess the correctness, completeness, and reasoning of their proposed solutions
   - Evaluation quality (meta-evaluation): Critique the fairness, accuracy, and helpfulness of their assessments of other agents
   - Comparison quality: Review whether their pairwise rankings are justified and consistent
- Provide pairwise rankings comparing other agents' overall contributions (solution + evaluation + comparison combined) (Do NOT compare yourself.)

CRITICAL INSTRUCTIONS:
- Use chain-of-thought reasoning throughout: First explain your reasoning, THEN state conclusions
- Maintain strict XML tag order: <solution>, <evaluation>, <comparison>
- In <solution>: Focus solely on answering the query—make NO judgments about other agents
- In <evaluation> and <comparison>: Exclude yourself (Agent {agent_id}) from all assessments
- Follow the exact XML formatting specified below

---

OUTPUT FORMAT (strict order required):

<solution>
[Provide your detailed, well-reasoned answer to the user query here. Use step-by-step reasoning where appropriate. Do NOT mention, compare, or evaluate other agents in this section.]
</solution>

<evaluation>
[Please review the previous turns of conversation from other agents, including their solutions, evaluations, and comparisons.
For each other agent's contribution, provide:
1. **Solution critique**: Analyze the quality, completeness, and reasoning of their proposed solution
2. **Meta-evaluation**: Assess whether their evaluations of other agents are fair, accurate, and constructive
3. **Comparison quality review**: Evaluate whether their pairwise rankings are justified and consistent

Should explain what you observe and why it matters for each critique (both solution critique and meta-evaluation of their assessment/comparison quality), and only then draw any judgment or classification.
- If there are no prior completions visible in the conversation history, write "N/A" here.]
</evaluation>

<comparison>
[Please review the previous turns of conversation from other agents. Carefully compare their overall contributions (solution + evaluation + comparison). Then, provide pairwise rankings or ties between all unordered pairs of the other agents’ completions. (e.g., "Agent 1 > Agent 2") Never compare yourself. 
Previous conversation format is as follows:
== Turn current_turn_idx/max_turns (Agent agent_id) ==
Agent agent_id's Solution:
solution text
Agent agent_id's Evaluation:
evaluation text
Agent agent_id's Comparison:
comparison text
== End of Turn ==

Output pairwise rankings using this format:
Agent X > Agent Y    [Agent X's overall contribution is stronger]
Agent A = Agent B    [Contributions are roughly equal in quality]
Agent M < Agent N    [Agent M's contribution is weaker]

...
Requirements:
- Only do this after fully evaluating the agents. 
- Compare ALL unordered pairs of other agents (Never include yourself agent-{agent_id} in any comparison. )
- One comparison per line
- Use only >, <, or = operators
- Base rankings on combined quality across solution, evaluation, and comparison
- If there are fewer than two other agents, write "N/A" here.]
</comparison>

Key Reminders:
- Use EXACTLY these three XML tags: <solution>, <evaluation>, <comparison>
- No additional wrapping tags, markdown code blocks, or commentary outside tags
- Never include "Agent {agent_id}" (yourself) in <evaluation> or <comparison> sections

Objective summary:  
Propose a high-quality solution; evaluate and compare other agents’ solution/evaluation/comparison content using the provided XML tags and order, always reasoning before reaching conclusions.
"""

VERIFIABLE_AGENT_SYSTEM_PROMPT = """You are Agent {agent_id} participating in a multi-agent self-play debate to solve a verifiable math problem.

Your objectives:
- Propose or refine a correct solution to the problem.
- In <solution>, include a final answer written in \\boxed{{...}} format.
- Evaluate other agents’ recent contributions (solution + critique quality), and compare other agents’ outputs.

OUTPUT FORMAT (use exact XML tags, in this order):

<solution>
Your detailed solution. You MUST include a final answer in \\boxed{{...}} format.
</solution>

<evaluation>
Evaluate other agents’ recent work. If there are no prior completions visible, write "N/A".
</evaluation>

<comparison>
Compare only OTHER agents visible in the history (never include yourself). If fewer than two other completions are visible, write "N/A".
</comparison>

Key reminders:
- Use EXACTLY these three XML tags, in strict order.
- Do NOT compare your own work in <comparison>.
"""


@dataclass
class ParsedResponse:
    """Parsed agent response."""

    solution: str
    evaluation: str
    comparisons: list[tuple[int, str, int]]  # List of (agent_a_id, op, agent_b_id) tuples
    raw_response: str
    comparison_text: str = ""
    author_id: int = -1
    observation: str = ""
    thinking: str = ""
    self_comparisons_dropped: int = 0


def parse_agent_response(
    response: str,
    *,
    author_id: int = -1,
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

    # Remove Qwen-style thinking blocks entirely, but keep them for logging/debugging.
    # Important: sometimes the model mentions literal "<solution>" in its preamble/thoughts,
    # and we must not treat those as the actual structured output.
    thinking_chunks = re.findall(r"<think>(.*?)</think>", response, flags=re.IGNORECASE | re.DOTALL)
    thinking = "\n\n".join(chunk.strip() for chunk in thinking_chunks if chunk.strip())
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.IGNORECASE | re.DOTALL).strip()

    # Prefer parsing the *last* complete XML block in the message.
    # This avoids accidentally parsing tag-shaped text in the model's preamble/reasoning.
    # Require tags to start on a new line (or start-of-string). This avoids matching
    # tag-shaped text embedded inside the model's preamble, e.g. "... like <solution> is first ...".
    block_pattern = re.compile(
        r"^\s*<solution>\s*(?P<solution>.*?)</solution>\s*"
        r"^\s*<evaluation>\s*(?P<evaluation>.*?)</evaluation>\s*"
        r"^\s*<comparison>\s*(?P<comparison>.*?)</comparison>\s*$",
        flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )
    block_matches = list(block_pattern.finditer(response))
    if block_matches:
        match = block_matches[-1]
        solution = match.group("solution").strip()
        evaluation = match.group("evaluation").strip()
        comparison_text = match.group("comparison").strip()
        raw_response = match.group(0).strip()
    else:
        # Fallback: older behavior where <comparison> might be missing.
        raw_response = response

        solution_matches = list(
            re.finditer(
                r"^\s*<solution>\s*(.*?)</solution>\s*$",
                response,
                flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
            )
        )
        if not solution_matches:
            raise ValueError(f"Missing <solution> tag in response: {response[:200]}")
        solution = solution_matches[-1].group(1).strip()

        evaluation_matches = list(
            re.finditer(
                r"^\s*<evaluation>\s*(.*?)</evaluation>\s*$",
                response,
                flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
            )
        )
        if not evaluation_matches:
            raise ValueError(f"Missing <evaluation> tag in response: {response[:200]}")
        evaluation = evaluation_matches[-1].group(1).strip()

        comp_matches = list(
            re.finditer(
                r"^\s*<comparison>\s*(.*?)</comparison>\s*$",
                response,
                flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
            )
        )
        comparison_text = comp_matches[-1].group(1).strip() if comp_matches else ""

    comparisons: list[tuple[int, str, int]] = []
    self_comparisons_dropped = 0
    if comparison_text:
        pairs = re.findall(r"Agent\s+(\d+)\s*([>=])\s*Agent\s+(\d+)", comparison_text)
        for agent_a, op, agent_b in pairs:
            a_id = int(agent_a)
            b_id = int(agent_b)
            if a_id == author_id or b_id == author_id:
                self_comparisons_dropped += 1
                continue
            comparisons.append((a_id, op, b_id))

    return ParsedResponse(
        solution=solution,
        evaluation=evaluation,
        comparisons=comparisons,
        raw_response=raw_response,
        comparison_text=comparison_text,
        author_id=author_id,
        observation=observation,
        thinking=thinking,
        self_comparisons_dropped=self_comparisons_dropped,
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
