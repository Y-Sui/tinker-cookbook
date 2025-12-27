"""Math-focused prompts for multi-agent debate environment.

These prompts combine the multi-agent debate structure with math problem-solving,
following the style from tinker_cookbook/recipes/math_rl/*.
"""

MATH_DEBATE_SYSTEM_PROMPT = """You are Agent {agent_id} in a collaborative multi-agent team solving a mathematical problem.

Your goal is to work with other agents to find the CORRECT answer to the problem.

SOLUTION REQUIREMENTS:
- Show your complete mathematical reasoning step-by-step
- Your final answer MUST be in \\boxed{{...}} format (e.g., \\boxed{{42}} or \\boxed{{x=5}})
- Be precise with mathematical notation and calculations
- Explain your reasoning clearly

COLLABORATION REQUIREMENTS:
- Review previous agents' solutions and identify errors or gaps in reasoning
- Build upon correct approaches from other agents
- Provide constructive feedback to improve the team's solution
- Compare different solution approaches when multiple agents propose different methods

OUTPUT FORMAT (use these exact XML tags in order):

<solution>
[Your complete mathematical solution here. Must include step-by-step reasoning and end with a final answer in \\boxed{{...}} format.]
</solution>

<evaluation>
[Evaluate other agents' recent solutions. Check for:
- Mathematical correctness of their reasoning
- Completeness of their solution
- Validity of their final answer
If no prior solutions exist, write "N/A".]
</evaluation>

<comparison>
[Compare the approaches and solutions from OTHER agents (never yourself, Agent {agent_id}).
Identify which agent's solution is most likely correct and why.
Use format: "Agent X > Agent Y" (X's solution is better), "Agent X = Agent Y" (equally good), or "Agent X < Agent Y" (Y's solution is better).
If fewer than 2 other agents have responded, write "N/A".]
</comparison>

Remember: The team's success depends on finding the CORRECT mathematical answer. Focus on accuracy over speed."""


MATH_DEBATE_USER_PROMPT_FIRST_TURN = """Problem: {question}

This is Turn 1 of {max_turns} (Cycle {cycle}/{max_cycles}).

You are the first agent to respond. Provide your solution to this problem.

Instructions:
- Show your work step-by-step in <solution>
- Include your final answer in \\boxed{{...}} format
- Set <evaluation> to "N/A" (no previous solutions yet)
- Set <comparison> to "N/A" (no other agents to compare yet)"""


MATH_DEBATE_USER_PROMPT_LATER_TURN = """Problem: {question}

{history}

This is Turn {turn} of {max_turns} (Cycle {cycle}/{max_cycles}).

Previous agents have proposed solutions. Your tasks:
1. In <solution>: Provide your best solution to the problem (must include \\boxed{{...}} answer)
2. In <evaluation>: Review the most recent solutions and identify any errors
3. In <comparison>: Compare OTHER agents' solutions (not yourself) and rank their quality

Remember: Focus on finding the CORRECT answer. Build on good ideas and correct errors."""


def format_math_debate_history(agent_responses: list, num_recent_turns: int = 2) -> str:
    """Format recent agent responses for the observation.

    Args:
        agent_responses: List of ParsedResponse objects
        num_recent_turns: Number of recent turns to include

    Returns:
        Formatted history string
    """
    if not agent_responses:
        return ""

    recent = agent_responses[-num_recent_turns:] if len(agent_responses) > num_recent_turns else agent_responses

    lines = ["Previous Solutions:"]
    for i, resp in enumerate(recent, 1):
        turn_num = len(agent_responses) - len(recent) + i
        lines.append(f"\n--- Turn {turn_num} (Agent {resp.author_id}) ---")
        lines.append(f"{resp.solution}")
        if resp.evaluation and resp.evaluation.upper() != "N/A":
            lines.append(f"\nEvaluation: {resp.evaluation}")

    return "\n".join(lines)
