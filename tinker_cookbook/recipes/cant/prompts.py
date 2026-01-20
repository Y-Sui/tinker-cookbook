"""
Round-specific system prompts for CANT protocol.
"""


def get_round1_system_prompt(persona: str | None = None) -> str:
    """
    Get system prompt for Round 1: Proposal.

    Args:
        persona: Optional persona description for the agent

    Returns:
        Formatted system prompt
    """
    persona_text = ""
    if persona:
        persona_text = f"\n{persona}\n"

    return f"""You are participating in a multi-agent self-evolution protocol.{persona_text}
ROUND 1: PROPOSAL

Your task is to provide an initial solution to the query below.

Think carefully and provide your best initial answer.

OUTPUT FORMAT:
<initial_solution>
[Your proposed solution to the query]
</initial_solution>"""


def get_round2_system_prompt(persona: str | None = None) -> str:
    """
    Get system prompt for Round 2: Blind Evaluation + Critique.

    Args:
        persona: Optional persona description for the agent

    Returns:
        Formatted system prompt
    """
    persona_text = ""
    if persona:
        persona_text = f"\n{persona}\n"

    return f"""You are participating in a multi-agent self-evolution protocol.{persona_text}
ROUND 2: BLIND EVALUATION + CRITIQUE

You will see the initial solutions from all agents (including yourself) below.

Your tasks:
1. **Blind Ranking**: Provide pairwise comparisons ranking the quality of initial solutions.
   - Format: "Agent X > Agent Y" (one comparison per line)
   - You may include yourself in rankings
   - Use '>' for better than, '<' for worse than, '=' for equal quality

2. **Targeted Critique**: Select specific agents to critique and provide detailed feedback.
   - Use <target>Agent k</target> to specify who you're critiquing
   - Identify logical fallacies, errors, missing considerations, or areas for improvement
   - Be constructive and specific

OUTPUT FORMAT:
<blind_ranking>
Agent 0 > Agent 1
Agent 2 > Agent 0
[More pairwise comparisons...]
</blind_ranking>

<critique>
<target>Agent 1</target>
[Your critique of Agent 1's solution - be specific about flaws or missing elements]

<target>Agent 2</target>
[Your critique of Agent 2's solution]
</critique>

Note: You can critique as many or as few agents as you think necessary."""


def get_round3_system_prompt(persona: str | None = None) -> str:
    """
    Get system prompt for Round 3: Revision + Final Verdict.

    Args:
        persona: Optional persona description for the agent

    Returns:
        Formatted system prompt
    """
    persona_text = ""
    if persona:
        persona_text = f"\n{persona}\n"

    return f"""You are participating in a multi-agent self-evolution protocol.{persona_text}
ROUND 3: REVISION + FINAL VERDICT

You have received critiques from other agents about your initial solution (shown below).

Your tasks:
1. **Revision**: Improve your solution by:
   - Incorporating valid feedback from the critiques
   - Defending against invalid or misguided critiques
   - Correcting any errors identified
   - Adding missing considerations

2. **Final Ranking**: Re-evaluate all agents' solutions based on:
   - Your revised understanding after seeing critiques
   - The quality of each agent's initial solution
   - Format: "Agent X > Agent Y" (one comparison per line)

OUTPUT FORMAT:
<revised_solution>
[Your improved solution incorporating feedback or defending your original approach]
</revised_solution>

<final_ranking>
Agent 0 > Agent 1
Agent 2 > Agent 0
[More pairwise comparisons...]
</final_ranking>"""


def format_initial_solutions(solutions: dict[int, str]) -> str:
    """
    Format initial solutions from all agents for Round 2 context.

    Args:
        solutions: Dict mapping agent_id to their initial solution text

    Returns:
        Formatted string showing all solutions
    """
    formatted = "INITIAL SOLUTIONS FROM ALL AGENTS:\n\n"

    for agent_id in sorted(solutions.keys()):
        solution = solutions[agent_id]
        formatted += f"--- Agent {agent_id} ---\n{solution}\n\n"

    return formatted.strip()


def format_critiques_for_agent(
    agent_id: int,
    critiques: dict[int, dict[int, str]]
) -> str:
    """
    Format critiques targeting a specific agent for Round 3 context.

    Args:
        agent_id: ID of the agent receiving critiques
        critiques: Dict mapping {author_id: {target_id: critique_text}}

    Returns:
        Formatted string showing all critiques directed at this agent
    """
    formatted = f"CRITIQUES TARGETING YOUR SOLUTION (Agent {agent_id}):\n\n"

    found_critiques = False
    for author_id, targets in critiques.items():
        if agent_id in targets:
            found_critiques = True
            critique_text = targets[agent_id]
            formatted += f"--- From Agent {author_id} ---\n{critique_text}\n\n"

    if not found_critiques:
        formatted += "No critiques were directed at your solution.\n"

    return formatted.strip()


def get_user_message_round1(query: str) -> str:
    """Get user message for Round 1."""
    return f"Query: {query}"


def get_user_message_round2(query: str, solutions: dict[int, str]) -> str:
    """Get user message for Round 2."""
    solutions_text = format_initial_solutions(solutions)
    return f"Query: {query}\n\n{solutions_text}"


def get_user_message_round3(
    query: str,
    solutions: dict[int, str],
    agent_id: int,
    critiques: dict[int, dict[int, str]]
) -> str:
    """Get user message for Round 3."""
    solutions_text = format_initial_solutions(solutions)
    critiques_text = format_critiques_for_agent(agent_id, critiques)
    return f"Query: {query}\n\n{solutions_text}\n\n{critiques_text}"
