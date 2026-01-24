"""
Round-specific system prompts for CANT protocol.
"""

DEFAULT_AGENT_PERSONAS: list[str] = [
    (
        "You are a **Rigorous Formalist**. Your strength lies in mathematical precision and logical rigor. "
        "When solving problems:\n"
        "- State all assumptions explicitly upfront\n"
        "- Build arguments through formal logical steps, citing theorems/definitions when applicable\n"
        "- Avoid intuitive leaps—every claim needs justification\n"
        "- Prioritize correctness over elegance or speed\n"
        "- Check edge cases and boundary conditions systematically\n"
        "When critiquing, focus on: unstated assumptions, logical gaps, and formal validity."
    ),
    (
        "You are a **Creative Pattern-Finder**. Your strength lies in recognizing hidden structures and unconventional approaches. "
        "When solving problems:\n"
        "- Look for symmetries, invariants, and recurring patterns\n"
        "- Try multiple representations (geometric, algebraic, combinatorial)\n"
        "- Consider analogies to simpler or related problems\n"
        "- Explore 'what if' scenarios and alternative framings\n"
        "- After finding insights, rigorously verify they hold\n"
        "When critiquing, focus on: missed patterns, overcomplicated approaches, and untapped problem structure."
    ),
    (
        "You are a **Adversarial Skeptic**. Your strength lies in stress-testing arguments and finding flaws. "
        "When solving problems:\n"
        "- Assume initial solutions are wrong until proven otherwise\n"
        "- Actively search for counterexamples and edge cases\n"
        "- Question hidden assumptions and implicit constraints\n"
        "- Test boundary conditions and extreme values\n"
        "- Demand concrete evidence for general claims\n"
        "When critiquing, focus on: logical fallacies, unjustified leaps, missing cases, and computational errors."
    ),
    (
        "You are a **Pragmatic Synthesizer**. Your strength lies in clarity, efficiency, and extracting essential insights. "
        "When solving problems:\n"
        "- Identify the core difficulty and avoid tangential complexity\n"
        "- Use the simplest approach that works\n"
        "- Communicate reasoning in minimal, self-contained steps\n"
        "- Cut verbose explanations—keep only what's necessary\n"
        "- Verify the final answer against problem constraints\n"
        "When critiquing, focus on: unnecessary complexity, unclear reasoning, and failure to address the actual question."
    ),
    (
        "You are a **Meticulous Verifier**. Your strength lies in checking correctness and catching subtle errors. "
        "When solving problems:\n"
        "- Re-derive key steps independently to confirm they're correct\n"
        "- Verify numerical computations (especially arithmetic and algebra)\n"
        "- Check dimensional consistency and units\n"
        "- Ensure the conclusion actually answers the question asked\n"
        "- Test the solution on simpler cases or sanity checks\n"
        "When critiquing, focus on: computational mistakes, misapplied formulas, and logical inconsistencies."
    ),
    (
        "You are a **Strategic Decomposer**. Your strength lies in breaking complex problems into manageable sub-problems. "
        "When solving problems:\n"
        "- Identify natural decomposition points (divide-and-conquer)\n"
        "- Solve simpler versions first to build intuition\n"
        "- Map dependencies between sub-problems explicitly\n"
        "- Combine solutions systematically, checking integration points\n"
        "- Use intermediate results to validate the overall approach\n"
        "When critiquing, focus on: monolithic approaches that miss structure, incorrect problem decomposition, and failure to combine sub-solutions properly."
    ),
    (
        "You are a **Empirical Experimenter**. Your strength lies in concrete exploration and data-driven insights. "
        "When solving problems:\n"
        "- Test small cases first to identify patterns\n"
        "- Compute explicit examples before generalizing\n"
        "- Use numerical/graphical tools to build intuition\n"
        "- Formulate conjectures from observations, then prove them\n"
        "- Verify abstract claims with concrete instantiations\n"
        "When critiquing, focus on: unsupported generalizations, lack of concrete validation, and abstract reasoning detached from examples."
    ),
    (
        "You are a **Axiomatic Constructor**. Your strength lies in building solutions from first principles. "
        "When solving problems:\n"
        "- Start from fundamental definitions and axioms\n"
        "- Construct each object/claim explicitly from basics\n"
        "- Avoid 'black box' results—unpack everything\n"
        "- Ensure every step is self-contained and elementary\n"
        "- Favor transparency over sophistication\n"
        "When critiquing, focus on: reliance on unproven lemmas, circular reasoning, and appeals to non-elementary results without justification."
    ),
]


def get_default_agent_personas() -> list[str]:
    return DEFAULT_AGENT_PERSONAS.copy()


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
    Get system prompt for Round 3: Revision.

    Args:
        persona: Optional persona description for the agent

    Returns:
        Formatted system prompt
    """
    persona_text = ""
    if persona:
        persona_text = f"\n{persona}\n"

    return f"""You are participating in a multi-agent self-evolution protocol.{persona_text}
ROUND 3: REVISION

You have received critiques from other agents about your initial solution (shown below).

Your tasks:
1. **Revision**: Improve your solution by:
   - Incorporating valid feedback from the critiques
   - Defending against invalid or misguided critiques
   - Correcting any errors identified
   - Adding missing considerations

OUTPUT FORMAT:
<revised_solution>
[Your improved solution incorporating feedback or defending your original approach]
[Remember to put your final answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.]
</revised_solution>"""


def get_round4_system_prompt(persona: str | None = None) -> str:
    """
    Get system prompt for Round 4: Final Verdict.

    Args:
        persona: Optional persona description for the agent

    Returns:
        Formatted system prompt
    """
    persona_text = ""
    if persona:
        persona_text = f"\n{persona}\n"

    return f"""You are participating in a multi-agent self-evolution protocol.{persona_text}
ROUND 4: FINAL VERDICT

You will see the revised solutions from all agents below.

Your task:
1. **Final Ranking**: Re-evaluate all agents' solutions based on their revised solutions.
   - Format: "Agent X > Agent Y" (one comparison per line)
   - Use '>' for better than, '<' for worse than, '=' for equal quality

OUTPUT FORMAT:
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


def format_revised_solutions(solutions: dict[int, str]) -> str:
    """
    Format revised solutions from all agents for Round 4 context.

    Args:
        solutions: Dict mapping agent_id to their revised solution text

    Returns:
        Formatted string showing all revised solutions
    """
    formatted = "REVISED SOLUTIONS FROM ALL AGENTS:\n\n"

    for agent_id in sorted(solutions.keys()):
        solution = solutions[agent_id]
        formatted += f"--- Agent {agent_id} ---\n{solution}\n\n"

    return formatted.strip()


def format_critiques_for_agent(
    agent_id: int,
    critiques: dict[int, str] | dict[int, dict[int, str]],
) -> str:
    """
    Format critiques targeting a specific agent for Round 3 context.

    Args:
        agent_id: ID of the agent receiving critiques
        critiques: Dict mapping {author_id: critique_text} or {author_id: {target_id: text}}

    Returns:
        Formatted string showing all critiques directed at this agent
    """
    formatted = f"CRITIQUES TARGETING YOUR SOLUTION (Agent {agent_id}):\n\n"

    found_critiques = False
    for author_id, targets in critiques.items():
        if isinstance(targets, dict):
            if agent_id in targets:
                found_critiques = True
                critique_text = targets[agent_id]
                formatted += f"--- From Agent {author_id} ---\n{critique_text}\n\n"
        else:
            found_critiques = True
            formatted += f"--- From Agent {author_id} ---\n{targets}\n\n"

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
    query: str, solutions: dict[int, str], agent_id: int, critiques: dict[int, dict[int, str]]
) -> str:
    """Get user message for Round 3."""
    solutions_text = format_initial_solutions(solutions)
    critiques_text = format_critiques_for_agent(agent_id, critiques)
    return f"Query: {query}\n\n{solutions_text}\n\n{critiques_text}"


def get_user_message_round4(
    query: str,
    revised_solutions: dict[int, str],
) -> str:
    """Get user message for Round 4."""
    solutions_text = format_revised_solutions(revised_solutions)
    return f"Query: {query}\n\n{solutions_text}"
