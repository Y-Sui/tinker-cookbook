"""Prompts and XML parsing utilities for multi-agent debate."""

import re
from dataclasses import dataclass

# Agent personas for diversity - each agent has a distinct reasoning style
# Temperature suggestions: lower for methodical (0.7), higher for creative (1.2)
AGENT_PERSONAS = {
    0: {
        "name": "The Methodical Analyst",
        "style": "You approach problems systematically and step-by-step. You break down complex problems into smaller parts, verify each step carefully, and prefer rigorous logical deductions over intuition. You are thorough and rarely skip steps.",
        "strength": "detailed verification and catching errors in reasoning chains",
        "suggested_temperature": 0.6,  # Lower temperature for more deterministic, careful reasoning
    },
    1: {
        "name": "The Creative Problem-Solver",
        "style": "You think outside the box and explore unconventional approaches. You look for elegant shortcuts, pattern recognition, and analogies to similar problems. You're willing to try multiple approaches and pivot quickly if one doesn't work.",
        "strength": "finding novel solutions and alternative approaches others might miss",
        "suggested_temperature": 1.0,  # Higher temperature for more creative exploration
    },
    2: {
        "name": "The Devil's Advocate",
        "style": "You are naturally skeptical and question assumptions. You actively look for counterexamples, edge cases, and potential flaws in reasoning. You stress-test solutions before accepting them and challenge others' conclusions constructively.",
        "strength": "identifying weaknesses, edge cases, and potential errors in proposed solutions",
        "suggested_temperature": 0.9,  # Moderate temperature for balanced critical thinking
    },
    3: {
        "name": "The Synthesizer",
        "style": "You excel at combining ideas from multiple sources. You look for common ground between different approaches, identify the best elements from various solutions, and build comprehensive answers that incorporate diverse perspectives.",
        "strength": "integrating multiple viewpoints and building consensus solutions",
        "suggested_temperature": 1.0,  # Standard temperature for balanced synthesis
    },
    4: {
        "name": "The First Principles Thinker",
        "style": "You always go back to fundamentals. You question whether the problem is being approached correctly from the start, verify definitions, and ensure basic assumptions hold. You prefer building solutions from foundational truths.",
        "strength": "ensuring correctness from the ground up and avoiding assumption-based errors",
        "suggested_temperature": 0.8,  # Lower temperature for careful foundational reasoning
    },
}


def get_agent_persona(agent_id: int) -> dict:
    """Get the persona for an agent, cycling through available personas."""
    return AGENT_PERSONAS[agent_id % len(AGENT_PERSONAS)]


def get_agent_temperature(agent_id: int) -> float:
    """Get the suggested temperature for an agent based on their persona.

    Returns the persona's suggested temperature, cycling through personas
    if agent_id exceeds the number of defined personas.
    """
    persona = get_agent_persona(agent_id)
    return persona.get("suggested_temperature", 1.0)


def format_persona_intro(agent_id: int) -> str:
    """Format the persona introduction for an agent."""
    persona = get_agent_persona(agent_id)
    return (
        f"Your reasoning style is '{persona['name']}': {persona['style']} "
        f"Your unique strength is {persona['strength']}."
    )


SUMMARIZER_SYSTEM_PROMPT = """
Write a concise, information-dense summary that preserves:
- The user question
- Each agent's solution
- Each agent's evaluations
Don't add new information. Output plain text only in the following format. 

Format: 

User Query: [The user query text]

Turn-by-turn summary:

Agent [first_agent_id]:
- Solution: []
- Evaluation: []

Agent [second_agent_id]:
- Solution: []
- Evaluation: []

Please summarize the debate in the above format.
"""

AGENT_SYSTEM_PROMPT = """
You are a participant in a multi-agent debate and reasoning system.

{persona_intro}

GOAL:
Collaborate to provide the best possible answer to the user query while evaluating all agents' contributions.

INPUT CONTEXT:
The input contains a User Query and a History of previous turns (solutions and evaluations from all agents).

INSTRUCTIONS:

1. **Construct Solution**:
   - Think deeply about the user query.
   - Formulate a comprehensive, nuanced, and accurate answer.
   - Focus only on the query; do not reference other agents in the solution.

2. **Evaluate All Agents**:
   - Review the "History" provided.
   - For EACH agent in history, analyze their solution accuracy and reasoning quality.
   - Provide balanced, objective critiques.

3. **Rank Agents (Pairwise Comparison)**:
   - Compare the overall quality of every pair of agents visible in history.
   - All agents can be included in rankings.
   - Provide brief justification for each comparison.

OUTPUT FORMAT:
Output response in these XML tags:

<solution>
[Answer to the user query.]
</solution>

<evaluation>
[If no agents in history, write "N/A".]
[Otherwise, for each agent: brief critique of their solution and reasoning quality.]
</evaluation>

<comparison>
[If fewer than 2 agents in history, write "N/A".]
[Compare pairs of agents. One per line. Format: Agent X > Agent Y: [brief reason]]
[Use >, <, or = operators. For ties, use = (e.g., "Agent 0 = Agent 1: both correct")]
[REQUIRED: Provide brief justification after each comparison.]
</comparison>
"""

VERIFIABLE_AGENT_SYSTEM_PROMPT = """
You are a math problem solver in a multi-agent discussion.

{persona_intro}

INPUT CONTEXT:
The input contains a User Query and a History of previous turns (if any). The User Query is a verifiable problem requiring a precise final answer in \\boxed{{answer}} format. The History contains all agents' solutions and evaluations from previous rounds.

INSTRUCTIONS:
Structure the response into three sections: Solution, Evaluation, and Comparison. Follow the instructions for each section carefully.

**1. Derive Solution**:
   - Use step-by-step chain-of-thought reasoning.
   - Verify every calculation and logical inference.
   - **CRITICAL**: MUST end solution with the final answer in LaTeX boxed format, e.g., \\boxed{{42}}.

**2. Evaluate All Agents**:
   - Review the "History of previous turns", which contains the full logs of all agents' solutions and evaluations.
   - Check every line of all agents' solutions and evaluations (if any).
   - Critique each agent on two fronts:
    (1) solution correctness, completeness, and reasoning quality - did they arrive at the right final answer and adequately justify it?
    (2) evaluation quality - did they fairly and accurately assess others? did they spot errors or hallucinate? is their evaluation aligned with their pairwise comparisons?

**3. Compare Agents (Pairwise)**:
   - Perform pairwise comparisons of all agents visible in history based on their solutions and evaluations.
   - Correctness is paramount. An agent with the correct final answer (derived correctly) > Agent with wrong answer.
   - If both agents' solutions are correct, compare their reasoning depth, justification, and evaluation quality.
   - **REQUIRED**: Provide brief justification for each comparison.

OUTPUT FORMAT:

<solution>
[Step-by-step solution. MUST end with \\boxed{{answer}}]
</solution>

<evaluation>
[Review all agents' solutions. Write "N/A" if no history.]
</evaluation>

<comparison>
[Rank pairs of agents. Format: Agent X > Agent Y: [brief reason]
[Write "N/A" if fewer than 2 agents in history.]
[Use >, <, or = operators.]
[REQUIRED: Provide brief justification after each comparison.]
</comparison>
""".strip()


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
    comparison_lines_total: int = 0
    comparison_lines_invalid: int = 0
    comparison_pairs_tie: int = 0


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
        ParsedResponse with extracted fields. If parsing fails or tags are incomplete
        (e.g., due to max_tokens cutoff), returns a ParsedResponse with placeholder
        values marked as [INCOMPLETE] or [PARSE_ERROR].
    """
    # Normalize common wrappers (e.g., markdown code fences) and strip Qwen-style thinking blocks.
    response = response.strip()
    # Remove fenced blocks like ```xml ... ```
    if response.startswith("```"):
        response = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", response)
        response = re.sub(r"\n```$", "", response)
        response = response.strip()

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
        # Fallback: parse individual tags, handling incomplete/malformed responses gracefully
        raw_response = response

        # Try to extract solution
        solution_matches = list(
            re.finditer(
                r"^\s*<solution>\s*(.*?)</solution>\s*$",
                response,
                flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
            )
        )
        if solution_matches:
            solution = solution_matches[-1].group(1).strip()
        else:
            # Handle incomplete/missing tag (e.g., max_tokens cutoff)
            incomplete_solution = re.search(
                r"<solution>\s*(.*?)(?:</solution>|$)",
                response,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if incomplete_solution:
                solution = "[INCOMPLETE] " + incomplete_solution.group(1).strip()
            else:
                solution = "[PARSE_ERROR: Missing <solution> tag]"

        # Try to extract evaluation
        evaluation_matches = list(
            re.finditer(
                r"^\s*<evaluation>\s*(.*?)</evaluation>\s*$",
                response,
                flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
            )
        )
        if evaluation_matches:
            evaluation = evaluation_matches[-1].group(1).strip()
        else:
            # Handle incomplete/missing tag
            incomplete_eval = re.search(
                r"<evaluation>\s*(.*?)(?:</evaluation>|$)",
                response,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if incomplete_eval:
                evaluation = "[INCOMPLETE] " + incomplete_eval.group(1).strip()
            else:
                evaluation = "N/A"

        # Try to extract comparison
        comp_matches = list(
            re.finditer(
                r"^\s*<comparison>\s*(.*?)</comparison>\s*$",
                response,
                flags=re.DOTALL | re.IGNORECASE | re.MULTILINE,
            )
        )
        if comp_matches:
            comparison_text = comp_matches[-1].group(1).strip()
        else:
            # Handle incomplete/missing tag
            incomplete_comp = re.search(
                r"<comparison>\s*(.*?)(?:</comparison>|$)",
                response,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if incomplete_comp:
                comparison_text = "[INCOMPLETE] " + incomplete_comp.group(1).strip()
            else:
                comparison_text = "N/A"

    comparisons: list[tuple[int, str, int]] = []
    self_comparisons_dropped = 0
    comparison_lines_total = 0
    comparison_lines_invalid = 0
    comparison_pairs_tie = 0
    if comparison_text:
        # Match "Agent 1 > Agent 2" or "Agent 1 = Agent 2" (also tolerates legacy "(R1)" round annotations)
        pair_pattern = re.compile(
            r"Agent\s+(\d+)(?:\s*\(R\d+\))?\s*([><])\s*Agent\s+(\d+)(?:\s*\(R\d+\))?",
            flags=re.IGNORECASE,
        )
        tie_symbol_pattern = re.compile(
            r"Agent\s+(\d+)(?:\s*\(R\d+\))?\s*(=|≈|==)\s*Agent\s+(\d+)(?:\s*\(R\d+\))?",
            flags=re.IGNORECASE,
        )
        tie_word_pattern = re.compile(
            r"Agent\s+(\d+)\s+and\s+Agent\s+(\d+)\s+(?:are|is)\s+(?:tied|equal)",
            flags=re.IGNORECASE,
        )
        lines = [line.strip() for line in comparison_text.splitlines() if line.strip()]
        comparison_lines_total = sum(1 for line in lines if line.lower() != "n/a")
        for line in lines:
            if line.lower() == "n/a":
                continue
            line_has_any = False
            # Match strict comparisons (> or <)
            for match in pair_pattern.finditer(line):
                line_has_any = True
                agent_a, op, agent_b = match.groups()
                a_id = int(agent_a)
                b_id = int(agent_b)
                if a_id == author_id or b_id == author_id:
                    self_comparisons_dropped += 1
                    continue
                comparisons.append((a_id, op, b_id))
            # Match ties with symbols (=, ≈, ==)
            for match in tie_symbol_pattern.finditer(line):
                line_has_any = True
                agent_a, tie_op, agent_b = match.groups()
                a_id = int(agent_a)
                b_id = int(agent_b)
                if a_id == author_id or b_id == author_id:
                    self_comparisons_dropped += 1
                    continue
                comparisons.append((a_id, "=", b_id))
                comparison_pairs_tie += 1
            # Match ties with words ("Agent X and Agent Y are tied")
            for match in tie_word_pattern.finditer(line):
                line_has_any = True
                agent_a, agent_b = match.groups()
                a_id = int(agent_a)
                b_id = int(agent_b)
                if a_id == author_id or b_id == author_id:
                    self_comparisons_dropped += 1
                    continue
                comparisons.append((a_id, "=", b_id))
                comparison_pairs_tie += 1
            if not line_has_any:
                comparison_lines_invalid += 1

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
        comparison_lines_total=comparison_lines_total,
        comparison_lines_invalid=comparison_lines_invalid,
        comparison_pairs_tie=comparison_pairs_tie,
    )
