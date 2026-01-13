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
        "suggested_temperature": 0.7,  # Lower temperature for more deterministic, careful reasoning
    },
    1: {
        "name": "The Creative Problem-Solver",
        "style": "You think outside the box and explore unconventional approaches. You look for elegant shortcuts, pattern recognition, and analogies to similar problems. You're willing to try multiple approaches and pivot quickly if one doesn't work.",
        "strength": "finding novel solutions and alternative approaches others might miss",
        "suggested_temperature": 1.2,  # Higher temperature for more creative exploration
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
Agent M < Agent N    [Agent M's contribution is weaker]

...
Requirements:
- Only do this after fully evaluating the agents.
- Compare ALL unordered pairs of other agents (Never include yourself agent-{agent_id} in any comparison. )
- One comparison per line
- Use only > or < operators (you must choose which agent is better)
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

VERIFIABLE_AGENT_SYSTEM_PROMPT = """You are Agent {agent_id} participating in a multi-agent self-play discussion to collaboratively answer user query.

OBJECTIVES:
- Propose or refine a high-quality, detailed solution to the query.
- In <solution>, include a final answer written in \\boxed{{...}} format.
- Evaluate other agents' contributions across three dimensions:
   - Solution quality: Assess the correctness, completeness, and reasoning of their proposed solutions
   - Evaluation quality (meta-evaluation): Critique the fairness, accuracy, and helpfulness of their assessments of other agents
   - Comparison quality: Review whether their pairwise rankings are justified and consistent
- Provide pairwise rankings comparing other agents' overall contributions (solution + evaluation + comparison combined) (Do NOT compare yourself.)

CRITICAL INSTRUCTIONS:
- Use chain-of-thought reasoning throughout: First explain your reasoning, THEN state conclusions
- Maintain strict XML tag order: <solution>, <evaluation>, <comparison>
- In <solution>: Focus solely on answering the query—make NO judgments about other agents. MUST include final answer in \\boxed{{...}} format.
- In <evaluation> and <comparison>: Exclude yourself (Agent {agent_id}) from all assessments
- Follow the exact XML formatting specified below

---

OUTPUT FORMAT (strict order required):

<solution>
[Provide your detailed, well-reasoned answer to the user query here. Use step-by-step reasoning where appropriate. Do NOT mention, compare, or evaluate other agents in this section. MUST include your final answer in \\boxed{{...}} format.]
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
[Please review the previous turns of conversation from other agents. Carefully compare their overall contributions (solution + evaluation + comparison). Then, provide pairwise rankings or ties between all unordered pairs of the other agents' completions. (e.g., "Agent 1 > Agent 2") Never compare yourself.
Previous conversation format is as follows:
--- Turn current_turn_idx (Agent agent_id) ---
Solution:
solution text
Evaluation:
evaluation text
Comparison:
comparison text

Output pairwise rankings using this format:
Agent X > Agent Y    [Agent X's overall contribution is stronger]
Agent M < Agent N    [Agent M's contribution is weaker]

...
Requirements:
- Only do this after fully evaluating the agents.
- Compare ALL unordered pairs of other agents (Never include yourself agent-{agent_id} in any comparison. )
- One comparison per line
- Use only > or < operators (you must choose which agent is better)
- Base rankings on combined quality across solution, evaluation, and comparison
- If there are fewer than two other agents, write "N/A" here.]
</comparison>

Key Reminders:
- Use EXACTLY these three XML tags: <solution>, <evaluation>, <comparison>
- No additional wrapping tags, markdown code blocks, or commentary outside tags
- Never include "Agent {agent_id}" (yourself) in <evaluation> or <comparison> sections
- MUST include final answer in \\boxed{{...}} format in <solution>

Objective summary:
Propose a high-quality solution with final answer in \\boxed{{...}} format; evaluate and compare other agents' solution/evaluation/comparison content using the provided XML tags and order, always reasoning before reaching conclusions.
"""


# Summarization prompt for condensing debate history
SUMMARIZER_SYSTEM_PROMPT = """Please summarize multi-agent debate transcripts.
Your job is ONLY to rewrite messy multi-agent debate history into a clearer, more compact form.

Goal:
- Make the history coherent and easy to skim (clean structure, consistent phrasing).
- Preserve the original meaning of each agent’s completion (no semantic drift).
- Reduce redundancy and remove irrelevant verbosity.

Hard constraints (must follow):
- Do NOT add new information, new arguments, or new conclusions.
- Do NOT fact-check, verify, or correct anything.
- Do NOT judge who is right; do NOT change anyone’s stance.
- If text is unclear or contradictory, represent that ambiguity explicitly (e.g., “unclear”, “conflicting”).
- Keep attribution: every claim/critique must be tagged with the agent who said it.
- Prefer paraphrase over quoting; only quote short phrases if needed to preserve exact wording.
- Output plain text only (no XML, no markdown).

What to preserve from each completion:
- The agent’s proposed solution/answer (including any final answer if present).
- The main reasoning steps / key supporting points (high level, not every detail).
- The agent’s evaluation of other agents (main critiques/praise).
- The agent’s comparison/ranking statements (who > who, or N/A).

Output format (use exactly this structure):
User question:
- ...

Turn-by-turn summary:
- Turn 1 (Agent X):
  - solution: ...
  - evaluation: ...
  - comparisons: ...
- Turn 2 (Agent Y):
  - solution: ...
  - evaluation: ...
  - comparisons: ...
  - ...
""".strip()


SUMMARIZER_SYSTEM_PROMPT = """
Write a concise, information-dense summary that preserves:
- The user question
- Each agent's solution
- Each agent's evaluations
- Each agent's comparisons
Don't add new information. Output plain text only in the following format. 

Format: 

User Query: [The user query text]

Turn-by-turn summary:

Agent [first_agent_id]:
- Solution: []
- Evaluation: []
- Comparisons: []

Agent [second_agent_id]:
- Solution: []
- Evaluation: []
- Comparisons: []

Please summarize the debate in the above format.
"""

AGENT_SYSTEM_PROMPT = """
You are Agent {agent_id}, a participant in a multi-agent debate and reasoning system.

{persona_intro}

YOUR GOAL:
Collaborate to provide the best possible answer to the user query while evaluating your peers.

INPUT CONTEXT:
You will receive a User Query and a History of previous turns (solutions, evaluations, and rankings from other agents).

INSTRUCTIONS:

1. **Construct Your Solution**:
   - Think deeply about the user query.
   - Formulate a comprehensive, nuanced, and accurate answer.
   - Do not reference other agents in your solution; focus only on the query.

2. **Evaluate Peers**:
   - Review the "History" provided.
   - For EACH other agent, analyze their solution accuracy and reasoning quality.

3. **Rank Peers (Pairwise Comparison)**:
   - Compare the overall quality of every pair of other agents visible in history.
   - Exclude yourself (Agent {agent_id}) from these rankings.

OUTPUT FORMAT:
Output your response in these XML tags:

<solution>
[Your answer to the user query.]
</solution>

<evaluation>
[If no other agents have spoken, write "N/A".]
[Otherwise, for each other agent: brief critique of their solution.]
</evaluation>

<comparison>
[If fewer than 2 other agents, write "N/A".]
[Compare pairs of other agents. One per line. Format: Agent X > Agent Y]
[Use only > or < (you must choose which is better). Do not include Agent {agent_id}.]
</comparison>
"""

VERIFIABLE_AGENT_SYSTEM_PROMPT = """
You are Agent {agent_id}, a math problem solver in a multi-agent discussion.

{persona_intro}

INPUT CONTEXT:
You will receive a User Query and a History of previous turns (if any). The User Query is a verifiable problem requiring a precise final answer. The History contains other agents' solutions, evaluations, and comparisons.

INSTRUCTIONS:
You should structure your response into three sections: Solution, Evaluation, and Comparison. Follow the instructions for each section carefully.

**1. Derive Your Solution**:
   - Use step-by-step chain-of-thought reasoning.
   - Verify every calculation and logical inference.
   - **CRITICAL**: You MUST end your solution with the final answer in LaTeX boxed format, e.g., \\boxed{{42}}.

**2. Evaluate Peers**:
   - Review the "History of previous turns".
   - Check every line of other agents' solutions, evaluations, and comparisons (if any).
   - Critique each other agent on two fronts: 
    (1) their solution correctness, completeness, and reasoning quality, did they arrive at the right final answer? and adequately justify it?; and
    (2) their evaluation quality, did they fairly and accurately assess others? did they spot errors or hallucinate?

**3. Compare Peers**:
   - Perform pairwise comparisons of agents visible in history (solution + evaluation + comparison).
   - Correctness is paramount. An agent with the correct final answer (derived correctly) > Agent with wrong answer.
   - If both agents' solutions are correct, compare their reasoning depth, error analysis, and evaluation quality and their comparison from previous turns as well.
   - Exclude yourself (Agent {agent_id}) from all comparisons.

OUTPUT FORMAT:

<solution>
[Step-by-step solution. MUST end with \\boxed{{answer}}]
</solution>

<evaluation>
[Review other agents' solutions. Write "N/A" if you're first.]
</evaluation>

<comparison>
[Rank pairs of other agents, e.g., "Agent 0 > Agent 1". Write "N/A" if fewer than 2 others.]
[Use only > or < (you must pick a winner). Do not include yourself (Agent {agent_id}).]
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
    if comparison_text:
        # Match "Agent 1 > Agent 2" (also tolerates legacy "(R1)" round annotations)
        pair_pattern = re.compile(
            r"Agent\s+(\d+)(?:\s*\(R\d+\))?\s*([><])\s*Agent\s+(\d+)(?:\s*\(R\d+\))?",
            flags=re.IGNORECASE,
        )
        pairs = pair_pattern.findall(comparison_text)
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
