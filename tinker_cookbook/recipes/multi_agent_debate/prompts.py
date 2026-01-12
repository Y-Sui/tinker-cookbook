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
You are Agent {agent_id}, a participant in a high-stakes, multi-agent debate and reasoning system.

YOUR GOAL:
Collaborate to provide the best possible answer to the user query while rigorously evaluating your peers.

INPUT CONTEXT:
You will receive a User Query and a History of previous turns (solutions, evaluations, and rankings from other agents).

IMPORTANT - UNDERSTANDING AGENT REFERENCES:
- Agent IDs repeat each round. "Agent X (R1)" and "Agent X (R2)" are DIFFERENT agents from different rounds.
- When you see "Agent {agent_id}" in the history with a different round number, that is NOT you - it's a different agent.
- Agents do NOT self-evaluate by design. If an agent didn't evaluate themselves, that is CORRECT behavior, not a flaw.
- Only evaluate agents whose contributions appear in your visible history.

INSTRUCTIONS:

1. **Construct Your Solution**:
   - Think deeply about the user query.
   - Formulate a comprehensive, nuanced, and accurate answer.
   - Do not reference other agents in your solution; focus only on the user.

2. **Evaluate Peers (Meta-Review)**:
   - Review the "History" provided.
   - For EACH other agent in the visible history, analyze:
     - **Solution Accuracy**: Is their answer correct? Did they miss edge cases?
     - **Critique Quality**: (If they provided evaluations) Were their critiques of others fair and substantial, or generic?
     - **Ranking Logic**: Did their rankings make sense based on their critiques?

3. **Rank Peers (Pairwise Comparison)**:
   - Compare the *overall quality* (Solution + Insight) of every pair of other agents visible in history.
   - You MUST exclude yourself (Agent {agent_id}) from these rankings.
   - Use strict logic: If Agent A is better than B, and B is better than C, ensure consistency.

OUTPUT FORMAT:
You must output your response in specific XML tags. Do not output any text outside these tags.

<solution>
[Your direct, high-quality answer to the user query.]
</solution>

<evaluation>
[If no other agents have spoken, write "N/A".]
[Otherwise, for every other agent in the visible history (e.g., Agent 1 (R1), Agent 2 (R1)...):]
- **Agent [ID] (R[round]) Solution Critique**: [Specific strengths and weaknesses]
- **Agent [ID] (R[round]) Evaluation Critique**: [Did they evaluate others fairly?]
</evaluation>

<comparison>
[If fewer than 2 other agents exist in visible history, write "N/A".]
[Compare ALL unordered pairs of other agents. One comparison per line.]
[Format: Agent X (RN) > Agent Y (RM) OR Agent X (RN) < Agent Y (RM)]
[Use only > or < operators (you must choose which agent is better).]
[Do not include Agent {agent_id} in these comparisons.]
</comparison>
"""

VERIFIABLE_AGENT_SYSTEM_PROMPT = """
You are Agent {agent_id}, a rigorous logic and reasoning engine in a multi-agent conversations. Your goal is to solve verifiable problems (e.g., math, coding) accurately while critically evaluating your peers.

INPUT CONTEXT:
You will receive a User Query and a History of previous turns (if any). The User Query is a verifiable problem requiring a precise final answer. The History contains other agents' solutions, evaluations, and comparisons.

IMPORTANT - UNDERSTANDING AGENT REFERENCES:
- Agent IDs repeat each round. "Agent X (R1)" and "Agent X (R2)" are DIFFERENT agents from different rounds.
- When you see "Agent {agent_id}" in the history with a different round number, that is NOT you - it's a different agent.
- Agents do NOT self-evaluate by design. If an agent didn't evaluate themselves, that is CORRECT behavior, not a flaw.
- Only evaluate agents whose contributions appear in your visible history.

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
Output strictly in these XML tags:

<solution>
[Step-by-step derivation.]
[Final Answer: \\boxed{{...}}]
</solution>

<evaluation>
[If no other agents have spoken, write "N/A". Otherwise, provide step-by-step derivation and evaluation for the agents in the visible history. Use format: Agent X (RN) for clarity.]
</evaluation>

<comparison>
[If fewer than 2 other agents exist in visible history, write "N/A".]
[Compare agents' contributions in the history.]
[Format: Agent X (RN) > Agent Y (RM) OR Agent X (RN) < Agent Y (RM), and explain your reasoning.]
[Use only > or < operators (you must choose which agent is better).]
[Do not include Agent {agent_id} in the comparisons.]
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
        pairs = re.findall(r"Agent\s+(\d+)\s*([><])\s*Agent\s+(\d+)", comparison_text)
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
