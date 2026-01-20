"""
Response parsers for extracting structured data from agent outputs in each round.
"""

import re
from dataclasses import dataclass
from typing import Sequence


@dataclass
class Round1Response:
    """Parsed response from Round 1 (Proposal)."""
    initial_solution: str
    author_id: int


@dataclass
class Round2Response:
    """Parsed response from Round 2 (Blind Evaluation + Critique)."""
    blind_ranking: list[tuple[int, str, int]]  # (agent_a, op, agent_b)
    critique_targets: list[int]  # Agent IDs being critiqued
    critique_texts: dict[int, str]  # {target_id: critique_text}
    author_id: int


@dataclass
class Round3Response:
    """Parsed response from Round 3 (Revision + Final Verdict)."""
    revised_solution: str
    final_ranking: list[tuple[int, str, int]]  # (agent_a, op, agent_b)
    author_id: int


def extract_xml_content(text: str, tag: str) -> str | None:
    """
    Extract content from XML-style tags.

    Args:
        text: Input text containing XML tags
        tag: Tag name to extract (without angle brackets)

    Returns:
        Content between opening and closing tags, or None if not found
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_pairwise_rankings(ranking_text: str) -> list[tuple[int, str, int]]:
    """
    Parse pairwise ranking statements in the format "Agent X > Agent Y".

    Args:
        ranking_text: Text containing ranking statements (one per line)

    Returns:
        List of (agent_a, operator, agent_b) tuples where operator is '>', '<', or '='

    Examples:
        "Agent 0 > Agent 1" -> [(0, '>', 1)]
        "Agent 2 < Agent 0" -> [(2, '<', 0)]
        "Agent 1 = Agent 3" -> [(1, '=', 3)]
    """
    rankings = []

    # Pattern matches: "Agent X {>|<|=} Agent Y"
    pattern = r"Agent\s+(\d+)\s*([><= ]+)\s*Agent\s+(\d+)"

    for line in ranking_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.search(pattern, line)
        if match:
            agent_a = int(match.group(1))
            operator_raw = match.group(2).strip()
            agent_b = int(match.group(3))

            # Normalize operator
            if '>' in operator_raw:
                operator = '>'
            elif '<' in operator_raw:
                operator = '<'
            elif '=' in operator_raw:
                operator = '='
            else:
                continue  # Invalid operator

            rankings.append((agent_a, operator, agent_b))

    return rankings


def parse_critique_targets(critique_text: str) -> tuple[list[int], dict[int, str]]:
    """
    Parse <target>Agent k</target> tags and extract associated critique text.

    Args:
        critique_text: Text containing <target> tags and critique content

    Returns:
        Tuple of (target_ids, critique_texts) where:
        - target_ids: List of agent IDs being critiqued
        - critique_texts: Dict mapping target_id to the critique text following their tag

    Example:
        Input:
            <target>Agent 1</target>
            Your logic is flawed...

            <target>Agent 2</target>
            You missed the constraint...

        Output:
            ([1, 2], {1: "Your logic is flawed...", 2: "You missed the constraint..."})
    """
    targets = []
    critique_texts = {}

    # Find all <target>Agent X</target> tags
    target_pattern = r"<target>\s*Agent\s+(\d+)\s*</target>"

    # Split by target tags to get critique segments
    segments = re.split(target_pattern, critique_text)

    # segments will be: [before_first_target, id1, text1, id2, text2, ...]
    # We process pairs starting from index 1
    for i in range(1, len(segments), 2):
        if i + 1 < len(segments):
            target_id = int(segments[i])
            critique_content = segments[i + 1].strip()

            targets.append(target_id)
            critique_texts[target_id] = critique_content

    return targets, critique_texts


def parse_round1_response(response: str, author_id: int) -> Round1Response:
    """
    Parse Round 1 response containing initial solution.

    Args:
        response: Agent's response text
        author_id: ID of the agent who generated this response

    Returns:
        Round1Response object with parsed content
    """
    initial_solution = extract_xml_content(response, "initial_solution")

    if initial_solution is None:
        # Fallback: use entire response if tag missing
        initial_solution = response.strip()

    return Round1Response(
        initial_solution=initial_solution,
        author_id=author_id,
    )


def parse_round2_response(response: str, author_id: int) -> Round2Response:
    """
    Parse Round 2 response containing blind ranking and critiques.

    Args:
        response: Agent's response text
        author_id: ID of the agent who generated this response

    Returns:
        Round2Response object with parsed content
    """
    # Extract blind ranking
    blind_ranking_text = extract_xml_content(response, "blind_ranking")
    blind_ranking = []
    if blind_ranking_text:
        blind_ranking = parse_pairwise_rankings(blind_ranking_text)

    # Extract critiques
    critique_text = extract_xml_content(response, "critique")
    critique_targets = []
    critique_texts = {}
    if critique_text:
        critique_targets, critique_texts = parse_critique_targets(critique_text)

    return Round2Response(
        blind_ranking=blind_ranking,
        critique_targets=critique_targets,
        critique_texts=critique_texts,
        author_id=author_id,
    )


def parse_round3_response(response: str, author_id: int) -> Round3Response:
    """
    Parse Round 3 response containing revised solution and final ranking.

    Args:
        response: Agent's response text
        author_id: ID of the agent who generated this response

    Returns:
        Round3Response object with parsed content
    """
    # Extract revised solution
    revised_solution = extract_xml_content(response, "revised_solution")
    if revised_solution is None:
        # Fallback: try "solution" tag
        revised_solution = extract_xml_content(response, "solution")
    if revised_solution is None:
        # Final fallback: use response before final_ranking
        final_ranking_tag = re.search(r"<final_ranking>", response)
        if final_ranking_tag:
            revised_solution = response[:final_ranking_tag.start()].strip()
        else:
            revised_solution = response.strip()

    # Extract final ranking
    final_ranking_text = extract_xml_content(response, "final_ranking")
    final_ranking = []
    if final_ranking_text:
        final_ranking = parse_pairwise_rankings(final_ranking_text)

    return Round3Response(
        revised_solution=revised_solution,
        final_ranking=final_ranking,
        author_id=author_id,
    )
