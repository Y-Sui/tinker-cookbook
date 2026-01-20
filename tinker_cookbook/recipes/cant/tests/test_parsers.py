"""Tests for response parsers."""

from tinker_cookbook.recipes.cant.parsers import (
    extract_xml_content,
    parse_pairwise_rankings,
    parse_critique_targets,
    parse_round1_response,
    parse_round2_response,
    parse_round3_response,
)


def test_extract_xml_content():
    """Test basic XML content extraction."""
    text = "<tag>content here</tag>"
    assert extract_xml_content(text, "tag") == "content here"

    # Multiline
    text = "<tag>\nline1\nline2\n</tag>"
    assert extract_xml_content(text, "tag") == "line1\nline2"

    # Not found
    assert extract_xml_content(text, "missing") is None


def test_parse_pairwise_rankings():
    """Test parsing of pairwise ranking statements."""
    ranking_text = """
    Agent 0 > Agent 1
    Agent 2 < Agent 0
    Agent 1 = Agent 3
    """

    rankings = parse_pairwise_rankings(ranking_text)

    assert len(rankings) == 3
    assert (0, '>', 1) in rankings
    assert (2, '<', 0) in rankings
    assert (1, '=', 3) in rankings


def test_parse_critique_targets():
    """Test parsing of <target> tags."""
    critique_text = """
    <target>Agent 1</target>
    Your logic is flawed because...

    <target>Agent 2</target>
    You missed the constraint that...
    """

    targets, texts = parse_critique_targets(critique_text)

    assert targets == [1, 2]
    assert 1 in texts
    assert 2 in texts
    assert "logic is flawed" in texts[1]
    assert "missed the constraint" in texts[2]


def test_parse_round1_response():
    """Test Round 1 response parsing."""
    response = """
    <initial_solution>
    This is my initial solution to the problem.
    </initial_solution>
    """

    parsed = parse_round1_response(response, author_id=0)

    assert parsed.author_id == 0
    assert "initial solution" in parsed.initial_solution


def test_parse_round1_response_fallback():
    """Test Round 1 parsing when tags are missing."""
    response = "Just a plain text solution without tags"

    parsed = parse_round1_response(response, author_id=1)

    assert parsed.author_id == 1
    assert parsed.initial_solution == response.strip()


def test_parse_round2_response():
    """Test Round 2 response parsing."""
    response = """
    <blind_ranking>
    Agent 0 > Agent 1
    Agent 2 > Agent 0
    </blind_ranking>

    <critique>
    <target>Agent 1</target>
    Your solution has a logical flaw.

    <target>Agent 2</target>
    You forgot to consider edge cases.
    </critique>
    """

    parsed = parse_round2_response(response, author_id=0)

    assert parsed.author_id == 0
    assert len(parsed.blind_ranking) == 2
    assert (0, '>', 1) in parsed.blind_ranking
    assert (2, '>', 0) in parsed.blind_ranking
    assert parsed.critique_targets == [1, 2]
    assert 1 in parsed.critique_texts
    assert 2 in parsed.critique_texts


def test_parse_round3_response():
    """Test Round 3 response parsing."""
    response = """
    <revised_solution>
    This is my revised solution incorporating feedback.
    </revised_solution>

    <final_ranking>
    Agent 0 > Agent 1
    Agent 2 > Agent 0
    </final_ranking>
    """

    parsed = parse_round3_response(response, author_id=1)

    assert parsed.author_id == 1
    assert "revised solution" in parsed.revised_solution
    assert len(parsed.final_ranking) == 2
    assert (0, '>', 1) in parsed.final_ranking
    assert (2, '>', 0) in parsed.final_ranking
