from tinker_cookbook.recipes.multi_agent_debate.prompts import parse_agent_response


def test_parse_agent_response_strips_think_and_preamble():
    raw = """
<think>
internal reasoning
</think>
Sure, here's my answer:
<thinking>short</thinking>
<solution>sol</solution>
<evaluation>N/A</evaluation>
<comparison>Agent 0 = Agent 1</comparison>
<consensus>NO</consensus>
<consensus_reason>reason</consensus_reason>
"""
    parsed = parse_agent_response(raw)
    assert parsed.solution == "sol"
    assert parsed.evaluation == "N/A"
    assert parsed.consensus_reached is False
    assert parsed.comparisons == [(0, "=", 1)]

