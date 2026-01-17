from tinker_cookbook.recipes.multi_agent_debate.core.prompts import parse_agent_response


def test_parse_agent_response_drops_self_comparisons():
    xml = """<solution>ok</solution>
<evaluation>ok</evaluation>
<comparison>
Agent 0 > Agent 1
Agent 1 > Agent 2
Agent 2 = Agent 0
</comparison>
<consensus>NO</consensus>
<consensus_reason>n/a</consensus_reason>
"""
    parsed = parse_agent_response(xml, author_id=0, observation="ctx")
    assert parsed.comparisons == [(1, ">", 2)]
    assert parsed.self_comparisons_dropped == 2


def test_parse_agent_response_round1_agent1_drops_all_comparisons():
    xml = """<solution>ok</solution>
<evaluation>ok</evaluation>
<comparison>
Agent 0 > Agent 2
Agent 0 > Agent 1
</comparison>
<consensus>NO</consensus>
<consensus_reason>n/a</consensus_reason>
"""
    # Round 1: Agent 1 should not produce comparisons at all.
    parsed = parse_agent_response(xml, author_id=1, observation="Round 1 of 3")
    assert parsed.comparisons == []


def test_parse_agent_response_round1_agent2_only_compares_prior_agents():
    xml = """<solution>ok</solution>
<evaluation>ok</evaluation>
<comparison>
Agent 0 > Agent 1
Agent 0 > Agent 2
Agent 0 > Agent 3
</comparison>
<consensus>NO</consensus>
<consensus_reason>n/a</consensus_reason>
"""
    # Round 1 at Agent 2's turn: only agents 0 and 1 have completions; comparisons must be among {0,1}.
    parsed = parse_agent_response(xml, author_id=2, observation="Round 1 of 3")
    assert parsed.comparisons == [(0, ">", 1)]
