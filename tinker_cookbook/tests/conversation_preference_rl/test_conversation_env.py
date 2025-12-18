import asyncio
import json
from types import SimpleNamespace

import pytest

tinker = pytest.importorskip("tinker")

from tinker_cookbook.recipes.conversation_preference_rl.env import (
    AgentResponse,
    MultiAgentConversationEnvGroupBuilder,
    MultiAgentCoordinator,
    PolicySelfComparisonJudge,
    parse_agent_response,
)
from tinker_cookbook.rl.preference_envs import TournamentPattern
from tinker_cookbook.rl.types import Trajectory


class FakeTokenizer:
    def decode(self, tokens):
        if isinstance(tokens, str):
            return tokens
        return "".join(chr(t) for t in tokens)


class FakeRenderer:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def build_generation_prompt(self, _convo):
        return tinker.ModelInput.from_ints([0])

    def get_stop_sequences(self):
        return ["\n}"]


class MockSamplingClient:
    def __init__(self, return_text: str):
        self.return_text = return_text
        self.called_with_prompt = None

    async def sample_async(self, prompt, sampling_params, num_samples):
        # Capture prompt for assertions
        self.called_with_prompt = prompt
        tokens = [ord(c) for c in self.return_text]
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=tokens)])


def test_parse_agent_response_from_markdown_json():
    text = """```json
{
  "reflection": "R",
  "evaluation": "E",
  "solution": "S"
}
```"""
    parsed = parse_agent_response(text)
    assert parsed.reflection == "R"
    assert parsed.evaluation == "E"
    assert parsed.solution == "S"


@pytest.mark.asyncio
async def test_coordinator_turns_and_completion():
    coord = MultiAgentCoordinator(num_agents=2, max_rounds=1, query="Q")
    await coord.submit_response(0, AgentResponse(reflection="r", evaluation="e", solution="s0"))
    assert coord.current_turn == 1
    await coord.submit_response(1, AgentResponse(reflection="r", evaluation="e", solution="s1"))
    assert coord.episode_done is True


def test_group_rewards_use_mocked_llm_judge():
    renderer = FakeRenderer()
    # Mock LLM to prefer solution B
    return_json = json.dumps({"reasoning": "B is better", "preference": "B"})
    sampling_client = MockSamplingClient(return_text=return_json)
    judge = PolicySelfComparisonJudge(sampling_client=sampling_client, renderer=renderer)

    builder = MultiAgentConversationEnvGroupBuilder(
        query="Q",
        num_agents=2,
        max_rounds=1,
        renderer=renderer,
        self_comparison_judge=judge,
        tournament_pattern=TournamentPattern.ALL_PAIRS_ONE_WAY,
    )
    envs = asyncio.run(builder.make_envs())
    coord = envs[0].coordinator  # shared coordinator
    coord.conversation_history = [
        AgentResponse(reflection="r", evaluation="e", solution="sol A"),
        AgentResponse(reflection="r", evaluation="e", solution="better sol B"),
    ]

    dummy_trajs = [
        Trajectory(transitions=[], final_ob=renderer.build_generation_prompt([])) for _ in envs
    ]
    rewards_and_metrics = asyncio.run(builder.compute_group_rewards(dummy_trajs, envs))
    rewards = [rm[0] for rm in rewards_and_metrics]

    # Agent 1 should win based on mocked judge preference
    assert rewards[1] > 0
    assert rewards[0] < 0
    # Confirm judge was invoked (prompt captured)
    assert sampling_client.called_with_prompt is not None
