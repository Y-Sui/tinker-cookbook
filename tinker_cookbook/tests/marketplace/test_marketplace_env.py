import asyncio
import pytest

tinker = pytest.importorskip("tinker")

from tinker_cookbook.recipes.multiplayer_rl.marketplace.env import (
    MarketplaceCoordinator,
    MarketplaceEnv,
    MarketplaceEnvGroupBuilder,
    ServiceProfile,
)


class FakeRenderer:
    """Lightweight stub to satisfy MarketplaceEnv constructor without touching tokenization."""

    def get_stop_sequences(self):
        return ["\n"]

    def build_generation_prompt(self, _convo):
        class Dummy:
            length = 0

        return Dummy()

    def parse_response(self, _action):
        return {"role": "assistant", "content": "stub"}, True


def _make_services():
    return [
        ServiceProfile(service_id=0, name="Fast", quality=1.0, base_price=10.0, cost=5.0, latency=0.1),
        ServiceProfile(service_id=1, name="Slow", quality=1.5, base_price=12.0, cost=6.0, latency=1.0),
    ]


def test_first_response_selects_fastest_service():
    coord = MarketplaceCoordinator(
        intent="Find dinner",
        services=_make_services(),
        market_rule="first_response",
        turn_budget=4,
        top_k=1,
    )
    coord.register_assistant_message("What's your offer?")
    # Fast service replies first; should finalize selection
    coord.register_service_message(service_id=0, content="Our price=9.5")
    assert coord.done is True
    assert coord.selected_service_id == 0
    metrics = coord.metrics()
    assert metrics["selected_service_id"] == 0
    assert coord.assistant_reward() != 0.0


def test_service_reward_only_for_selected():
    coord = MarketplaceCoordinator(
        intent="Need flowers",
        services=_make_services(),
        market_rule="auction",
        turn_budget=4,
        top_k=2,
    )
    coord.register_assistant_message("Please give your best price.")
    coord.register_service_message(1, "price=11.0")
    coord.finalize_selection(service_id=1, price=11.0)
    assert coord.service_reward(1) > 0.0
    assert coord.service_reward(0) == 0.0


def test_group_rewards_match_selected_service():
    renderer = FakeRenderer()
    builder = MarketplaceEnvGroupBuilder(
        intent="Book cleaning",
        services=_make_services(),
        renderer=renderer,
        market_rule="auction",
        turn_budget=4,
        top_k=2,
        fixed_service_policy=None,
    )
    envs = asyncio.run(builder.make_envs())
    # Manually mark a selection on shared coordinator
    coord = envs[0].coordinator  # type: ignore[attr-defined]
    coord.finalize_selection(service_id=1, price=11.0)
    rewards_and_metrics = asyncio.run(builder.compute_group_rewards([], envs))
    rewards = [rm[0] for rm in rewards_and_metrics]
    # Only one service (id=1) should receive positive reward
    assert any(r > 0 for r in rewards)
    assert rewards.count(0.0) == len(rewards) - 1
