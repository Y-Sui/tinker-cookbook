import asyncio
import math
import random
import re
from dataclasses import dataclass
from typing import Literal, Sequence

import chz
import tinker
from tinker import types
from tinker_cookbook.completers import StopCondition, TinkerMessageCompleter
from tinker_cookbook.renderers import Message, Renderer, ensure_text, get_renderer
from tinker_cookbook.rl import types as rl_types
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

MarketRule = Literal["first_response", "top_k_quality", "auction"]

DEFAULT_INTENTS = [
    "Find a family-friendly Italian restaurant with outdoor seating.",
    "Book a cleaning service for a 2-bedroom apartment next Tuesday.",
    "I need same-day flower delivery with a $60 budget.",
    "Locate a laptop repair shop that can replace a battery this week.",
    "Find a dog-walking service near downtown available tomorrow morning.",
]


@dataclass
class ServiceProfile:
    service_id: int
    name: str
    quality: float
    base_price: float
    cost: float
    latency: float


def _format_service_summary(profile: ServiceProfile) -> str:
    return (
        f"{profile.service_id}: {profile.name} (quality≈{profile.quality:.2f}, "
        f"base_price={profile.base_price:.2f}, latency={profile.latency:.2f}s)"
    )


class MarketplaceCoordinator:
    """Coordinates assistant and service agents inside one marketplace."""

    def __init__(
        self,
        intent: str,
        services: list[ServiceProfile],
        market_rule: MarketRule,
        turn_budget: int,
        top_k: int,
    ):
        self.intent = intent
        self.services = services
        self.market_rule = market_rule
        self.turn_budget = turn_budget
        self.top_k = top_k

        self.condition = asyncio.Condition()
        self.next_role: Literal["assistant", "service"] = "assistant"
        self.current_service_pos = 0
        self.done = False
        self.turn_count = 0
        self.selected_service_id: int | None = None
        self.selected_price: float | None = None
        self.messages: list[tuple[str, str]] = []  # (role, content)
        self.last_offers: dict[int, float] = {}
        self.service_order: list[int] = [s.service_id for s in self._service_order()]

    def _service_order(self) -> list[ServiceProfile]:
        if self.market_rule == "top_k_quality":
            ordered = sorted(self.services, key=lambda s: s.quality, reverse=True)
            return ordered[: max(1, self.top_k)]
        return sorted(self.services, key=lambda s: s.latency)

    def _service_by_id(self, service_id: int) -> ServiceProfile:
        for svc in self.services:
            if svc.service_id == service_id:
                return svc
        raise KeyError(f"Unknown service_id {service_id}")

    def _parsed_price(self, content: str) -> float | None:
        match = re.search(r"price[=:] ?([0-9]+(?:\.[0-9]+)?)", content, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    def _select_if_first_response(self, service_id: int, content: str) -> None:
        if self.market_rule != "first_response":
            return
        if self.selected_service_id is None:
            self.selected_service_id = service_id
            self.selected_price = self.last_offers.get(service_id) or self._service_by_id(
                service_id
            ).base_price
            self.done = True
            self.next_role = "assistant"

    def register_assistant_message(self, content: str) -> None:
        self.messages.append(("assistant", content))
        self.turn_count += 1
        if self.turn_count >= self.turn_budget:
            self.done = True
            self.next_role = "assistant"

    def register_service_message(self, service_id: int, content: str) -> None:
        self.messages.append((f"service_{service_id}", content))
        maybe_price = self._parsed_price(content)
        if maybe_price is not None:
            self.last_offers[service_id] = maybe_price
        self._select_if_first_response(service_id, content)

    def finalize_selection(self, service_id: int, price: float | None) -> None:
        self.selected_service_id = service_id
        self.selected_price = price if price is not None else self.last_offers.get(
            service_id
        ) or self._service_by_id(service_id).base_price
        self.done = True
        self.next_role = "assistant"

    def _assistant_prompt(self) -> list[Message]:
        summaries = "\\n".join(
            _format_service_summary(self._service_by_id(sid)) for sid in self.service_order
        )
        sys_prompt = (
            "You are an assistant representing the customer. Negotiate quickly, ask clarifying questions, "
            "and select a service using 'Select: <service_id> price=<number>'. "
            "Services may bias toward faster responses. Available services:\\n"
            f"{summaries}"
        )
        convo: list[Message] = [{"role": "system", "content": sys_prompt}]
        for role, content in self.messages:
            if role == "assistant":
                convo.append({"role": "assistant", "content": content})
            else:
                convo.append({"role": "user", "content": f"{role}: {content}"})
        if not self.messages:
            convo.append(
                {"role": "user", "content": f"Customer intent: {self.intent}. Start by asking a question."}
            )
        return convo

    def _service_prompt(self, service_id: int) -> list[Message]:
        profile = self._service_by_id(service_id)
        sys_prompt = (
            f"You are {profile.name} (id={profile.service_id}). Offer prices near {profile.base_price:.2f} "
            f"with cost floor {profile.cost:.2f}. Respond concisely and close the deal fast."
        )
        convo: list[Message] = [{"role": "system", "content": sys_prompt}]
        for role, content in self.messages:
            if role == "assistant":
                convo.append({"role": "user", "content": content})
            elif role == f"service_{service_id}":
                convo.append({"role": "assistant", "content": content})
        if not self.messages:
            convo.append({"role": "user", "content": f"Customer intent: {self.intent}"})
        return convo

    def assistant_observation(self, renderer: Renderer) -> tinker.ModelInput:
        return renderer.build_generation_prompt(self._assistant_prompt())

    def service_observation(self, renderer: Renderer, service_id: int) -> tinker.ModelInput:
        return renderer.build_generation_prompt(self._service_prompt(service_id))

    def metrics(self) -> dict[str, float | int | str]:
        if self.selected_service_id is None:
            return {"turns": self.turn_count, "selected_service": "none"}
        profile = self._service_by_id(self.selected_service_id)
        price = self.selected_price or profile.base_price
        welfare = profile.quality - 0.05 * price - 0.1 * profile.latency
        return {
            "turns": self.turn_count,
            "selected_service": profile.name,
            "selected_service_id": profile.service_id,
            "price": price,
            "quality": profile.quality,
            "latency": profile.latency,
            "welfare": welfare,
        }

    def assistant_reward(self) -> float:
        if self.selected_service_id is None:
            return -0.5
        profile = self._service_by_id(self.selected_service_id)
        price = self.selected_price or profile.base_price
        welfare = profile.quality - 0.05 * price - 0.1 * profile.latency
        welfare -= 0.01 * self.turn_count
        return welfare

    def service_reward(self, service_id: int) -> float:
        if self.selected_service_id != service_id:
            return 0.0
        profile = self._service_by_id(service_id)
        price = self.selected_price or profile.base_price
        return price - profile.cost

    def _ready(self, role: Literal["assistant", "service"], service_id: int | None) -> bool:
        if self.done:
            return True
        if role == "assistant":
            return self.next_role == "assistant"
        return self.next_role == "service" and self.service_order[self.current_service_pos] == service_id

    async def wait_for_turn(
        self, role: Literal["assistant", "service"], service_id: int | None = None
    ) -> None:
        async with self.condition:
            await self.condition.wait_for(lambda: self._ready(role, service_id))

    async def advance_after_assistant(self) -> None:
        async with self.condition:
            if not self.done:
                self.next_role = "service"
                self.current_service_pos = 0
            self.condition.notify_all()

    async def advance_after_service(self) -> None:
        async with self.condition:
            if not self.done:
                self.current_service_pos += 1
                if self.current_service_pos >= len(self.service_order):
                    self.next_role = "assistant"
            self.condition.notify_all()


class MarketplaceEnv(Env):
    def __init__(
        self,
        role: Literal["assistant", "service"],
        service_id: int | None,
        coordinator: MarketplaceCoordinator,
        renderer: Renderer,
        fixed_service_policy: TinkerMessageCompleter | None,
    ):
        self.role = role
        self.service_id = service_id
        self.coordinator = coordinator
        self.renderer = renderer
        self.fixed_service_policy = fixed_service_policy

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        if self.role == "assistant":
            return self.coordinator.assistant_observation(self.renderer), self.stop_condition
        assert self.service_id is not None
        await self.coordinator.wait_for_turn("service", self.service_id)
        return self.coordinator.service_observation(self.renderer, self.service_id), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        if self.coordinator.done:
            return self._done_step()
        if self.role == "assistant":
            return await self._assistant_step(action)
        assert self.service_id is not None
        return await self._service_step(action)

    async def _assistant_step(self, action: Action) -> StepResult:
        await self.coordinator.wait_for_turn("assistant")
        action_message, _ = self.renderer.parse_response(action)
        content = ensure_text(action_message["content"])

        selection_match = re.search(r"Select: ?(\d+)", content, flags=re.IGNORECASE)
        maybe_service_id = int(selection_match.group(1)) if selection_match else None
        maybe_price = self.coordinator._parsed_price(content)

        self.coordinator.register_assistant_message(content)
        if maybe_service_id is not None:
            self.coordinator.finalize_selection(maybe_service_id, maybe_price)
        await self.coordinator.advance_after_assistant()

        reward = self.coordinator.assistant_reward() if self.coordinator.done else -0.01
        metrics: dict[str, float | int | str] = {}
        if self.coordinator.done:
            metrics.update(self.coordinator.metrics())

        if self.coordinator.done:
            return StepResult(
                reward=reward,
                episode_done=True,
                next_observation=types.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics=metrics,
            )

        return StepResult(
            reward=reward,
            episode_done=False,
            next_observation=self.coordinator.assistant_observation(self.renderer),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )

    async def _service_step(self, action: Action) -> StepResult:
        await self.coordinator.wait_for_turn("service", self.service_id)
        if self.coordinator.done:
            return self._done_step()

        content: str
        if self.fixed_service_policy is not None:
            prompt = self.coordinator._service_prompt(self.service_id)
            resp = await self.fixed_service_policy(prompt)
            content = ensure_text(resp["content"])
        else:
            action_message, _ = self.renderer.parse_response(action)
            content = ensure_text(action_message["content"])

        self.coordinator.register_service_message(self.service_id, content)
        await self.coordinator.advance_after_service()

        if self.coordinator.done:
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=types.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics=self.coordinator.metrics(),
            )

        return StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=self.coordinator.service_observation(self.renderer, self.service_id),
            next_stop_condition=self.stop_condition,
            metrics={},
        )

    def _done_step(self) -> StepResult:
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=types.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=self.coordinator.metrics(),
        )


@dataclass
class MarketplaceEnvGroupBuilder(EnvGroupBuilder):
    intent: str
    services: list[ServiceProfile]
    renderer: Renderer
    market_rule: MarketRule
    turn_budget: int
    top_k: int
    fixed_service_policy: TinkerMessageCompleter | None

    async def make_envs(self) -> Sequence[Env]:
        coordinator = MarketplaceCoordinator(
            intent=self.intent,
            services=self.services,
            market_rule=self.market_rule,
            turn_budget=self.turn_budget,
            top_k=self.top_k,
        )
        envs: list[Env] = [
            MarketplaceEnv(
                role="assistant",
                service_id=None,
                coordinator=coordinator,
                renderer=self.renderer,
                fixed_service_policy=None,
            )
        ]
        for service_id in coordinator.service_order:
            svc = coordinator._service_by_id(service_id)
            envs.append(
                MarketplaceEnv(
                    role="service",
                    service_id=svc.service_id,
                    coordinator=coordinator,
                    renderer=self.renderer,
                    fixed_service_policy=self.fixed_service_policy,
                )
            )
        return envs

    def logging_tags(self) -> list[str]:
        return ["marketplace", self.market_rule]

    async def compute_group_rewards(
        self, trajectory_group: list[rl_types.Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, dict[str, float | int | str]]]:
        # Reward only the selected service; assistant reward is delivered step-wise.
        coordinator = env_group[0].coordinator  # shared across envs
        rewards: list[float] = []
        metrics = coordinator.metrics()
        for env in env_group:
            if isinstance(env, MarketplaceEnv) and env.role == "service":
                rewards.append(coordinator.service_reward(env.service_id))
            else:
                rewards.append(0.0)
        metrics_list = [metrics for _ in rewards]
        return list(zip(rewards, metrics_list))


class MarketplaceDataset(RLDataset):
    def __init__(
        self,
        intents: list[str],
        renderer: Renderer,
        batch_size: int,
        market_rule: MarketRule,
        num_services: int,
        turn_budget: int,
        top_k: int,
        fixed_service_policy: TinkerMessageCompleter | None,
    ):
        self.intents = intents
        self.renderer = renderer
        self.batch_size = batch_size
        self.market_rule = market_rule
        self.num_services = num_services
        self.turn_budget = turn_budget
        self.top_k = top_k
        self.fixed_service_policy = fixed_service_policy

    def _sample_service_pool(self, seed: int) -> list[ServiceProfile]:
        rng = random.Random(seed)
        services: list[ServiceProfile] = []
        for i in range(self.num_services):
            quality = rng.uniform(0.5, 1.5)
            base_price = rng.uniform(5.0, 40.0)
            cost = max(1.0, base_price - rng.uniform(2.0, 10.0))
            latency = rng.uniform(0.1, 1.5)
            services.append(
                ServiceProfile(
                    service_id=i,
                    name=f"Service-{i}",
                    quality=quality,
                    base_price=base_price,
                    cost=cost,
                    latency=latency,
                )
            )
        return services

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        builders: list[EnvGroupBuilder] = []
        for i in range(self.batch_size):
            idx = index * self.batch_size + i
            if idx >= len(self.intents):
                break
            services = self._sample_service_pool(seed=idx)
            builders.append(
                MarketplaceEnvGroupBuilder(
                    intent=self.intents[idx],
                    services=services,
                    renderer=self.renderer,
                    market_rule=self.market_rule,
                    turn_budget=self.turn_budget,
                    top_k=self.top_k,
                    fixed_service_policy=self.fixed_service_policy,
                )
            )
        return builders

    def __len__(self) -> int:
        return math.ceil(len(self.intents) / self.batch_size)


@chz.chz
class MarketplaceDatasetBuilder(RLDatasetBuilder):
    model_name: str
    renderer_name: str
    batch_size: int
    num_markets_train: int
    num_markets_test: int
    num_services: int = 2
    turn_budget: int = 6
    market_rule: MarketRule = "first_response"
    top_k: int = 2
    base_url: str | None = None
    service_base_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    def _construct_fixed_service_policy(
        self, renderer: Renderer
    ) -> TinkerMessageCompleter:
        service_client = tinker.ServiceClient(base_url=self.base_url)
        sampling_client = service_client.create_sampling_client(base_model=self.service_base_model)
        return TinkerMessageCompleter(
            sampling_client=sampling_client, renderer=renderer, max_tokens=48, stop_condition=renderer.get_stop_sequences()
        )

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        if self.market_rule not in ("first_response", "top_k_quality", "auction"):
            raise ValueError(
                f"Unknown market_rule {self.market_rule}. "
                "Choose from first_response, top_k_quality, auction."
            )
        tokenizer = get_tokenizer(self.model_name)
        renderer = get_renderer(self.renderer_name, tokenizer)

        rng = random.Random(0)
        intents = list(DEFAULT_INTENTS)
        while len(intents) < self.num_markets_train + self.num_markets_test:
            intents.append(rng.choice(DEFAULT_INTENTS))
        train_intents = intents[: self.num_markets_train]
        test_intents = intents[self.num_markets_train : self.num_markets_train + self.num_markets_test]

        fixed_service_policy = self._construct_fixed_service_policy(renderer)

        train_dataset = MarketplaceDataset(
            intents=train_intents,
            renderer=renderer,
            batch_size=self.batch_size,
            market_rule=self.market_rule,
            num_services=self.num_services,
            turn_budget=self.turn_budget,
            top_k=self.top_k,
            fixed_service_policy=None,  # self-play for training
        )
        test_dataset = MarketplaceDataset(
            intents=test_intents,
            renderer=renderer,
            batch_size=len(test_intents),
            market_rule=self.market_rule,
            num_services=self.num_services,
            turn_budget=self.turn_budget,
            top_k=self.top_k,
            fixed_service_policy=fixed_service_policy,  # fixed services for eval
        )
        return train_dataset, test_dataset
