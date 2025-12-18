# Marketplace Multi-Agent RL (Assistant ↔ Services)

This recipe mirrors the Magentic Marketplace setup: a customer-facing assistant negotiates with multiple services in a two-sided market. We support three market rules:
- `first_response`: the first service reply wins (captures latency bias).
- `top_k_quality`: only the top-k (by quality) services are surfaced.
- `auction`: all services can respond; the assistant chooses.

Run a small self-play training loop:
```bash
python -m tinker_cookbook.recipes.multiplayer_rl.marketplace.train \
  market_rule=first_response \
  num_services=2 \
  batch_size=8 \
  num_markets_train=32 \
  num_markets_test=4
```

Key pieces:
- `env.py` defines `MarketplaceCoordinator` (shared state), per-role `MarketplaceEnv` (assistant + services), and `MarketplaceDatasetBuilder`. Training is self-play; evaluation uses fixed service bots.
- Observations are rendered with the configured renderer/tokenizer. The assistant acts with `Select: <service_id> price=<number>` to close a deal; services respond with offers (parsed for `price=`).
- Rewards: assistant gets welfare (quality – price – latency penalty), with a small per-turn penalty; services profit from accepted deals, otherwise 0.

Turn budget, number of services, and market rule are exposed via the CLI. Test splits pit the assistant against fixed service policies; training uses self-play.
