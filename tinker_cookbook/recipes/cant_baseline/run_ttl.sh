# single turn
python -m tinker_cookbook.recipes.cant_baseline.train \
    model_name="Qwen/Qwen3-8B" \
    renderer_name="qwen3_disable_thinking" \
    max_tokens=8096 \
    env_type="verifiable" \
    num_agents=4 \
    batch_size=64 \
    num_train_datapoints=8000 \
    epoch=15 \
    eval_every=4 \
    save_every=4 \
    train_datasets=math500,aime2024,aime2025 \
    test_datasets=aime2024,aime2025,math500 \
    wandb_project="CANT-01-20" \
    ground_truth_mode="single_turn" \
    num_agents=1 \
    wandb_name="Qwen3-8B-disable-thinking-ttl-baseline-single-turn"


# multi-turn
python -m tinker_cookbook.recipes.cant_baseline.train \
    model_name="Qwen/Qwen3-8B" \
    renderer_name="qwen3_disable_thinking" \
    max_tokens=8096 \
    env_type="verifiable" \
    num_agents=4 \
    batch_size=64 \
    num_train_datapoints=8000 \
    epoch=15 \
    eval_every=4 \
    save_every=4 \
    train_datasets=math500,aime2024,aime2025 \
    test_datasets=aime2024,aime2025,math500 \
    wandb_project="CANT-01-20" \
    ground_truth_mode="multi_round" \
    num_agents=4 \
    wandb_name="Qwen3-8B-disable-thinking-ttl-baseline-multi-round"