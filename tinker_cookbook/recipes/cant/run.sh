python -m tinker_cookbook.recipes.cant.train \
    model_name="Qwen/Qwen3-8B" \
    renderer_name="qwen3_disable_thinking" \
    max_tokens=8096 \
    env_type="verifiable" \
    num_agents=4 \
    batch_size=64 \
    epoch=1 \
    eval_every=20 \
    save_every=20 \
    train_datasets=math \
    test_datasets=aime2024,aime2025,math500 \
    wandb_project="CANT-01-20" \
    wandb_name="Qwen3-8B-disable-thinking-train-on-math"