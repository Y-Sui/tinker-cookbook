python -m tinker_cookbook.recipes.cant.train \
    model_name="Qwen/Qwen3-8B" \
    renderer_name="qwen3" \
    max_tokens=16384 \
    env_type="verifiable" \
    num_agents=4 \
    batch_size=32 \
    num_train_datapoints=8000 \
    epoch=1 \
    eval_every=0 \
    save_every=5 \
    train_datasets=deepmath,aime2024,aime2025,gpqa, \
    wandb_project="CANT-01-25" \
    wandb_name="Qwen3-8B-disable-thinking-train-on-deepmath"