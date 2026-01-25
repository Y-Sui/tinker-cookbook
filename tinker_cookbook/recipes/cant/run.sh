python -m tinker_cookbook.recipes.cant.train \
    model_name="Qwen/Qwen3-8B" \
    renderer_name="qwen3" \
    max_tokens=16384 \
    env_type="verifiable" \
    num_agents=4 \
    batch_size=16 \
    num_train_datapoints=8000 \
    epoch=1 \
    eval_every=0 \
    save_every=5 \
    train_datasets=deepmath,math500,aime2024,aime2025,gpqa, \
    wandb_project="CANT-01-25" \
    wandb_name="Qwen3-8B-disable-thinking-train-on-deepmath"


python -m tinker_cookbook.recipes.cant.train \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    renderer_name="qwen3" \
    max_tokens=16384 \
    env_type="verifiable" \
    num_agents=4 \
    batch_size=16 \
    num_train_datapoints=8000 \
    epoch=1 \
    eval_every=0 \
    save_every=5 \
    train_datasets=deepmath,math500,aime2024,aime2025,gpqa, \
    wandb_project="CANT-01-25" \
    wandb_name="Qwen3-4B-train-on-deepmath"


python -m tinker_cookbook.recipes.cant.train \
    model_name="meta-llama/Llama-3.1-8B" \
    renderer_name="llama3" \
    max_tokens=16384 \
    env_type="verifiable" \
    num_agents=4 \
    batch_size=16 \
    num_train_datapoints=8000 \
    epoch=1 \
    eval_every=0 \
    save_every=5 \
    train_datasets=deepmath,math500,aime2024,aime2025,gpqa, \
    wandb_project="CANT-01-25" \
    wandb_name="Llama-3.1-8B-train-on-deepmath"
    

python -m tinker_cookbook.recipes.cant.train \
    model_name="meta-llama/Llama-3.2-3B" \
    renderer_name="llama3" \
    max_tokens=16384 \
    env_type="verifiable" \
    num_agents=4 \
    batch_size=16 \
    num_train_datapoints=8000 \
    epoch=1 \
    eval_every=0 \
    save_every=5 \
    train_datasets=deepmath,math500,aime2024,aime2025,gpqa, \
    wandb_project="CANT-01-25" \
    wandb_name="Llama-3.2-3B-train-on-deepmath"
    