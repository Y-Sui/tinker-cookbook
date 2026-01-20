python -m tinker_cookbook.recipes.cant.train \
    model_name="Qwen/Qwen3-8B" \
    renderer_name="qwen3_disable_thinking" \
    max_tokens=8096 \
    env_type="verifiable" \
    num_agents=4 \
    batch_size=16 \
    epoch=30 \
    verifiable_dataset_path="tinker_cookbook/data/aime2024_sample.jsonl" \
    verifiable_problem_field="query" \
    verifiable_answer_field="answer" \
    wandb_project="CANT-01-20" \
    wandb_name="Updated-Qwen3-8B-disable-thinking-aime2024-new"