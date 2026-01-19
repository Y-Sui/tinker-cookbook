
python -m tinker_cookbook.recipes.multi_agent_debate.train \
    model_name="Qwen/Qwen3-8B" \
    renderer_name="qwen3_disable_thinking" \
    max_tokens=8096 \
    env="verifiable" \
    num_agents=3 \
    max_rounds=3 \
    batch_size=64 \
    num_train_datapoints=-1 \
    epoch=6 \
    learning_rate=3e-5 \
    eval_every=2 \
    save_every=10 \
    max_parallel_evals=64 \
    summarize_history=False \
    summarize_model="Qwen/Qwen3-4B-Instruct-2507" \
    num_groups_to_log=1 \
    log_full_transcript=True \
    verifiable_dataset_path=tinker_cookbook/data/math500_sample.jsonl \
    verifiable_problem_field="query" \
    verifiable_answer_field="answer" \
    verifiable_grader="sympy" \
    wandb_project="CANT-01-17" \
    wandb_name="Updated-Qwen3-8B-decay-format-penalty-disable-thinking-0117-math500" \
    enable_format_penalty=True \
    disable_eval=True \


python -m tinker_cookbook.recipes.multi_agent_debate.train \
    env="verifiable" \
    model_name="Qwen/Qwen3-8B" \
    renderer_name="qwen3_disable_thinking" \
    max_tokens=4096 \
    num_agents=3 \
    max_rounds=3 \
    batch_size=16 \
    num_train_datapoints=-1 \
    epoch=1 \
    learning_rate=3e-5 \
    use_cosine_lr_schedule=False \
    eval_every=5 \
    save_every=100 \
    max_parallel_evals=64 \
    verifiable_dataset_name="math" \
    verifiable_dataset_split="train" \
    verifiable_eval_dataset_name="math500" \
    verifiable_eval_dataset_split="test" \
    max_questions=-1 \
    verifiable_eval_max_questions=30 \
    num_groups_to_log=1 \
    log_full_transcript=False \
    wandb_project="CANT-01-20" \
    wandb_name="Updated-Qwen3-8B-disable-thinking-math500" \