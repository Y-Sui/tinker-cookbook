python -m tinker_cookbook.recipes.multi_agent_debate.train \
    dataset_path="tinker_cookbook/example_data/nonverifiable_queries.jsonl" \
    dataset_field="query" \
    num_agents=2 \
    model_name="meta-llama/Llama-3.2-1B"


python3 -m tinker_cookbook.recipes.multi_agent_debate.train \
    model_name="Qwen/Qwen3-8B" \
    num_agents=3 \
    dataset_path=tinker_cookbook/example_data/nonverifiable_queries.jsonl \
    batch_size=16 \
    max_rounds=3 \
    max_tokens=8196 \
    reward_mode="win_minus_loss"


python -m tinker_cookbook.recipes.multi_agent_debate.train batch_size=1 num_train_datapoints=1 num_test_datapoints=0 eval_every=0 num_groups_to_log=1 max_rounds=3 history_rounds=-1 model_name="Qwen/Qwen3-8B" log_full_transcript=True