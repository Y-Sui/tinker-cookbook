python -m tinker_cookbook.recipes.multi_agent_debate.train \
    dataset_path="tinker_cookbook/example_data/nonverifiable_queries.jsonl" \
    dataset_field="query" \
    num_agents=2 \
    model_name="meta-llama/Llama-3.2-1B"


python3 -m tinker_cookbook.recipes.multi_agent_debate.train model_name="Qwen/Qwen3-8B" num_agents=3 dataset_path=tinker_cookbook/example_data/nonverifiable_queries.jsonl batch_size=16 max_rounds=3