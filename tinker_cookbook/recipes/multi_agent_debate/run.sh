python -m tinker_cookbook.recipes.multi_agent_debate.train \
    hf_dataset_name="lighteval/mmlu" \
    hf_dataset_split="abstract_algebra" \
    hf_dataset_question_field="question" \
    hf_dataset_split="test" \
    max_questions=1000 \
    num_agents=2 \
    model_name="meta-llama/Llama-3.2-1B"
