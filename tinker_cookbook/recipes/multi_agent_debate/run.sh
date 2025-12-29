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

# test cases for debugging, showing the entire training trajectories of the multi-agent debate, set the num_groups_to_log to 1 (log only 1 group of trajectories, means only one query), set num_train_datapoints to 1 (train on only 1 datapoint), set num_test_datapoints to 0 (no evaluation), set eval_every to 0 (no evaluation during training), set batch_size to 1 (train on only 1 datapoint per step)
python -m tinker_cookbook.recipes.multi_agent_debate.train \
    batch_size=1 \
    num_train_datapoints=1 \
    num_test_datapoints=0 \
    eval_every=0 \
    num_groups_to_log=1 \
    max_rounds=3 \
    num_agents=3 \
    model_name="Qwen/Qwen3-8B" \
    log_full_transcript=True \
    dataset_path=tinker_cookbook/data/longwriter_6k_sample.jsonl \
    dataset_field="query" \
    max_tokens=8196 \
    reward_mode="win_minus_loss" \
    summarize_history=True \
    history_rounds=2


python -m tinker_cookbook.recipes.multi_agent_debate.train \
    batch_size=1 \
    num_train_datapoints=1 \
    num_test_datapoints=0 \
    eval_every=0 \
    num_groups_to_log=1 \
    max_rounds=3 \
    num_agents=3 \
    model_name="Qwen/Qwen3-8B" \
    log_full_transcript=True \
    dataset_path=tinker_cookbook/data/longwriter_6k_sample.jsonl \
    dataset_field="query" \
    max_tokens=8196 \
    reward_mode="win_minus_loss" \
    summarize_history=False \
    history_rounds=2


# metrics saved in ~/tinker-examples/multi-agent-debate/Qwen/Qwen3-8B-debate-3agents-16groups-3e-05lr-2025-12-23-14-17/metrics.jsonl
python3 -m tinker_cookbook.recipes.multi_agent_debate.train \
    model_name="Qwen/Qwen3-8B" \
    num_agents=3 \
    max_rounds=3 \
    dataset_path=tinker_cookbook/data/longwriter_6k_sample.jsonl \
    batch_size=16 \
    max_tokens=8196 \
    num_train_datapoints=1000 \
    num_test_datapoints=250 \
    max_questions=1500 \
    reward_mode="win_minus_loss" \
    num_test_datapoints=20 \
    log_full_transcript=True \
    num_groups_to_log=1 \
    wandb_name="multi-agent-debate-longwriter-6k"



#!/bin/bash
# Quick test script for multi-agent debate environments
# Tests both non-verifiable and verifiable environments with minimal setup

set -e  # Exit on error

echo "========================================"
echo "Testing Non-Verifiable Environment"
echo "========================================"
python -m tinker_cookbook.recipes.multi_agent_debate.train \
    env="non-verifiable" \
    batch_size=1 \
    num_train_datapoints=1 \
    num_test_datapoints=0 \
    eval_every=0 \
    num_groups_to_log=1 \
    max_rounds=3 \
    num_agents=3 \
    model_name="Qwen/Qwen3-8B" \
    log_full_transcript=True \
    non_verifiable_dataset_path=tinker_cookbook/data/longwriter_6k_sample.jsonl \
    non_verifiable_dataset_field="query" \
    max_tokens=4096 \
    wandb_project=""

echo ""
echo "========================================"
echo "Testing Verifiable Environment"
echo "========================================"
python -m tinker_cookbook.recipes.multi_agent_debate.train \
    env="verifiable" \
    batch_size=16 \
    num_train_datapoints=10 \
    num_test_datapoints=4 \
    eval_every=2 \
    num_groups_to_log=10 \
    max_rounds=3 \
    num_agents=3 \
    model_name="Qwen/Qwen3-8B" \
    log_full_transcript=True \
    verifiable_dataset_path=tinker_cookbook/data/aime2024_sample.jsonl \
    verifiable_problem_field="query" \
    verifiable_answer_field="answer" \
    verifiable_grader="sympy" \
    verifiable_eval_mode="direct" \
    async_config.max_steps_off_policy=3 \
    async_config.groups_per_batch=16 \
    max_tokens=8196 \
    wandb_project=""

echo ""
echo "========================================"
echo "All tests completed successfully!"
echo "========================================"
