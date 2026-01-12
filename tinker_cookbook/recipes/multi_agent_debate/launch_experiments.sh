#!/bin/bash

SESSION="debate-training-01-12"

# Check if session already exists (check before set -e to avoid early exit)
if tmux has-session -t $SESSION 2>/dev/null; then
    echo "Session $SESSION already exists. Attaching..."
    tmux attach-session -t $SESSION
    exit 0
fi

echo "Creating new tmux session: $SESSION"

set -e  # Enable exit on error after the session check

# Create new session with first window (Experiment 1)
tmux new-session -d -s $SESSION -n "exp1-nodebug-both"
tmux send-keys -t $SESSION:0 'conda activate tinker' C-m
tmux send-keys -t $SESSION:0 'python -m tinker_cookbook.recipes.multi_agent_debate.train \
    model_name="Qwen/Qwen3-8B" \
    renderer_name="qwen3_disable_thinking" \
    max_tokens=8096 \
    env="verifiable" \
    num_agents=3 \
    max_rounds=3 \
    history_rounds=2 \
    batch_size=16 \
    num_train_datapoints=-1 \
    epoch=3 \
    learning_rate=3e-5 \
    eval_every=2 \
    save_every=10 \
    max_parallel_evals=64 \
    summarize_history=False \
    summarize_model="Qwen/Qwen3-4B-Instruct-2507" \
    num_groups_to_log=1 \
    log_full_transcript=True \
    verifiable_dataset_path=tinker_cookbook/data/aime2025_sample.jsonl \
    verifiable_problem_field="query" \
    verifiable_answer_field="answer" \
    verifiable_grader="sympy" \
    wandb_project="CANT-01-12" \
    wandb_name="Qwen3-8B-decay-format-penalty-disable-thinking-0112" \
    enable_format_penalty=True' C-m

# # Window 2: Experiment 2 - with thinking, 8096 tokens, both features
# tmux new-window -t $SESSION:1 -n "exp2-think-8k-both"
# tmux send-keys -t $SESSION:1 'conda activate tinker' C-m
# tmux send-keys -t $SESSION:1 'python -m tinker_cookbook.recipes.multi_agent_debate.train \
#     model_name="Qwen/Qwen3-8B" \
#     renderer_name="qwen3" \
#     max_tokens=8096 \
#     env="verifiable" \
#     num_agents=3 \
#     max_rounds=3 \
#     history_rounds=2 \
#     batch_size=16 \
#     num_train_datapoints=-1 \
#     epoch=3 \
#     learning_rate=3e-5 \
#     eval_every=2 \
#     save_every=10 \
#     max_parallel_evals=64 \
#     summarize_history=False \
#     summarize_model="Qwen/Qwen3-4B-Instruct-2507" \
#     num_groups_to_log=1 \
#     log_full_transcript=True \
#     verifiable_dataset_path=tinker_cookbook/data/aime2025_sample.jsonl \
#     verifiable_problem_field="query" \
#     verifiable_answer_field="answer" \
#     verifiable_grader="sympy" \
#     wandb_project="CANT-01-12" \
#     wandb_name="Qwen3-8B-decay-format-penalty-thinking-8096" \
# #     enable_format_penalty=True' C-m

# Window 3: Experiment 3 - with thinking, 16392 tokens, both features
tmux new-window -t $SESSION:2 -n "exp3-think-16k-both"
tmux send-keys -t $SESSION:2 'conda activate tinker' C-m
tmux send-keys -t $SESSION:2 'python -m tinker_cookbook.recipes.multi_agent_debate.train \
    model_name="Qwen/Qwen3-8B" \
    renderer_name="qwen3" \
    max_tokens=16392 \
    env="verifiable" \
    num_agents=3 \
    max_rounds=3 \
    history_rounds=2 \
    batch_size=16 \
    num_train_datapoints=-1 \
    epoch=3 \
    learning_rate=3e-5 \
    eval_every=2 \
    save_every=10 \
    max_parallel_evals=64 \
    summarize_history=False \
    summarize_model="Qwen/Qwen3-4B-Instruct-2507" \
    num_groups_to_log=1 \
    log_full_transcript=True \
    verifiable_dataset_path=tinker_cookbook/data/aime2025_sample.jsonl \
    verifiable_problem_field="query" \
    verifiable_answer_field="answer" \
    verifiable_grader="sympy" \
    wandb_project="CANT-01-12" \
    wandb_name="Qwen3-8B-decay-format-penalty-thinking-16392-0112" \
    enable_format_penalty=True' C-m

# # Window 4: Experiment 4 - no thinking, only decay enabled
# tmux new-window -t $SESSION:3 -n "exp4-nodebug-decay"
# tmux send-keys -t $SESSION:3 'conda activate tinker' C-m
# tmux send-keys -t $SESSION:3 'python -m tinker_cookbook.recipes.multi_agent_debate.train \
#     model_name="Qwen/Qwen3-8B" \
#     renderer_name="qwen3_disable_thinking" \
#     max_tokens=8096 \
#     env="verifiable" \
#     num_agents=3 \
#     max_rounds=3 \
#     history_rounds=2 \
#     batch_size=16 \
#     num_train_datapoints=-1 \
#     epoch=3 \
#     learning_rate=3e-5 \
#     eval_every=2 \
#     save_every=10 \
#     max_parallel_evals=64 \
#     summarize_history=False \
#     summarize_model="Qwen/Qwen3-4B-Instruct-2507" \
#     num_groups_to_log=1 \
#     log_full_transcript=True \
#     verifiable_dataset_path=tinker_cookbook/data/aime2025_sample.jsonl \
#     verifiable_problem_field="query" \
#     verifiable_answer_field="answer" \
#     verifiable_grader="sympy" \
#     wandb_project="CANT-01-12" \
#     wandb_name="Qwen3-8B-decay-disable-thinking" \
# #     enable_format_penalty=False' C-m

# # Window 5: Experiment 5 - no thinking, only format penalty enabled
# tmux new-window -t $SESSION:4 -n "exp5-nodebug-format"
# tmux send-keys -t $SESSION:4 'conda activate tinker' C-m
# tmux send-keys -t $SESSION:4 'python -m tinker_cookbook.recipes.multi_agent_debate.train \
#     model_name="Qwen/Qwen3-8B" \
#     renderer_name="qwen3_disable_thinking" \
#     max_tokens=8096 \
#     env="verifiable" \
#     num_agents=3 \
#     max_rounds=3 \
#     history_rounds=2 \
#     batch_size=16 \
#     num_train_datapoints=-1 \
#     epoch=3 \
#     learning_rate=3e-5 \
#     eval_every=2 \
#     save_every=10 \
#     max_parallel_evals=64 \
#     summarize_history=False \
#     summarize_model="Qwen/Qwen3-4B-Instruct-2507" \
#     num_groups_to_log=1 \
#     log_full_transcript=True \
#     verifiable_dataset_path=tinker_cookbook/data/aime2025_sample.jsonl \
#     verifiable_problem_field="query" \
#     verifiable_answer_field="answer" \
#     verifiable_grader="sympy" \
#     wandb_project="CANT-01-12" \
#     wandb_name="Qwen3-8B-format-penalty-disable-thinking" \
# #     enable_format_penalty=True' C-m

# # Window 6: Experiment 6 - with thinking, only decay enabled
# tmux new-window -t $SESSION:5 -n "exp6-think-decay"
# tmux send-keys -t $SESSION:5 'conda activate tinker' C-m
# tmux send-keys -t $SESSION:5 'python -m tinker_cookbook.recipes.multi_agent_debate.train \
#     model_name="Qwen/Qwen3-8B" \
#     renderer_name="qwen3" \
#     max_tokens=16392 \
#     env="verifiable" \
#     num_agents=3 \
#     max_rounds=3 \
#     history_rounds=2 \
#     batch_size=16 \
#     num_train_datapoints=-1 \
#     epoch=3 \
#     learning_rate=3e-5 \
#     eval_every=2 \
#     save_every=10 \
#     max_parallel_evals=64 \
#     summarize_history=False \
#     summarize_model="Qwen/Qwen3-4B-Instruct-2507" \
#     num_groups_to_log=1 \
#     log_full_transcript=True \
#     verifiable_dataset_path=tinker_cookbook/data/aime2025_sample.jsonl \
#     verifiable_problem_field="query" \
#     verifiable_answer_field="answer" \
#     verifiable_grader="sympy" \
#     wandb_project="CANT-01-12" \
#     wandb_name="Qwen3-8B-decay-thinking-16392" \
# #     enable_format_penalty=False' C-m

# # Window 7: Experiment 7 - with thinking, only format penalty enabled
# tmux new-window -t $SESSION:6 -n "exp7-think-format"
# tmux send-keys -t $SESSION:6 'conda activate tinker' C-m
# tmux send-keys -t $SESSION:6 'python -m tinker_cookbook.recipes.multi_agent_debate.train \
#     model_name="Qwen/Qwen3-8B" \
#     renderer_name="qwen3" \
#     max_tokens=16392 \
#     env="verifiable" \
#     num_agents=3 \
#     max_rounds=3 \
#     history_rounds=2 \
#     batch_size=16 \
#     num_train_datapoints=-1 \
#     epoch=3 \
#     learning_rate=3e-5 \
#     eval_every=2 \
#     save_every=10 \
#     max_parallel_evals=64 \
#     summarize_history=False \
#     summarize_model="Qwen/Qwen3-4B-Instruct-2507" \
#     num_groups_to_log=1 \
#     log_full_transcript=True \
#     verifiable_dataset_path=tinker_cookbook/data/aime2025_sample.jsonl \
#     verifiable_problem_field="query" \
#     verifiable_answer_field="answer" \
#     verifiable_grader="sympy" \
#     wandb_project="CANT-01-12" \
#     wandb_name="Qwen3-8B-format-penalty-thinking-16392" \
# #     enable_format_penalty=True' C-m

# Select first window and attach
tmux select-window -t $SESSION:0
echo "Attaching to session $SESSION..."
tmux attach-session -t $SESSION
