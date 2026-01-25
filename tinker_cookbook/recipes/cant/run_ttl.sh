#!/bin/bash

# ==============================================================================
# CANT Checkpoint Evaluation Script (Parallel Execution)
# ==============================================================================
# This script evaluates all checkpoints from the Qwen3-8B-disable-thinking 
# training run on all available tasks. It runs evaluations IN PARALLEL for
# maximum throughput.
#
# Hardcoded Configuration:
#   - Checkpoints: multi_agent_debate/runs/Qwen3-8B-disable-thinking-train-on-deepmath
#   - Model: Qwen/Qwen3-8B
#   - Renderer: qwen3_disable_thinking
#   - Tasks: aime_2024, aime_2025, gpqa, math (inspect_evals prefix)
#   - Checkpoint interval: Every 5 steps
#
# Usage:
#   ./run_ttl.sh
#
# Parallel Execution:
#   - Evaluations run in parallel (default: max 4 concurrent jobs)
#   - Adjust max_parallel variable in script to change concurrency
#   - Requires GNU parallel (recommended) or falls back to xargs
#   - Individual logs saved to: <checkpoints_path>/eval_logs/step_*.log
#
# Output:
#   - Logs: multi_agent_debate/runs/Qwen3-8B-disable-thinking-train-on-deepmath/eval_logs/
#   - Summary table with success/failure status for each checkpoint
# ==============================================================================

# Configuration
# Hardcoded checkpoints path
checkpoints_path="/ndata/yuansui/tinker-cookbook/tinker_cookbook/recipes/multi_agent_debate/runs/Qwen3-8B-disable-thinking-train-on-deepmath"

# All available tasks to evaluate on (using inspect_evals naming convention)
# Note: Task names use underscores (aime_2024) not just numbers (aime2024)
tasks="inspect_evals/aime_2024,inspect_evals/aime_2025,inspect_evals/gpqa,inspect_evals/math"

save_every=5  # Checkpoints are saved every 5 steps
max_parallel=4  # Maximum number of parallel evaluations (adjust based on available resources)

# Validation
if [ ! -d "$checkpoints_path" ]; then
    echo "Error: Directory not found: $checkpoints_path"
    exit 1
fi

checkpoints_file="$checkpoints_path/checkpoints.jsonl"
if [ ! -f "$checkpoints_file" ]; then
    echo "Error: checkpoints.jsonl not found at: $checkpoints_file"
    exit 1
fi

# Model configuration (hardcoded based on the training run)
model_name="Qwen/Qwen3-8B"  # Base model from training run
renderer_name="qwen3_disable_thinking"  # Renderer used in training

# Create logs directory for parallel execution
logs_dir="$checkpoints_path/eval_logs"
mkdir -p "$logs_dir"

echo "=================================================="
echo "CANT Checkpoint Evaluation (Parallel)"
echo "=================================================="
echo "Checkpoints path: $checkpoints_path"
echo "Model name: $model_name"
echo "Renderer name: $renderer_name"
echo "Tasks: $tasks"
echo "Save frequency: every $save_every steps"
echo "Max parallel jobs: $max_parallel"
echo "Logs directory: $logs_dir"
echo "=================================================="
echo ""

# Parse checkpoints.jsonl and extract sampler_path entries
# Filter for checkpoints that have sampler_path and match the save_every interval
checkpoint_paths=()
checkpoint_steps=()

while IFS= read -r line; do
    # Extract sampler_path and batch/step number
    sampler_path=$(echo "$line" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('sampler_path', ''))" 2>/dev/null)
    batch=$(echo "$line" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('batch', -1))" 2>/dev/null)
    
    # Check if this checkpoint should be evaluated (has sampler_path and matches save interval)
    if [ -n "$sampler_path" ] && [ "$batch" -ge 0 ]; then
        # Check if batch is a multiple of save_every
        if [ $((batch % save_every)) -eq 0 ]; then
            checkpoint_paths+=("$sampler_path")
            checkpoint_steps+=("$batch")
        fi
    fi
done < "$checkpoints_file"

num_checkpoints=${#checkpoint_paths[@]}
echo "Found $num_checkpoints checkpoints to evaluate (every $save_every steps)"
echo ""

if [ $num_checkpoints -eq 0 ]; then
    echo "Warning: No valid checkpoints found in $checkpoints_file"
    echo "Make sure checkpoints have 'sampler_path' field and 'batch' field"
    exit 1
fi

# Function to evaluate a single checkpoint
evaluate_checkpoint() {
    local i=$1
    local model_path=$2
    local step=$3
    local log_file="$logs_dir/step_${step}.log"
    
    {
        echo "=================================================="
        echo "Evaluating checkpoint $((i+1))/$num_checkpoints"
        echo "Step: $step"
        echo "Model path: $model_path"
        echo "=================================================="
        
        # Build the command
        cmd="python -m tinker_cookbook.recipes.cant.eval.run_offline_evals"
        cmd="$cmd model_path=\"$model_path\""
        cmd="$cmd task_names=\"$tasks\""
        cmd="$cmd use_cant_protocol=true"
        cmd="$cmd num_agents=4"
        
        # Add model and renderer parameters
        cmd="$cmd model_name=\"$model_name\""
        cmd="$cmd renderer_name=\"$renderer_name\""
        
        echo "Running: $cmd"
        echo ""
        
        # Execute the evaluation
        eval $cmd
        exit_code=$?
        
        if [ $exit_code -ne 0 ]; then
            echo "Error: Evaluation failed for step $step (exit code: $exit_code)"
        else
            echo "Success: Evaluation completed for step $step"
        fi
        
        echo ""
    } 2>&1 | tee "$log_file"
    
    return $exit_code
}

# Export variables and function for parallel execution
export -f evaluate_checkpoint
export tasks model_name renderer_name logs_dir num_checkpoints

# Evaluate checkpoints in parallel
echo "Starting parallel evaluation of $num_checkpoints checkpoints..."
echo "Logs will be written to: $logs_dir/step_*.log"
echo ""

# Use GNU parallel if available, otherwise fall back to xargs
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel with max $max_parallel jobs"
    for i in "${!checkpoint_paths[@]}"; do
        echo "$i ${checkpoint_paths[$i]} ${checkpoint_steps[$i]}"
    done | parallel -j $max_parallel --colsep ' ' evaluate_checkpoint {1} {2} {3}
else
    echo "GNU parallel not found, using xargs with max $max_parallel jobs"
    for i in "${!checkpoint_paths[@]}"; do
        echo "$i ${checkpoint_paths[$i]} ${checkpoint_steps[$i]}"
    done | xargs -P $max_parallel -I {} bash -c '
        read -r idx path step <<< "{}"
        evaluate_checkpoint "$idx" "$path" "$step"
    '
fi

echo ""
echo "=================================================="
echo "Evaluation complete!"
echo "=================================================="
echo "Total checkpoints evaluated: $num_checkpoints"
echo "Logs location: $logs_dir"
echo ""
echo "Summary of results:"
for step in "${checkpoint_steps[@]}"; do
    log_file="$logs_dir/step_${step}.log"
    if [ -f "$log_file" ]; then
        if grep -q "Success: Evaluation completed" "$log_file"; then
            echo "  ✓ Step $step: SUCCESS"
        else
            echo "  ✗ Step $step: FAILED (check $log_file)"
        fi
    else
        echo "  ? Step $step: NO LOG FOUND"
    fi
done
echo "=================================================="