python -m tinker_cookbook.recipes.cant_baseline.tinker_eval \
    model_name="Qwen/Qwen3-8B" \
    renderer_name="qwen3_disable_thinking" \
    datasets="aime2024,aime2025"

python -m tinker_cookbook.recipes.cant_baseline.openrouter_eval \
    model_name="Qwen/Qwen3-8B" \
    renderer_name="qwen3_disable_thinking" \
    datasets="aime2024,aime2025,math500" 