# Math Competition Datasets

This directory contains math problem datasets for evaluation.

## Format

Each JSONL file contains one problem per line in JSON format:

```json
{
  "problem": "The problem statement",
  "answer": "The ground truth answer (in extractable format)",
  "id": "unique_problem_id",
  "year": 2024
}
```

## Available Datasets

### AIME 2024 (`aime_2024.jsonl`)
- American Invitational Mathematics Examination 2024
- Example file provided: `aime_2024_example.jsonl`
- Format: Standard AIME problems with numeric answers

### AIME 2025 (`aime_2025.jsonl`)
- American Invitational Mathematics Examination 2025
- Example file provided: `aime_2025_example.jsonl`
- Format: Standard AIME problems with numeric answers

## Usage

The multi-agent debate evaluation system will automatically load these datasets when configured:

```python
from tinker_cookbook.recipes.multi_agent_debate.math_debate_dataset import (
    AIME2024EvalBuilder,
    AIME2025EvalBuilder,
)

# For AIME 2024
builder = AIME2024EvalBuilder(
    batch_size=1,
    num_test_datapoints=30,
    model_name="Qwen/Qwen3-8B-Instruct",
    renderer_name="qwen3",
)

# For AIME 2025
builder = AIME2025EvalBuilder(
    batch_size=1,
    num_test_datapoints=30,
    model_name="Qwen/Qwen3-8B-Instruct",
    renderer_name="qwen3",
)
```

## Adding Custom Datasets

To add your own math problems:

1. Create a new JSONL file in this directory
2. Each line should be a JSON object with at minimum:
   - `problem`: The problem text
   - `answer`: The answer (extractable via sympy or math_verify)
3. Load it using the generic loader:

```python
from tinker_cookbook.recipes.multi_agent_debate.math_debate_datasets import (
    load_jsonl_math_problems
)

problems = load_jsonl_math_problems(
    "data/my_custom_dataset.jsonl",
    problem_field="problem",
    answer_field="answer",
    dataset_name="my_dataset",
)
```

## Notes

- Answers should be in a format that can be extracted and verified (numeric, algebraic expressions, etc.)
- The system uses `\\boxed{}` format for model outputs
- Grading is done via sympy by default, with math_verify as an alternative
