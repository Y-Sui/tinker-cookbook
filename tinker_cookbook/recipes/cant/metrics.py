"""
Grading utilities for CANT evaluation.

Note: Primary evaluation is now handled by Inspect AI tasks in eval/inspect_tasks.py.
This module contains only low-level grading utilities used by custom scorers.
"""

# Re-export grading utilities from math_rl for backward compatibility
from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)

__all__ = [
    "extract_boxed",
    "grade_answer",
    "grade_answer_math_verify",
    "run_with_timeout_signal",
]
