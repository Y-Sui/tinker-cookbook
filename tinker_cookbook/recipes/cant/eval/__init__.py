"""
CANT evaluation using Inspect AI.

This package provides Inspect AI-based evaluation for the CANT recipe,
supporting both CANT protocol (multi-agent discussion) and baseline
(single-turn) evaluation modes.
"""

from tinker_cookbook.recipes.cant.eval.evaluators import CANTEvaluatorBuilder

__all__ = [
    "CANTEvaluatorBuilder",
]
