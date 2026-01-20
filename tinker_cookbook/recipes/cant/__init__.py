"""
CANT (Critique and Revision) Framework

A three-round self-evolution system featuring:
- Round 1: Initial proposal generation
- Round 2: Blind evaluation + targeted critique
- Round 3: Revision + final verdict

Rewards based on persuasion success (Bradley-Terry scoring), solution quality,
consensus alignment, and improvement acceptance.
"""

from tinker_cookbook.recipes.cant.env import CANTEnv, CANTEnvGroupBuilder
from tinker_cookbook.recipes.cant.verifiable_env import VerifiableCANTEnv, VerifiableCANTEnvGroupBuilder

__all__ = [
    "CANTEnv",
    "CANTEnvGroupBuilder",
    "VerifiableCANTEnv",
    "VerifiableCANTEnvGroupBuilder",
]
