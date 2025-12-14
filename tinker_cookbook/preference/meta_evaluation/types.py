"""
Schema definitions for multi-agent meta-evaluation.

This module provides Pydantic models for:
- MetaEvaluation: Output from an agent's evaluation with meta-critique
- ComparisonHistory: History of evaluations for a specific comparison
- AgentPersona: Agent configuration with persona and system prompt
"""

from pydantic import BaseModel, Field
from typing import Literal


class AgentPersona(BaseModel):
    """Defines agent persona with system prompt."""

    agent_id: str
    persona_name: str  # "innovator", "critic", "synthesizer"
    system_prompt: str


class MetaEvaluation(BaseModel):
    """Meta-evaluation output from an agent.

    Contains both the preference judgment and meta-critique of previous evaluations.
    """

    round_id: int
    agent_id: str

    # Core evaluation
    thinking: str = Field(description="Reasoning process for this evaluation")
    comparison_result: Literal["A", "B", "Tie"] = Field(
        description="Which completion is better: A, B, or Tie"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the judgment (0.0 to 1.0)"
    )

    # Meta-evaluation (critiques)
    critique_of_previous_judge: str | None = Field(
        default=None,
        description="Critique of how the previous agent evaluated (None for first round)",
    )
    critique_of_solutions: str = Field(
        description="Analysis of both solutions A and B"
    )

    # Convergence signal
    consensus_reached: bool = Field(
        default=False, description="Whether agent believes consensus is reached"
    )


class ComparisonHistory(BaseModel):
    """History of evaluations for a specific comparison (or similar comparisons).

    Tracks how different agents have evaluated a comparison across rounds,
    enabling meta-learning from judgment evolution.
    """

    comparison_signature: str  # Hash or fingerprint of comparison
    evaluations: list[MetaEvaluation] = Field(default_factory=list)

    def add_evaluation(self, eval: MetaEvaluation) -> None:
        """Add a new evaluation to the history."""
        self.evaluations.append(eval)

    def to_context_str(self) -> str:
        """Format history for LLM context.

        Returns a formatted string showing the evolution of evaluations,
        suitable for inclusion in an agent's prompt.
        """
        if not self.evaluations:
            return "(No previous evaluations)"

        text = "### Previous Evaluation History:\n\n"
        for eval in self.evaluations:
            text += f"--- Round {eval.round_id} by {eval.agent_id.upper()} ---\n"
            text += f"Decision: {eval.comparison_result} (confidence: {eval.confidence:.2f})\n"
            text += f"Reasoning: {eval.thinking[:200]}...\n"
            if eval.critique_of_previous_judge:
                text += (
                    f"Meta-critique: {eval.critique_of_previous_judge[:150]}...\n"
                )
            text += "\n"

        return text
