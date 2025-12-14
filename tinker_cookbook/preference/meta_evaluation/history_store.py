"""
Storage for per-comparison evaluation histories.

This module provides ComparisonHistoryStore which:
- Stores evaluation histories in-memory for fast lookup
- Persists to JSONL for checkpoint resumption
- Computes comparison signatures for similarity matching
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict

from tinker_cookbook.preference.meta_evaluation.types import (
    ComparisonHistory,
    MetaEvaluation,
)
from tinker_cookbook.preference.types import Comparison
from tinker_cookbook.renderers import Message

logger = logging.getLogger(__name__)


class ComparisonHistoryStore:
    """
    Stores per-comparison evaluation history.

    Design: In-memory dict with JSONL persistence for checkpointing.
    Signature is based on prompt only, so similar prompts share history.
    """

    def __init__(self, log_dir: str | None = None):
        """
        Initialize the history store.

        Args:
            log_dir: Optional directory to persist histories. If provided,
                    histories will be saved to {log_dir}/comparison_histories.jsonl
        """
        self.histories: Dict[str, ComparisonHistory] = {}
        self.log_dir = Path(log_dir) if log_dir else None

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.history_file = self.log_dir / "comparison_histories.jsonl"
            self._load_from_disk()

    def _compute_signature(self, comparison: Comparison) -> str:
        """
        Compute hash signature for comparison.

        Note: We hash the prompt only, not completions, so similar
        prompts share history (per user requirement: "track how judgments
        for similar comparisons evolved").

        Args:
            comparison: The comparison to compute signature for

        Returns:
            16-character hex signature
        """
        prompt_text = self._messages_to_text(comparison.prompt_conversation)
        return hashlib.sha256(prompt_text.encode()).hexdigest()[:16]

    def _messages_to_text(self, messages: list[Message]) -> str:
        """Serialize messages for hashing."""
        return json.dumps(
            [{"role": m["role"], "content": m["content"]} for m in messages]
        )

    def get_history(self, comparison: Comparison) -> ComparisonHistory:
        """
        Get history for comparison (creates empty if not exists).

        Args:
            comparison: The comparison to get history for

        Returns:
            ComparisonHistory object (possibly empty)
        """
        signature = self._compute_signature(comparison)

        if signature not in self.histories:
            self.histories[signature] = ComparisonHistory(
                comparison_signature=signature
            )

        return self.histories[signature]

    def add_evaluation(
        self,
        comparison: Comparison,
        evaluation: MetaEvaluation,
    ) -> None:
        """
        Add evaluation to comparison history.

        Args:
            comparison: The comparison being evaluated
            evaluation: The meta-evaluation result
        """
        history = self.get_history(comparison)
        history.add_evaluation(evaluation)

        # Persist to disk
        if self.log_dir:
            self._append_to_disk(history)

    def _append_to_disk(self, history: ComparisonHistory) -> None:
        """Append history to JSONL file."""
        with open(self.history_file, "a") as f:
            f.write(history.model_dump_json() + "\n")

    def _load_from_disk(self) -> None:
        """Load histories from JSONL file."""
        if not self.history_file.exists():
            return

        with open(self.history_file, "r") as f:
            for line in f:
                if line.strip():
                    history = ComparisonHistory.model_validate_json(line)
                    self.histories[history.comparison_signature] = history

        logger.info(f"Loaded {len(self.histories)} comparison histories")
