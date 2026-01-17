# Multi-Agent Debate Code Polish Recommendations

## Overview
This document outlines recommendations to improve code organization, maintainability, and clarity in the multi_agent_debate recipe.

## Current Structure Analysis

### Files (12 total):
```
multi_agent_debate/
├── base_env.py              # Base classes (701 lines)
├── coordinator.py           # Turn coordination (189 lines)
├── env.py                   # Non-verifiable debate (391 lines)
├── verifiable_env.py        # Verifiable math debate (673 lines)
├── prompts.py               # Prompts & parsing (376 lines)
├── utils.py                 # Utilities (159 lines)
├── evaluator.py             # Custom evaluator (249 lines)
├── loaders.py               # Data loading (96 lines)
├── train.py                 # Training script (272 lines)
├── eval.py                  # Eval-only script (137 lines)
├── inference.py             # OpenRouter self-play (410 lines)
└── doc/                     # Documentation
```

### Strengths:
1. ✅ Good separation between verifiable/non-verifiable environments
2. ✅ Clean base class abstraction (base_env.py) eliminates duplication
3. ✅ Comprehensive documentation and docstrings
4. ✅ Well-designed coordinator with parallel generation
5. ✅ Robust reward system v2 (soft vote ratio + consensus alignment)
6. ✅ Flexible persona system for agent diversity

### Areas for Improvement:

## 1. Module Organization (Priority: HIGH)

### Issue:
- No `__init__.py` - makes imports verbose and unclear
- 12 files at root level - hard to navigate
- No clear separation between core logic, training, and evaluation

### Recommended Structure:
```
multi_agent_debate/
├── __init__.py              # Public API exports
├── README.md                # Quick start guide
│
├── core/                    # Core debate logic (NEW)
│   ├── __init__.py
│   ├── coordinator.py
│   ├── prompts.py
│   └── rewards.py          # Extracted from base_env.py
│
├── environments/            # Environment implementations (NEW)
│   ├── __init__.py
│   ├── base.py             # Renamed from base_env.py
│   ├── debate.py           # Renamed from env.py
│   └── verifiable.py       # Renamed from verifiable_env.py
│
├── evaluation/              # Evaluation components (NEW)
│   ├── __init__.py
│   └── evaluator.py
│
├── data/                    # Data loading (NEW)
│   ├── __init__.py
│   └── loaders.py
│
├── scripts/                 # Runnable scripts (NEW)
│   ├── train.py
│   ├── eval.py
│   └── openrouter_selfplay.py  # Renamed from inference.py
│
├── utils.py                 # Keep at root
└── doc/                     # Keep as is
```

### Benefits:
- Clear separation of concerns
- Easier navigation and discoverability
- Cleaner imports: `from .core import coordinator` vs `from . import coordinator`
- Separates user-facing scripts from library code

## 2. File Naming (Priority: MEDIUM)

### Issues:
- `inference.py` → Actually OpenRouter self-play, not generic inference
- `env.py` → Generic name, should be `debate.py` or `non_verifiable.py`
- `base_env.py` → Could be `base.py` in environments/ subdirectory

### Recommendations:
```python
# Before
from tinker_cookbook.recipes.multi_agent_debate.env import MultiAgentDebateEnv
from tinker_cookbook.recipes.multi_agent_debate.verifiable_env import VerifiableMultiAgentDebateEnv

# After (with new structure)
from tinker_cookbook.recipes.multi_agent_debate.environments import (
    MultiAgentDebateEnv,
    VerifiableMultiAgentDebateEnv,
)
```

## 3. Code Organization Within Files (Priority: HIGH)

### Issue: Reward System Complexity
`base_env.py` contains 700+ lines with complex reward computation logic mixed with environment code.

### Recommendation: Extract Reward System

**Current (base_env.py):**
```python
class BaseMultiAgentEnvGroupBuilder:
    def _collect_votes(self, coordinator): ...         # 60 lines
    def _compute_generator_rewards(self, votes): ...   # 50 lines
    def _compute_judge_rewards(self, votes): ...       # 140 lines
    def _compute_rewards_v2(self, ...): ...            # 70 lines
```

**Proposed (core/rewards.py):**
```python
@dataclass
class Vote:
    """A single pairwise comparison vote."""
    source_agent: int
    source_turn: int
    winner_agent: int
    winner_step: int
    loser_agent: int
    loser_step: int


class RewardSystem:
    """V2 reward system: soft vote ratio + consensus alignment."""

    def collect_votes(self, coordinator, num_agents) -> list[Vote]:
        """Parse all <comparison> tags and build vote registry."""
        ...

    def compute_generator_rewards(
        self, votes: list[Vote], num_steps_per_agent: dict
    ) -> dict[int, list[float]]:
        """Compute soft vote ratio rewards for generator tokens."""
        ...

    def compute_judge_rewards(
        self, votes: list[Vote], coordinator, num_steps_per_agent: dict
    ) -> tuple[dict[int, list[float]], int]:
        """Compute consensus-aligned rewards for judge tokens."""
        ...

    def compute_rewards_v2(
        self, trajectory_group, env_group, num_agents
    ) -> tuple[dict, dict, dict]:
        """Main entry point for v2 reward computation."""
        ...
```

**Benefits:**
- Clearer separation of concerns
- Easier to test reward logic independently
- Reduces base_env.py from 700 → ~400 lines
- Makes reward system reusable

## 4. Add Missing Documentation (Priority: MEDIUM)

### Missing Files:

#### A. `__init__.py` with Public API
```python
"""Multi-agent debate for reinforcement learning.

This recipe implements multi-agent debate with pairwise comparison rewards.
Supports both verifiable (math) and non-verifiable (open-ended) tasks.

Quick Start:
    from tinker_cookbook.recipes.multi_agent_debate import train

    # Run training
    python -m tinker_cookbook.recipes.multi_agent_debate.scripts.train \\
        env="verifiable" \\
        num_agents=3 \\
        max_rounds=3
"""

# Core components
from .core.coordinator import MultiAgentCoordinator
from .core.prompts import (
    parse_agent_response,
    ParsedResponse,
    AGENT_SYSTEM_PROMPT,
    VERIFIABLE_AGENT_SYSTEM_PROMPT,
)

# Environments
from .environments.debate import (
    MultiAgentDebateEnv,
    MultiAgentDebateDataset,
    MultiAgentDebateDatasetBuilder,
)
from .environments.verifiable import (
    VerifiableMultiAgentDebateEnv,
    VerifiableMathDebateDataset,
    VerifiableMathDebateDatasetBuilder,
    VerifiableMathProblem,
)

# Evaluation
from .evaluation.evaluator import MultiAgentDebateEvaluator

# Data loading
from .data.loaders import (
    load_questions_from_jsonl,
    load_math_problems_from_jsonl,
)

__all__ = [
    # Core
    "MultiAgentCoordinator",
    "parse_agent_response",
    "ParsedResponse",
    # Environments
    "MultiAgentDebateEnv",
    "MultiAgentDebateDataset",
    "MultiAgentDebateDatasetBuilder",
    "VerifiableMultiAgentDebateEnv",
    "VerifiableMathDebateDataset",
    "VerifiableMathDebateDatasetBuilder",
    "VerifiableMathProblem",
    # Evaluation
    "MultiAgentDebateEvaluator",
    # Data
    "load_questions_from_jsonl",
    "load_math_problems_from_jsonl",
]
```

#### B. `README.md` - Quick Start Guide
```markdown
# Multi-Agent Debate

Multi-agent debate with pairwise comparison rewards for reinforcement learning.

## Features

- **Dual Environment Support**: Verifiable (math) and non-verifiable (open-ended) tasks
- **V2 Reward System**: Soft vote ratio (generator) + consensus alignment (judge)
- **Parallel Generation**: All agents generate simultaneously within each round
- **Agent Personas**: 5 distinct reasoning styles for diversity
- **Online Test-Time Learning**: Train and evaluate on the same dataset
- **Sequence Extension**: O(T) compute scaling via KV-cache preservation

## Quick Start

### Training (Verifiable Math)
```bash
python -m tinker_cookbook.recipes.multi_agent_debate.scripts.train \
    env="verifiable" \
    dataset_path="tinker_cookbook/data/aime2024_sample.jsonl" \
    num_agents=3 \
    max_rounds=3 \
    batch_size=16
```

### Evaluation Only
```bash
python -m tinker_cookbook.recipes.multi_agent_debate.scripts.eval \
    checkpoint_path="tinker://..." \
    dataset_path="tinker_cookbook/data/aime2024_sample.jsonl" \
    num_agents=3
```

### OpenRouter Self-Play (No Training)
```bash
export OPENROUTER_API_KEY=...
python -m tinker_cookbook.recipes.multi_agent_debate.scripts.openrouter_selfplay \
    env="verifiable" \
    policy_model="openai/gpt-4o-mini" \
    num_agents=3
```

## Architecture

See `doc/algorithm_flowchart.md` and `doc/math_formulation.md` for details.

### Reward System V2

- **Generator Rewards** (for `<solution>` and `<evaluation>` tokens):
  - Soft vote ratio: R_gen = sum(votes_for) / count(votes)
  - Range: [-1, +1], provides dense gradient signal

- **Judge Rewards** (for `<comparison>` tokens):
  - Consensus alignment: R_judge = +1 if aligned with majority, -1 if not
  - Encourages accurate peer evaluation

### Agent Personas

5 distinct reasoning styles (cycled for num_agents > 5):
1. The Methodical Analyst (temp=0.6)
2. The Creative Problem-Solver (temp=1.0)
3. The Devil's Advocate (temp=0.9)
4. The Synthesizer (temp=1.0)
5. The First Principles Thinker (temp=0.8)

## Configuration

Key hyperparameters in `scripts/train.py`:

```python
@chz.chz
class CLIConfig:
    # Model
    model_name: str = "Qwen/Qwen3-8B"
    renderer_name: str = "qwen3_disable_thinking"

    # Debate
    num_agents: int = 3
    max_rounds: int = 3

    # Training
    batch_size: int = 16
    learning_rate: float = 3e-5

    # Reward weights
    lambda_gen: float = 1.0
    lambda_judge: float = 1.0
```

## Directory Structure

```
multi_agent_debate/
├── core/              # Coordinator, prompts, rewards
├── environments/      # Debate environments
├── evaluation/        # Evaluators
├── data/              # Data loaders
├── scripts/           # Training/eval scripts
├── utils.py           # Utilities
└── doc/               # Algorithm docs
```
```

## 5. Improve Type Hints (Priority: LOW)

### Issues:
- Some functions missing return type hints
- Use of `object` type in some places

### Recommendations:
```python
# Before
def get_step_idx_before_turn(agent_id, turn_idx, num_agents):
    ...

# After
def get_step_idx_before_turn(agent_id: int, turn_idx: int, num_agents: int) -> int:
    """Find the step index for agent_id's most recent turn before turn_idx.

    Returns -1 if the agent hasn't acted yet.
    """
    ...
```

## 6. Consolidate Constants (Priority: LOW)

### Issue:
Constants scattered across files (FORMAT_PENALTY, FORMAT_EXEMPT_TURNS, etc.)

### Recommendation:
```python
# core/constants.py
"""Constants for multi-agent debate reward system."""

# Reward system
FORMAT_PENALTY = -0.5
FORMAT_EXEMPT_TURNS = 2
DEFAULT_LAMBDA_GEN = 1.0
DEFAULT_LAMBDA_JUDGE = 1.0

# Stop conditions
STOP_SEQUENCES = ["</comparison>"]
```

## Implementation Priority

### Phase 1: High Impact, Low Risk
1. ✅ Add `__init__.py` with public API exports
2. ✅ Add `README.md` with quick start guide
3. ✅ Extract reward system to `core/rewards.py`

### Phase 2: Medium Impact, Medium Risk
4. ⚠️ Reorganize into subdirectories (core/, environments/, etc.)
5. ⚠️ Rename files (inference.py → openrouter_selfplay.py, env.py → debate.py)
6. ⚠️ Update all imports throughout codebase

### Phase 3: Low Impact, Low Risk
7. 📝 Add missing type hints
8. 📝 Consolidate constants

## Testing Strategy

After each phase:
1. Run existing tests: `pytest tinker_cookbook/tests/test_multi_agent_debate*.py`
2. Run smoke test: Train for 1 batch on small dataset
3. Verify imports work: `python -c "from tinker_cookbook.recipes.multi_agent_debate import *"`

## Conclusion

**Recommended Approach:** Start with Phase 1 (high impact, low risk), then evaluate if Phase 2 is worth the migration effort.

**Key Benefits:**
- Better code organization and discoverability
- Easier to onboard new contributors
- Cleaner public API
- Improved testability
- Better separation of concerns

**Migration Risk:** Phase 2 requires updating imports across the codebase and may break existing user scripts. Consider deprecation path if needed.
