from typing import TYPE_CHECKING

from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.utils import logtree

if TYPE_CHECKING:
    from tinker_cookbook.recipes.multi_agent_debate.env import MultiAgentCoordinator

from .prompts import AGENT_SYSTEM_PROMPT

# Stop when we see the closing tag of the last required field
STOP_CONDITION = ["</comparison>"]


def _log_debate_transcript(coordinator: "MultiAgentCoordinator") -> None:
    with logtree.scope_header("Debate Transcript"):
        logtree.log_text(f"Question: {coordinator.state.question}")
        turns = coordinator.state.agent_responses
        if not turns:
            logtree.log_text("(No responses captured)")
            return
        for turn_idx, response in enumerate(turns, start=1):
            with logtree.scope_header(f"Turn {turn_idx}"):
                with logtree.scope_header(f"Agent {response.author_id}"):
                    with logtree.scope_details("System prompt"):
                        logtree.log_text(AGENT_SYSTEM_PROMPT.format(agent_id=response.author_id))
                    if response.observation:
                        with logtree.scope_details("Observation (context)"):
                            logtree.log_text(response.observation)
                    if response.solution:
                        with logtree.scope_details("Solution"):
                            logtree.log_text(response.solution)
                    if response.evaluation:
                        with logtree.scope_details("Evaluation"):
                            logtree.log_text(response.evaluation)
                    if response.comparison_text:
                        with logtree.scope_details("Comparison"):
                            logtree.log_text(response.comparison_text)
                    with logtree.scope_details("Raw response"):
                        logtree.log_text(response.raw_response)


def _get_debate_stop_condition(renderer: Renderer) -> StopCondition:
    """
    Pick stop sequences that are compatible with the active renderer.

    - Token-based renderers (e.g. `llama3`) return `list[int]` stop tokens; we must not mix in strings.
    - Text-based renderers (e.g. `role_colon`) return `list[str]` stop sequences; we can add debate-specific tags.
    """
    renderer_stop = renderer.get_stop_sequences()
    if not renderer_stop:
        return STOP_CONDITION
    if isinstance(renderer_stop[0], int):
        return renderer_stop
    # De-duplicate while preserving order
    return list(dict.fromkeys([*renderer_stop, *STOP_CONDITION]))


def _get_summarizer_stop_condition(renderer: Renderer) -> StopCondition:
    renderer_stop = renderer.get_stop_sequences()
    return renderer_stop or []
