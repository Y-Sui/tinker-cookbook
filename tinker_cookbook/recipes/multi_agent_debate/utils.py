from typing import TYPE_CHECKING

from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.utils import logtree

if TYPE_CHECKING:
    from tinker_cookbook.recipes.multi_agent_debate.core.coordinator import MultiAgentCoordinator

from .core.prompts import AGENT_SYSTEM_PROMPT, format_persona_intro

# Stop when we see the closing tag of the last required field
STOP_CONDITION = ["</comparison>"]


def get_step_idx_before_turn(agent_id: int, turn_idx: int, num_agents: int) -> int:
    """
    Find the step index for agent_id's most recent turn before turn_idx.

    Returns -1 if the agent hasn't acted yet.

    Example: agent_id=0, turn_idx=5, num_agents=3
    - Agent 0's turns: 0, 3, 6, ...
    - Most recent before 5 is turn 3
    - Step index = 3 // 3 = 1
    """
    # Find agent's most recent turn before turn_idx
    if agent_id >= turn_idx:
        return -1  # Agent hasn't acted yet

    # Count how many complete rounds have passed
    current_round = turn_idx // num_agents

    # If agent_id < (turn_idx % num_agents), they've acted in current round
    # Otherwise, their last action was in the previous round
    if agent_id < (turn_idx % num_agents):
        most_recent_turn = current_round * num_agents + agent_id
    else:
        most_recent_turn = (current_round - 1) * num_agents + agent_id

    if most_recent_turn < 0:
        return -1

    # Convert turn index to step index for this agent
    step_idx = most_recent_turn // num_agents
    return step_idx


def log_debate_transcript(
    coordinator: "MultiAgentCoordinator",
    *,
    metrics_by_agent: list[dict[str, float]] | None = None,
) -> None:
    with logtree.scope_header("Debate Transcript"):
        logtree.log_text(f"Question: {coordinator.state.question}")
        num_agents = coordinator.state.num_agents
        logtree.log_text(f"History policy: last round only ({num_agents} turns).")
        turns = coordinator.state.agent_responses
        if not turns:
            logtree.log_text("(No responses captured)")
            return
        num_cycles = (len(turns) + num_agents - 1) // num_agents
        for cycle_idx in range(num_cycles):
            cycle_start = cycle_idx * num_agents
            cycle_end = cycle_start + num_agents
            cycle_turns = turns[cycle_start:cycle_end]
            with logtree.scope_header(f"Cycle {cycle_idx + 1}"):
                for offset, response in enumerate(cycle_turns):
                    turn_idx = cycle_start + offset + 1
                    with logtree.scope_header(f"Turn {turn_idx} (Agent {response.author_id})"):
                        with logtree.scope_details("System prompt"):
                            persona_intro = format_persona_intro(response.author_id)
                            logtree.log_text(
                                AGENT_SYSTEM_PROMPT.format(
                                    agent_id=response.author_id,
                                    persona_intro=persona_intro,
                                )
                            )
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
        if metrics_by_agent:
            with logtree.scope_header("Evaluation Metrics"):
                for agent_id, metrics in enumerate(metrics_by_agent):
                    with logtree.scope_header(f"Agent {agent_id}"):
                        logtree.log_text(str(metrics))


def log_direct_evaluation(
    problem: str, response_text: str, parsed_solution: str, metrics: dict[str, float]
) -> None:
    """Log direct evaluation results including system prompt, completion, and solution."""
    with logtree.scope_header("Direct Evaluation"):
        with logtree.scope_details("System prompt"):
            logtree.log_text(
                "Solve the following math problem. Write your final answer in \\boxed{} format."
            )
        with logtree.scope_details("Problem"):
            logtree.log_text(problem)
        with logtree.scope_details("Completion (raw response)"):
            logtree.log_text(response_text)
        with logtree.scope_details("Parsed solution"):
            logtree.log_text(parsed_solution)
        with logtree.scope_details("Metrics"):
            logtree.log_text(str(metrics))


def log_debate_evaluation_final_solutions(
    agent_solutions: list[tuple[int, str | None, str, dict[str, float]]],
) -> None:
    """Log final solutions and metrics for each agent in debate evaluation.

    Args:
        agent_solutions: List of (agent_id, solution_text, parsed_answer, metrics) tuples
    """
    with logtree.scope_header("Final Solutions and Metrics"):
        for agent_id, solution_text, parsed_answer, metrics in agent_solutions:
            with logtree.scope_header(f"Agent {agent_id}"):
                if solution_text is None:
                    logtree.log_text("(No response)")
                else:
                    with logtree.scope_details("Final solution text"):
                        logtree.log_text(solution_text)
                    with logtree.scope_details("Parsed answer"):
                        logtree.log_text(parsed_answer)
                logtree.log_text(f"Metrics: {metrics}")


def get_debate_stop_condition(renderer: Renderer) -> StopCondition:
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


def get_summarizer_stop_condition(renderer: Renderer) -> StopCondition:
    renderer_stop = renderer.get_stop_sequences()
    return renderer_stop or []
