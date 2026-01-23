"""
Generic CANT protocol solver for Inspect AI.

This module provides a generic solver that wraps any Inspect AI task with
the CANT 4-round multi-agent discussion protocol. The solver can be composed
with any existing task without modification.

Usage:
    # Wrap any Inspect AI task with CANT protocol
    from inspect_ai import get_task

    task = get_task("inspect_evals/gsm8k")
    cant_task = with_cant_protocol(task, num_agents=4)
"""

import asyncio
import logging

from inspect_ai import Task, task, task_with
from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.solver import Generate, Solver, generate, solver
from inspect_ai.solver import TaskState

from tinker_cookbook.recipes.cant.coordinator import CANTCoordinator
from tinker_cookbook.recipes.cant.prompts import (
    get_default_agent_personas,
    get_round1_system_prompt,
    get_round2_system_prompt,
    get_round3_system_prompt,
    get_round4_system_prompt,
    get_user_message_round1,
    get_user_message_round2,
    get_user_message_round3,
    get_user_message_round4,
)

logger = logging.getLogger(__name__)


@solver
def cant_protocol_solver(num_agents: int = 4) -> Solver:
    """
    Generic solver that runs CANT 4-round multi-agent protocol.

    This solver can be composed with ANY Inspect AI task. It runs the full
    CANT discussion protocol on the input, generates a solution through
    multi-agent collaboration, and returns the highest-ranked solution.

    The solver is designed to be used as the first solver in a chain,
    followed by any task-specific solvers.

    Args:
        num_agents: Number of agents to participate in the discussion

    Returns:
        Solver function that executes the CANT protocol

    Example:
        @task
        def my_task():
            return Task(
                dataset=my_dataset,
                solver=[cant_protocol_solver(num_agents=4), generate()],
                scorer=my_scorer(),
            )
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        """
        Run CANT protocol and return highest-ranked solution.

        1. Round 0: Each agent proposes initial solution
        2. Round 1: Agents rank and critique solutions
        3. Round 2: Agents revise their solutions
        4. Round 3: Agents provide final rankings
        5. Return highest-ranked solution for standard evaluation
        """
        question = state.input_text

        # Initialize coordinator
        coordinator = CANTCoordinator(
            question=question,
            num_agents=num_agents,
        )

        # Get agent personas
        personas = get_default_agent_personas()

        # Run 4 rounds for all agents
        try:
            await _run_cant_protocol(coordinator, num_agents, personas, generate, state)
        except Exception as e:
            logger.error(f"Error running CANT protocol: {e}", exc_info=True)
            state.output = ModelOutput.from_content(model="cant", content="[ERROR]")
            return state

        # Select final answer (highest ranked)
        final_answer = _select_highest_ranked_solution(coordinator)

        # Return for standard evaluation
        state.output = ModelOutput.from_content(model="cant", content=final_answer)
        return state

    return solve


async def _run_cant_protocol(
    coordinator: CANTCoordinator,
    num_agents: int,
    personas: list[str],
    generate: Generate,
    state: TaskState,
) -> None:
    """Execute full 4-round CANT protocol."""

    from copy import deepcopy

    # Round 0: Initial proposals (parallel agent execution)
    agent_states = []
    for agent_id in range(num_agents):
        system_prompt = get_round1_system_prompt(personas[agent_id % len(personas)])
        user_message = get_user_message_round1(coordinator.question)

        # Create a clean state for this agent with just the prompt
        agent_state = deepcopy(state)
        agent_state.messages = [ChatMessageUser(content=system_prompt + "\n\n" + user_message)]
        agent_states.append(agent_state)

    # Generate all responses in parallel
    responses = await asyncio.gather(*[generate(agent_state) for agent_state in agent_states])

    # Store responses
    for agent_id, response in enumerate(responses):
        response_text = response.output.completion if response.output else ""
        coordinator.add_round1_response(agent_id, response_text)

    # Advance to Round 1
    coordinator.advance_round()

    # Round 1: Critique and ranking (parallel agent execution)
    agent_states = []
    for agent_id in range(num_agents):
        system_prompt = get_round2_system_prompt(personas[agent_id % len(personas)])
        user_message = get_user_message_round2(
            coordinator.question, coordinator.get_initial_solutions()
        )

        agent_state = deepcopy(state)
        agent_state.messages = [ChatMessageUser(content=system_prompt + "\n\n" + user_message)]
        agent_states.append(agent_state)

    # Generate all responses in parallel
    responses = await asyncio.gather(*[generate(agent_state) for agent_state in agent_states])

    # Store responses
    for agent_id, response in enumerate(responses):
        response_text = response.output.completion if response.output else ""
        coordinator.add_round2_response(agent_id, response_text)

    # Advance to Round 2
    coordinator.advance_round()

    # Round 2: Revision (parallel agent execution)
    agent_states = []
    for agent_id in range(num_agents):
        system_prompt = get_round3_system_prompt(personas[agent_id % len(personas)])
        user_message = get_user_message_round3(
            coordinator.question,
            coordinator.get_initial_solutions(),
            agent_id,
            coordinator.critique_texts,
        )

        agent_state = deepcopy(state)
        agent_state.messages = [ChatMessageUser(content=system_prompt + "\n\n" + user_message)]
        agent_states.append(agent_state)

    # Generate all responses in parallel
    responses = await asyncio.gather(*[generate(agent_state) for agent_state in agent_states])

    # Store responses
    for agent_id, response in enumerate(responses):
        response_text = response.output.completion if response.output else ""
        coordinator.add_round3_response(agent_id, response_text)

    # Advance to Round 3
    coordinator.advance_round()

    # Round 3: Final ranking (parallel agent execution)
    agent_states = []
    for agent_id in range(num_agents):
        system_prompt = get_round4_system_prompt(personas[agent_id % len(personas)])
        user_message = get_user_message_round4(
            coordinator.question,
            coordinator.revised_solutions,
        )

        agent_state = deepcopy(state)
        agent_state.messages = [ChatMessageUser(content=system_prompt + "\n\n" + user_message)]
        agent_states.append(agent_state)

    # Generate all responses in parallel
    responses = await asyncio.gather(*[generate(agent_state) for agent_state in agent_states])

    # Store responses
    for agent_id, response in enumerate(responses):
        response_text = response.output.completion if response.output else ""
        coordinator.add_round4_response(agent_id, response_text)

    # Mark protocol complete
    coordinator.advance_round()

    # Round 1: Critique and ranking
    for agent_id in range(num_agents):
        system_prompt = get_round2_system_prompt(personas[agent_id % len(personas)])
        user_message = get_user_message_round2(
            coordinator.question, coordinator.get_initial_solutions()
        )

        agent_state = state
        agent_state.messages = [ChatMessageUser(content=system_prompt + "\n\n" + user_message)]
        response = await generate(agent_state)

        response_text = response.output.completion if response.output else ""
        coordinator.add_round2_response(agent_id, response_text)

    # Advance to Round 2
    coordinator.advance_round()

    # Round 2: Revision
    for agent_id in range(num_agents):
        system_prompt = get_round3_system_prompt(personas[agent_id % len(personas)])
        user_message = get_user_message_round3(
            coordinator.question,
            coordinator.get_initial_solutions(),
            agent_id,
            coordinator.critique_texts,
        )

        agent_state = state
        agent_state.messages = [ChatMessageUser(content=system_prompt + "\n\n" + user_message)]
        response = await generate(agent_state)

        response_text = response.output.completion if response.output else ""
        coordinator.add_round3_response(agent_id, response_text)

    # Advance to Round 3
    coordinator.advance_round()

    # Round 3: Final ranking
    for agent_id in range(num_agents):
        system_prompt = get_round4_system_prompt(personas[agent_id % len(personas)])
        user_message = get_user_message_round4(
            coordinator.question,
            coordinator.revised_solutions,
        )

        agent_state = state
        agent_state.messages = [ChatMessageUser(content=system_prompt + "\n\n" + user_message)]
        response = await generate(agent_state)

        response_text = response.output.completion if response.output else ""
        coordinator.add_round4_response(agent_id, response_text)

    # Mark protocol complete
    coordinator.advance_round()


def _select_highest_ranked_solution(coordinator: CANTCoordinator) -> str:
    """
    Select solution with best final ranking.

    Computes which solution received the best rankings across all agents
    and returns that solution text.

    The scoring system works as follows:
    - Each agent provides a ranking for all solutions (1 = best, 2 = second, etc.)
    - We convert ranks to scores where higher is better: score = num_agents - rank + 1
    - The solution with the highest cumulative score is selected

    Args:
        coordinator: Coordinator containing all protocol state

    Returns:
        Text of the highest-ranked solution
    """
    if not coordinator.final_rankings or not coordinator.revised_solutions:
        # Fallback to first initial solution if protocol failed
        if coordinator.initial_solutions:
            return list(coordinator.initial_solutions.values())[0]
        return "[NO SOLUTION GENERATED]"

    # Count ranking positions for each agent
    agent_scores: dict[int, float] = {i: 0.0 for i in range(coordinator.num_agents)}

    for ranker_id, rankings in coordinator.final_rankings.items():
        # rankings is list of (agent_id, justification, rank)
        for agent_id, _, rank in rankings:
            # Lower rank is better (1 is best)
            # Convert to score where higher is better
            agent_scores[agent_id] += coordinator.num_agents - rank + 1

    # Find agent with highest score
    if not agent_scores:
        # Fallback
        return list(coordinator.revised_solutions.values())[0]

    best_agent_id = max(agent_scores.items(), key=lambda x: x[1])[0]

    # Return their revised solution
    return coordinator.revised_solutions.get(best_agent_id, "[NO SOLUTION]")


def with_cant_protocol(task: Task, num_agents: int = 4) -> Task:
    """
    Wrap any Inspect AI task with CANT protocol solver.

    This function takes any existing Inspect AI task and wraps it with the
    CANT 4-round multi-agent discussion protocol. The CANT protocol runs first
    to generate solutions through collaboration, then the task's original
    solver and scorer are applied.

    This allows you to evaluate ANY Inspect AI task using the CANT protocol
    without modifying the task definition.

    Args:
        task: Inspect AI Task object to wrap
        num_agents: Number of agents for CANT protocol (default: 4)

    Returns:
        New Task with CANT protocol prepended to solver chain

    Example:
        # Wrap standard GSM8K task
        from inspect_ai import get_task

        gsm8k = get_task("inspect_evals/gsm8k")
        cant_gsm8k = with_cant_protocol(gsm8k, num_agents=4)

        # Wrap custom task
        my_task = Task(dataset=..., solver=..., scorer=...)
        cant_my_task = with_cant_protocol(my_task, num_agents=4)
    """
    # Get original solver (default to generate() if none)
    original_solver = task.solver if task.solver else generate()

    # Create solver chain: CANT protocol → original solver
    # The CANT solver will populate state.output with the highest-ranked solution
    # The original solver can then process this output if needed
    new_solver = [cant_protocol_solver(num_agents), original_solver]

    # Return new task with wrapped solver
    # Keep everything else from the original task (dataset, scorer, etc.)
    return Task(
        dataset=task.dataset,
        solver=new_solver,
        scorer=task.scorer,  # Keep original scorer - don't override
        name=f"cant_{task.name}_{num_agents}agents",
        sandbox=task.sandbox,
        epochs=task.epochs,
        fail_on_error=task.fail_on_error,
        message_limit=task.message_limit,
    )


def create_cant_task(task_name: str, num_agents: int = 4):
    """
    Create a CANT protocol version of an Inspect AI task.

    Returns a module path string that Inspect AI can use to load the wrapper task.
    The wrapper task is created as a module-level function that can be imported.

    Args:
        task_name: Base task name (e.g., "inspect_evals/gsm8k")
        num_agents: Number of agents for CANT protocol

    Returns:
        String module path to a task function (for use with Inspect AI)

    Note:
        Since we can't dynamically create importable tasks, we need to use
        a different approach. For CANT protocol mode, we'll use solver parameter
        override instead of task wrapping.
    """
    # For now, return the base task name with a solver override marker
    # The actual solver will be set via eval_async parameters
    return task_name
