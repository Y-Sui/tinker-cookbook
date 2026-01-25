"""Memory buffering for CANT using LLM summarization (google/gemini-flash-1.5)."""

import asyncio
import logging

from dotenv import load_dotenv

from tinker_cookbook.completers import OpenRouterMessageCompleter

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Hardcoded constants for consistency
DEFAULT_SUMMARIZATION_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_MAX_SOLUTION_TOKENS = 3000  # Target tokens per solution in buffered context
MAX_RETRIES = 3

SUMMARIZATION_PROMPT = """You are compressing a mathematical solution for multi-agent discussion.

Original solution:
{solution}

Create a comprehensive summary (~{target_tokens} tokens) that preserves:
1. The complete mathematical approach and ALL key reasoning steps
2. ALL important intermediate results and calculations
3. The final answer in \\boxed{{}} format - CRITICAL: You MUST preserve the exact \\boxed{{answer}} if present
4. Any assumptions or constraints identified

This summary will be judged by expert agents. Preserve all critical reasoning - do NOT omit important steps.

IMPORTANT: If the original solution contains \\boxed{{answer}}, your summary MUST also include \\boxed{{answer}} with the same answer.

Comprehensive Summary:"""


async def summarize_solution_openrouter(
    solution: str,
    openrouter_completer: OpenRouterMessageCompleter,
    target_tokens: int = DEFAULT_MAX_SOLUTION_TOKENS,
    max_retries: int = MAX_RETRIES,
) -> str:
    """Summarize solution via OpenRouter LLM with retry logic."""
    estimated_tokens = len(solution) // 4

    # If already under target, return as-is
    if estimated_tokens <= target_tokens:
        return solution

    prompt = SUMMARIZATION_PROMPT.format(solution=solution, target_tokens=target_tokens)

    # Retry loop with exponential backoff
    for attempt in range(max_retries):
        try:
            response = await openrouter_completer([{"role": "user", "content": prompt}])

            summary = response["content"]

            # Handle list content (multimodal messages)
            if isinstance(summary, list):
                summary = "".join(
                    [c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in summary]
                )

            return summary

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(
                    f"Summarization attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Summarization failed after {max_retries} attempts")
                raise Exception(
                    f"Failed to summarize solution after {max_retries} attempts. Last error: {e}"
                ) from e


async def buffer_solutions(
    solutions: dict[int, str],
    openrouter_completer: OpenRouterMessageCompleter | None,
    max_solution_tokens: int = DEFAULT_MAX_SOLUTION_TOKENS,
    use_llm_summarization: bool = True,
    eval_mode: bool = False,  # Aggressive compression for eval (1K vs 4K)
) -> dict[int, str]:
    """Buffer solutions via LLM (parallel) or return full."""
    target_tokens = 1000 if eval_mode else max_solution_tokens
    if not use_llm_summarization:
        return solutions

    if openrouter_completer is None:
        raise ValueError("use_llm_summarization=True requires OPENROUTER_API_KEY")

    async def summarize_one(agent_id: int, sol: str) -> tuple[int, str]:
        summary = await summarize_solution_openrouter(sol, openrouter_completer, target_tokens)
        return agent_id, summary

    results = await asyncio.gather(*[summarize_one(aid, sol) for aid, sol in solutions.items()])
    return {agent_id: summary for agent_id, summary in results}


# NEW: Critique summarization (shorter target)
CRITIQUE_SUMMARY_PROMPT = """You are compressing a critique for multi-agent context.

Original critique:
{critique}

Create a detailed, structured summary (~{target_tokens} tokens) that preserves:
1. The target agent identity and the specific claims being challenged
2. Each distinct flaw or counterargument (no merging of separate points)
3. Any evidence, examples, or calculations cited
4. The logical chain of the critique (why the flaw matters to the conclusion)
5. Any proposed fixes, alternatives, or missing considerations

Keep the same stance and tone. Do not invent new arguments. Do not drop key details.

Summary:"""


async def summarize_critique_openrouter(
    critique: str,
    openrouter_completer: OpenRouterMessageCompleter,
    target_tokens: int = 500,  # Critiques are shorter
    max_retries: int = MAX_RETRIES,
) -> str:
    """Summarize critique via OpenRouter LLM."""
    estimated_tokens = len(critique) // 4
    if estimated_tokens <= target_tokens:
        return critique

    prompt = CRITIQUE_SUMMARY_PROMPT.format(critique=critique, target_tokens=target_tokens)

    for attempt in range(max_retries):
        try:
            response = await openrouter_completer([{"role": "user", "content": prompt}])
            summary = response["content"]
            if isinstance(summary, list):
                summary = "".join(
                    [c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in summary]
                )
            return summary
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)
            else:
                raise Exception(f"Critique summarization failed: {e}") from e


async def buffer_critiques(
    critique_texts: dict[int, dict[int, str]],  # {author: {target: text}}
    openrouter_completer: OpenRouterMessageCompleter | None,
    max_critique_tokens: int = 500,
    use_llm_summarization: bool = True,
    eval_mode: bool = False,
) -> dict[int, dict[int, str]]:
    """Buffer all critiques in parallel."""
    if not use_llm_summarization:
        return critique_texts

    if openrouter_completer is None:
        raise ValueError("use_llm_summarization=True requires OPENROUTER_API_KEY")

    target_tokens = 250 if eval_mode else max_critique_tokens

    async def summarize_critique_pair(
        author: int, target_critiques: dict[int, str]
    ) -> tuple[int, dict[int, str]]:
        buffered_targets = {}
        for target, text in target_critiques.items():
            summary = await summarize_critique_openrouter(text, openrouter_completer, target_tokens)
            buffered_targets[target] = summary
        return author, buffered_targets

    # Flatten all critiques for parallel processing
    results = await asyncio.gather(
        *[summarize_critique_pair(author, targets) for author, targets in critique_texts.items()]
    )

    return {author: targets for author, targets in results}
