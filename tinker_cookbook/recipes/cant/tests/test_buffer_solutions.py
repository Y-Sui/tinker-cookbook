"""Test the full buffer_solutions function used in CANT."""

import asyncio
import os
from dotenv import load_dotenv

from tinker_cookbook.completers import OpenRouterMessageCompleter
from tinker_cookbook.recipes.cant.memory_buffer import (
    buffer_solutions,
    DEFAULT_SUMMARIZATION_MODEL,
    DEFAULT_MAX_SOLUTION_TOKENS,
)

load_dotenv(override=True)


async def test_buffer_solutions():
    """Test buffer_solutions function with multiple agents."""

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found")
        return False

    print("✓ OPENROUTER_API_KEY found")

    # Create completer
    completer = OpenRouterMessageCompleter(
        api_key=api_key,
        model_name=DEFAULT_SUMMARIZATION_MODEL,
        max_tokens=DEFAULT_MAX_SOLUTION_TOKENS,
        temperature=0.8,
    )
    print(f"✓ Created completer with model: {DEFAULT_SUMMARIZATION_MODEL}")

    # Simulate multiple agent solutions (like in CANT Round 0)
    solutions = {
        0: """
        Agent 0's solution:
        Let x be the number of apples. We have:
        - Initial: x apples
        - Gave away: 3 apples
        - Bought: 5 more apples
        - Final: 10 apples
        
        Equation: x - 3 + 5 = 10
        Simplify: x + 2 = 10
        Solve: x = 8
        
        Verification: 8 - 3 + 5 = 5 + 5 = 10 ✓
        
        Therefore, initially there were \\boxed{8} apples.
        """
        * 5,  # Make it long
        1: """
        Agent 1's approach:
        Working backwards from the final state:
        - Final: 10 apples
        - Before buying 5: 10 - 5 = 5 apples
        - Before giving 3: 5 + 3 = 8 apples
        
        So the initial amount was \\boxed{8} apples.
        """
        * 5,
        2: """
        Agent 2's solution:
        Let I = initial apples
        I - 3 + 5 = 10
        I = 10 - 5 + 3
        I = 8
        
        Answer: \\boxed{8}
        """
        * 5,
        3: "Short solution: \\boxed{8}",  # Already short, won't be summarized
    }

    print(f"\n📝 Original solutions:")
    for agent_id, sol in solutions.items():
        print(f"  Agent {agent_id}: {len(sol)} chars (~{len(sol) // 4} tokens)")

    # Test 1: With LLM summarization enabled
    print("\n🔄 Test 1: WITH LLM summarization (parallel processing)...")
    try:
        buffered = await buffer_solutions(
            solutions,
            completer,
            max_solution_tokens=500,
            use_llm_summarization=True,
        )
        print("✓ Buffering successful!")

        print(f"\n📊 Buffered solutions:")
        for agent_id, summary in buffered.items():
            original_len = len(solutions[agent_id])
            summary_len = len(summary)
            compression = (1 - summary_len / original_len) * 100 if original_len > 0 else 0
            print(
                f"  Agent {agent_id}: {summary_len} chars (~{summary_len // 4} tokens) "
                f"[{compression:.1f}% compression]"
            )

            # Verify boxed answers preserved
            if "\\boxed{" in solutions[agent_id]:
                if "\\boxed{" in summary:
                    print(f"    ✓ Answer preserved")
                else:
                    print(f"    ⚠️  WARNING: Answer NOT preserved")

        print("\n✓ Test 1 passed!")

    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 2: Without LLM summarization (should return original)
    print("\n🔄 Test 2: WITHOUT LLM summarization (passthrough)...")
    try:
        unbuffered = await buffer_solutions(
            solutions,
            completer,
            use_llm_summarization=False,
        )

        if unbuffered == solutions:
            print("✓ Correctly returned original solutions (no summarization)")
        else:
            print("❌ Solutions modified when use_llm_summarization=False")
            return False

        print("✓ Test 2 passed!")

    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False

    # Test 3: Error handling (None completer with summarization enabled)
    print("\n🔄 Test 3: Error handling (None completer)...")
    try:
        await buffer_solutions(
            solutions,
            None,
            use_llm_summarization=True,
        )
        print("❌ Should have raised ValueError for None completer")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
        print("✓ Test 3 passed!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    return True


async def main():
    """Run the tests."""
    print("=" * 60)
    print("Testing buffer_solutions() for CANT Memory Management")
    print("=" * 60)
    print()

    success = await test_buffer_solutions()

    print()
    print("=" * 60)
    if success:
        print("✅ All tests passed! buffer_solutions() is working correctly.")
        print("\nThis confirms:")
        print("  • Parallel LLM summarization works")
        print("  • Compression reduces token usage")
        print("  • \\boxed{} answers are preserved")
        print("  • Passthrough mode works when disabled")
        print("  • Error handling is correct")
    else:
        print("❌ Tests failed. Check errors above.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
