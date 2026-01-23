"""Quick test to verify OpenRouter LLM summarization works."""

import asyncio
import os
from dotenv import load_dotenv

from tinker_cookbook.completers import OpenRouterMessageCompleter
from tinker_cookbook.recipes.cant.memory_buffer import (
    summarize_solution_openrouter,
    DEFAULT_SUMMARIZATION_MODEL,
    DEFAULT_MAX_SOLUTION_TOKENS,
)

load_dotenv(override=True)


async def test_summarization():
    """Test that OpenRouter summarization works end-to-end."""

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        print("Please set it in .env file or export it")
        return False

    print("✓ OPENROUTER_API_KEY found")

    # Create completer
    try:
        completer = OpenRouterMessageCompleter(
            api_key=api_key,
            model_name=DEFAULT_SUMMARIZATION_MODEL,
            max_tokens=DEFAULT_MAX_SOLUTION_TOKENS,
            temperature=0.8,
        )
        print(f"✓ Created OpenRouterMessageCompleter with model: {DEFAULT_SUMMARIZATION_MODEL}")
    except Exception as e:
        print(f"❌ Failed to create completer: {e}")
        return False

    # Test solution (long enough to trigger summarization)
    test_solution = (
        """
    Let me solve this step by step.
    
    First, I need to understand the problem. We have a quadratic equation:
    ax^2 + bx + c = 0
    
    Step 1: Identify the coefficients
    a = 2, b = -5, c = 3
    
    Step 2: Apply the quadratic formula
    x = (-b ± sqrt(b^2 - 4ac)) / (2a)
    
    Step 3: Calculate the discriminant
    Δ = b^2 - 4ac
    Δ = (-5)^2 - 4(2)(3)
    Δ = 25 - 24
    Δ = 1
    
    Step 4: Since Δ > 0, we have two real roots
    x1 = (5 + sqrt(1)) / 4 = (5 + 1) / 4 = 6/4 = 3/2
    x2 = (5 - sqrt(1)) / 4 = (5 - 1) / 4 = 4/4 = 1
    
    Step 5: Verify the solutions
    For x = 3/2:
    2(3/2)^2 - 5(3/2) + 3 = 2(9/4) - 15/2 + 3 = 9/2 - 15/2 + 6/2 = 0 ✓
    
    For x = 1:
    2(1)^2 - 5(1) + 3 = 2 - 5 + 3 = 0 ✓
    
    Therefore, the solutions are \\boxed{x = 1 \\text{ or } x = \\frac{3}{2}}.
    """
        * 3
    )  # Make it longer to ensure summarization triggers

    print(f"✓ Test solution length: {len(test_solution)} chars (~{len(test_solution) // 4} tokens)")

    # Test summarization
    try:
        print("\n🔄 Calling OpenRouter to summarize...")
        summary = await summarize_solution_openrouter(
            test_solution,
            completer,
            target_tokens=500,
            max_retries=3,
        )
        print("✓ Summarization successful!")
        print(f"\n📝 Summary length: {len(summary)} chars (~{len(summary) // 4} tokens)")
        print(f"\n📄 Summary content:\n{'-' * 60}\n{summary}\n{'-' * 60}")

        # Check that boxed answer is preserved
        if "\\boxed{" in test_solution:
            if "\\boxed{" in summary:
                print("\n✓ \\boxed{} answer preserved in summary")
            else:
                print("\n⚠️  WARNING: \\boxed{} answer NOT preserved in summary")

        return True

    except Exception as e:
        print(f"❌ Summarization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run the test."""
    print("=" * 60)
    print("Testing OpenRouter LLM Summarization for CANT")
    print("=" * 60)
    print()

    success = await test_summarization()

    print()
    print("=" * 60)
    if success:
        print("✅ All checks passed! Summarization is working.")
    else:
        print("❌ Test failed. Check errors above.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
