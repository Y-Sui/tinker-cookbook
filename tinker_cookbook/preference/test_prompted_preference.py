"""
Simple test case for prompted preference model.

Usage:
    python test_prompted_preference.py
"""

import asyncio
import os
from pathlib import Path

# Load .env file
env_path = Path.cwd() / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

from tinker_cookbook.preference.prompted_preference import (
    PromptedPreferenceModelBuilder,
)
from tinker_cookbook.preference.types import Comparison


async def test_prompted_preference():
    """Test the prompted preference model with a simple comparison."""

    # Create a simple test comparison
    comparison = Comparison(
        prompt_conversation=[{"role": "user", "content": "What is the capital of France?"}],
        completion_A=[{"role": "assistant", "content": "The capital of France is Paris."}],
        completion_B=[{"role": "assistant", "content": "I don't know."}],
    )

    # Create the preference model builder
    builder = PromptedPreferenceModelBuilder(
        model_name="anthropic/claude-sonnet-4.5",
        api_provider="openrouter",
    )

    # Build the preference model
    print("Creating preference model...")
    preference_model = builder()

    # Evaluate the comparison
    print("\nEvaluating comparison...")
    print("Prompt:", comparison.prompt_conversation[0]["content"])
    print("Completion A:", comparison.completion_A[0]["content"])
    print("Completion B:", comparison.completion_B[0]["content"])
    print()

    # Get the preference score
    score = await preference_model(comparison)

    print(f"\nPreference score: {score}")
    print("Interpretation:")
    if score < 0:
        print("  → Completion A is preferred")
    elif score > 0:
        print("  → Completion B is preferred")
    else:
        print("  → Tie (both are equally good)")

    # Expected: Completion A should be preferred (score = -1.0)
    assert score == -1.0, f"Expected A to be preferred (score=-1.0), got {score}"
    print("\n✓ Test passed!")


async def test_tie_case():
    """Test a case where completions should be tied."""

    comparison = Comparison(
        prompt_conversation=[{"role": "user", "content": "Say hello"}],
        completion_A=[{"role": "assistant", "content": "Hello!"}],
        completion_B=[{"role": "assistant", "content": "Hi there!"}],
    )

    builder = PromptedPreferenceModelBuilder(
        model_name="anthropic/claude-sonnet-4.5",
        api_provider="openrouter",
    )

    preference_model = builder()

    print("\n--- Testing Tie Case ---")
    print("Prompt:", comparison.prompt_conversation[0]["content"])
    print("Completion A:", comparison.completion_A[0]["content"])
    print("Completion B:", comparison.completion_B[0]["content"])

    score = await preference_model(comparison)
    print(f"\nPreference score: {score}")

    # This might be a tie (0.0) or slight preference either way
    print("✓ Tie case evaluated successfully!")


async def test_custom_prompt():
    """Test with a custom evaluation prompt."""

    comparison = Comparison(
        prompt_conversation=[{"role": "user", "content": "Write a short poem about coding"}],
        completion_A=[
            {
                "role": "assistant",
                "content": "Code flows like a river,\nBugs make me shiver,\nDebug forever.",
            }
        ],
        completion_B=[{"role": "assistant", "content": "print('hello world')"}],
    )

    custom_prompt = """You are evaluating creative writing quality.

Consider:
- Creativity and originality
- Relevance to the prompt
- Engagement and appeal

Determine which completion better fulfills the creative writing request."""

    builder = PromptedPreferenceModelBuilder(
        model_name="anthropic/claude-sonnet-4.5",
        api_provider="openrouter",
        preference_prompt=custom_prompt,
    )

    preference_model = builder()

    print("\n--- Testing Custom Prompt ---")
    print("Using custom evaluation criteria for creative writing")

    score = await preference_model(comparison)
    print(f"\nPreference score: {score}")

    # Completion A (poem) should be preferred over B (code snippet)
    print("✓ Custom prompt test completed!")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Prompted Preference Model")
    print("=" * 60)

    try:
        # Test 1: Clear preference
        await test_prompted_preference()

        # Test 2: Tie case
        await test_tie_case()

        # Test 3: Custom prompt
        await test_custom_prompt()

        print("\n" + "=" * 60)
        print("All tests completed successfully! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
