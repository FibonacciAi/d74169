#!/usr/bin/env python3
"""
RLM Prime Search Example
========================

Demonstrates recursive prime hunting using the d74169 Prime Sonar
combined with Recursive Language Models (RLMs).

The RLM intelligently decomposes large search spaces and recursively
zooms into interesting regions.

Requirements:
    pip install d74169 rlm

Usage:
    # Set your API key
    export ANTHROPIC_API_KEY="your-key-here"
    # or
    export OPENAI_API_KEY="your-key-here"

    # Run the example
    python examples/rlm_prime_search.py
"""

import os
import sys

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo_without_rlm():
    """
    Demonstrate the Prime Sonar tools without requiring an LLM.
    This shows what the RLM has access to.
    """
    print("=" * 60)
    print("d74169 RLM Tools Demo (No LLM Required)")
    print("=" * 60)

    from d74169.rlm_agent import create_repl_tools

    tools = create_repl_tools()

    # 1. Basic prime scan
    print("\n[1] Scanning primes in [1, 200]")
    print("-" * 40)
    result = tools['scan_primes'](1, 200)
    print(f"Found {result['count']} primes")
    print(f"Density: {result['density']:.4f}")
    print(f"First 15: {result['primes'][:15]}")

    # 2. Twin primes
    print("\n[2] Twin primes in [1, 200]")
    print("-" * 40)
    twins = tools['find_twin_primes'](1, 200)
    print(f"Found {twins['count']} twin pairs")
    print(f"Pairs: {twins['twins'][:10]}...")
    print(f"Average gap between twins: {twins['avg_gap']:.1f}")

    # 3. Prime gaps analysis
    print("\n[3] Prime gap analysis [1, 200]")
    print("-" * 40)
    gaps = tools['find_prime_gaps'](1, 200)
    print(f"Max gap: {gaps['max_gap']}")
    print(f"Avg gap: {gaps['avg_gap']:.2f}")
    print(f"Gap distribution: {gaps['gap_distribution']}")

    # 4. Sophie Germain primes
    print("\n[4] Sophie Germain primes [1, 100]")
    print("-" * 40)
    germain = tools['find_sophie_germain'](1, 100)
    print(f"Found {germain['count']} Sophie Germain primes")
    print(f"(p, 2p+1) pairs: {germain['germain']}")

    # 5. Goldbach decomposition
    print("\n[5] Goldbach decomposition of 100")
    print("-" * 40)
    goldbach = tools['goldbach_decompose'](100)
    print(f"100 = p + q has {goldbach['count']} solutions:")
    for p, q in goldbach['pairs']:
        print(f"  100 = {p} + {q}")

    # 6. Region comparison
    print("\n[6] Comparing prime density across regions")
    print("-" * 40)
    comparison = tools['compare_regions'](
        (1, 1000),
        (10000, 11000),
        (100000, 101000)
    )
    print(f"{'Region':<20} {'Density':<12} {'Twins':<8} {'Avg Gap':<10}")
    print("-" * 50)
    for region, stats in comparison.items():
        print(f"{region:<20} {stats['prime_density']:.6f}   {stats['twin_count']:<8} {stats['avg_gap']:.2f}")

    # 7. Chunking for large ranges
    print("\n[7] Chunking strategy for large ranges")
    print("-" * 40)
    chunks = tools['chunk_range'](1, 100000, num_chunks=5)
    print(f"Range [1, 100000] split into {len(chunks)} chunks:")
    for i, (start, end) in enumerate(chunks):
        print(f"  Chunk {i+1}: [{start}, {end}]")

    # 8. Raw primality scores
    print("\n[8] Top primality scores in [90, 110]")
    print("-" * 40)
    scores = tools['score_integers'](90, 110)
    print("Highest scoring integers (higher = more likely prime):")
    for n, score in scores['scores'][:10]:
        is_prime = n in [97, 101, 103, 107, 109]
        marker = "*" if is_prime else " "
        print(f"  {n:4d}: {score:8.4f} {marker}")
    print("  (* = actually prime)")

    return tools


def demo_with_rlm():
    """
    Demonstrate full RLM integration with recursive prime hunting.
    Requires API key for Anthropic or OpenAI.
    """
    print("\n" + "=" * 60)
    print("d74169 RLM Agent Demo (Requires API Key)")
    print("=" * 60)

    # Check for API key
    has_anthropic = os.environ.get('ANTHROPIC_API_KEY')
    has_openai = os.environ.get('OPENAI_API_KEY')

    if not has_anthropic and not has_openai:
        print("\nNo API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        print("Skipping RLM demo.")
        return None

    try:
        from d74169.rlm_agent import PrimeSonarRLM
    except ImportError as e:
        print(f"\nCould not import RLM: {e}")
        print("Install with: pip install rlm")
        return None

    # Configure based on available API
    if has_anthropic:
        backend = "anthropic"
        model = "claude-sonnet-4-20250514"
    else:
        backend = "openai"
        model = "gpt-4o"

    print(f"\nUsing backend: {backend}, model: {model}")

    # Create agent
    agent = PrimeSonarRLM(
        backend=backend,
        model=model,
        max_depth=2,
        verbose=True
    )

    # Example queries
    queries = [
        "Find all twin primes between 1 and 500, and identify any interesting patterns in their distribution.",
        "What is the largest prime gap under 10,000? Show me the primes before and after the gap.",
        "Compare prime density in the ranges [1,1000], [10000,11000], and [100000,101000]. Which has the highest twin prime density?",
    ]

    print("\n" + "-" * 60)
    print("Running example query...")
    print("-" * 60)

    # Run first query as demo
    query = queries[0]
    print(f"\nQuery: {query}\n")

    try:
        result = agent.analyze(query)
        print(f"\nResponse:\n{result.response}")
        print(f"\n[Execution time: {result.execution_time:.2f}s]")
    except Exception as e:
        print(f"Error running RLM query: {e}")
        return None

    return agent


def demo_hunt_patterns():
    """
    Demonstrate pattern hunting capabilities.
    """
    print("\n" + "=" * 60)
    print("Pattern Hunting Demo")
    print("=" * 60)

    # Check for API
    if not os.environ.get('ANTHROPIC_API_KEY') and not os.environ.get('OPENAI_API_KEY'):
        print("\nSkipping (no API key)")
        return

    try:
        from d74169.rlm_agent import PrimeSonarRLM

        backend = "anthropic" if os.environ.get('ANTHROPIC_API_KEY') else "openai"
        model = "claude-sonnet-4-20250514" if backend == "anthropic" else "gpt-4o"

        agent = PrimeSonarRLM(backend=backend, model=model, max_depth=2)

        print("\nHunting for twin prime clusters up to 10,000...")
        result = agent.hunt("twin_clusters", max_n=10000)
        print(f"\n{result.response[:1000]}...")  # First 1000 chars

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="d74169 RLM Prime Search Demo")
    parser.add_argument("--no-llm", action="store_true",
                        help="Run only the non-LLM demo (no API key needed)")
    parser.add_argument("--hunt", action="store_true",
                        help="Run pattern hunting demo")
    args = parser.parse_args()

    # Always run the tools demo
    tools = demo_without_rlm()

    if not args.no_llm:
        # Try RLM demo if API key available
        demo_with_rlm()

        if args.hunt:
            demo_hunt_patterns()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
