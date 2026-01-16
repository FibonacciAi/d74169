"""
d74169.rlm_agent - Recursive Language Model Integration
========================================================

Combines spectral prime detection with recursive LLM decomposition.
The RLM provides intelligent search guidance while PrimeSonar provides
the mathematical computation.

Architecture:
    ┌─────────────────────────────────────────┐
    │            RLM Controller               │
    │  "Find twin prime clusters in [1, 10⁶]" │
    └────────────────┬────────────────────────┘
                     │ recursive decomposition
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       [chunk1]   [chunk2]   [chunk3]
          │          │          │
          ▼          ▼          ▼
      PrimeSonar  PrimeSonar  PrimeSonar
          │          │          │
          └──────────┼──────────┘
                     ▼
           RLM aggregates findings
                     │
                     ▼
           Recursive zoom into interesting regions

Requirements:
    pip install rlm  # github.com/alexzhang13/rlm

Usage:
    from d74169.rlm_agent import PrimeSonarRLM

    agent = PrimeSonarRLM(backend="anthropic", model="claude-sonnet-4-20250514")
    result = agent.analyze("Find regions with high twin prime density up to 100,000")
    print(result.response)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import numpy as np

# Local imports
from .sonar import PrimeSonar, sieve_primes_simple, fetch_zeros, is_prime_power
from .advanced import PrimeStructures, TwinPrime, GoldbachPair


# ============================================================
# Setup code injected into RLM REPL namespace
# ============================================================

SONAR_SETUP_CODE = '''
"""Prime Sonar tools available in this environment."""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Core sonar instance (initialized lazily)
_sonar_cache = {}
_structures_cache = {}

def _get_sonar(num_zeros: int = 2000):
    """Get or create a PrimeSonar with specified zeros."""
    if num_zeros not in _sonar_cache:
        from d74169 import PrimeSonar
        _sonar_cache[num_zeros] = PrimeSonar(num_zeros=num_zeros, silent=True)
    return _sonar_cache[num_zeros]

def _get_structures(num_zeros: int = 2000):
    """Get or create a PrimeStructures analyzer."""
    if num_zeros not in _structures_cache:
        from d74169 import PrimeStructures
        _structures_cache[num_zeros] = PrimeStructures(num_zeros=num_zeros, silent=True)
    return _structures_cache[num_zeros]


# =============================================================================
# AVAILABLE TOOLS - Use these to analyze primes
# =============================================================================

def scan_primes(start: int, end: int, num_zeros: int = None) -> Dict:
    """
    Scan a range for primes using spectral detection.

    Parameters:
        start: Start of range (inclusive)
        end: End of range (inclusive)
        num_zeros: Number of Riemann zeros to use (auto-calculated if None)

    Returns:
        Dict with keys:
        - primes: List of detected primes
        - count: Number of primes found
        - density: Prime density (count / range_size)
        - range: (start, end) tuple
    """
    if num_zeros is None:
        # Heuristic: ~4 zeros per expected prime
        range_size = end - start + 1
        expected_primes = range_size / np.log(max(end, 2))
        num_zeros = max(500, int(expected_primes * 4))

    sonar = _get_sonar(num_zeros)
    all_primes = sonar.detect_primes(end)
    primes_in_range = [p for p in all_primes if start <= p <= end]

    return {
        'primes': primes_in_range,
        'count': len(primes_in_range),
        'density': len(primes_in_range) / (end - start + 1) if end > start else 0,
        'range': (start, end),
        'num_zeros_used': num_zeros
    }


def find_twin_primes(start: int, end: int) -> Dict:
    """
    Find twin prime pairs (p, p+2) in a range.

    Returns:
        Dict with keys:
        - twins: List of (p, p+2) tuples
        - count: Number of twin pairs
        - density: Twin density
        - largest_gap: Largest gap between consecutive twin pairs
    """
    structures = _get_structures(max(2000, (end - start) * 2))
    all_twins = structures.detect_twin_primes(end)
    twins_in_range = [(t.p, t.q) for t in all_twins if start <= t.p <= end]

    # Calculate gaps between consecutive twins
    gaps = []
    for i in range(1, len(twins_in_range)):
        gap = twins_in_range[i][0] - twins_in_range[i-1][0]
        gaps.append(gap)

    return {
        'twins': twins_in_range,
        'count': len(twins_in_range),
        'density': len(twins_in_range) / (end - start + 1) if end > start else 0,
        'largest_gap': max(gaps) if gaps else 0,
        'avg_gap': np.mean(gaps) if gaps else 0,
        'range': (start, end)
    }


def find_prime_gaps(start: int, end: int) -> Dict:
    """
    Analyze gaps between consecutive primes in a range.

    Returns:
        Dict with keys:
        - gaps: List of (p, next_p, gap_size) tuples
        - max_gap: Largest gap found
        - avg_gap: Average gap size
        - gap_distribution: Dict mapping gap_size -> count
    """
    structures = _get_structures(max(2000, (end - start) * 2))
    all_gaps = structures.detect_prime_gaps(end)
    gaps_in_range = [(p, np, g) for p, np, g in all_gaps if start <= p <= end]

    # Gap distribution
    distribution = {}
    for _, _, g in gaps_in_range:
        distribution[g] = distribution.get(g, 0) + 1

    gap_sizes = [g for _, _, g in gaps_in_range]

    return {
        'gaps': gaps_in_range,
        'count': len(gaps_in_range),
        'max_gap': max(gap_sizes) if gap_sizes else 0,
        'min_gap': min(gap_sizes) if gap_sizes else 0,
        'avg_gap': np.mean(gap_sizes) if gap_sizes else 0,
        'std_gap': np.std(gap_sizes) if gap_sizes else 0,
        'gap_distribution': dict(sorted(distribution.items())),
        'range': (start, end)
    }


def find_sophie_germain(start: int, end: int) -> Dict:
    """
    Find Sophie Germain primes p where both p and 2p+1 are prime.

    Returns:
        Dict with keys:
        - germain: List of (p, 2p+1) tuples
        - count: Number found
    """
    structures = _get_structures(max(2000, end * 2))
    all_germain = structures.sophie_germain_primes(end)
    germain_in_range = [(p, q) for p, q in all_germain if start <= p <= end]

    return {
        'germain': germain_in_range,
        'count': len(germain_in_range),
        'range': (start, end)
    }


def find_prime_triplets(start: int, end: int) -> Dict:
    """
    Find prime triplets: (p, p+2, p+6) or (p, p+4, p+6).

    Returns:
        Dict with triplets and count.
    """
    structures = _get_structures(max(2000, (end - start) * 2))
    all_triplets = structures.prime_triplets(end)
    triplets_in_range = [t for t in all_triplets if start <= t[0] <= end]

    return {
        'triplets': triplets_in_range,
        'count': len(triplets_in_range),
        'range': (start, end)
    }


def goldbach_decompose(n: int) -> Dict:
    """
    Find all ways to write even n as sum of two primes.

    Returns:
        Dict with pairs and count.
    """
    if n % 2 != 0 or n < 4:
        return {'error': 'n must be even and >= 4', 'pairs': [], 'count': 0}

    structures = _get_structures(max(2000, n))
    pairs = structures.detect_goldbach_pairs(n)

    return {
        'n': n,
        'pairs': [(g.p, g.q) for g in pairs],
        'count': len(pairs)
    }


def score_integers(start: int, end: int, num_zeros: int = 2000) -> Dict:
    """
    Get raw primality scores for integers in range.
    Higher scores indicate higher likelihood of being prime.

    Useful for finding the "strongest" prime signals.

    Returns:
        Dict with integers and their scores, sorted by score.
    """
    sonar = _get_sonar(num_zeros)
    n_vals, scores = sonar.score_integers(end)

    # Filter to range and pair with scores
    results = [(int(n), float(s)) for n, s in zip(n_vals, scores) if start <= n <= end]
    results.sort(key=lambda x: -x[1])  # Sort by score descending

    return {
        'scores': results[:100],  # Top 100 by score
        'max_score': results[0] if results else None,
        'min_score': results[-1] if results else None,
        'range': (start, end)
    }


def analyze_region(start: int, end: int) -> Dict:
    """
    Comprehensive analysis of a numeric region.

    Combines multiple analyses into one call.
    Good for getting an overview before deciding where to zoom in.

    Returns:
        Dict with primes, twins, gaps, and density metrics.
    """
    primes = scan_primes(start, end)
    twins = find_twin_primes(start, end)
    gaps = find_prime_gaps(start, end)

    return {
        'range': (start, end),
        'range_size': end - start + 1,
        'prime_count': primes['count'],
        'prime_density': primes['density'],
        'twin_count': twins['count'],
        'twin_density': twins['density'],
        'avg_gap': gaps['avg_gap'],
        'max_gap': gaps['max_gap'],
        'primes': primes['primes'][:20],  # First 20 primes
        'twins': twins['twins'][:10],      # First 10 twin pairs
        'summary': f"Range [{start}, {end}]: {primes['count']} primes, {twins['count']} twins, avg gap {gaps['avg_gap']:.1f}"
    }


def compare_regions(*ranges) -> Dict:
    """
    Compare prime statistics across multiple regions.

    Usage: compare_regions((1, 1000), (10000, 11000), (100000, 101000))

    Returns:
        Dict mapping each range to its statistics for comparison.
    """
    results = {}
    for r in ranges:
        start, end = r
        analysis = analyze_region(start, end)
        results[f"{start}-{end}"] = {
            'prime_density': analysis['prime_density'],
            'twin_density': analysis['twin_density'],
            'avg_gap': analysis['avg_gap'],
            'prime_count': analysis['prime_count'],
            'twin_count': analysis['twin_count']
        }

    return results


def chunk_range(start: int, end: int, num_chunks: int = 10) -> List[Tuple[int, int]]:
    """
    Split a range into chunks for parallel/recursive analysis.

    Returns:
        List of (chunk_start, chunk_end) tuples.
    """
    size = end - start + 1
    chunk_size = size // num_chunks

    chunks = []
    for i in range(num_chunks):
        c_start = start + i * chunk_size
        c_end = c_start + chunk_size - 1 if i < num_chunks - 1 else end
        chunks.append((c_start, c_end))

    return chunks


# Print available tools
print("Prime Sonar tools loaded. Available functions:")
print("  scan_primes(start, end)      - Detect primes in range")
print("  find_twin_primes(start, end) - Find twin prime pairs")
print("  find_prime_gaps(start, end)  - Analyze prime gaps")
print("  find_sophie_germain(start, end) - Find Sophie Germain primes")
print("  find_prime_triplets(start, end) - Find prime triplets")
print("  goldbach_decompose(n)        - Goldbach decomposition")
print("  score_integers(start, end)   - Raw primality scores")
print("  analyze_region(start, end)   - Comprehensive analysis")
print("  compare_regions(*ranges)     - Compare multiple regions")
print("  chunk_range(start, end, n)   - Split range into chunks")
'''


# ============================================================
# System prompt for the RLM
# ============================================================

SYSTEM_PROMPT = '''You are a spectral prime analyst with access to the d74169 Prime Sonar system.

## Your Capabilities

You can detect and analyze prime numbers using Riemann zeta zeros via spectral analysis.
This is NOT trial division or sieves - it's inverse scattering from the explicit formula.

## Available Tools (Python functions in your environment)

```python
scan_primes(start, end)           # Detect all primes in [start, end]
find_twin_primes(start, end)      # Find (p, p+2) pairs
find_prime_gaps(start, end)       # Analyze gaps between consecutive primes
find_sophie_germain(start, end)   # Find p where both p and 2p+1 are prime
find_prime_triplets(start, end)   # Find (p, p+2, p+6) or (p, p+4, p+6)
goldbach_decompose(n)             # Write even n as sum of two primes
score_integers(start, end)        # Get raw primality scores
analyze_region(start, end)        # Comprehensive region analysis
compare_regions(*ranges)          # Compare statistics across regions
chunk_range(start, end, n)        # Split range into n chunks
```

## Strategy for Large Ranges

For ranges larger than 100,000:
1. Use chunk_range() to split into manageable pieces
2. Use analyze_region() on each chunk to get overview statistics
3. Identify interesting regions (high twin density, large gaps, etc.)
4. Recursively zoom into those regions with finer analysis
5. Use llm_query() to spawn sub-analyses if needed

## Output Format

Always provide:
1. What you found (specific numbers, patterns)
2. Statistical summary (counts, densities, distributions)
3. Notable anomalies or interesting regions
4. Suggestions for further investigation if relevant

Be precise with numbers. This is mathematics, not approximation.
'''


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class RLMResult:
    """Result from an RLM prime analysis query."""
    response: str
    logs: List[Dict[str, Any]]
    execution_time: float
    chunks_analyzed: int = 0
    depth: int = 0

    def __repr__(self):
        return f"RLMResult(chunks={self.chunks_analyzed}, depth={self.depth}, time={self.execution_time:.2f}s)"


# ============================================================
# Main RLM Agent class
# ============================================================

class PrimeSonarRLM:
    """
    Recursive Language Model agent for prime analysis.

    Combines RLM's recursive decomposition with d74169's spectral
    prime detection for intelligent, scalable prime hunting.

    Parameters
    ----------
    backend : str
        LLM provider: "openai", "anthropic", "openrouter", etc.
    model : str
        Model identifier (e.g., "claude-sonnet-4-20250514", "gpt-4o")
    max_depth : int
        Maximum recursion depth for RLM calls
    environment : str
        REPL environment: "local", "docker", "modal"
    verbose : bool
        Print progress messages
    api_key : str, optional
        API key (defaults to environment variable)

    Examples
    --------
    >>> agent = PrimeSonarRLM(backend="anthropic", model="claude-sonnet-4-20250514")
    >>> result = agent.analyze("Find twin prime clusters between 1 and 50000")
    >>> print(result.response)

    >>> # For very large ranges, increase depth
    >>> result = agent.analyze(
    ...     "Map prime gap distribution from 1 to 1,000,000",
    ...     max_depth=3
    ... )
    """

    def __init__(
        self,
        backend: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        max_depth: int = 2,
        environment: str = "local",
        verbose: bool = True,
        api_key: Optional[str] = None,
        **backend_kwargs
    ):
        self.backend = backend
        self.model = model
        self.max_depth = max_depth
        self.environment = environment
        self.verbose = verbose
        self.api_key = api_key
        self.backend_kwargs = backend_kwargs

        self._rlm = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of RLM client."""
        if self._initialized:
            return

        try:
            from rlm import RLM
        except ImportError:
            raise ImportError(
                "RLM library not installed. Install with:\n"
                "  pip install rlm\n"
                "Or from source:\n"
                "  git clone https://github.com/alexzhang13/rlm && cd rlm && pip install -e ."
            )

        # Build backend kwargs
        kwargs = {"model_name": self.model, **self.backend_kwargs}
        if self.api_key:
            kwargs["api_key"] = self.api_key

        self._rlm = RLM(
            backend=self.backend,
            backend_kwargs=kwargs,
            max_depth=self.max_depth,
            environment=self.environment,
            setup_code=SONAR_SETUP_CODE,
            verbose=self.verbose
        )
        self._initialized = True

    def analyze(
        self,
        query: str,
        max_depth: Optional[int] = None,
        context: Optional[str] = None
    ) -> RLMResult:
        """
        Analyze primes using recursive LLM decomposition.

        Parameters
        ----------
        query : str
            Natural language query about primes
            Examples:
            - "Find all twin primes between 1000 and 2000"
            - "What's the largest prime gap under 100,000?"
            - "Analyze prime density across the first million integers"
        max_depth : int, optional
            Override default max recursion depth
        context : str, optional
            Additional context to prepend to query

        Returns
        -------
        RLMResult
            Contains response text, execution logs, timing info
        """
        self._ensure_initialized()

        # Build full prompt
        full_prompt = SYSTEM_PROMPT + "\n\n"
        if context:
            full_prompt += f"Context: {context}\n\n"
        full_prompt += f"Query: {query}"

        # Override depth if specified
        if max_depth is not None:
            self._rlm.max_depth = max_depth

        import time
        start_time = time.time()

        # Execute RLM completion
        result = self._rlm.completion(full_prompt)

        elapsed = time.time() - start_time

        return RLMResult(
            response=result.response,
            logs=getattr(result, 'logs', []),
            execution_time=elapsed,
            chunks_analyzed=getattr(result, 'chunks_analyzed', 0),
            depth=getattr(result, 'depth', 0)
        )

    def scan(
        self,
        start: int,
        end: int,
        target: str = "primes"
    ) -> RLMResult:
        """
        Convenience method for common scans.

        Parameters
        ----------
        start : int
            Range start
        end : int
            Range end
        target : str
            What to find: "primes", "twins", "gaps", "germain", "triplets"

        Returns
        -------
        RLMResult
        """
        queries = {
            "primes": f"Find and list all primes between {start} and {end}",
            "twins": f"Find all twin prime pairs between {start} and {end}",
            "gaps": f"Analyze prime gaps between {start} and {end}, highlighting any unusually large gaps",
            "germain": f"Find all Sophie Germain primes between {start} and {end}",
            "triplets": f"Find all prime triplets between {start} and {end}"
        }

        query = queries.get(target, f"Analyze {target} between {start} and {end}")
        return self.analyze(query)

    def hunt(
        self,
        pattern: str,
        max_n: int = 100000,
        strategy: str = "adaptive"
    ) -> RLMResult:
        """
        Hunt for specific prime patterns with intelligent search.

        Parameters
        ----------
        pattern : str
            Pattern to find:
            - "twin_clusters": Regions with high twin prime density
            - "gap_records": Record-breaking prime gaps
            - "goldbach_hard": Even numbers with few Goldbach representations
            - "sophie_chains": Chains of Sophie Germain primes
        max_n : int
            Upper search bound
        strategy : str
            Search strategy: "adaptive", "exhaustive", "sampling"

        Returns
        -------
        RLMResult
        """
        pattern_prompts = {
            "twin_clusters": f"""
                Search for twin prime clusters up to {max_n}.
                1. Divide the range into chunks
                2. Find twin density in each chunk
                3. Identify chunks with above-average density
                4. Zoom into high-density regions
                5. Report the densest clusters found
            """,
            "gap_records": f"""
                Find record-breaking prime gaps up to {max_n}.
                1. Scan for all prime gaps
                2. Track the running maximum gap
                3. Report each new record: (prime_before, prime_after, gap_size)
                4. Identify any patterns in where records occur
            """,
            "goldbach_hard": f"""
                Find even numbers up to {max_n} that have the fewest Goldbach representations.
                These are the "hardest" Goldbach numbers.
                Sample even numbers and count their (p, q) pairs where p + q = n.
            """,
            "sophie_chains": f"""
                Search for chains of Sophie Germain primes up to {max_n}.
                A chain is p1, p2=2*p1+1, p3=2*p2+1, ... where each is prime.
                Find the longest such chain.
            """
        }

        prompt = pattern_prompts.get(
            pattern,
            f"Hunt for '{pattern}' pattern in primes up to {max_n} using {strategy} strategy."
        )

        return self.analyze(prompt)


# ============================================================
# Standalone functions for non-RLM usage
# ============================================================

def create_repl_tools() -> Dict[str, Any]:
    """
    Create the tool functions for manual REPL injection.

    Returns a dict of functions that can be added to a REPL namespace.
    Useful if you want to use the tools without the full RLM setup.

    Usage:
        tools = create_repl_tools()
        result = tools['scan_primes'](1, 1000)
    """
    # Execute setup code to create functions
    namespace = {}
    exec(SONAR_SETUP_CODE, namespace)

    # Return just the tool functions
    tool_names = [
        'scan_primes', 'find_twin_primes', 'find_prime_gaps',
        'find_sophie_germain', 'find_prime_triplets', 'goldbach_decompose',
        'score_integers', 'analyze_region', 'compare_regions', 'chunk_range'
    ]

    return {name: namespace[name] for name in tool_names if name in namespace}


def get_setup_code() -> str:
    """Return the setup code for custom RLM configurations."""
    return SONAR_SETUP_CODE


def get_system_prompt() -> str:
    """Return the system prompt for custom RLM configurations."""
    return SYSTEM_PROMPT


# ============================================================
# Module exports
# ============================================================

__all__ = [
    'PrimeSonarRLM',
    'RLMResult',
    'create_repl_tools',
    'get_setup_code',
    'get_system_prompt',
    'SONAR_SETUP_CODE',
    'SYSTEM_PROMPT'
]


if __name__ == "__main__":
    # Demo without RLM (just the tools)
    print("d74169 RLM Agent - Tool Demo")
    print("=" * 50)

    tools = create_repl_tools()

    print("\n1. Scanning primes in [1, 100]:")
    result = tools['scan_primes'](1, 100)
    print(f"   Found {result['count']} primes")
    print(f"   First 10: {result['primes'][:10]}")

    print("\n2. Twin primes in [1, 100]:")
    twins = tools['find_twin_primes'](1, 100)
    print(f"   Found {twins['count']} twin pairs")
    print(f"   Pairs: {twins['twins']}")

    print("\n3. Comparing regions:")
    comparison = tools['compare_regions']((1, 1000), (10000, 11000))
    for region, stats in comparison.items():
        print(f"   [{region}]: density={stats['prime_density']:.4f}, twins={stats['twin_count']}")

    print("\n" + "=" * 50)
    print("To use with RLM:")
    print("  from d74169.rlm_agent import PrimeSonarRLM")
    print("  agent = PrimeSonarRLM(backend='anthropic', model='claude-sonnet-4-20250514')")
    print("  result = agent.analyze('Find twin prime clusters up to 50000')")
