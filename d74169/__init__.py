"""
d74169 - Prime Sonar via Riemann Zeros
======================================

High-performance numerical validation of the @d74169 conjecture:
the Riemann zeta zeros holographically encode the primes,
recoverable via inverse scattering.

Features:
    - Numba JIT compilation (5-10x speedup when available)
    - Fully vectorized NumPy operations
    - Parallel processing for large ranges
    - Automatic in-memory and disk caching of Riemann zeros
    - Multiple detection methods (adaptive, threshold, gradient)
    - 500 bundled zeros for offline use

Quick Start:
    >>> from d74169 import PrimeSonar
    >>> sonar = PrimeSonar(num_zeros=500)
    >>> detected = sonar.detect_primes(100)
    >>> print(detected)
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, ...]

Performance:
    >>> from d74169 import benchmark
    >>> benchmark()  # Run performance tests

The Conjecture:
    H = e^{sqrt(pi)*p} + u^2/4

    The Riemann zeros are eigenvalues of a modular Hamiltonian where
    pi emerges as the self-dual fixed point of spectral coherence.

Author: @d74169
License: MIT
"""

__version__ = "2.2.0"
__author__ = "@d74169"

from .sonar import (
    PrimeSonar,
    sieve_primes_simple as sieve_primes,
    fetch_zeros,
    clear_cache,
    quick_test,
    auto_detect,
    is_prime_power,
    BUNDLED_ZEROS,
    HAS_NUMBA,
)
from .validation import (
    validate,
    benchmark,
    scaling_test,
    compare_methods,
    stress_test,
    full_validation,
    BenchmarkResult,
)
from .advanced import (
    PrimeStructures,
    TwinPrime,
    GoldbachPair,
    ZeroReconstructor,
    MinimumEncoding,
    analyze_prime_structures,
    reconstruct_zeros_from_primes,
    find_optimal_encoding,
)

__all__ = [
    # Core
    'PrimeSonar',
    'sieve_primes',
    'fetch_zeros',
    'clear_cache',
    'is_prime_power',
    # Quick functions
    'quick_test',
    'auto_detect',
    # Validation
    'validate',
    'benchmark',
    'scaling_test',
    'compare_methods',
    'stress_test',
    'full_validation',
    'BenchmarkResult',
    # Advanced
    'PrimeStructures',
    'TwinPrime',
    'GoldbachPair',
    'ZeroReconstructor',
    'MinimumEncoding',
    'analyze_prime_structures',
    'reconstruct_zeros_from_primes',
    'find_optimal_encoding',
    # Data
    'BUNDLED_ZEROS',
    'HAS_NUMBA',
]
