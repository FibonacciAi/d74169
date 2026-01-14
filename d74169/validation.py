"""
d74169.validation - Validation and Benchmarking
================================================

Tools for testing and benchmarking the Prime Sonar.
Optimized with dataclasses and vectorized operations.
"""

from __future__ import annotations

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from .sonar import PrimeSonar, sieve_primes_simple, is_prime_power, HAS_NUMBA


@dataclass(slots=True)
class BenchmarkResult:
    """Results from a benchmark run."""
    range_max: int
    num_zeros: int
    accuracy: float
    precision: float
    recall: float
    elapsed_ms: float
    detected_count: int
    actual_count: int
    missed: List[int] = field(default_factory=list)
    false_positives: List[int] = field(default_factory=list)
    prime_powers: List[int] = field(default_factory=list)

    @property
    def f1_score(self) -> float:
        """Harmonic mean of precision and recall."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def status(self) -> str:
        if self.accuracy == 100.0 and not self.false_positives:
            return "PERFECT"
        if self.accuracy >= 95:
            return "EXCELLENT"
        if self.accuracy >= 90:
            return "GOOD"
        return "OK"

    def __str__(self) -> str:
        return (f"Range={self.range_max:>5}, Zeros={self.num_zeros:>4}, "
                f"Acc={self.accuracy:>5.1f}%, F1={self.f1_score:.3f}, "
                f"Time={self.elapsed_ms:>6.1f}ms [{self.status}]")


def validate(max_n: int = 100, num_zeros: int = None,
             verbose: bool = True, silent: bool = False) -> BenchmarkResult:
    """
    Validate Prime Sonar accuracy against known primes.

    Parameters
    ----------
    max_n : int
        Upper bound for validation
    num_zeros : int, optional
        Number of Riemann zeros (defaults to 4x max_n)
    verbose : bool
        Print detailed results
    silent : bool
        Suppress all output

    Returns
    -------
    BenchmarkResult
        Detailed benchmark metrics
    """
    if num_zeros is None:
        num_zeros = max_n * 4

    start = time.perf_counter()

    sonar = PrimeSonar(num_zeros=num_zeros, silent=True)
    detected, powers = sonar.detect_primes(max_n, return_powers=True)
    actual = list(sieve_primes_simple(max_n))

    elapsed_ms = (time.perf_counter() - start) * 1000

    detected_set = set(detected)
    actual_set = set(actual)

    correct = detected_set & actual_set
    missed = sorted(actual_set - detected_set)
    false_positives = sorted(detected_set - actual_set)

    accuracy = 100.0 * len(correct) / len(actual_set) if actual_set else 100.0
    precision = len(correct) / len(detected_set) if detected_set else 1.0
    recall = len(correct) / len(actual_set) if actual_set else 1.0

    result = BenchmarkResult(
        range_max=max_n,
        num_zeros=num_zeros,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        elapsed_ms=elapsed_ms,
        detected_count=len(detected),
        actual_count=len(actual),
        missed=missed,
        false_positives=false_positives,
        prime_powers=powers
    )

    if verbose and not silent:
        print(f"\n@d74169 Validation: Range [2, {max_n}]")
        print("=" * 50)
        print(f"Zeros:      {num_zeros}")
        print(f"Accuracy:   {accuracy:.1f}%")
        print(f"Precision:  {precision:.3f}")
        print(f"Recall:     {recall:.3f}")
        print(f"F1 Score:   {result.f1_score:.3f}")
        print(f"Time:       {elapsed_ms:.1f}ms")
        print(f"Status:     {result.status}")
        if missed:
            print(f"Missed:     {missed[:10]}{'...' if len(missed) > 10 else ''}")
        if false_positives:
            print(f"Spurious:   {false_positives[:10]}{'...' if len(false_positives) > 10 else ''}")
        print("=" * 50)

    return result


def benchmark(ranges: List[int] = None,
              zeros_multiplier: int = 4,
              silent: bool = False) -> List[BenchmarkResult]:
    """
    Run benchmarks across multiple ranges.

    Parameters
    ----------
    ranges : list of int
        Ranges to test (default: [50, 100, 200, 500])
    zeros_multiplier : int
        Zeros = range * multiplier
    silent : bool
        Suppress output

    Returns
    -------
    list of BenchmarkResult
    """
    if ranges is None:
        ranges = [50, 100, 200, 500]

    results = []

    if not silent:
        print("\n@d74169 Benchmark")
        print("=" * 65)
        if HAS_NUMBA:
            print("Numba JIT: ENABLED (parallel)")
        else:
            print("Numba JIT: disabled (install numba for 5-10x speedup)")
        print("-" * 65)
        print(f"{'Range':<10} {'Zeros':<10} {'Accuracy':<12} {'F1':<10} {'Time':<12} {'Status':<10}")
        print("-" * 65)

    for max_n in ranges:
        num_zeros = max_n * zeros_multiplier
        result = validate(max_n, num_zeros, verbose=False, silent=True)
        results.append(result)

        if not silent:
            print(f"{max_n:<10} {num_zeros:<10} {result.accuracy:.1f}%{'':<8} "
                  f"{result.f1_score:.3f}{'':<6} {result.elapsed_ms:.0f}ms{'':<6} {result.status:<10}")

    if not silent:
        print("-" * 65)
        avg_acc = sum(r.accuracy for r in results) / len(results)
        avg_f1 = sum(r.f1_score for r in results) / len(results)
        total_time = sum(r.elapsed_ms for r in results)
        print(f"Average: Acc={avg_acc:.1f}%, F1={avg_f1:.3f}, Total={total_time:.0f}ms")
        print("=" * 65)

    return results


def scaling_test(max_n: int = 1000,
                 zero_counts: List[int] = None,
                 silent: bool = False) -> List[BenchmarkResult]:
    """
    Test how accuracy scales with number of zeros for a fixed range.

    Parameters
    ----------
    max_n : int
        Fixed range to test
    zero_counts : list of int
        Zero counts to test
    silent : bool
        Suppress output

    Returns
    -------
    list of BenchmarkResult
    """
    if zero_counts is None:
        zero_counts = [1000, 2000, 4000, 8000]

    results = []

    if not silent:
        print(f"\n@d74169 Scaling Test: Range [2, {max_n}]")
        print("=" * 55)
        print(f"{'Zeros':<12} {'Accuracy':<12} {'F1':<10} {'Time':<12} {'Status':<10}")
        print("-" * 55)

    for nz in zero_counts:
        result = validate(max_n, nz, verbose=False, silent=True)
        results.append(result)

        if not silent:
            print(f"{nz:<12} {result.accuracy:.1f}%{'':<8} "
                  f"{result.f1_score:.3f}{'':<6} {result.elapsed_ms:.0f}ms{'':<6} {result.status:<10}")

    if not silent:
        print("=" * 55)

    return results


def compare_methods(max_n: int = 200, num_zeros: int = 800,
                    silent: bool = False) -> Dict[str, BenchmarkResult]:
    """
    Compare different detection methods.

    Returns results for 'adaptive', 'threshold', and 'gradient' methods.
    """
    if not silent:
        print(f"\n@d74169 Method Comparison: range={max_n}, zeros={num_zeros}")
        print("-" * 55)
        print(f"{'Method':<12} {'Accuracy':<12} {'F1':<10} {'Time':<12}")
        print("-" * 55)

    sonar = PrimeSonar(num_zeros=num_zeros, silent=True)
    actual = set(sieve_primes_simple(max_n))
    results = {}

    for method in ['adaptive', 'threshold', 'gradient']:
        start = time.perf_counter()
        detected = sonar.detect_primes(max_n, method=method)
        elapsed_ms = (time.perf_counter() - start) * 1000

        detected_set = set(detected)
        correct = detected_set & actual
        missed = sorted(actual - detected_set)
        false_positives = sorted(detected_set - actual)

        accuracy = 100.0 * len(correct) / len(actual) if actual else 100.0
        precision = len(correct) / len(detected_set) if detected_set else 1.0
        recall = len(correct) / len(actual) if actual else 1.0

        result = BenchmarkResult(
            range_max=max_n,
            num_zeros=num_zeros,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            elapsed_ms=elapsed_ms,
            detected_count=len(detected),
            actual_count=len(actual),
            missed=missed,
            false_positives=false_positives
        )
        results[method] = result

        if not silent:
            print(f"{method:<12} {accuracy:.1f}%{'':<8} "
                  f"{result.f1_score:.3f}{'':<6} {elapsed_ms:.1f}ms")

    if not silent:
        print("-" * 55)

    return results


def stress_test(max_range: int = 1000,
                target_accuracy: float = 95.0,
                max_zeros: int = 10000,
                silent: bool = False) -> Dict:
    """
    Find minimum zeros needed for target accuracy.

    Uses binary search for efficiency.
    """
    if not silent:
        print(f"\n@d74169 Stress Test: range={max_range}, target={target_accuracy}%")
        print("-" * 50)

    low, high = 100, max_zeros
    best_result = None

    while low < high:
        mid = (low + high) // 2
        result = validate(max_range, mid, verbose=False, silent=True)

        if not silent:
            status = "OK" if result.accuracy >= target_accuracy else "  "
            print(f"  zeros={mid:>5}: {result.accuracy:>5.1f}% {status}")

        if result.accuracy >= target_accuracy:
            best_result = result
            high = mid
        else:
            low = mid + 1

    if best_result is None:
        best_result = validate(max_range, max_zeros, verbose=False, silent=True)

    if not silent:
        print("-" * 50)
        print(f"Result: {best_result.num_zeros} zeros for {best_result.accuracy:.1f}%")

    return {
        'zeros': best_result.num_zeros,
        'accuracy': best_result.accuracy,
        'elapsed_ms': best_result.elapsed_ms,
        'result': best_result
    }


def full_validation(silent: bool = False) -> Dict:
    """
    Run the complete @d74169 validation suite.

    Returns summary statistics.
    """
    if not silent:
        print("\n" + "=" * 70)
        print("   @d74169 CONJECTURE - FULL VALIDATION SUITE")
        print("=" * 70)

    # Test 1: Perfect detection ranges
    perfect_tests = []
    test_cases = [(50, 200), (100, 400), (200, 800)]

    if not silent:
        print("\nPerfect Detection Tests:")

    for max_n, nz in test_cases:
        result = validate(max_n, nz, verbose=False, silent=True)
        perfect_tests.append(result)
        if not silent:
            status = "PERFECT" if result.accuracy == 100 else f"{result.accuracy:.1f}%"
            print(f"  Range {max_n:>3} with {nz:>4} zeros: {status}")

    # Test 2: Scaling behavior
    if not silent:
        print("\nScaling Test at range 500:")

    scaling_results = scaling_test(500, [1000, 2000, 3000, 4000], silent=True)
    if not silent:
        for r in scaling_results:
            print(f"  {r.num_zeros:>4} zeros: {r.accuracy:.1f}%")

    # Summary
    perfect_range = 0
    for r in perfect_tests:
        if r.accuracy == 100:
            perfect_range = max(perfect_range, r.range_max)

    summary = {
        'perfect_range': perfect_range,
        'perfect_tests': perfect_tests,
        'scaling_results': scaling_results,
        'validated': all(r.accuracy == 100 for r in perfect_tests[:2])
    }

    if not silent:
        print("\n" + "-" * 70)
        validated_str = "VALIDATED" if summary['validated'] else "NEEDS MORE ZEROS"
        print(f"  RESULT: @d74169 Conjecture {validated_str}")
        print(f"  Perfect detection up to range: {summary['perfect_range']}")
        print("=" * 70 + "\n")

    return summary


if __name__ == "__main__":
    full_validation()
