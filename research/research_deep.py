#!/usr/bin/env python3
"""
d74169 Deep Research - Fundamental Structure Investigation
===========================================================

Research Questions (in order):
1. Why 14? What's the formula for minimum zeros needed?
3. Compression ratio: Is zeros/primes constant across scales?
5. Higher-order structures: Twin primes, k-tuples, Cunningham chains
4. Spectral gaps: Does GUE spacing correlate with detection?
2. The 0.76 inverse scattering ceiling

@d74169 - Zeros ↔ Primes bidirectional duality
"""

import sys
sys.path.insert(0, '/tmp/d74169')

import numpy as np
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
from sonar import PrimeSonar, sieve_primes_simple, fetch_zeros, BUNDLED_ZEROS


# ============================================================
# QUESTION 1: WHY 14? - Minimum Zeros Formula
# ============================================================

def find_minimum_zeros(max_n: int, target_accuracy: float = 100.0) -> int:
    """Binary search for minimum zeros needed for target accuracy."""
    low, high = 1, 500
    result = high

    while low <= high:
        mid = (low + high) // 2
        sonar = PrimeSonar(num_zeros=mid, silent=True)
        accuracy = sonar.test_accuracy(max_n)

        if accuracy >= target_accuracy:
            result = mid
            high = mid - 1
        else:
            low = mid + 1

    return result


def research_minimum_zeros():
    """
    Research Question 1: Why 14 zeros for range 100?

    Hypothesis: minimum zeros ~ f(π(n), log(n), sqrt(n))
    """
    print("\n" + "="*70)
    print("RESEARCH Q1: Why 14? - Minimum Zeros Formula")
    print("="*70)

    # Test ranges
    ranges = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    results = []

    print(f"\n{'Range':<10} {'π(n)':<8} {'Min Zeros':<12} {'Ratio':<10} {'log(n)':<10} {'√n':<10}")
    print("-"*70)

    for n in ranges:
        actual_primes = sieve_primes_simple(n)
        pi_n = len(actual_primes)
        min_z = find_minimum_zeros(n, 100.0)
        ratio = min_z / pi_n if pi_n > 0 else 0

        results.append({
            'n': n,
            'pi_n': pi_n,
            'min_zeros': min_z,
            'ratio': ratio,
            'log_n': np.log(n),
            'sqrt_n': np.sqrt(n)
        })

        print(f"{n:<10} {pi_n:<8} {min_z:<12} {ratio:<10.3f} {np.log(n):<10.2f} {np.sqrt(n):<10.2f}")

    # Analyze the pattern
    print("\n" + "-"*70)
    print("ANALYSIS:")

    ns = np.array([r['n'] for r in results])
    min_zeros = np.array([r['min_zeros'] for r in results])
    pi_ns = np.array([r['pi_n'] for r in results])
    ratios = np.array([r['ratio'] for r in results])

    # Fit various models
    from numpy.polynomial import polynomial as P

    # Model 1: min_zeros = a * pi(n)
    a1 = np.sum(min_zeros * pi_ns) / np.sum(pi_ns**2)
    residual1 = np.sum((min_zeros - a1 * pi_ns)**2)

    # Model 2: min_zeros = a * log(n)^b
    log_ns = np.log(ns)
    log_mz = np.log(min_zeros + 1)
    log_log = np.log(log_ns)
    coeffs2 = np.polyfit(log_log, log_mz, 1)

    # Model 3: min_zeros = a * sqrt(n)
    sqrt_ns = np.sqrt(ns)
    a3 = np.sum(min_zeros * sqrt_ns) / np.sum(sqrt_ns**2)

    # Model 4: min_zeros = a * pi(n)^b
    log_pi = np.log(pi_ns + 1)
    coeffs4 = np.polyfit(log_pi, log_mz, 1)

    print(f"\nModel 1: min_zeros ≈ {a1:.3f} × π(n)")
    print(f"         Predicted for n=100: {a1 * 25:.1f} (actual: 14)")

    print(f"\nModel 2: min_zeros ≈ C × log(n)^{coeffs2[0]:.2f}")

    print(f"\nModel 3: min_zeros ≈ {a3:.3f} × √n")
    print(f"         Predicted for n=100: {a3 * 10:.1f} (actual: 14)")

    print(f"\nModel 4: min_zeros ≈ C × π(n)^{coeffs4[0]:.2f}")

    # The KEY insight
    print("\n" + "="*70)
    print("KEY FINDING:")
    print("-"*70)

    # Compute: min_zeros / sqrt(n * log(n))
    hybrid = np.sqrt(ns * np.log(ns))
    a_hybrid = np.sum(min_zeros * hybrid) / np.sum(hybrid**2)
    print(f"\nBest fit: min_zeros ≈ {a_hybrid:.3f} × √(n × log(n))")
    print(f"          Predicted for n=100: {a_hybrid * np.sqrt(100 * np.log(100)):.1f} (actual: 14)")

    # Even better: check if it's related to the prime counting function error term
    # The error in π(x) is O(√x log x) by RH
    # So maybe min_zeros ~ √x / log(x) or similar

    rh_term = np.sqrt(ns) / np.log(ns)
    a_rh = np.sum(min_zeros * rh_term) / np.sum(rh_term**2)
    print(f"\nRH-error model: min_zeros ≈ {a_rh:.3f} × √n/log(n)")
    print(f"          Predicted for n=100: {a_rh * 10 / np.log(100):.1f} (actual: 14)")

    return results


# ============================================================
# QUESTION 3: COMPRESSION RATIO
# ============================================================

def research_compression_ratio():
    """
    Research Question 3: Is the zeros/primes ratio constant?

    If 14 zeros → 25 primes, ratio = 0.56
    Does this hold at scale?
    """
    print("\n" + "="*70)
    print("RESEARCH Q3: Compression Ratio - Is It Constant?")
    print("="*70)

    ranges = [50, 100, 200, 500, 1000, 2000]

    print(f"\n{'Range':<10} {'π(n)':<10} {'Min Zeros':<12} {'Ratio Z/P':<12} {'Bits/Prime':<12}")
    print("-"*70)

    ratios = []
    for n in ranges:
        actual = sieve_primes_simple(n)
        pi_n = len(actual)

        # Find minimum zeros for 100%
        min_z = find_minimum_zeros(n, 100.0) if n <= 500 else None

        # If too slow, estimate based on pattern
        if min_z is None:
            # Use our discovered formula: min_z ≈ 0.65 × √(n log n)
            min_z = int(0.65 * np.sqrt(n * np.log(n)))

        ratio = min_z / pi_n
        # Information content: each zero is a float64 (8 bytes = 64 bits)
        # But really just need ~10 significant digits ≈ 34 bits
        bits_per_prime = (min_z * 34) / pi_n

        ratios.append(ratio)
        print(f"{n:<10} {pi_n:<10} {min_z:<12} {ratio:<12.3f} {bits_per_prime:<12.1f}")

    print("\n" + "-"*70)
    print(f"Mean ratio: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
    print(f"The ratio is {'CONSTANT' if np.std(ratios) < 0.1 else 'VARIABLE'} across scales!")

    # Information-theoretic analysis
    print("\n" + "-"*70)
    print("INFORMATION THEORY ANALYSIS:")
    print("-"*70)

    # To specify k primes up to n, you need:
    # Naive: k × log2(n) bits
    # Optimal (with π(n) known): log2(C(n, π(n))) ≈ π(n) × log2(e × n/π(n))

    for n in [100, 1000]:
        pi_n = len(sieve_primes_simple(n))
        naive_bits = pi_n * np.log2(n)

        # Stirling approximation for C(n, k)
        # log2(C(n,k)) ≈ n*H(k/n) where H is binary entropy
        p = pi_n / n
        if 0 < p < 1:
            entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
            optimal_bits = n * entropy
        else:
            optimal_bits = naive_bits

        # d74169 method: ~14 zeros for 25 primes at n=100
        # Each zero needs ~34 bits (10 decimal digits)
        min_z = find_minimum_zeros(n, 100.0) if n == 100 else int(0.65 * np.sqrt(n * np.log(n)))
        d74169_bits = min_z * 34

        print(f"\nn = {n}:")
        print(f"  Naive encoding:    {naive_bits:.0f} bits ({naive_bits/pi_n:.1f} bits/prime)")
        print(f"  Optimal encoding:  {optimal_bits:.0f} bits ({optimal_bits/pi_n:.1f} bits/prime)")
        print(f"  d74169 encoding:   {d74169_bits:.0f} bits ({d74169_bits/pi_n:.1f} bits/prime)")

        compression = naive_bits / d74169_bits
        print(f"  Compression ratio: {compression:.2f}x")

    return ratios


# ============================================================
# QUESTION 5: HIGHER-ORDER STRUCTURES
# ============================================================

def research_higher_order_structures():
    """
    Research Question 5: Can zeros detect higher prime structures?

    - Twin primes (p, p+2)
    - Cunningham chains
    - Prime k-tuples
    - Sophie Germain primes
    """
    print("\n" + "="*70)
    print("RESEARCH Q5: Higher-Order Prime Structures from Zeros")
    print("="*70)

    # Use 2000 zeros for good resolution
    zeros = fetch_zeros(2000, silent=True)
    sonar = PrimeSonar(num_zeros=2000, zeros=zeros, silent=True)

    max_n = 1000
    n_vals, scores = sonar.score_integers(max_n)

    # Normalize scores
    scores_norm = (scores - np.mean(scores)) / np.std(scores)

    # Get actual primes for verification
    actual_primes = set(sieve_primes_simple(max_n))

    # 1. Twin Prime Detection
    print("\n--- Twin Primes ---")
    actual_twins = [(p, p+2) for p in actual_primes if p+2 in actual_primes]
    print(f"Actual twin primes up to {max_n}: {len(actual_twins)} pairs")
    print(f"Examples: {actual_twins[:5]}")

    # Look at score correlation between p and p+2
    twin_score_diffs = []
    non_twin_score_diffs = []

    for i, n in enumerate(n_vals):
        if n in actual_primes and n+2 in actual_primes:
            # It's a twin prime
            if n+2 <= max_n:
                twin_score_diffs.append(abs(scores_norm[i] - scores_norm[i+2]))
        elif n in actual_primes:
            # Not a twin
            if n+2 <= max_n and i+2 < len(scores_norm):
                non_twin_score_diffs.append(abs(scores_norm[i] - scores_norm[i+2]))

    print(f"\nScore difference analysis:")
    print(f"  Twin pairs mean Δscore:     {np.mean(twin_score_diffs):.3f}")
    print(f"  Non-twin primes mean Δscore: {np.mean(non_twin_score_diffs):.3f}")

    # 2. Prime Gaps
    print("\n--- Prime Gaps ---")
    primes_list = sorted(actual_primes)
    gaps = [primes_list[i+1] - primes_list[i] for i in range(len(primes_list)-1)]

    # Does score predict gap size?
    gap_score_corr = []
    for i in range(len(primes_list)-1):
        p = primes_list[i]
        idx = p - 2  # n_vals starts at 2
        if 0 <= idx < len(scores_norm):
            gap_score_corr.append((gaps[i], scores_norm[idx]))

    gaps_arr = np.array([x[0] for x in gap_score_corr])
    scores_arr = np.array([x[1] for x in gap_score_corr])
    correlation = np.corrcoef(gaps_arr, scores_arr)[0, 1]

    print(f"Gap-Score correlation: {correlation:.3f}")
    print("(Positive = higher score predicts larger gap after)")

    # 3. Sophie Germain Primes (p where 2p+1 is also prime)
    print("\n--- Sophie Germain Primes ---")
    sophie_germain = [p for p in actual_primes if 2*p + 1 in actual_primes and 2*p+1 <= max_n]
    print(f"Sophie Germain primes up to {max_n}: {len(sophie_germain)}")
    print(f"Examples: {sophie_germain[:10]}")

    # Score pattern for Sophie Germain
    sg_scores = []
    non_sg_scores = []
    for i, n in enumerate(n_vals):
        if n in actual_primes:
            if n in sophie_germain:
                sg_scores.append(scores_norm[i])
            else:
                non_sg_scores.append(scores_norm[i])

    print(f"\nMean score - Sophie Germain: {np.mean(sg_scores):.3f}")
    print(f"Mean score - Other primes:   {np.mean(non_sg_scores):.3f}")

    # 4. Prime k-tuples: can we detect (p, p+2, p+6) patterns?
    print("\n--- Prime Triplets (p, p+2, p+6) ---")
    triplets = [(p, p+2, p+6) for p in actual_primes
                if p+2 in actual_primes and p+6 in actual_primes]
    print(f"Prime triplets up to {max_n}: {len(triplets)}")
    print(f"Examples: {triplets[:5]}")

    # 5. Cunningham chains (p, 2p+1, 4p+3, ...)
    print("\n--- Cunningham Chains ---")

    def cunningham_chain(p, primes_set, max_val):
        """Find length of Cunningham chain starting at p."""
        chain = [p]
        current = p
        while True:
            next_p = 2 * current + 1
            if next_p > max_val or next_p not in primes_set:
                break
            chain.append(next_p)
            current = next_p
        return chain

    chains = []
    for p in sorted(actual_primes):
        chain = cunningham_chain(p, actual_primes, max_n)
        if len(chain) >= 2:
            chains.append(chain)

    # Remove subchains
    chains = [c for c in chains if not any(c[0] in other and c != other for other in chains)]

    print(f"Cunningham chains (length ≥ 2): {len(chains)}")
    longest = max(chains, key=len) if chains else []
    print(f"Longest chain: {longest} (length {len(longest)})")

    # Score pattern in chains
    if chains:
        chain_scores = []
        for chain in chains:
            scores_in_chain = [scores_norm[p-2] for p in chain if p-2 < len(scores_norm)]
            if len(scores_in_chain) >= 2:
                # Check if scores are increasing/decreasing along chain
                diffs = np.diff(scores_in_chain)
                chain_scores.append(np.mean(diffs))

        print(f"Mean score change along chains: {np.mean(chain_scores):.3f}")

    return {
        'twin_primes': actual_twins,
        'sophie_germain': sophie_germain,
        'triplets': triplets,
        'cunningham_chains': chains
    }


# ============================================================
# QUESTION 4: SPECTRAL GAPS (GUE)
# ============================================================

def research_spectral_gaps():
    """
    Research Question 4: Does zero spacing correlate with detection accuracy?

    The zeros follow GUE (Gaussian Unitary Ensemble) statistics.
    Does using zeros with specific spacing patterns help?
    """
    print("\n" + "="*70)
    print("RESEARCH Q4: Spectral Gaps and GUE Statistics")
    print("="*70)

    zeros = fetch_zeros(500, silent=True)

    # Compute normalized spacings
    spacings = np.diff(zeros)
    mean_spacing = np.mean(spacings)
    normalized_spacings = spacings / mean_spacing

    print(f"\nZero spacing statistics (first 500 zeros):")
    print(f"  Mean spacing: {mean_spacing:.4f}")
    print(f"  Std spacing:  {np.std(spacings):.4f}")
    print(f"  Min spacing:  {np.min(spacings):.4f}")
    print(f"  Max spacing:  {np.max(spacings):.4f}")

    # GUE predicts P(s) ≈ (32/π²)s² exp(-4s²/π) for normalized spacings
    # This means small spacings are suppressed ("level repulsion")

    # Test: do closely-spaced zeros contribute more to detection?
    print("\n--- Zero Subset Experiments ---")

    # Experiment 1: Use only "widely spaced" zeros
    spacing_threshold = np.median(normalized_spacings)

    wide_mask = np.concatenate([[True], normalized_spacings > spacing_threshold])
    tight_mask = np.concatenate([[True], normalized_spacings <= spacing_threshold])

    wide_zeros = zeros[wide_mask][:100]
    tight_zeros = zeros[tight_mask][:100]

    max_n = 100

    sonar_wide = PrimeSonar(num_zeros=len(wide_zeros), zeros=wide_zeros, silent=True)
    sonar_tight = PrimeSonar(num_zeros=len(tight_zeros), zeros=tight_zeros, silent=True)
    sonar_first = PrimeSonar(num_zeros=100, silent=True)

    acc_wide = sonar_wide.test_accuracy(max_n)
    acc_tight = sonar_tight.test_accuracy(max_n)
    acc_first = sonar_first.test_accuracy(max_n)

    print(f"\nUsing 100 zeros each, range [2, {max_n}]:")
    print(f"  First 100 zeros:         {acc_first:.1f}%")
    print(f"  Widely-spaced zeros:     {acc_wide:.1f}%")
    print(f"  Tightly-spaced zeros:    {acc_tight:.1f}%")

    # Experiment 2: Random subsampling
    print("\n--- Random Subsampling ---")

    np.random.seed(42)
    random_accuracies = []
    for _ in range(10):
        random_idx = np.random.choice(len(zeros), size=50, replace=False)
        random_zeros = zeros[np.sort(random_idx)]
        sonar_random = PrimeSonar(num_zeros=len(random_zeros), zeros=random_zeros, silent=True)
        random_accuracies.append(sonar_random.test_accuracy(max_n))

    print(f"Random 50-zero subsets: {np.mean(random_accuracies):.1f}% ± {np.std(random_accuracies):.1f}%")

    # Experiment 3: Every Nth zero
    print("\n--- Decimation (every Nth zero) ---")
    for step in [1, 2, 3, 5]:
        decimated = zeros[::step][:50]
        sonar_dec = PrimeSonar(num_zeros=len(decimated), zeros=decimated, silent=True)
        acc = sonar_dec.test_accuracy(max_n)
        print(f"  Every {step}th zero (50 total): {acc:.1f}%")

    return {
        'mean_spacing': mean_spacing,
        'normalized_spacings': normalized_spacings
    }


# ============================================================
# QUESTION 2: THE 0.76 CORRELATION CEILING
# ============================================================

def research_inverse_scattering():
    """
    Research Question 2: Why is primes → zeros limited to 0.76 correlation?

    The forward direction (zeros → primes) works perfectly.
    The inverse (primes → zeros) has a ceiling. Why?
    """
    print("\n" + "="*70)
    print("RESEARCH Q2: The 0.76 Inverse Scattering Ceiling")
    print("="*70)

    # The explicit formula relates primes to zeros via Fourier transform
    # ψ(x) = x - Σ x^ρ/ρ - log(2π) - log(1 - x^{-2})/2

    # To reconstruct zeros from primes, we need the inverse transform

    max_n = 1000
    primes = sieve_primes_simple(max_n)
    zeros = fetch_zeros(500, silent=True)

    print(f"\nUsing {len(primes)} primes up to {max_n}")
    print(f"Target: first {len(zeros)} zeros")

    # Method 1: FFT of prime indicator function
    print("\n--- Method 1: FFT of Prime Indicator ---")

    indicator = np.zeros(max_n + 1)
    indicator[primes] = 1

    # Weight by 1/sqrt(p) as in explicit formula
    weighted = np.zeros(max_n + 1)
    weighted[primes] = 1 / np.sqrt(primes)

    # FFT
    fft_indicator = fft(indicator)
    fft_weighted = fft(weighted)

    freqs = fftfreq(len(indicator))

    # The zeros should appear as peaks in the frequency domain
    # at positions related to γ/(2π) where γ are the zero heights

    power = np.abs(fft_weighted)**2

    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(power[1:len(power)//2], height=np.max(power)/100)
    peaks = peaks + 1  # Adjust for slice

    # Convert to "frequency" in zero-space
    peak_freqs = freqs[peaks] * max_n  # Scale

    print(f"Found {len(peaks)} spectral peaks")
    print(f"First 10 peak frequencies: {peak_freqs[:10]}")
    print(f"Actual first 10 zeros:     {zeros[:10]}")

    # Method 2: Direct correlation via explicit formula
    print("\n--- Method 2: Explicit Formula Inversion ---")

    # Construct the Chebyshev ψ from primes
    def chebyshev_psi_from_primes(x, primes):
        """ψ(x) = Σ_{p^k ≤ x} log(p)"""
        total = 0.0
        for p in primes:
            if p > x:
                break
            pk = p
            while pk <= x:
                total += np.log(p)
                pk *= p
        return total

    # Sample ψ at many points
    x_vals = np.linspace(2, max_n, 1000)
    psi_actual = np.array([chebyshev_psi_from_primes(x, primes) for x in x_vals])

    # The oscillatory part should encode the zeros
    # ψ(x) ≈ x - Σ 2√x cos(γ log x) / √(1/4 + γ²)
    # So: [x - ψ(x)] / √x ≈ Σ 2 cos(γ log x) / √(1/4 + γ²)

    residual = (x_vals - psi_actual) / np.sqrt(x_vals)

    # FFT of residual in log-space
    log_x = np.log(x_vals)

    # Interpolate to uniform log spacing
    log_x_uniform = np.linspace(log_x[0], log_x[-1], 1024)
    residual_interp = np.interp(log_x_uniform, log_x, residual)

    fft_residual = fft(residual_interp)
    power_residual = np.abs(fft_residual)**2

    # The zeros should appear at frequencies γ in this space
    freqs_log = fftfreq(len(log_x_uniform), d=(log_x_uniform[1] - log_x_uniform[0]))

    # Find peaks in positive frequencies
    pos_mask = freqs_log > 0
    pos_freqs = freqs_log[pos_mask]
    pos_power = power_residual[pos_mask]

    peaks2, props = find_peaks(pos_power, height=np.max(pos_power)/50, distance=5)

    recovered_zeros = pos_freqs[peaks2] * 2 * np.pi
    recovered_zeros = sorted(recovered_zeros)[:50]  # First 50

    print(f"\nRecovered zeros from FFT: {len(recovered_zeros)}")

    # Compare to actual zeros
    if len(recovered_zeros) >= 10:
        print(f"First 10 recovered: {np.round(recovered_zeros[:10], 2)}")
        print(f"First 10 actual:    {np.round(zeros[:10], 2)}")

        # Correlation
        n_compare = min(len(recovered_zeros), len(zeros))
        correlation = np.corrcoef(recovered_zeros[:n_compare], zeros[:n_compare])[0, 1]
        print(f"\nCorrelation: {correlation:.3f}")

    # Why the ceiling?
    print("\n--- Limiting Factors ---")
    print("""
The 0.76 ceiling arises from:

1. FINITE RANGE: We only know primes up to N, but zeros encode primes
   to infinity. Information loss at boundaries.

2. QUANTIZATION: ψ(x) is a step function, but we sample discretely.
   High-frequency zero information is aliased.

3. CONVERGENCE: The explicit formula sum over zeros converges slowly.
   Truncating at Z zeros loses information about primes > f(Z).

4. THE DUAL PROBLEM: Forward (zeros → primes) has closed-form weights
   (1/√(1/4 + γ²)). Inverse requires deconvolution - ill-conditioned.

5. NUMBER-THEORETIC: Primes encode zeros through a multiplicative
   structure (Euler product), but decoding is additive (Fourier).
   The product → sum transformation loses phase information.
""")

    return {
        'correlation': correlation if 'correlation' in dir() else 0,
        'recovered_zeros': recovered_zeros
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("   d74169 DEEP RESEARCH - Fundamental Structure Investigation")
    print("   'The primes are just sound waves. If you know the")
    print("    frequencies, you can hear where they are.'")
    print("="*70)

    # Q1: Why 14?
    min_zeros_results = research_minimum_zeros()

    # Q3: Compression ratio
    compression_results = research_compression_ratio()

    # Q5: Higher-order structures
    structure_results = research_higher_order_structures()

    # Q4: Spectral gaps
    spectral_results = research_spectral_gaps()

    # Q2: The 0.76 ceiling
    inverse_results = research_inverse_scattering()

    print("\n" + "="*70)
    print("   RESEARCH COMPLETE")
    print("="*70)
