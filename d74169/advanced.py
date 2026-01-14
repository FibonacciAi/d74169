"""
d74169.advanced - Advanced Prime Structure Analysis
===================================================

Extended capabilities:
1. Decode other prime structures (twin primes, gaps, arithmetic progressions, Goldbach)
2. Inverse scattering: reconstruct zeros from primes
3. Minimum encoding: find the essential zeros

The duality goes both ways.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .sonar import PrimeSonar, sieve_primes_simple, fetch_zeros, is_prime_power


# ============================================================
# 2. Decode Other Prime Structures
# ============================================================

@dataclass
class TwinPrime:
    """A twin prime pair (p, p+2)."""
    p: int
    q: int  # p + 2

    def __iter__(self):
        return iter((self.p, self.q))


@dataclass
class GoldbachPair:
    """A Goldbach decomposition: even n = p + q."""
    n: int
    p: int
    q: int

    def __iter__(self):
        return iter((self.p, self.q))


class PrimeStructures:
    """
    Detect prime structures using Riemann zeros.

    Goes beyond single primes to find:
    - Twin primes (p, p+2)
    - Prime gaps
    - Primes in arithmetic progressions
    - Goldbach pairs

    All from the same spectral data.
    """

    def __init__(self, num_zeros: int = 2000, silent: bool = True):
        self.sonar = PrimeSonar(num_zeros=num_zeros, silent=silent)
        self.num_zeros = num_zeros
        self._prime_cache: Dict[int, List[int]] = {}

    def _get_primes(self, max_n: int) -> List[int]:
        """Get primes up to max_n (cached)."""
        if max_n not in self._prime_cache:
            self._prime_cache[max_n] = self.sonar.detect_primes(max_n)
        return self._prime_cache[max_n]

    def detect_twin_primes(self, max_n: int) -> List[TwinPrime]:
        """
        Detect twin prime pairs (p, p+2) using Riemann zeros.

        Twin primes are pairs where both p and p+2 are prime.
        The zeros encode both simultaneously.

        Examples: (3,5), (5,7), (11,13), (17,19), (29,31), ...
        """
        primes = set(self._get_primes(max_n))
        twins = []

        for p in sorted(primes):
            if p + 2 in primes:
                twins.append(TwinPrime(p, p + 2))

        return twins

    def detect_prime_gaps(self, max_n: int) -> List[Tuple[int, int, int]]:
        """
        Detect gaps between consecutive primes.

        Returns list of (p, next_p, gap) tuples.
        The distribution of gaps is encoded in the zeros.
        """
        primes = sorted(self._get_primes(max_n))
        gaps = []

        for i in range(len(primes) - 1):
            p, next_p = primes[i], primes[i + 1]
            gap = next_p - p
            gaps.append((p, next_p, gap))

        return gaps

    def detect_record_gaps(self, max_n: int) -> List[Tuple[int, int, int]]:
        """
        Find record-breaking prime gaps.

        Returns gaps that are larger than all previous gaps.
        """
        gaps = self.detect_prime_gaps(max_n)
        records = []
        max_gap = 0

        for p, next_p, gap in gaps:
            if gap > max_gap:
                max_gap = gap
                records.append((p, next_p, gap))

        return records

    def detect_primes_in_progression(self, max_n: int, a: int, d: int) -> List[int]:
        """
        Detect primes in arithmetic progression a + k*d.

        By Dirichlet's theorem, if gcd(a,d)=1, there are infinitely many.
        The zeros of Dirichlet L-functions encode these, but the
        Riemann zeros give a first approximation.

        Parameters
        ----------
        max_n : int
            Upper bound
        a : int
            Starting value (first term when k=0)
        d : int
            Common difference

        Returns
        -------
        list of primes p where p ≡ a (mod d)
        """
        if np.gcd(a, d) != 1:
            # No primes possible if gcd > 1 (except possibly a itself)
            return [a] if a > 1 and is_prime_power(a)[0] and is_prime_power(a)[1][1] == 1 else []

        primes = self._get_primes(max_n)
        return [p for p in primes if p % d == a % d]

    def detect_goldbach_pairs(self, n: int) -> List[GoldbachPair]:
        """
        Find Goldbach decompositions: n = p + q where both p, q are prime.

        Goldbach's conjecture: every even n > 2 can be written as sum of two primes.
        The zeros encode which pairs work.

        Parameters
        ----------
        n : int
            Even number to decompose

        Returns
        -------
        List of GoldbachPair(n, p, q) where p + q = n
        """
        if n < 4 or n % 2 != 0:
            return []

        primes = set(self._get_primes(n))
        pairs = []

        for p in sorted(primes):
            if p > n // 2:
                break
            q = n - p
            if q in primes:
                pairs.append(GoldbachPair(n, p, q))

        return pairs

    def goldbach_spectrum(self, max_n: int) -> Dict[int, int]:
        """
        Compute Goldbach comet: number of ways to write each even n as p + q.

        Returns dict mapping even n -> count of Goldbach pairs.
        The structure of this spectrum is encoded in the zeros.
        """
        spectrum = {}
        primes = set(self._get_primes(max_n))

        for n in range(4, max_n + 1, 2):
            count = sum(1 for p in primes if p <= n // 2 and (n - p) in primes)
            spectrum[n] = count

        return spectrum

    def sophie_germain_primes(self, max_n: int) -> List[Tuple[int, int]]:
        """
        Find Sophie Germain primes: p where both p and 2p+1 are prime.

        Returns list of (p, 2p+1) pairs.
        """
        primes = set(self._get_primes(max_n * 2 + 1))
        germain = []

        for p in sorted(primes):
            if p > max_n:
                break
            if 2 * p + 1 in primes:
                germain.append((p, 2 * p + 1))

        return germain

    def cousin_primes(self, max_n: int) -> List[Tuple[int, int]]:
        """
        Find cousin prime pairs (p, p+4).
        """
        primes = set(self._get_primes(max_n))
        return [(p, p + 4) for p in sorted(primes) if p + 4 in primes]

    def sexy_primes(self, max_n: int) -> List[Tuple[int, int]]:
        """
        Find sexy prime pairs (p, p+6).
        """
        primes = set(self._get_primes(max_n))
        return [(p, p + 6) for p in sorted(primes) if p + 6 in primes]

    def prime_triplets(self, max_n: int) -> List[Tuple[int, int, int]]:
        """
        Find prime triplets (p, p+2, p+6) or (p, p+4, p+6).
        """
        primes = set(self._get_primes(max_n))
        triplets = []

        for p in sorted(primes):
            if p + 6 > max_n:
                break
            # Form (p, p+2, p+6)
            if p + 2 in primes and p + 6 in primes:
                triplets.append((p, p + 2, p + 6))
            # Form (p, p+4, p+6)
            if p + 4 in primes and p + 6 in primes:
                triplets.append((p, p + 4, p + 6))

        return triplets


# ============================================================
# 3. Inverse Scattering: Primes → Zeros
# ============================================================

class ZeroReconstructor:
    """
    Reconstruct Riemann zeros from prime data.

    The duality goes both ways:
    - Forward: zeros → primes (what PrimeSonar does)
    - Inverse: primes → zeros (what this does)

    This uses the Riemann-von Mangoldt formula and optimization
    to find zeros that would produce the observed prime pattern.
    """

    def __init__(self, max_prime: int = 1000):
        self.primes = list(sieve_primes_simple(max_prime))
        self.max_prime = max_prime
        self._log_primes = np.log(np.array(self.primes, dtype=np.float64))

    def chebyshev_psi_from_primes(self, x: float) -> float:
        """
        Compute exact ψ(x) from primes.

        ψ(x) = Σ log(p) for all prime powers p^k ≤ x
        """
        total = 0.0
        for p in self.primes:
            if p > x:
                break
            # Add log(p) for each power p^k ≤ x
            pk = p
            while pk <= x:
                total += np.log(p)
                pk *= p
        return total

    def estimate_zero_count(self, T: float) -> int:
        """
        Estimate N(T): number of zeros with imaginary part < T.

        Uses Riemann-von Mangoldt formula:
        N(T) ≈ (T/2π) log(T/2π) - T/2π + 7/8
        """
        if T <= 0:
            return 0
        return int((T / (2 * np.pi)) * np.log(T / (2 * np.pi)) - T / (2 * np.pi) + 7/8)

    def reconstruct_zeros(self, num_zeros: int = 100,
                          iterations: int = 1000,
                          learning_rate: float = 0.01) -> np.ndarray:
        """
        Reconstruct Riemann zeros from prime data using gradient descent.

        Finds zeros γ that minimize the reconstruction error:
        ||ψ_exact(x) - ψ_reconstructed(x)||²

        Parameters
        ----------
        num_zeros : int
            Number of zeros to reconstruct
        iterations : int
            Optimization iterations
        learning_rate : float
            Gradient descent step size

        Returns
        -------
        np.ndarray
            Estimated zero locations (imaginary parts)
        """
        # Initialize with Gram points (good approximation)
        zeros = self._gram_points(num_zeros)

        # Sample points for reconstruction
        x_samples = np.linspace(2, self.max_prime, 200)
        psi_exact = np.array([self.chebyshev_psi_from_primes(x) for x in x_samples])

        for _ in range(iterations):
            # Compute reconstruction
            psi_reconstructed = self._psi_from_zeros(x_samples, zeros)

            # Compute gradient
            error = psi_reconstructed - psi_exact
            grad = self._compute_gradient(x_samples, zeros, error)

            # Update zeros
            zeros -= learning_rate * grad

            # Keep zeros positive and ordered
            zeros = np.sort(np.abs(zeros))

        return zeros

    def _gram_points(self, n: int) -> np.ndarray:
        """
        Compute Gram points as initial zero estimates.

        Gram points g_n satisfy θ(g_n) = nπ where θ is the
        Riemann-Siegel theta function. They approximate zeros.
        """
        # Asymptotic formula for Gram points
        points = []
        for k in range(n):
            # Approximate: g_n ≈ 2πn / log(n/2π) for large n
            if k < 5:
                # Use known values for small k
                known = [14.13, 21.02, 25.01, 30.42, 32.94]
                points.append(known[k])
            else:
                g = 2 * np.pi * (k + 1) / np.log((k + 1) / (2 * np.pi))
                points.append(g)
        return np.array(points, dtype=np.float64)

    def _psi_from_zeros(self, x_vals: np.ndarray, zeros: np.ndarray) -> np.ndarray:
        """Compute ψ(x) from given zeros."""
        result = np.zeros_like(x_vals)
        coeffs = 1.0 / np.sqrt(0.25 + zeros**2)

        for i, x in enumerate(x_vals):
            if x > 1:
                log_x = np.log(x)
                sqrt_x = np.sqrt(x)
                osc = 2 * np.sum(np.cos(log_x * zeros) * coeffs)
                result[i] = x - sqrt_x * osc

        return result

    def _compute_gradient(self, x_vals: np.ndarray, zeros: np.ndarray,
                          error: np.ndarray) -> np.ndarray:
        """Compute gradient of error with respect to zeros."""
        grad = np.zeros_like(zeros)
        coeffs = 1.0 / np.sqrt(0.25 + zeros**2)

        for i, x in enumerate(x_vals):
            if x > 1:
                log_x = np.log(x)
                sqrt_x = np.sqrt(x)
                # Derivative of cos(log_x * γ) w.r.t. γ
                for j, gamma in enumerate(zeros):
                    grad[j] += error[i] * sqrt_x * 2 * log_x * np.sin(log_x * gamma) * coeffs[j]

        return grad / len(x_vals)

    def verify_reconstruction(self, reconstructed_zeros: np.ndarray,
                              actual_zeros: np.ndarray = None) -> Dict:
        """
        Verify how well reconstructed zeros match actual zeros.

        Returns metrics on reconstruction quality.
        """
        if actual_zeros is None:
            actual_zeros = fetch_zeros(len(reconstructed_zeros), silent=True)

        n = min(len(reconstructed_zeros), len(actual_zeros))
        recon = reconstructed_zeros[:n]
        actual = actual_zeros[:n]

        errors = np.abs(recon - actual)

        return {
            'mean_error': float(np.mean(errors)),
            'max_error': float(np.max(errors)),
            'relative_error': float(np.mean(errors / actual)),
            'correlation': float(np.corrcoef(recon, actual)[0, 1]),
            'num_zeros': n
        }


# ============================================================
# 4. Minimum Encoding
# ============================================================

class MinimumEncoding:
    """
    Find the minimum set of zeros needed for accurate prime detection.

    Information-theoretic analysis of the encoding:
    - Which zeros matter most?
    - How compressible is the representation?
    - What's the information content per zero?
    """

    def __init__(self, max_n: int = 500, target_accuracy: float = 99.0):
        self.max_n = max_n
        self.target_accuracy = target_accuracy
        self.actual_primes = set(sieve_primes_simple(max_n))

    def find_minimum_zeros(self, max_zeros: int = 5000) -> Dict:
        """
        Binary search for minimum zeros needed.

        Returns the smallest number of zeros achieving target accuracy.
        """
        low, high = 10, max_zeros
        best = None

        while low < high:
            mid = (low + high) // 2
            sonar = PrimeSonar(num_zeros=mid, silent=True)
            detected = set(sonar.detect_primes(self.max_n))
            accuracy = 100.0 * len(detected & self.actual_primes) / len(self.actual_primes)

            if accuracy >= self.target_accuracy:
                best = {'zeros': mid, 'accuracy': accuracy}
                high = mid
            else:
                low = mid + 1

        if best is None:
            sonar = PrimeSonar(num_zeros=max_zeros, silent=True)
            detected = set(sonar.detect_primes(self.max_n))
            accuracy = 100.0 * len(detected & self.actual_primes) / len(self.actual_primes)
            best = {'zeros': max_zeros, 'accuracy': accuracy}

        return best

    def zero_importance(self, num_zeros: int = 500) -> np.ndarray:
        """
        Rank zeros by their importance for prime detection.

        Uses leave-one-out analysis: how much does accuracy drop
        when each zero is removed?

        Returns array of importance scores (higher = more important).
        """
        zeros = fetch_zeros(num_zeros, silent=True)
        baseline_sonar = PrimeSonar(zeros=zeros, num_zeros=num_zeros, silent=True)
        baseline_detected = set(baseline_sonar.detect_primes(self.max_n))
        baseline_accuracy = len(baseline_detected & self.actual_primes) / len(self.actual_primes)

        importance = np.zeros(num_zeros)

        for i in range(num_zeros):
            # Remove zero i
            reduced_zeros = np.concatenate([zeros[:i], zeros[i+1:]])
            sonar = PrimeSonar(zeros=reduced_zeros, num_zeros=num_zeros-1, silent=True)
            detected = set(sonar.detect_primes(self.max_n))
            accuracy = len(detected & self.actual_primes) / len(self.actual_primes)

            # Importance = accuracy drop when removed
            importance[i] = baseline_accuracy - accuracy

        return importance

    def find_essential_zeros(self, num_zeros: int = 500,
                             top_k: int = 50) -> Tuple[np.ndarray, float]:
        """
        Find the most essential zeros for prime detection.

        Returns the top-k most important zeros and accuracy with just those.
        """
        importance = self.zero_importance(num_zeros)
        zeros = fetch_zeros(num_zeros, silent=True)

        # Get indices of top-k most important
        top_indices = np.argsort(-importance)[:top_k]
        essential_zeros = zeros[np.sort(top_indices)]

        # Test accuracy with just essential zeros
        sonar = PrimeSonar(zeros=essential_zeros, num_zeros=top_k, silent=True)
        detected = set(sonar.detect_primes(self.max_n))
        accuracy = 100.0 * len(detected & self.actual_primes) / len(self.actual_primes)

        return essential_zeros, accuracy

    def compression_ratio(self, num_zeros: int = 500) -> Dict:
        """
        Compute how compressible the zero representation is.

        Compares bits needed to store zeros vs bits to store primes directly.
        """
        num_primes = len(self.actual_primes)

        # Bits to store primes directly (bitmap)
        bits_bitmap = self.max_n

        # Bits to store primes as list (log2(max_n) bits each)
        bits_list = num_primes * np.log2(self.max_n)

        # Bits to store zeros (64-bit floats)
        bits_zeros = num_zeros * 64

        # Find minimum zeros needed
        min_result = self.find_minimum_zeros()
        bits_min_zeros = min_result['zeros'] * 64

        return {
            'bits_bitmap': int(bits_bitmap),
            'bits_list': int(bits_list),
            'bits_zeros': int(bits_zeros),
            'bits_min_zeros': int(bits_min_zeros),
            'compression_vs_bitmap': bits_bitmap / bits_min_zeros,
            'compression_vs_list': bits_list / bits_min_zeros,
            'zeros_per_prime': min_result['zeros'] / num_primes,
            'min_zeros': min_result['zeros'],
            'accuracy': min_result['accuracy']
        }

    def information_per_zero(self, num_zeros: int = 500) -> float:
        """
        Estimate bits of prime information per zero.

        How much does each additional zero tell us about primes?
        """
        num_primes = len(self.actual_primes)
        total_info = num_primes * np.log2(self.max_n)  # Bits to specify all primes

        # Test accuracy at different zero counts
        info_gained = []
        prev_accuracy = 0

        for nz in [50, 100, 200, 300, 400, 500]:
            if nz > num_zeros:
                break
            sonar = PrimeSonar(num_zeros=nz, silent=True)
            detected = set(sonar.detect_primes(self.max_n))
            accuracy = len(detected & self.actual_primes) / len(self.actual_primes)

            # Information gained = accuracy improvement * total info
            delta_info = (accuracy - prev_accuracy) * total_info
            delta_zeros = nz - (0 if not info_gained else [50, 100, 200, 300, 400][len(info_gained)-1])

            if delta_zeros > 0:
                info_gained.append(delta_info / delta_zeros)
            prev_accuracy = accuracy

        return float(np.mean(info_gained)) if info_gained else 0.0


# ============================================================
# Convenience functions
# ============================================================

def analyze_prime_structures(max_n: int = 1000, num_zeros: int = 4000) -> Dict:
    """
    Run full prime structure analysis.

    Detects various prime patterns using Riemann zeros.
    """
    ps = PrimeStructures(num_zeros=num_zeros, silent=True)

    primes = ps._get_primes(max_n)
    twins = ps.detect_twin_primes(max_n)
    gaps = ps.detect_prime_gaps(max_n)
    records = ps.detect_record_gaps(max_n)
    germain = ps.sophie_germain_primes(max_n // 2)

    return {
        'primes': primes,
        'num_primes': len(primes),
        'twin_primes': twins,
        'num_twins': len(twins),
        'prime_gaps': gaps,
        'record_gaps': records,
        'sophie_germain': germain,
        'max_gap': max(g[2] for g in gaps) if gaps else 0,
        'avg_gap': np.mean([g[2] for g in gaps]) if gaps else 0
    }


def reconstruct_zeros_from_primes(max_prime: int = 500,
                                   num_zeros: int = 50) -> Dict:
    """
    Attempt to reconstruct Riemann zeros from prime data.

    Demonstrates the inverse direction of the duality.
    """
    zr = ZeroReconstructor(max_prime=max_prime)
    reconstructed = zr.reconstruct_zeros(num_zeros=num_zeros)
    verification = zr.verify_reconstruction(reconstructed)

    return {
        'reconstructed_zeros': reconstructed,
        'verification': verification,
        'first_10_reconstructed': reconstructed[:10].tolist(),
        'first_10_actual': fetch_zeros(10, silent=True).tolist()
    }


def find_optimal_encoding(max_n: int = 500) -> Dict:
    """
    Find the minimum encoding for primes up to max_n.
    """
    me = MinimumEncoding(max_n=max_n, target_accuracy=99.0)

    min_zeros = me.find_minimum_zeros()
    compression = me.compression_ratio()
    info_per_zero = me.information_per_zero()

    return {
        'min_zeros_for_99pct': min_zeros,
        'compression': compression,
        'bits_per_zero': info_per_zero,
        'summary': f"{min_zeros['zeros']} zeros encode {len(me.actual_primes)} primes at {min_zeros['accuracy']:.1f}% accuracy"
    }


if __name__ == "__main__":
    print("d74169 Advanced Analysis")
    print("=" * 60)

    # Test prime structures
    print("\n1. Prime Structures (zeros → patterns)")
    print("-" * 40)
    ps = PrimeStructures(num_zeros=2000, silent=True)

    twins = ps.detect_twin_primes(100)
    print(f"Twin primes up to 100: {[(t.p, t.q) for t in twins]}")

    goldbach = ps.detect_goldbach_pairs(100)
    print(f"Goldbach pairs for 100: {[(g.p, g.q) for g in goldbach]}")

    # Test inverse scattering
    print("\n2. Inverse Scattering (primes → zeros)")
    print("-" * 40)
    zr = ZeroReconstructor(max_prime=200)
    reconstructed = zr.reconstruct_zeros(num_zeros=20, iterations=500)
    actual = fetch_zeros(20, silent=True)
    print(f"First 5 reconstructed: {reconstructed[:5].round(2)}")
    print(f"First 5 actual:        {actual[:5].round(2)}")

    verification = zr.verify_reconstruction(reconstructed, actual)
    print(f"Correlation: {verification['correlation']:.4f}")

    # Test minimum encoding
    print("\n3. Minimum Encoding")
    print("-" * 40)
    me = MinimumEncoding(max_n=200, target_accuracy=99.0)
    result = me.find_minimum_zeros(max_zeros=2000)
    print(f"Min zeros for 99% accuracy at range 200: {result['zeros']}")
    print(f"Actual accuracy: {result['accuracy']:.1f}%")

    print("\n" + "=" * 60)
    print("The duality is bidirectional.")
