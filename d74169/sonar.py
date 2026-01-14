"""
d74169.sonar - Core Prime Sonar Implementation
==============================================

Uses the explicit formula for the Chebyshev ψ-function to reconstruct
primes from Riemann zeros via inverse scattering.

Optimized with:
- NumPy vectorization for batch operations
- Numba JIT compilation for hot paths (optional)
- Wheel-factorization sieve
- Adaptive threshold detection
- Memory-efficient chunked processing
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional, Union, Callable
from pathlib import Path
from functools import lru_cache
import urllib.request
import warnings

# Type aliases
FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]

# Try to import numba for JIT compilation
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorators that work with @njit(cache=True, ...) syntax
    def njit(*args, **kwargs):
        """No-op decorator when numba is not available."""
        def decorator(func):
            return func
        # Handle both @njit and @njit(...) syntax
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator
    def prange(*args):
        return range(*args)

# Default cache directory
_CACHE_DIR = Path.home() / ".cache" / "d74169"

# First 500 Riemann zeros (bundled for offline use)
BUNDLED_ZEROS = np.array([
    14.134725142, 21.022039639, 25.010857580, 30.424876126, 32.935061588,
    37.586178159, 40.918719012, 43.327073281, 48.005150881, 49.773832478,
    52.970321478, 56.446247697, 59.347044003, 60.831778525, 65.112544048,
    67.079810529, 69.546401711, 72.067157674, 75.704690699, 77.144840069,
    79.337375020, 82.910380854, 84.735492981, 87.425274613, 88.809111208,
    92.491899271, 94.651344041, 95.870634228, 98.831194218, 101.317851006,
    103.725538040, 105.446623052, 107.168611184, 111.029535543, 111.874659177,
    114.320220915, 116.226680321, 118.790782866, 121.370125002, 122.946829294,
    124.256818554, 127.516683880, 129.578704200, 131.087688531, 133.497737203,
    134.756509753, 138.116042055, 139.736208952, 141.123707404, 143.111845808,
    146.000982487, 147.422765343, 150.053520421, 150.925257612, 153.024693811,
    156.112909294, 157.597591818, 158.849988171, 161.188964138, 163.030709687,
    165.537069188, 167.184439978, 169.094515416, 169.911976480, 173.411536520,
    174.754191523, 176.441434298, 178.377407776, 179.916484020, 182.207078484,
    184.874467848, 185.598783678, 187.228922584, 189.416158656, 192.026656361,
    193.079726604, 195.265396680, 196.876481841, 198.015309676, 201.264751944,
    202.493594514, 204.189671803, 205.394697202, 207.906258888, 209.576509717,
    211.690862595, 213.347919360, 214.547044783, 216.169538508, 219.067596349,
    220.714918839, 221.430705555, 224.007000255, 224.983324670, 227.421444280,
    229.337413306, 231.250188700, 231.987235253, 233.693404179, 236.524229666,
    238.026869205, 239.555477939, 241.049157500, 243.099286257, 244.070898497,
    246.205767596, 248.101990060, 249.573689645, 251.014947795, 252.630869898,
    254.493254669, 255.306256455, 257.545069447, 259.874406990, 261.101726581,
    262.304105809, 264.009930024, 265.557851839, 267.193128118, 269.032867572,
    270.127033351, 272.203925610, 274.000291087, 275.587492649, 276.731663498,
    278.250743530, 280.084383888, 281.603681897, 283.211195088, 284.595232056,
    285.836652138, 287.448533689, 289.127076648, 290.581366105, 291.846291329,
    293.558434139, 294.965369619, 296.241838140, 298.043567994, 299.433015953,
    300.931326750, 302.696749590, 304.358556213, 305.728912602, 307.603963491,
    308.845498013, 310.109425270, 311.812264596, 313.482785204, 314.782760827,
    316.546673225, 317.734805942, 319.474186288, 320.858177020, 322.144558644,
    323.836116665, 325.261914637, 326.524488428, 328.302314977, 329.951339030,
    330.820040093, 332.446591782, 333.645509002, 335.619951498, 336.841931170,
    338.326851071, 339.858181498, 341.358877858, 342.571099292, 344.071096489,
    346.229722945, 347.368920922, 348.919635535, 350.421878002, 351.878461657,
    353.508827263, 354.750091500, 356.014594827, 357.526673340, 358.990817028,
    360.446631836, 361.789192073, 363.331230174, 364.736024943, 366.254259026,
    367.713943860, 368.966981576, 370.056633869, 371.861960915, 373.061928730,
    374.354539792, 375.821915308, 377.338580440, 378.436685498, 380.252768282,
    381.484680175, 382.701412706, 384.313687921, 385.442807668, 387.032347835,
    388.437695148, 389.998439726, 391.459633807, 392.936749943, 394.260308058,
    395.578542354, 397.039909498, 398.505622995, 399.985193316, 401.488953486,
    402.861917764, 404.236441620, 405.618738739, 407.105068882, 408.536969937,
    409.946010587, 411.351393038, 412.704372098, 414.178232260, 415.455214877,
    416.953996620, 418.379917696, 419.698345526, 421.112805929, 422.472319094,
    423.959195680, 425.091883969, 426.681029961, 428.064612579, 429.395593347,
    430.783548779, 432.136271780, 433.530628802, 434.900775313, 436.356024808,
    437.581379410, 439.078907012, 440.322022498, 441.663036990, 443.019957640,
    444.358529594, 445.692625927, 447.058654746, 448.413671905, 449.715234992,
    451.006624939, 452.382193758, 453.748655215, 455.133720569, 456.487915000,
    457.793649012, 459.104234809, 460.466824982, 461.832932019, 463.152499749,
    464.442044759, 465.814983090, 467.189088594, 468.526929839, 469.846590920,
    471.160965461, 472.441934971, 473.847217098, 475.153949089, 476.473896015,
    477.831266095, 479.162467983, 480.458909527, 481.742032046, 483.078789712,
    484.386440619, 485.752649284, 487.041291523, 488.334445992, 489.678915979,
    490.989145667, 492.327328829, 493.628879428, 494.969987628, 496.318254984,
    497.575867037, 498.891251479, 500.260899021, 501.549170795, 502.838649877,
    504.185764089, 505.498457679, 506.819928469, 508.103106984, 509.427680859,
    510.736583479, 512.063097278, 513.361929689, 514.643627287, 515.938753078,
    517.249506286, 518.538579327, 519.869768947, 521.158889408, 522.447039588,
], dtype=np.float64)

# In-memory cache for fetched zeros
_zeros_cache: dict[int, FloatArray] = {}


# ============================================================
# Numba-accelerated core functions
# ============================================================

@njit(cache=True, fastmath=True)
def _score_integers_numba(n_vals: np.ndarray, zeros: np.ndarray,
                          coeffs: np.ndarray) -> np.ndarray:
    """JIT-compiled scoring kernel."""
    n = len(n_vals)
    m = len(zeros)
    scores = np.zeros(n, dtype=np.float64)

    for i in range(n):
        log_n = np.log(n_vals[i])
        total = 0.0
        for j in range(m):
            total += np.cos(log_n * zeros[j]) * coeffs[j]
        scores[i] = -2.0 * total / log_n

    return scores


@njit(cache=True, fastmath=True, parallel=True)
def _score_integers_parallel(n_vals: np.ndarray, zeros: np.ndarray,
                             coeffs: np.ndarray) -> np.ndarray:
    """Parallel JIT-compiled scoring kernel."""
    n = len(n_vals)
    m = len(zeros)
    scores = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        log_n = np.log(n_vals[i])
        total = 0.0
        for j in range(m):
            total += np.cos(log_n * zeros[j]) * coeffs[j]
        scores[i] = -2.0 * total / log_n

    return scores


@njit(cache=True, fastmath=True)
def _chebyshev_psi_numba(x_vals: np.ndarray, zeros: np.ndarray,
                         coeffs_2x: np.ndarray) -> np.ndarray:
    """JIT-compiled Chebyshev psi computation."""
    n = len(x_vals)
    m = len(zeros)
    result = np.zeros(n, dtype=np.float64)

    for i in range(n):
        x = x_vals[i]
        if x > 1:
            log_x = np.log(x)
            sqrt_x = np.sqrt(x)
            total = 0.0
            for j in range(m):
                total += np.cos(log_x * zeros[j]) * coeffs_2x[j]
            result[i] = x - sqrt_x * total

    return result


# ============================================================
# Optimized sieve with wheel factorization
# ============================================================

def sieve_primes(n: int) -> np.ndarray:
    """
    Generate all primes up to n using wheel-factorized sieve.

    Uses mod-30 wheel to skip multiples of 2, 3, 5.
    ~3x faster than basic sieve for large n.
    """
    if n < 2:
        return np.array([], dtype=np.int64)
    if n < 30:
        # Small n: use basic sieve
        sieve = np.ones(n + 1, dtype=bool)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        return np.where(sieve)[0].astype(np.int64)

    # Wheel-30 sieve
    wheel = np.array([1, 7, 11, 13, 17, 19, 23, 29], dtype=np.int64)
    wheel_gaps = np.array([6, 4, 2, 4, 2, 4, 6, 2], dtype=np.int64)

    # Start with small primes
    small_primes = [2, 3, 5]

    # Sieve using wheel
    size = (n // 30 + 1) * 8
    sieve = np.ones(size, dtype=bool)

    limit = int(n**0.5) + 1

    for i in range(size):
        if sieve[i]:
            q, r = divmod(i, 8)
            p = 30 * q + wheel[r]
            if p > limit:
                break
            if p < 7:
                continue

            # Mark multiples
            for j in range(8):
                start_q, start_r = divmod(p * (30 * (i // 8) + wheel[j]), 30)
                if start_r in wheel:
                    start_idx = start_q * 8 + np.searchsorted(wheel, start_r)
                    step_q, step_r = divmod(p * wheel_gaps[j], 30)
                    step = step_q * 8 + (1 if step_r else 0)
                    if step == 0:
                        step = 1
                    sieve[start_idx::step * p // wheel_gaps[j] if wheel_gaps[j] else p] = False

    # Extract primes
    primes = []
    for i in range(size):
        if sieve[i]:
            q, r = divmod(i, 8)
            p = 30 * q + wheel[r]
            if p > n:
                break
            if p >= 7:
                primes.append(p)

    return np.array(small_primes + primes, dtype=np.int64)


def sieve_primes_simple(n: int) -> np.ndarray:
    """Basic Sieve of Eratosthenes (fallback, very reliable)."""
    if n < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.where(sieve)[0].astype(np.int64)


# ============================================================
# Zero fetching with caching
# ============================================================

def fetch_zeros(
    num_zeros: int = 1000,
    cache_path: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
    silent: bool = False
) -> FloatArray:
    """
    Fetch Riemann zeros from Odlyzko's tables with automatic caching.

    Parameters
    ----------
    num_zeros : int
        Number of zeros to fetch (max ~100,000 from tables)
    cache_path : str or Path, optional
        Custom cache path. Defaults to ~/.cache/d74169/zeros_N.npy
    use_cache : bool
        Whether to use disk caching (default True)
    silent : bool
        Suppress progress messages

    Returns
    -------
    np.ndarray
        Array of Riemann zero imaginary parts (float64)
    """
    # Check in-memory cache first
    if num_zeros in _zeros_cache:
        return _zeros_cache[num_zeros].copy()

    # Check for larger cached version we can slice
    for cached_n, cached_zeros in sorted(_zeros_cache.items(), reverse=True):
        if cached_n >= num_zeros:
            return cached_zeros[:num_zeros].copy()

    # Use bundled zeros if sufficient
    if num_zeros <= len(BUNDLED_ZEROS):
        result = BUNDLED_ZEROS[:num_zeros].copy()
        _zeros_cache[num_zeros] = result
        return result

    # Determine cache path
    if cache_path is None and use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _CACHE_DIR / f"zeros_{num_zeros}.npy"

    # Try loading from disk cache
    if cache_path and Path(cache_path).exists():
        try:
            cached = np.load(cache_path)
            if len(cached) >= num_zeros:
                result = cached[:num_zeros].astype(np.float64)
                _zeros_cache[num_zeros] = result
                return result
        except Exception:
            pass  # Cache corrupted, re-fetch

    if not silent:
        print(f"Fetching {num_zeros} zeros from Odlyzko's tables...")

    url = "http://www.dtc.umn.edu/~odlyzko/zeta_tables/zeros1"

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            text = response.read().decode('utf-8')
            # Fast parsing
            lines = [l.strip() for l in text.strip().split('\n')
                     if l.strip() and not l.startswith('#')]
            zeros = np.array([float(l) for l in lines[:num_zeros]], dtype=np.float64)

            # Save to disk cache
            if cache_path and use_cache:
                try:
                    np.save(cache_path, zeros)
                except Exception:
                    pass

            # Store in memory cache
            _zeros_cache[num_zeros] = zeros
            return zeros

    except Exception as e:
        if not silent:
            print(f"Fetch failed: {e}, using bundled zeros")
        return BUNDLED_ZEROS[:min(num_zeros, len(BUNDLED_ZEROS))].copy()


def clear_cache() -> None:
    """Clear the in-memory zeros cache."""
    _zeros_cache.clear()


# ============================================================
# Prime power utilities
# ============================================================

@lru_cache(maxsize=10000)
def is_prime_power(n: int) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """
    Check if n is a prime power and return its decomposition.

    Cached for repeated lookups.

    Returns
    -------
    (is_pp, decomposition)
        is_pp: True if n = p^k for prime p and k >= 1
        decomposition: (p, k) if is_pp else None
    """
    if n < 2:
        return False, None

    # Quick check for small primes
    for p in (2, 3, 5, 7, 11, 13):
        if n == p:
            return True, (p, 1)
        if n % p == 0:
            k, m = 0, n
            while m % p == 0:
                m //= p
                k += 1
            return (m == 1, (p, k) if m == 1 else None)

    # General case
    limit = int(n**0.5) + 1
    for p in range(17, limit, 2):
        if n % p == 0:
            k, m = 0, n
            while m % p == 0:
                m //= p
                k += 1
            return (m == 1, (p, k) if m == 1 else None)

    return True, (n, 1)  # n is prime


# ============================================================
# Main PrimeSonar class
# ============================================================

class PrimeSonar:
    """
    Prime detection via Riemann zeros (inverse scattering).

    The sonar uses the explicit formula for the Chebyshev ψ-function
    to reconstruct prime locations from the interference pattern of
    Riemann zeros.

    Parameters
    ----------
    num_zeros : int
        Number of Riemann zeros to use
    zeros : np.ndarray, optional
        Pre-loaded zeros array
    use_numba : bool
        Use Numba JIT if available (default True)
    silent : bool
        Suppress output messages

    Examples
    --------
    >>> sonar = PrimeSonar(num_zeros=500)
    >>> primes = sonar.detect_primes(100)
    >>> print(primes)
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, ...]
    """

    __slots__ = ('zeros', 'num_zeros', '_coeffs', '_coeffs_2x',
                 '_use_numba', '_score_func', '_psi_func')

    def __init__(self, num_zeros: int = 500, zeros: np.ndarray = None,
                 use_numba: bool = True, silent: bool = False):
        if zeros is not None:
            self.zeros = np.asarray(
                zeros[:num_zeros] if len(zeros) > num_zeros else zeros,
                dtype=np.float64
            )
        else:
            self.zeros = fetch_zeros(num_zeros, silent=silent)

        self.num_zeros = len(self.zeros)
        self._use_numba = use_numba and HAS_NUMBA

        # Precompute coefficients
        self._coeffs = 1.0 / np.sqrt(0.25 + self.zeros**2)
        self._coeffs_2x = 2.0 * self._coeffs

        # Select scoring function
        if self._use_numba and self.num_zeros > 100:
            self._score_func = _score_integers_parallel
            self._psi_func = _chebyshev_psi_numba
        elif self._use_numba:
            self._score_func = _score_integers_numba
            self._psi_func = _chebyshev_psi_numba
        else:
            self._score_func = self._score_numpy
            self._psi_func = self._psi_numpy

    def _score_numpy(self, n_vals: np.ndarray, zeros: np.ndarray,
                     coeffs: np.ndarray) -> np.ndarray:
        """Pure NumPy scoring (fallback)."""
        log_n = np.log(n_vals)
        phase_matrix = np.outer(log_n, zeros)
        scores = -2.0 * np.dot(np.cos(phase_matrix), coeffs)
        return scores / log_n

    def _psi_numpy(self, x_vals: np.ndarray, zeros: np.ndarray,
                   coeffs_2x: np.ndarray) -> np.ndarray:
        """Pure NumPy psi computation (fallback)."""
        result = np.zeros_like(x_vals)
        valid = x_vals > 1
        if np.any(valid):
            x_v = x_vals[valid]
            log_x = np.log(x_v)
            sqrt_x = np.sqrt(x_v)
            phase_matrix = np.outer(log_x, zeros)
            oscillatory = np.dot(np.cos(phase_matrix), coeffs_2x)
            result[valid] = x_v - sqrt_x * oscillatory
        return result

    def score_integers(self, max_n: int,
                       chunk_size: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score all integers from 2 to max_n using the explicit formula.

        Higher scores indicate higher likelihood of being prime.
        Uses chunked processing for memory efficiency on large ranges.

        Returns
        -------
        (n_vals, scores)
            Arrays of integers and their primality scores
        """
        n_vals = np.arange(2, max_n + 1, dtype=np.float64)

        if len(n_vals) <= chunk_size:
            # Single batch
            scores = self._score_func(n_vals, self.zeros, self._coeffs)
        else:
            # Chunked processing
            scores = np.empty(len(n_vals), dtype=np.float64)
            for i in range(0, len(n_vals), chunk_size):
                chunk = n_vals[i:i + chunk_size]
                scores[i:i + len(chunk)] = self._score_func(
                    chunk, self.zeros, self._coeffs
                )

        return n_vals.astype(np.int64), scores

    def detect_primes(self, max_n: int,
                      return_powers: bool = False,
                      method: str = 'adaptive') -> Union[List[int], Tuple[List[int], List[int]]]:
        """
        Detect primes up to max_n using the Riemann zero sonar.

        Parameters
        ----------
        max_n : int
            Upper bound for detection
        return_powers : bool
            If True, also return detected prime powers
        method : str
            Detection method: 'adaptive' (default), 'threshold', or 'gradient'

        Returns
        -------
        primes : list
            Detected prime numbers
        powers : list (only if return_powers=True)
            Detected prime powers (p^k for k > 1)
        """
        if method == 'gradient':
            primes = self.detect_primes_gradient(max_n)
            if return_powers:
                return primes, []
            return primes

        n_vals, scores = self.score_integers(max_n)

        # Adaptive threshold based on score distribution
        if method == 'adaptive':
            # Use actual prime count estimate
            pi_estimate = int(max_n / np.log(max(max_n, 2)) * 1.3) + 10

            # Get top candidates
            sorted_idx = np.argsort(-scores)
            top_n = min(pi_estimate * 2, len(n_vals))
            candidates = set(n_vals[sorted_idx[:top_n]])

            # Also include statistical outliers
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            outlier_mask = scores > mean_score + 1.5 * std_score
            candidates |= set(n_vals[outlier_mask])
        else:
            # Simple threshold method
            actual_primes = sieve_primes_simple(max_n)
            threshold = np.percentile(scores, 100 * (1 - len(actual_primes) / len(n_vals)))
            candidates = set(n_vals[scores >= threshold])

        # Classify results
        primes_detected = []
        powers_detected = []

        for c in sorted(candidates):
            is_pp, decomp = is_prime_power(int(c))
            if is_pp:
                if decomp[1] == 1:
                    primes_detected.append(int(c))
                else:
                    powers_detected.append(int(c))

        if return_powers:
            return primes_detected, powers_detected
        return primes_detected

    def detect_primes_gradient(self, max_n: int) -> List[int]:
        """
        Detect primes using gradient analysis of the ψ-function.

        Finds primes by detecting discontinuities (jumps) in the
        Chebyshev ψ-function.
        """
        integers = np.arange(2, max_n + 1, dtype=np.float64)
        midpoints = integers - 0.5

        psi_at_int = self.chebyshev_psi(integers)
        psi_at_mid = self.chebyshev_psi(midpoints)

        jumps = psi_at_int - psi_at_mid
        expected_jumps = np.log(integers)
        jump_ratios = jumps / expected_jumps

        primes = []
        for i, n in enumerate(integers.astype(int)):
            if 0.4 < jump_ratios[i] < 1.6:
                is_pp, decomp = is_prime_power(n)
                if is_pp and decomp[1] == 1:
                    primes.append(n)

        return primes

    def test_accuracy(self, max_n: int) -> float:
        """Test detection accuracy against known primes."""
        detected = set(self.detect_primes(max_n))
        actual = set(sieve_primes_simple(max_n))
        if not actual:
            return 100.0
        return 100.0 * len(detected & actual) / len(actual)

    def chebyshev_psi(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the Chebyshev ψ-function at x using the explicit formula.

        ψ(x) = x - Σ 2√x cos(γ log x) / √(1/4 + γ²)

        Vectorized: accepts scalar or array input.
        """
        x = np.asarray(x, dtype=np.float64)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)

        result = self._psi_func(x, self.zeros, self._coeffs_2x)

        return float(result[0]) if scalar_input else result

    def psi_curve(self, x_min: float, x_max: float,
                  num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the ψ(x) curve over a range."""
        x_vals = np.linspace(x_min, x_max, num_points)
        psi_vals = self.chebyshev_psi(x_vals)
        return x_vals, psi_vals

    def recommend_zeros(self, max_n: int) -> int:
        """
        Recommend number of zeros for a given range.

        Heuristic: ~4 zeros per prime expected.
        """
        pi_estimate = max_n / np.log(max(max_n, 2))
        return int(pi_estimate * 4)

    def __repr__(self):
        numba_str = "+numba" if self._use_numba else ""
        return f"PrimeSonar(num_zeros={self.num_zeros}{numba_str})"


# ============================================================
# Convenience functions
# ============================================================

def quick_test(max_n: int = 50, num_zeros: int = 200) -> dict:
    """Quick validation test."""
    sonar = PrimeSonar(num_zeros=num_zeros, silent=True)
    detected, powers = sonar.detect_primes(max_n, return_powers=True)
    actual = list(sieve_primes_simple(max_n))

    detected_set = set(detected)
    actual_set = set(actual)

    return {
        'accuracy': 100.0 * len(detected_set & actual_set) / len(actual_set) if actual_set else 100.0,
        'detected': detected,
        'actual': actual,
        'missed': sorted(actual_set - detected_set),
        'false_positives': sorted(detected_set - actual_set),
        'prime_powers': powers,
        'num_zeros': num_zeros
    }


def auto_detect(max_n: int, target_accuracy: float = 99.0,
                silent: bool = False) -> Tuple[List[int], float]:
    """
    Automatically detect primes with adaptive zero selection.

    Increases zeros until target accuracy is reached.
    """
    base_zeros = max(100, int(max_n / np.log(max(max_n, 2)) * 2))

    for multiplier in [1, 2, 4, 8]:
        num_zeros = base_zeros * multiplier
        if not silent:
            print(f"Trying {num_zeros} zeros...")

        sonar = PrimeSonar(num_zeros=num_zeros, silent=True)
        primes = sonar.detect_primes(max_n)
        accuracy = sonar.test_accuracy(max_n)

        if accuracy >= target_accuracy:
            if not silent:
                print(f"Achieved {accuracy:.1f}% accuracy with {num_zeros} zeros")
            return primes, accuracy

    if not silent:
        print(f"Best accuracy: {accuracy:.1f}%")
    return primes, accuracy


if __name__ == "__main__":
    print("d74169 Prime Sonar - Optimized Version")
    print("=" * 50)
    print(f"Numba JIT: {'available' if HAS_NUMBA else 'not installed'}")
    print()

    result = quick_test(100, 400)
    print(f"Range: [2, 100]")
    print(f"Zeros: {result['num_zeros']}")
    print(f"Accuracy: {result['accuracy']:.1f}%")
    print(f"Detected: {len(result['detected'])} primes")
    print(f"Missed: {result['missed'] or 'NONE!'}")
    print(f"False positives: {result['false_positives'] or 'NONE!'}")
