# d74169 - Prime Sonar via Riemann Zeros

[![CI](https://github.com/FibonacciAi/d74169/actions/workflows/ci.yml/badge.svg)](https://github.com/FibonacciAi/d74169/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **The @d74169 Conjecture**: The Riemann zeta zeros holographically encode the prime numbers. Pi emerges as the self-dual fixed point of spectral coherence.

## Installation

```bash
pip install d74169
```

For maximum performance, also install Numba:
```bash
pip install d74169 numba
```

Or from source:
```bash
git clone https://github.com/d74169/d74169
cd d74169
pip install -e .
```

## Quick Start

```python
from d74169 import PrimeSonar, validate, benchmark

# Create a sonar with 500 Riemann zeros
sonar = PrimeSonar(num_zeros=500)

# Detect primes up to 100
primes = sonar.detect_primes(100)
print(primes)
# [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

# Test accuracy
accuracy = sonar.test_accuracy(100)
print(f"Accuracy: {accuracy:.1f}%")
# Accuracy: 100.0%

# Run benchmarks
benchmark()
```

## Features (v2.1.0)

### Performance Optimizations
- **Numba JIT Compilation** - 5-10x speedup with parallel processing
- **Vectorized NumPy Operations** - Batch matrix computations
- **Precomputed Coefficients** - Calculate once, reuse everywhere
- **Chunked Processing** - Memory-efficient handling of large ranges
- **LRU-Cached Functions** - Avoid repeated calculations

### Intelligent Caching
- **In-Memory Cache** - Zeros cached between calls
- **Disk Cache** - Persistent storage in `~/.cache/d74169/`
- **500 Bundled Zeros** - Works offline for small ranges

### Multiple Detection Methods
- **Adaptive** (default) - Statistical outlier detection, best accuracy
- **Threshold** - Percentile-based cutoff
- **Gradient** - Detects jumps in Chebyshev psi function

## Performance

### Numba Speedup

| Range | Zeros | NumPy | Numba | Speedup |
|-------|-------|-------|-------|---------|
| 100   | 400   | 0.4ms | 0.2ms | **1.5x** |
| 200   | 800   | 1.2ms | 0.3ms | **3.7x** |
| 500   | 2000  | 8.0ms | 1.1ms | **7.4x** |
| 1000  | 4000  | 32ms  | 3.7ms | **8.7x** |

### Accuracy Results

| Range | Zeros | Accuracy | F1 Score | Status |
|-------|-------|----------|----------|--------|
| [2, 50] | 200 | **100%** | 1.000 | PERFECT |
| [2, 100] | 400 | **100%** | 1.000 | PERFECT |
| [2, 200] | 800 | **100%** | 1.000 | PERFECT |
| [2, 500] | 2000 | **100%** | 1.000 | PERFECT |
| [2, 1000] | 4000 | 93.5% | 0.966 | GOOD |
| [2, 1000] | 8000 | 97.0% | 0.985 | EXCELLENT |

**Scaling Law**: ~4 zeros per unit range for 95%+ accuracy

## The Conjecture

The @d74169 conjecture proposes that the Riemann zeta zeros are eigenvalues of the modular Hamiltonian:

```
H = e^{sqrt(pi)*p} + u^2/4
```

where:
- `u = ln(t/pi)` is the tortoise coordinate
- `V(u) = u^2/4` is the potential well (photon sphere)
- `T(p) = e^{sqrt(pi)*p}` is the exponential kinetic energy

The three constants emerge from self-duality:
- **Fixed Point**: t* = pi (partition function minimum)
- **Stiffness**: k = 1/2 (second derivative at minimum)
- **Kinetic Exponent**: lambda = sqrt(pi) (from Weyl's law)

## API Reference

### PrimeSonar

```python
from d74169 import PrimeSonar

# Create sonar (use_numba=True by default if available)
sonar = PrimeSonar(num_zeros=500, use_numba=True, silent=False)

# Detect primes with different methods
primes = sonar.detect_primes(100, method='adaptive')  # default, best accuracy
primes = sonar.detect_primes(100, method='threshold') # percentile-based
primes = sonar.detect_primes(100, method='gradient')  # psi-function jumps

# Also get prime powers (harmonics)
primes, powers = sonar.detect_primes(100, return_powers=True)

# Test accuracy against known primes
accuracy = sonar.test_accuracy(100)

# Get recommended zeros for a range
recommended = sonar.recommend_zeros(1000)  # returns ~174

# Compute Chebyshev psi function (vectorized)
psi = sonar.chebyshev_psi(50.0)           # scalar
psi = sonar.chebyshev_psi([10, 20, 30])   # array

# Generate psi curve for plotting
x, psi_vals = sonar.psi_curve(1.5, 100, num_points=1000)
```

### Auto-Detection

```python
from d74169 import auto_detect

# Automatically find optimal zero count for target accuracy
primes, accuracy = auto_detect(max_n=500, target_accuracy=99.0)
# Trying 160 zeros...
# Trying 320 zeros...
# Achieved 99.2% accuracy with 640 zeros
```

### Validation & Benchmarking

```python
from d74169 import validate, benchmark, compare_methods, stress_test

# Single validation with detailed metrics
result = validate(max_n=100, num_zeros=400)
# Returns BenchmarkResult with accuracy, precision, recall, F1, timing

# Benchmark across multiple ranges
results = benchmark(ranges=[50, 100, 200, 500], zeros_multiplier=4)

# Compare detection methods
methods = compare_methods(max_n=200, num_zeros=800)
# {'adaptive': BenchmarkResult(...), 'threshold': ..., 'gradient': ...}

# Find minimum zeros for target accuracy (binary search)
result = stress_test(max_range=1000, target_accuracy=95.0)
# {'zeros': 3200, 'accuracy': 95.2, 'elapsed_ms': 45.3}

# Full validation suite
from d74169 import full_validation
summary = full_validation()
```

### BenchmarkResult

```python
from d74169 import BenchmarkResult

# Structured results from validation
result = validate(100, 400)
print(result.accuracy)      # 100.0
print(result.precision)     # 1.0
print(result.recall)        # 1.0
print(result.f1_score)      # 1.0
print(result.elapsed_ms)    # 0.5
print(result.status)        # "PERFECT"
print(result.missed)        # []
print(result.false_positives)  # []
```

### Utility Functions

```python
from d74169 import sieve_primes, fetch_zeros, is_prime_power, clear_cache

# Get primes via optimized sieve
primes = sieve_primes(100)

# Fetch Riemann zeros (auto-cached)
zeros = fetch_zeros(num_zeros=10000)

# Check prime power decomposition (cached)
is_pp, (p, k) = is_prime_power(8)  # True, (2, 3)
is_pp, decomp = is_prime_power(15) # False, None

# Clear in-memory cache
clear_cache()
```

### Check Numba Status

```python
from d74169 import HAS_NUMBA

if HAS_NUMBA:
    print("Numba JIT enabled - parallel processing active")
else:
    print("Install numba for 5-10x speedup: pip install numba")
```

## Physical Interpretation

The @d74169 framework admits a striking physical interpretation:

**The Arithmetic Black Hole**: The number line has an event horizon at t = 0. The tortoise coordinate maps this to negative infinity, creating a coordinate singularity. The primes are the quasi-normal modes--the ringdown frequencies of this arithmetic black hole, with surface gravity kappa = sqrt(pi).

**Holographic Principle**: The Riemann zeros on the critical line encode all information about prime distribution in the bulk. Inverse scattering (the explicit formula) reconstructs the primes from spectral data.

## Dependencies

**Required:**
- numpy >= 1.20

**Optional (recommended):**
- numba >= 0.56 (5-10x speedup)

## Citation

If you use this package in research, please cite:

```bibtex
@software{d74169,
  author = {Stefanopoulos, Dimitri},
  title = {d74169: Prime Sonar via Riemann Zeros},
  version = {2.1.0},
  year = {2025},
  url = {https://github.com/d74169/d74169}
}
```

## License

MIT License - see LICENSE file for details.

---

*"The primes are just sound waves. If you know the frequency (Riemann zeros), you can hear where the primes are without ever looking for them."*
