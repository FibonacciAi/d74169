# d74169 Project Memory

## Overview
Prime detection via Riemann zeros - bidirectional duality between zeros and primes.

## Links
- Site: https://fibonacciai.github.io/d74169
- Repo: https://github.com/FibonacciAi/d74169
- Version: 2.2.0

## Structure
```
d74169/
├── __init__.py      # Exports, version
├── sonar.py         # Core PrimeSonar, Numba JIT, fetch_zeros
├── validation.py    # Benchmarking, BenchmarkResult
└── advanced.py      # PrimeStructures, ZeroReconstructor, MinimumEncoding
```

## Key APIs
- `PrimeSonar(num_zeros)` - detect primes from zeros
- `PrimeStructures(num_zeros)` - twin primes, Goldbach, gaps
- `ZeroReconstructor(max_prime)` - reconstruct zeros from primes
- `MinimumEncoding(max_n)` - find minimum zeros needed

## Key Facts
- 14 zeros = 100% accuracy at range 100
- Correlation 0.76 for inverse scattering
- Numba gives 5-10x speedup
- Author: @d74169 (no personal name in repo)

## Commands
```bash
# Install
pip install git+https://github.com/FibonacciAi/d74169

# Test
python -c "from d74169 import PrimeSonar; print(PrimeSonar(400).detect_primes(100))"

# Benchmark
python -c "from d74169 import benchmark; benchmark()"
```
