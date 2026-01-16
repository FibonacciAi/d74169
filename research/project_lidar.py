#!/usr/bin/env python3
"""
PROJECT LIDAR: Super-Resolution Prime Detection
=================================================
Goal: Use L1-Regularization (Lasso) to recover primes up to n=1000
      using only 80 zeros (instead of ~126 required by direct method)

Mechanism: Exploit the SPARSITY of primes (~168 out of 1000) via
           compressed sensing / sparse recovery techniques.

The explicit formula gives us:
    ψ(n) ≈ n - 2√n × Σ cos(γ×log(n))/|ρ|

We can rewrite detection as: Find sparse vector x where x[n]=1 iff n is prime
Using measurement matrix A constructed from Riemann zeros.
"""

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PROJECT LIDAR: Super-Resolution Prime Detection")
print("=" * 70)

# Load zeros
ZEROS = np.load('/Users/dimitristefanopoulos/d74169_tests/riemann_zeros_master_v2.npy')

def sieve(n):
    s = [True] * (n+1)
    s[0] = s[1] = False
    for i in range(2, int(n**0.5)+1):
        if s[i]:
            for j in range(i*i, n+1, i):
                s[j] = False
    return [i for i in range(n+1) if s[i]]

# === BUILD THE MEASUREMENT MATRIX ===
print("\n[1] BUILDING SPECTRAL MEASUREMENT MATRIX")
print("=" * 70)

def build_measurement_matrix(N, num_zeros):
    """
    Build measurement matrix A where A[j,n] = cos(γ_j × log(n)) / |ρ_j|

    The signal: x[n] = 1 if n is prime, 0 otherwise
    The measurement: y[j] = Σ_n A[j,n] × x[n] = spectral response at zero j
    """
    gamma = ZEROS[:num_zeros]
    weights = 1.0 / np.sqrt(0.25 + gamma**2)

    A = np.zeros((num_zeros, N-1))  # n from 2 to N
    for j in range(num_zeros):
        for n in range(2, N+1):
            A[j, n-2] = np.cos(gamma[j] * np.log(n)) * weights[j]

    return A

def build_enhanced_matrix(N, num_zeros):
    """
    Enhanced matrix with both cos and sin components
    Plus multi-scale features
    """
    gamma = ZEROS[:num_zeros]
    weights = 1.0 / np.sqrt(0.25 + gamma**2)

    # 2 × num_zeros rows (cos and sin)
    rows = 2 * num_zeros
    A = np.zeros((rows, N-1))

    for j in range(num_zeros):
        for n in range(2, N+1):
            phase = gamma[j] * np.log(n)
            A[j, n-2] = np.cos(phase) * weights[j]
            A[num_zeros + j, n-2] = np.sin(phase) * weights[j]

    return A

# Target range
N = 1000
true_primes = set(sieve(N))
num_primes = len([p for p in true_primes if 2 <= p <= N])
print(f"Target: N = {N}, true primes: {num_primes}")

# Test with 80 zeros (vs ~126 needed for direct method)
NUM_ZEROS = 80

# Build measurement matrices
A_basic = build_measurement_matrix(N, NUM_ZEROS)
A_enhanced = build_enhanced_matrix(N, NUM_ZEROS)

print(f"Basic matrix shape: {A_basic.shape}")
print(f"Enhanced matrix shape: {A_enhanced.shape}")

# === CREATE THE SPARSE SIGNAL (ground truth) ===
x_true = np.zeros(N-1)
for p in true_primes:
    if 2 <= p <= N:
        x_true[p-2] = 1

print(f"Signal sparsity: {int(sum(x_true))} / {len(x_true)} = {sum(x_true)/len(x_true):.3f}")

# === SIMULATE MEASUREMENTS ===
print("\n[2] SPARSE RECOVERY VIA LASSO")
print("=" * 70)

# Compute "measurements" (sum over primes)
y_basic = A_basic @ x_true
y_enhanced = A_enhanced @ x_true

# Add small noise to make it realistic
noise_level = 0.01
y_basic_noisy = y_basic + np.random.randn(len(y_basic)) * noise_level * np.std(y_basic)
y_enhanced_noisy = y_enhanced + np.random.randn(len(y_enhanced)) * noise_level * np.std(y_enhanced)

# === LASSO RECOVERY ===
print("\nAttempting L1-regularized recovery...")

def lasso_recovery(A, y, alpha_range=None):
    """Use Lasso to recover sparse prime indicator"""
    scaler = StandardScaler()
    A_scaled = scaler.fit_transform(A.T).T

    if alpha_range is None:
        # Use cross-validation to find optimal alpha
        model = LassoCV(cv=5, max_iter=10000, n_jobs=-1)
    else:
        model = Lasso(alpha=alpha_range, max_iter=10000)

    model.fit(A_scaled.T, np.zeros(A.shape[1]))  # This doesn't make sense, let me fix

    return model

# Actually, we need to reframe this problem correctly.
# The Lasso approach should be: given measurements y, recover x

def sparse_prime_recovery(A, y, true_x, alphas=[1e-5, 1e-4, 1e-3, 1e-2, 0.1]):
    """
    Attempt sparse recovery of prime indicator vector.

    y = A @ x  →  find sparse x given y and A

    This is underdetermined (more unknowns than equations),
    but L1 regularization exploits sparsity.
    """
    results = []

    for alpha in alphas:
        # Lasso: minimize ||y - Ax||² + α||x||₁
        # A has shape (measurements, unknowns), Lasso expects (samples, features)
        # So A (80 × 999) means 80 samples, 999 features
        model = Lasso(alpha=alpha, max_iter=20000, positive=True)  # primes are non-negative
        model.fit(A, y)  # A is (80, 999), y is (80,) - correct!

        x_recovered = model.coef_

        # Threshold to get binary prediction
        threshold = np.percentile(x_recovered, 100 * (1 - sum(true_x)/len(true_x)) - 5)
        x_pred = (x_recovered > threshold).astype(int)

        # Calculate metrics
        pred_primes = set(n+2 for n in range(len(x_pred)) if x_pred[n] > 0)
        actual_primes = set(n+2 for n in range(len(true_x)) if true_x[n] > 0)

        tp = len(pred_primes & actual_primes)
        fp = len(pred_primes - actual_primes)
        fn = len(actual_primes - pred_primes)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'alpha': alpha,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detected': len(pred_primes),
            'x_recovered': x_recovered
        })

        print(f"α={alpha:.0e}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, detected={len(pred_primes)}")

    return results

print("\n--- Basic Matrix (80 zeros, cos only) ---")
results_basic = sparse_prime_recovery(A_basic, y_basic_noisy, x_true)

print("\n--- Enhanced Matrix (80 zeros, cos+sin) ---")
results_enhanced = sparse_prime_recovery(A_enhanced, y_enhanced_noisy, x_true)

# === ITERATIVE REFINEMENT ===
print("\n[3] ITERATIVE REFINEMENT (Reweighted L1)")
print("=" * 70)

def reweighted_lasso(A, y, true_x, max_iter=5):
    """
    Iteratively Reweighted L1 (IRL1) for enhanced sparsity.
    At each iteration, reweight to penalize larger coefficients less.
    """
    x_est = np.ones(A.shape[1]) * 0.5
    epsilon = 1e-3

    for iteration in range(max_iter):
        # Compute weights: w[i] = 1 / (|x[i]| + ε)
        weights = 1.0 / (np.abs(x_est) + epsilon)

        # Scale columns of A by weights
        A_weighted = A / weights

        # Solve weighted Lasso
        model = Lasso(alpha=1e-3, max_iter=10000, positive=True)
        model.fit(A_weighted, y)  # A_weighted is (measurements, unknowns)

        x_est = model.coef_ / weights

        # Threshold
        threshold = np.percentile(x_est, 83)  # ~17% are primes
        x_pred = (x_est > threshold).astype(int)

        # Metrics
        pred_primes = set(n+2 for n in range(len(x_pred)) if x_pred[n] > 0)
        actual_primes = set(n+2 for n in range(len(true_x)) if true_x[n] > 0)

        tp = len(pred_primes & actual_primes)
        fp = len(pred_primes - actual_primes)
        fn = len(actual_primes - pred_primes)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Iteration {iteration+1}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    return x_est

print("\nReweighted L1 with enhanced matrix:")
x_reweighted = reweighted_lasso(A_enhanced, y_enhanced_noisy, x_true)

# Skip OMP - focus on Lasso comparison

# === COMPARE WITH DIRECT METHOD ===
print("\n[5] COMPARISON: LIDAR vs DIRECT METHOD")
print("=" * 70)

def direct_detection(N, num_zeros, zeros):
    """The original d74169 direct method"""
    gamma = zeros[:num_zeros]
    weights = 1.0 / np.sqrt(0.25 + gamma**2)

    scores = []
    for n in range(2, N+1):
        log_n = np.log(n)
        score = -2 * np.sum(np.cos(gamma * log_n) * weights) / log_n
        scores.append((n, score))

    # Adaptive threshold
    scores.sort(key=lambda x: -x[1])
    target_count = int(1.3 * N / np.log(N))
    detected = set(n for n, s in scores[:target_count])

    return detected

# Test direct method at different zero counts
print("\nDirect method accuracy by zero count:")
for nz in [40, 60, 80, 100, 126, 150]:
    if nz > len(ZEROS):
        continue
    detected = direct_detection(N, nz, ZEROS)

    tp = len(detected & true_primes)
    fp = len(detected - true_primes)
    fn = len(true_primes - detected)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  {nz:3d} zeros: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

# === SUPER-RESOLUTION TEST ===
print("\n[6] SUPER-RESOLUTION: CAN 80 ZEROS MATCH 126?")
print("=" * 70)

# The key question: can compressed sensing with 80 zeros match direct with 126?
direct_80 = direct_detection(N, 80, ZEROS)
direct_126 = direct_detection(N, 126, ZEROS)

def evaluate(detected, truth, name):
    tp = len(detected & truth)
    fp = len(detected - truth)
    fn = len(truth - detected)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
    print(f"{name}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
    return f1

print("\nBaseline comparison:")
f1_direct_80 = evaluate(direct_80, true_primes, "Direct (80 zeros)")
f1_direct_126 = evaluate(direct_126, true_primes, "Direct (126 zeros)")

# Best Lasso result
best_lasso = max(results_enhanced, key=lambda x: x['f1'])
print(f"Best Lasso (80 zeros): P={best_lasso['precision']:.3f}, R={best_lasso['recall']:.3f}, F1={best_lasso['f1']:.3f}")

# === FINAL SUMMARY ===
print("\n" + "=" * 70)
print("PROJECT LIDAR: SUMMARY")
print("=" * 70)

improvement = (best_lasso['f1'] - f1_direct_80) / f1_direct_80 * 100 if f1_direct_80 > 0 else 0
gap_closed = (best_lasso['f1'] - f1_direct_80) / (f1_direct_126 - f1_direct_80) * 100 if (f1_direct_126 - f1_direct_80) > 0 else 0

print(f"""
RESULTS:
- Target: N = {N} (168 primes)
- Zeros available: 80 (vs 126 needed for 100% direct)

Direct Method:
  80 zeros → F1 = {f1_direct_80:.3f}
  126 zeros → F1 = {f1_direct_126:.3f}

LIDAR (L1 Sparse Recovery):
  80 zeros → F1 = {best_lasso['f1']:.3f}

SUPER-RESOLUTION GAIN:
  Improvement over direct: {improvement:.1f}%
  Gap to 126-zero closed: {gap_closed:.1f}%

CONCLUSION:
{"SUCCESS: L1 regularization achieves super-resolution!" if best_lasso['f1'] > f1_direct_80 else "More tuning needed - try different matrix designs."}
""")

print("[@d74169] Project Lidar complete.")
