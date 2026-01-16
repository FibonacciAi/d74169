#!/usr/bin/env python3
"""
PROJECT HIGHWAY: Targeted Resonance at Scale
==============================================
Goal: Scan the Δ=2310 highway at range n=10^6 (one million)
      If the correlation remains r≈1.0, we've discovered a way to
      find massive primes using low-frequency (few zeros) data.

Mechanism: Primes separated by primorial P_k# share phase resonance:
           γ × log(p₂/p₁) = γ × log(1 + 2310/p₁) ≈ 0 for large p₁

This means distant primes on the same "highway" look IDENTICAL spectrally!
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sympy import isprime, primerange
import time

print("=" * 70)
print("PROJECT HIGHWAY: Targeted Resonance at Scale")
print("=" * 70)

# Load zeros
ZEROS = np.load('/Users/dimitristefanopoulos/d74169_tests/riemann_zeros_master_v2.npy')

# === SPECTRAL FINGERPRINT ===
def spectral_fingerprint(n, zeros, num_zeros=100):
    """
    Multi-scale spectral fingerprint for integer n.
    Uses first num_zeros Riemann zeros.
    """
    log_n = np.log(n)
    gamma = zeros[:num_zeros]
    weights = 1.0 / np.sqrt(0.25 + gamma**2)

    # Phase-based fingerprint (more sensitive)
    phases = gamma * log_n
    cos_vals = np.cos(phases) * weights
    sin_vals = np.sin(phases) * weights

    return np.concatenate([cos_vals, sin_vals])

def spectral_score(n, zeros, num_zeros=50):
    """The d74169 score"""
    log_n = np.log(n)
    gamma = zeros[:num_zeros]
    weights = 1.0 / np.sqrt(0.25 + gamma**2)
    return -2 * np.sum(np.cos(gamma * log_n) * weights) / log_n

# === THE PRIMORIAL HIGHWAYS ===
print("\n[1] PRIMORIAL HIGHWAY DEFINITIONS")
print("=" * 70)

PRIMORIALS = {
    'P2': 2,           # 2
    'P3': 6,           # 2×3
    'P5': 30,          # 2×3×5
    'P7': 210,         # 2×3×5×7
    'P11': 2310,       # 2×3×5×7×11
    'P13': 30030,      # 2×3×5×7×11×13
}

for name, val in PRIMORIALS.items():
    print(f"  {name}# = {val}")

# === SCAN RANGES ===
print("\n[2] HIGHWAY CORRELATION AT DIFFERENT SCALES")
print("=" * 70)

def find_highway_pairs(delta, start, end, max_pairs=100):
    """Find prime pairs (p, p+delta) in range [start, end]"""
    pairs = []
    # Use sympy's prime generator for efficiency
    for p in primerange(start, end):
        if p + delta <= end and isprime(p + delta):
            pairs.append((p, p + delta))
            if len(pairs) >= max_pairs:
                break
    return pairs

def compute_highway_correlation(delta, start, end, num_zeros=50, max_pairs=50):
    """
    Compute fingerprint correlation for prime pairs on the Δ highway.
    """
    t0 = time.time()
    pairs = find_highway_pairs(delta, start, end, max_pairs)
    t1 = time.time()

    if len(pairs) < 5:
        return None, 0, t1-t0

    correlations = []
    for p1, p2 in pairs:
        fp1 = spectral_fingerprint(p1, ZEROS, num_zeros)
        fp2 = spectral_fingerprint(p2, ZEROS, num_zeros)
        r, _ = pearsonr(fp1, fp2)
        if not np.isnan(r):
            correlations.append(r)

    if len(correlations) < 5:
        return None, len(pairs), t1-t0

    return np.mean(correlations), len(pairs), t1-t0

# Test at increasing scales
print(f"\n{'Scale':<12} {'Δ=2':<12} {'Δ=6':<12} {'Δ=30':<12} {'Δ=210':<12} {'Δ=2310':<12}")
print("-" * 72)

SCALES = [
    (1000, 10000, "10^3-10^4"),
    (10000, 100000, "10^4-10^5"),
    (100000, 500000, "10^5-5×10^5"),
    (500000, 1000000, "5×10^5-10^6"),
]

results_by_scale = {}

for start, end, label in SCALES:
    row = [label]
    results_by_scale[label] = {}

    for delta in [2, 6, 30, 210, 2310]:
        corr, pairs, elapsed = compute_highway_correlation(delta, start, end, num_zeros=50, max_pairs=30)
        if corr is not None:
            row.append(f"{corr:.4f}({pairs})")
            results_by_scale[label][delta] = corr
        else:
            row.append(f"N/A")
            results_by_scale[label][delta] = None

    print(f"{row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12} {row[5]:<12}")

# === THE BIG TEST: Δ=2310 AT ONE MILLION ===
print("\n[3] THE BIG TEST: Δ=2310 HIGHWAY AT n=10^6")
print("=" * 70)

print("\nSearching for prime pairs (p, p+2310) near n=1,000,000...")

# Find pairs around 10^6
big_pairs = find_highway_pairs(2310, 900000, 1100000, max_pairs=50)
print(f"Found {len(big_pairs)} pairs in range [900000, 1100000]")

if big_pairs:
    print(f"\nFirst 5 pairs:")
    for p1, p2 in big_pairs[:5]:
        print(f"  ({p1}, {p2})")

    # Compute correlations
    correlations_2310 = []
    scores_p1 = []
    scores_p2 = []

    for p1, p2 in big_pairs:
        fp1 = spectral_fingerprint(p1, ZEROS, 100)
        fp2 = spectral_fingerprint(p2, ZEROS, 100)
        r, _ = pearsonr(fp1, fp2)
        correlations_2310.append(r)

        scores_p1.append(spectral_score(p1, ZEROS, 50))
        scores_p2.append(spectral_score(p2, ZEROS, 50))

    print(f"\nΔ=2310 Highway Statistics at n≈10^6:")
    print(f"  Mean correlation: {np.mean(correlations_2310):.6f}")
    print(f"  Std correlation:  {np.std(correlations_2310):.6f}")
    print(f"  Min correlation:  {np.min(correlations_2310):.6f}")
    print(f"  Max correlation:  {np.max(correlations_2310):.6f}")

    # Score correlation
    score_corr, _ = pearsonr(scores_p1, scores_p2)
    print(f"\n  Score correlation (p₁ vs p₂): {score_corr:.6f}")

# === PHASE ANALYSIS ===
print("\n[4] PHASE DRIFT ANALYSIS")
print("=" * 70)

def analyze_phase_drift(p1, p2, num_zeros=20):
    """
    Analyze how phases drift between p1 and p2.
    Δφ_j = γ_j × log(p₂/p₁) = γ_j × log(1 + (p₂-p₁)/p₁)
    """
    delta = p2 - p1
    log_ratio = np.log(p2 / p1)

    gamma = ZEROS[:num_zeros]
    phase_drifts = gamma * log_ratio

    return phase_drifts

if big_pairs:
    print("\nPhase drift for first pair on Δ=2310 highway:")
    p1, p2 = big_pairs[0]
    drifts = analyze_phase_drift(p1, p2, 20)

    print(f"  Pair: ({p1}, {p2})")
    print(f"  Δ/p₁ = {2310/p1:.6f}")
    print(f"  log(p₂/p₁) = {np.log(p2/p1):.6f}")
    print(f"\n  Zero    γ        Δφ (rad)    Δφ (cycles)")
    print("  " + "-" * 45)

    for j in range(10):
        drift_rad = drifts[j]
        drift_cycles = drift_rad / (2 * np.pi)
        print(f"  {j+1:2d}    {ZEROS[j]:7.2f}    {drift_rad:8.5f}    {drift_cycles:8.5f}")

    # Compare with small primes
    print("\n  Compare with small pair (11, 11+2310=2321):")
    small_drifts = analyze_phase_drift(11, 2321, 20)
    print(f"  Δ/p₁ = {2310/11:.4f}")
    print(f"  log(p₂/p₁) = {np.log(2321/11):.4f}")
    print(f"  Phase drift for γ₁: {small_drifts[0]:.4f} rad = {small_drifts[0]/(2*np.pi):.4f} cycles")

# === LOW-FREQUENCY DETECTION STRATEGY ===
print("\n[5] LOW-FREQUENCY PRIME DETECTION STRATEGY")
print("=" * 70)

def highway_prime_detector(known_prime, delta, candidates, num_zeros=20):
    """
    Given a known prime p₁, detect if p₁+delta is prime
    by comparing spectral fingerprints.

    The hypothesis: if r(p₁, p₂) > threshold, then p₂ is likely prime.
    """
    fp_known = spectral_fingerprint(known_prime, ZEROS, num_zeros)

    detections = []
    for candidate in candidates:
        fp_cand = spectral_fingerprint(candidate, ZEROS, num_zeros)
        r, _ = pearsonr(fp_known, fp_cand)

        # Also check score
        score = spectral_score(candidate, ZEROS, num_zeros)

        detections.append({
            'n': candidate,
            'correlation': r,
            'score': score,
            'is_prime': isprime(candidate)
        })

    return detections

# Test: given a prime near 10^6, can we detect its highway partner?
if big_pairs:
    test_prime = big_pairs[0][0]
    print(f"\nTest: Starting from known prime p = {test_prime}")

    # Check candidates around p + 2310
    target = test_prime + 2310
    candidates = list(range(target - 10, target + 11))

    detections = highway_prime_detector(test_prime, 2310, candidates, num_zeros=50)

    print(f"\nCandidates near {target}:")
    print(f"{'n':<12} {'Correlation':<12} {'Score':<12} {'Prime?':<8}")
    print("-" * 44)

    for d in detections:
        prime_marker = "YES" if d['is_prime'] else "no"
        print(f"{d['n']:<12} {d['correlation']:<12.6f} {d['score']:<12.4f} {prime_marker:<8}")

    # Can correlation alone identify the prime?
    primes_in_range = [d for d in detections if d['is_prime']]
    non_primes = [d for d in detections if not d['is_prime']]

    if primes_in_range and non_primes:
        avg_prime_corr = np.mean([d['correlation'] for d in primes_in_range])
        avg_non_prime_corr = np.mean([d['correlation'] for d in non_primes])

        print(f"\nPrime avg correlation: {avg_prime_corr:.6f}")
        print(f"Non-prime avg corr:    {avg_non_prime_corr:.6f}")
        print(f"Separation:            {avg_prime_corr - avg_non_prime_corr:.6f}")

# === VISUALIZATION ===
print("\n[6] VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#0a0e1a')

for ax in axes.flat:
    ax.set_facecolor('#131a2e')
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#2d3a5a')

fig.suptitle('PROJECT HIGHWAY: Δ=2310 Resonance at Scale', fontsize=16, color='#c4b5fd', fontweight='bold')

# Panel 1: Correlation vs Scale
ax1 = axes[0, 0]
scales_x = [1, 2, 3, 4]  # 10^3, 10^4, 10^5, 10^6 ranges
scale_labels = ['10³-10⁴', '10⁴-10⁵', '10⁵-5×10⁵', '5×10⁵-10⁶']

for delta, color, label in [(2, '#ef4444', 'Δ=2'), (30, '#10b981', 'Δ=30'), (2310, '#8b5cf6', 'Δ=2310')]:
    y_vals = []
    for scale_label in ['10^3-10^4', '10^4-10^5', '10^5-5×10^5', '5×10^5-10^6']:
        if scale_label in results_by_scale and delta in results_by_scale[scale_label]:
            val = results_by_scale[scale_label][delta]
            y_vals.append(val if val is not None else np.nan)
        else:
            y_vals.append(np.nan)
    ax1.plot(scales_x, y_vals, 'o-', color=color, linewidth=2, markersize=8, label=label)

ax1.axhline(0.999, color='#fbbf24', linestyle='--', alpha=0.5, label='r=0.999')
ax1.set_xticks(scales_x)
ax1.set_xticklabels(scale_labels, color='#94a3b8')
ax1.set_ylabel('Fingerprint Correlation', color='#94a3b8')
ax1.set_title('Highway Correlation vs Scale', color='white')
ax1.legend(facecolor='#131a2e', edgecolor='#2d3a5a', labelcolor='#94a3b8')
ax1.set_ylim([0.98, 1.001])

# Panel 2: Phase drift distribution
ax2 = axes[0, 1]
if big_pairs:
    all_drifts = []
    for p1, p2 in big_pairs[:20]:
        drifts = analyze_phase_drift(p1, p2, 50)
        all_drifts.extend(drifts)

    ax2.hist(np.array(all_drifts) % (2*np.pi), bins=50, color='#06b6d4', alpha=0.7, edgecolor='white')
    ax2.axvline(np.pi, color='#ef4444', linewidth=2, linestyle='--', label='π')
    ax2.set_xlabel('Phase Drift (mod 2π)', color='#94a3b8')
    ax2.set_ylabel('Count', color='#94a3b8')
    ax2.set_title('Phase Drift Distribution at n≈10⁶', color='white')
    ax2.legend(facecolor='#131a2e', edgecolor='#2d3a5a', labelcolor='#94a3b8')

# Panel 3: Score correlation p1 vs p2
ax3 = axes[1, 0]
if big_pairs and len(scores_p1) > 5:
    ax3.scatter(scores_p1, scores_p2, c='#10b981', alpha=0.6, s=50)
    ax3.plot([min(scores_p1), max(scores_p1)], [min(scores_p1), max(scores_p1)],
             'r--', linewidth=2, label=f'r = {score_corr:.4f}')
    ax3.set_xlabel('Score(p₁)', color='#94a3b8')
    ax3.set_ylabel('Score(p₂ = p₁ + 2310)', color='#94a3b8')
    ax3.set_title('Score Correlation on Δ=2310 Highway', color='white')
    ax3.legend(facecolor='#131a2e', edgecolor='#2d3a5a', labelcolor='#94a3b8')

# Panel 4: Correlation histogram
ax4 = axes[1, 1]
if big_pairs:
    ax4.hist(correlations_2310, bins=20, color='#8b5cf6', alpha=0.7, edgecolor='white')
    ax4.axvline(np.mean(correlations_2310), color='#fbbf24', linewidth=2,
                label=f'Mean = {np.mean(correlations_2310):.4f}')
    ax4.set_xlabel('Fingerprint Correlation', color='#94a3b8')
    ax4.set_ylabel('Count', color='#94a3b8')
    ax4.set_title('Δ=2310 Correlation Distribution at n≈10⁶', color='white')
    ax4.legend(facecolor='#131a2e', edgecolor='#2d3a5a', labelcolor='#94a3b8')

plt.tight_layout(rect=[0, 0, 1, 0.95])
output = '/private/tmp/d74169_repo/research/project_highway.png'
plt.savefig(output, dpi=150, facecolor='#0a0e1a', bbox_inches='tight')
print(f"\nSaved: {output}")

# === FINAL SUMMARY ===
print("\n" + "=" * 70)
print("PROJECT HIGHWAY: SUMMARY")
print("=" * 70)

if big_pairs:
    mean_corr = np.mean(correlations_2310)
    persistence = "YES" if mean_corr > 0.99 else "PARTIAL" if mean_corr > 0.9 else "NO"

    print(f"""
RESULTS:

Target: Δ=2310 highway at n ≈ 10⁶
Pairs found: {len(big_pairs)}
Mean fingerprint correlation: {mean_corr:.6f}

CORRELATION PERSISTENCE: {persistence}

IMPLICATIONS:
""")

    if mean_corr > 0.99:
        print("""
BREAKTHROUGH: The highway resonance PERSISTS at million-scale!

This means:
1. Primes separated by 2310 remain SPECTRALLY INDISTINGUISHABLE
2. We can potentially find massive primes using only low-frequency data
3. The primorial structure creates "tunnels" through prime space

PRACTICAL APPLICATION:
Given a known large prime p, check if p + 2310 is prime by:
- Computing fingerprint correlation (should be ~1.0 for primes)
- If correlation drops significantly, p + 2310 is composite
""")
    else:
        print(f"""
The correlation decays from ~1.0 to ~{mean_corr:.3f} at scale 10⁶.

This indicates:
1. Phase drift accumulates: Δφ = γ × log(1 + 2310/p) becomes significant
2. The V1 fingerprint (sum-based) may need refinement for large scales
3. Consider using V2 fingerprint or phase-normalized methods
""")

print("\n[@d74169] Project Highway complete.")
