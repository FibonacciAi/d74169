#!/usr/bin/env python3
"""
d74169 Fibonacci-Riemann Connection Research
=============================================
Exploring whether φ, Fibonacci numbers, or related sequences
have special properties in the spectral prime landscape.

Research Questions:
1. Do Fibonacci primes have special spectral signatures?
2. Does φ appear in Riemann zero spacings?
3. Do Fibonacci numbers have distinctive d74169 scores?
4. Is there a φ-related pattern in prime fingerprints?
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

print("=" * 70)
print("@d74169 FIBONACCI-RIEMANN CONNECTION RESEARCH")
print("=" * 70)

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618033988749895
PSI = (1 - np.sqrt(5)) / 2  # ≈ -0.618033988749895

print(f"\nφ (Golden Ratio) = {PHI:.15f}")
print(f"ψ = 1 - φ = {PSI:.15f}")
print(f"φ² = φ + 1 = {PHI**2:.15f}")

# Load zeros - set ZEROS_PATH env var or update default path
import os
ZEROS_PATH = os.environ.get('ZEROS_PATH', '/Users/dimitristefanopoulos/d74169_tests/riemann_zeros_master_v2.npy')
ZEROS = np.load(ZEROS_PATH)

def sieve(n):
    s = [True] * (n+1)
    s[0] = s[1] = False
    for i in range(2, int(n**0.5)+1):
        if s[i]:
            for j in range(i*i, n+1, i):
                s[j] = False
    return set(i for i in range(n+1) if s[i])

def fibonacci_sequence(n_terms):
    """Generate first n Fibonacci numbers"""
    fibs = [1, 1]
    while len(fibs) < n_terms:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs

def is_fibonacci(n):
    """Check if n is a Fibonacci number"""
    # n is Fibonacci iff 5n² ± 4 is a perfect square
    def is_perfect_square(x):
        s = int(np.sqrt(x))
        return s * s == x
    return is_perfect_square(5 * n * n + 4) or is_perfect_square(5 * n * n - 4)

def d74169_score(n, num_zeros=50):
    """Compute d74169 interference score"""
    log_n = np.log(n)
    gamma = ZEROS[:num_zeros]
    return np.sum(np.cos(gamma * log_n) / np.sqrt(0.25 + gamma**2))

def spectral_fingerprint(n, num_zeros=20):
    """Individual zero contributions"""
    log_n = np.log(n)
    fp = []
    for i in range(num_zeros):
        gamma = ZEROS[i]
        fp.append(np.cos(gamma * log_n) / np.sqrt(0.25 + gamma**2))
        fp.append(np.sin(gamma * log_n) / np.sqrt(0.25 + gamma**2))
    return np.array(fp)

primes = sieve(100000)
fibs = fibonacci_sequence(50)

# === EXPERIMENT 1: Fibonacci Primes ===
print("\n" + "=" * 70)
print("[1] FIBONACCI PRIMES: SPECTRAL SIGNATURES")
print("=" * 70)

# Fibonacci primes: F_n where F_n is prime
fib_primes = [f for f in fibs if f in primes and f > 1]
print(f"\nFibonacci primes found: {fib_primes[:15]}...")

# Compare to random primes of similar size
random_primes = [p for p in sorted(primes) if p <= max(fib_primes)]
np.random.seed(42)
sample_random = np.random.choice(random_primes, min(len(fib_primes), len(random_primes)), replace=False)

# Compute scores
fib_prime_scores = [d74169_score(p) for p in fib_primes if p < 10000]
random_prime_scores = [d74169_score(p) for p in sample_random if p < 10000]

print(f"\nFibonacci prime scores (n={len(fib_prime_scores)}):")
print(f"  Mean: {np.mean(fib_prime_scores):.4f}")
print(f"  Std:  {np.std(fib_prime_scores):.4f}")

print(f"\nRandom prime scores (n={len(random_prime_scores)}):")
print(f"  Mean: {np.mean(random_prime_scores):.4f}")
print(f"  Std:  {np.std(random_prime_scores):.4f}")

# Fingerprint correlation within Fibonacci primes
fib_fps = [spectral_fingerprint(p) for p in fib_primes if p < 5000]
if len(fib_fps) >= 2:
    fib_corrs = []
    for i in range(len(fib_fps)):
        for j in range(i+1, len(fib_fps)):
            c, _ = pearsonr(fib_fps[i], fib_fps[j])
            if not np.isnan(c):
                fib_corrs.append(c)
    print(f"\nFibonacci prime intra-correlation: {np.mean(fib_corrs):.4f}")

# === EXPERIMENT 2: Golden Ratio in Zero Spacings ===
print("\n" + "=" * 70)
print("[2] GOLDEN RATIO IN RIEMANN ZERO SPACINGS")
print("=" * 70)

# Compute consecutive zero spacings
spacings = np.diff(ZEROS[:1000])
mean_spacing = np.mean(spacings)

print(f"\nMean zero spacing: {mean_spacing:.6f}")
print(f"φ = {PHI:.6f}")
print(f"Ratio mean_spacing/φ: {mean_spacing/PHI:.6f}")

# Look for φ in spacing ratios
spacing_ratios = spacings[1:] / spacings[:-1]
phi_matches = np.abs(spacing_ratios - PHI) < 0.1
phi_inv_matches = np.abs(spacing_ratios - 1/PHI) < 0.1

print(f"\nSpacing ratios close to φ (±0.1): {np.sum(phi_matches)} / {len(spacing_ratios)} ({100*np.mean(phi_matches):.1f}%)")
print(f"Spacing ratios close to 1/φ (±0.1): {np.sum(phi_inv_matches)} / {len(spacing_ratios)} ({100*np.mean(phi_inv_matches):.1f}%)")

# Check if φ appears more than random expectation
# Random uniform would give ~10% in any ±0.1 window
print(f"Expected by chance (~10% window): {len(spacing_ratios) * 0.1:.0f}")

# === EXPERIMENT 3: Fibonacci Numbers (All) ===
print("\n" + "=" * 70)
print("[3] ALL FIBONACCI NUMBERS: SPECTRAL PROPERTIES")
print("=" * 70)

# Score all Fibonacci numbers
fib_scores = [(f, d74169_score(f), f in primes) for f in fibs[2:30] if f > 1]

print("\nFibonacci number scores:")
print(f"{'F_n':<12} {'Score':<12} {'Prime?':<8}")
print("-" * 35)
for f, score, is_prime in fib_scores[:15]:
    marker = "✓" if is_prime else ""
    print(f"{f:<12} {score:<12.4f} {marker}")

# Compare Fibonacci vs non-Fibonacci of similar size
fib_set = set(fibs)
all_fib_scores = [d74169_score(f) for f in fibs[4:25]]  # F_5 to F_25
non_fib_scores = [d74169_score(n) for n in range(10, 1000) if n not in fib_set][:len(all_fib_scores)]

print(f"\nFibonacci number mean score: {np.mean(all_fib_scores):.4f}")
print(f"Non-Fibonacci mean score: {np.mean(non_fib_scores):.4f}")

# === EXPERIMENT 4: φ-Scaled Primes ===
print("\n" + "=" * 70)
print("[4] φ-SCALED PATTERNS IN PRIMES")
print("=" * 70)

# Check if primes at φ-related positions have special properties
# Test: p and floor(p*φ) or floor(p/φ)
prime_list = sorted([p for p in primes if p < 1000])

phi_pairs = []
for p in prime_list[:50]:
    p_phi = int(p * PHI)
    p_phi_inv = int(p / PHI)
    if p_phi in primes:
        phi_pairs.append((p, p_phi, 'φ'))
    if p_phi_inv in primes and p_phi_inv > 1:
        phi_pairs.append((p, p_phi_inv, '1/φ'))

print(f"\nPrime pairs at φ-ratio: {len(phi_pairs)}")
print("Examples:")
for p1, p2, ratio in phi_pairs[:10]:
    actual_ratio = p2/p1 if ratio == 'φ' else p1/p2
    print(f"  ({p1}, {p2}): ratio = {actual_ratio:.4f} (target {PHI if ratio=='φ' else 1/PHI:.4f})")

# Fingerprint correlation of φ-related prime pairs
if phi_pairs:
    phi_pair_corrs = []
    for p1, p2, _ in phi_pairs[:20]:
        fp1 = spectral_fingerprint(p1)
        fp2 = spectral_fingerprint(p2)
        c, _ = pearsonr(fp1, fp2)
        if not np.isnan(c):
            phi_pair_corrs.append(c)
    if phi_pair_corrs:
        print(f"\nφ-ratio prime pair fingerprint correlation: {np.mean(phi_pair_corrs):.4f}")

# === EXPERIMENT 5: Zeckendorf Representation ===
print("\n" + "=" * 70)
print("[5] ZECKENDORF REPRESENTATION OF PRIMES")
print("=" * 70)

def zeckendorf(n):
    """Return Zeckendorf representation (sum of non-consecutive Fibonacci numbers)"""
    fibs_local = [1, 2]
    while fibs_local[-1] < n:
        fibs_local.append(fibs_local[-1] + fibs_local[-2])

    rep = []
    remaining = n
    for f in reversed(fibs_local):
        if f <= remaining:
            rep.append(f)
            remaining -= f
        if remaining == 0:
            break
    return rep

# Compute Zeckendorf representation length for primes vs composites
prime_zeck_lens = [len(zeckendorf(p)) for p in range(2, 200) if p in primes]
comp_zeck_lens = [len(zeckendorf(c)) for c in range(4, 200) if c not in primes]

print(f"\nZeckendorf representation length:")
print(f"  Primes (2-200): mean = {np.mean(prime_zeck_lens):.3f}")
print(f"  Composites (4-200): mean = {np.mean(comp_zeck_lens):.3f}")

# Is there correlation between Zeckendorf length and d74169 score?
nums = list(range(2, 200))
zeck_lens = [len(zeckendorf(n)) for n in nums]
scores = [d74169_score(n, num_zeros=14) for n in nums]

zeck_score_corr, _ = pearsonr(zeck_lens, scores)
print(f"\nCorrelation (Zeckendorf length vs d74169 score): {zeck_score_corr:.4f}")

# === EXPERIMENT 6: φ in Phase Relationships ===
print("\n" + "=" * 70)
print("[6] φ IN SPECTRAL PHASE RELATIONSHIPS")
print("=" * 70)

# Check if phase differences between consecutive primes relate to φ
consecutive_primes = [(prime_list[i], prime_list[i+1]) for i in range(min(50, len(prime_list)-1))]

phase_ratios_gamma1 = []
for p1, p2 in consecutive_primes:
    phase1 = (ZEROS[0] * np.log(p1)) % (2 * np.pi)
    phase2 = (ZEROS[0] * np.log(p2)) % (2 * np.pi)
    if phase1 > 0:
        phase_ratios_gamma1.append(phase2 / phase1)

# How often does phase ratio approximate φ?
phi_phase_matches = [r for r in phase_ratios_gamma1 if 1.5 < r < 1.75]
print(f"\nPhase ratios (γ₁) near φ [1.5, 1.75]: {len(phi_phase_matches)} / {len(phase_ratios_gamma1)}")

# Check log ratio
log_ratios = [np.log(p2/p1) for p1, p2 in consecutive_primes]
log_phi = np.log(PHI)
print(f"\nlog(φ) = {log_phi:.6f}")
print(f"Mean log(p_{{n+1}}/p_n): {np.mean(log_ratios):.6f}")

# === EXPERIMENT 7: Fibonacci Index Primes ===
print("\n" + "=" * 70)
print("[7] FIBONACCI-INDEX PRIMES: F_p WHERE p IS PRIME")
print("=" * 70)

# F_p for prime p
prime_indices = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
fib_at_prime_index = []
for p in prime_indices:
    if p < len(fibs):
        fib_at_prime_index.append((p, fibs[p-1]))  # F_p (1-indexed traditionally)

print("\nF_p for prime p:")
print(f"{'p':<6} {'F_p':<15} {'F_p prime?':<12} {'Score':<10}")
print("-" * 45)
for p, f_p in fib_at_prime_index:
    is_prime = f_p in primes
    score = d74169_score(f_p) if f_p < 50000 else float('nan')
    print(f"{p:<6} {f_p:<15} {'Yes' if is_prime else 'No':<12} {score:<10.4f}")

# === EXPERIMENT 8: φ^n and Prime Distribution ===
print("\n" + "=" * 70)
print("[8] φ^n SEQUENCE AND PRIME PROXIMITY")
print("=" * 70)

# Check how close φ^n gets to primes
phi_powers = [int(PHI**n) for n in range(2, 25)]
print("\nφ^n and nearest primes:")
print(f"{'n':<4} {'φ^n':<12} {'Nearest Prime':<15} {'Distance':<10}")
print("-" * 45)

for n in range(2, 20):
    phi_n = int(PHI**n)
    # Find nearest prime
    for d in range(100):
        if phi_n + d in primes:
            nearest = phi_n + d
            dist = d
            break
        if phi_n - d in primes and phi_n - d > 1:
            nearest = phi_n - d
            dist = -d
            break
    else:
        nearest = "?"
        dist = "?"
    print(f"{n:<4} {phi_n:<12} {nearest:<15} {dist:<10}")

# === VISUALIZATION ===
print("\n" + "=" * 70)
print("[9] GENERATING VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor('#0a0e1a')

for ax in axes.flat:
    ax.set_facecolor('#131a2e')
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#2d3a5a')

fig.suptitle('@d74169 Fibonacci-Riemann Connections', fontsize=16, color='#c4b5fd', fontweight='bold')

# Panel 1: Fibonacci vs Random Prime Scores
ax1 = axes[0, 0]
if fib_prime_scores and random_prime_scores:
    ax1.hist(fib_prime_scores, bins=10, alpha=0.7, color='#fbbf24', label='Fibonacci Primes', density=True)
    ax1.hist(random_prime_scores, bins=15, alpha=0.7, color='#06b6d4', label='Random Primes', density=True)
ax1.set_xlabel('d74169 Score', color='#94a3b8')
ax1.set_ylabel('Density', color='#94a3b8')
ax1.set_title('Fibonacci vs Random Prime Scores', color='white')
ax1.legend(facecolor='#131a2e', edgecolor='#2d3a5a', labelcolor='#94a3b8')

# Panel 2: Zero Spacing Ratios
ax2 = axes[0, 1]
ax2.hist(spacing_ratios[:500], bins=50, alpha=0.7, color='#8b5cf6', edgecolor='white', linewidth=0.5)
ax2.axvline(PHI, color='#fbbf24', linewidth=2, linestyle='--', label=f'φ = {PHI:.3f}')
ax2.axvline(1/PHI, color='#10b981', linewidth=2, linestyle='--', label=f'1/φ = {1/PHI:.3f}')
ax2.set_xlabel('Spacing Ratio γ_{n+1}/γ_n', color='#94a3b8')
ax2.set_ylabel('Count', color='#94a3b8')
ax2.set_title('Zero Spacing Ratios', color='white')
ax2.legend(facecolor='#131a2e', edgecolor='#2d3a5a', labelcolor='#94a3b8')

# Panel 3: Fibonacci Number Scores
ax3 = axes[0, 2]
fib_nums = fibs[4:20]
fib_scores_plot = [d74169_score(f) for f in fib_nums]
colors = ['#fbbf24' if f in primes else '#ef4444' for f in fib_nums]
ax3.bar(range(len(fib_nums)), fib_scores_plot, color=colors, alpha=0.8)
ax3.set_xticks(range(len(fib_nums)))
ax3.set_xticklabels([str(f) for f in fib_nums], rotation=45, ha='right', fontsize=8)
ax3.set_xlabel('Fibonacci Number', color='#94a3b8')
ax3.set_ylabel('d74169 Score', color='#94a3b8')
ax3.set_title('Fibonacci Number Scores (Gold=Prime)', color='white')

# Panel 4: Zeckendorf Length vs Score
ax4 = axes[1, 0]
prime_mask = [n in primes for n in nums]
ax4.scatter([zeck_lens[i] for i in range(len(nums)) if prime_mask[i]],
            [scores[i] for i in range(len(nums)) if prime_mask[i]],
            c='#10b981', alpha=0.6, s=30, label='Primes')
ax4.scatter([zeck_lens[i] for i in range(len(nums)) if not prime_mask[i]],
            [scores[i] for i in range(len(nums)) if not prime_mask[i]],
            c='#ef4444', alpha=0.4, s=20, label='Composites')
ax4.set_xlabel('Zeckendorf Length', color='#94a3b8')
ax4.set_ylabel('d74169 Score', color='#94a3b8')
ax4.set_title(f'Zeckendorf vs Score (r={zeck_score_corr:.3f})', color='white')
ax4.legend(facecolor='#131a2e', edgecolor='#2d3a5a', labelcolor='#94a3b8')

# Panel 5: φ^n proximity to primes
ax5 = axes[1, 1]
ns = list(range(2, 20))
phi_ns = [PHI**n for n in ns]
distances = []
for phi_n in phi_ns:
    phi_n_int = int(phi_n)
    for d in range(1000):
        if phi_n_int + d in primes:
            distances.append(d)
            break
        if phi_n_int - d in primes and phi_n_int - d > 1:
            distances.append(d)
            break
    else:
        distances.append(100)

ax5.bar(ns, distances, color='#8b5cf6', alpha=0.8)
ax5.set_xlabel('n', color='#94a3b8')
ax5.set_ylabel('Distance to Nearest Prime', color='#94a3b8')
ax5.set_title('φⁿ Distance to Nearest Prime', color='white')

# Panel 6: Summary Statistics
ax6 = axes[1, 2]
ax6.axis('off')

summary_text = f"""
FIBONACCI-RIEMANN FINDINGS

φ (Golden Ratio) = {PHI:.6f}

Zero Spacing Analysis:
• Mean spacing: {mean_spacing:.4f}
• Ratios near φ: {100*np.mean(phi_matches):.1f}%
• Ratios near 1/φ: {100*np.mean(phi_inv_matches):.1f}%

Fibonacci Primes:
• Found: {len(fib_primes)} in first 50 Fibonacci
• Mean score: {np.mean(fib_prime_scores):.4f}
• Intra-correlation: {np.mean(fib_corrs) if fib_corrs else 'N/A':.4f}

Zeckendorf Analysis:
• Score correlation: {zeck_score_corr:.4f}
• Prime mean length: {np.mean(prime_zeck_lens):.2f}
• Composite mean length: {np.mean(comp_zeck_lens):.2f}

φ-Related Prime Pairs: {len(phi_pairs)}
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace', color='#94a3b8')

plt.tight_layout(rect=[0, 0, 1, 0.95])
output = os.path.join(os.path.dirname(__file__), 'project_fibonacci.png')
plt.savefig(output, dpi=150, facecolor='#0a0e1a', bbox_inches='tight')
print(f"\nSaved: {output}")

# === KEY FINDINGS ===
print("\n" + "=" * 70)
print("KEY FINDINGS: FIBONACCI-RIEMANN CONNECTIONS")
print("=" * 70)

print(f"""
1. FIBONACCI PRIMES
   • {len(fib_primes)} Fibonacci primes found: {fib_primes[:8]}...
   • Mean score: {np.mean(fib_prime_scores):.4f} vs random primes: {np.mean(random_prime_scores):.4f}
   • Intra-class correlation: {np.mean(fib_corrs) if fib_corrs else 'N/A':.4f}

2. GOLDEN RATIO IN ZERO SPACINGS
   • φ appears in {100*np.mean(phi_matches):.1f}% of spacing ratios (vs ~10% random)
   • 1/φ appears in {100*np.mean(phi_inv_matches):.1f}% of spacing ratios
   • {"SIGNAL DETECTED" if np.mean(phi_matches) > 0.15 else "No significant signal"}

3. ZECKENDORF REPRESENTATION
   • Correlation with d74169 score: {zeck_score_corr:.4f}
   • {"WEAK SIGNAL" if abs(zeck_score_corr) > 0.1 else "No signal"}

4. φ-RATIO PRIME PAIRS
   • Found {len(phi_pairs)} pairs where p₂/p₁ ≈ φ or 1/φ
   • Fingerprint correlation: {np.mean(phi_pair_corrs) if phi_pair_corrs else 'N/A':.4f}

5. OVERALL ASSESSMENT
   The Fibonacci connection appears {"SIGNIFICANT" if np.mean(phi_matches) > 0.15 or abs(zeck_score_corr) > 0.15 else "WEAK but worth further investigation"}.
""")

print("\n[@d74169] Fibonacci-Riemann research complete.")
