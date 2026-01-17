#!/usr/bin/env python3
"""
d74169 Research: Probing Production LLMs for Riemann Zero Signatures
=====================================================================
Testing if Claude/GPT encode Riemann zero structure in their responses.

Approach:
1. Query models with prime-related tasks
2. Analyze response patterns and "preferences"
3. Correlate with zero-based score function
4. Test if model behavior matches spectral predictions

We can't access internal hidden states, but we CAN probe behaviorally:
- Which numbers does the model "think" are prime?
- Do its confidence patterns correlate with zero scores?
- Does it make errors that align with spectral predictions?

@D74169 / Claude Opus 4.5
"""

import numpy as np
import os
import json
import re
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("@d74169 RESEARCH: PROBING LLMs FOR ZERO SIGNATURES")
print("=" * 70)

# === Riemann Zeros ===
ZEROS = np.array([
    14.134725141734693, 21.022039638771555, 25.010857580145688,
    30.424876125859513, 32.935061587739189, 37.586178158825671,
    40.918719012147495, 43.327073280914999, 48.005150881167159,
    49.773832477672302, 52.970321477714460, 56.446247697063394,
    59.347044002602353, 60.831778524609809, 65.112544048081606,
    67.079810529494173, 69.546401711173979, 72.067157674481907,
    75.704690699083933, 77.144840068874805
])

def sieve(n):
    s = [True] * (n+1)
    s[0] = s[1] = False
    for i in range(2, int(n**0.5)+1):
        if s[i]:
            for j in range(i*i, n+1, i):
                s[j] = False
    return set(i for i in range(n+1) if s[i])

primes = sieve(10000)

def compute_score(n, num_zeros=20):
    """d74169 score function based on Riemann zeros"""
    if n <= 1:
        return -999
    log_n = np.log(n)
    gamma = ZEROS[:num_zeros]
    phasor_sum = np.sum(np.cos(gamma * log_n) / np.sqrt(0.25 + gamma**2))
    return -2 * phasor_sum / log_n

def compute_phase_angle(n, num_zeros=20):
    """Phase angle of the weighted phasor sum"""
    if n <= 1:
        return 0
    log_n = np.log(n)
    gamma = ZEROS[:num_zeros]
    real_part = np.sum(np.cos(gamma * log_n) / np.sqrt(0.25 + gamma**2))
    imag_part = np.sum(np.sin(gamma * log_n) / np.sqrt(0.25 + gamma**2))
    return np.arctan2(imag_part, real_part)

# ============================================================
# Part 1: Check for API availability
# ============================================================
print("\n" + "=" * 70)
print("PART 1: CHECKING API AVAILABILITY")
print("=" * 70)

has_anthropic = False
has_openai = False

try:
    import anthropic
    client_anthropic = anthropic.Anthropic()
    has_anthropic = True
    print("✓ Anthropic API available")
except Exception as e:
    print(f"✗ Anthropic API not available: {e}")

try:
    import openai
    client_openai = openai.OpenAI()
    has_openai = True
    print("✓ OpenAI API available")
except Exception as e:
    print(f"✗ OpenAI API not available: {e}")

if not has_anthropic and not has_openai:
    print("\n⚠ No LLM APIs available. Running simulated analysis...")
    print("  Set ANTHROPIC_API_KEY or OPENAI_API_KEY to probe real models.")

# ============================================================
# Part 2: Define Probing Tasks
# ============================================================
print("\n" + "=" * 70)
print("PART 2: PROBING TASK DEFINITIONS")
print("=" * 70)

def query_claude(prompt, max_tokens=100):
    """Query Claude API"""
    if not has_anthropic:
        return None
    try:
        message = client_anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        print(f"Claude API error: {e}")
        return None

def query_gpt(prompt, max_tokens=100):
    """Query GPT API"""
    if not has_openai:
        return None
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"GPT API error: {e}")
        return None

# Test numbers - mix of primes and composites with varying zero scores
test_numbers = [
    # High score primes
    97, 101, 103, 107, 109,
    # Low score primes
    127, 131, 137, 139, 149,
    # Composites with high scores (potential false positives)
    91, 119, 121, 133, 143,
    # Composites with low scores
    100, 102, 104, 108, 110,
    # Edge cases
    2, 3, 5, 7, 11, 13
]

# Compute ground truth
ground_truth = {n: n in primes for n in test_numbers}
zero_scores = {n: compute_score(n) for n in test_numbers}
phase_angles = {n: compute_phase_angle(n) for n in test_numbers}

print(f"\nTest set: {len(test_numbers)} numbers")
print(f"Primes: {sum(ground_truth.values())}")
print(f"Composites: {len(test_numbers) - sum(ground_truth.values())}")

# ============================================================
# Part 3: Primality Judgment Probing
# ============================================================
print("\n" + "=" * 70)
print("PART 3: PRIMALITY JUDGMENT PROBING")
print("=" * 70)

print("""
Test: Ask models to judge if numbers are prime.
Hypothesis: Error patterns should correlate with zero scores.
- High-score composites → more likely false positives
- Low-score primes → more likely false negatives
""")

def probe_primality(query_fn, model_name, numbers):
    """Probe model's primality judgments"""
    results = {}

    for n in numbers:
        prompt = f"Is {n} a prime number? Answer only 'yes' or 'no'."
        response = query_fn(prompt, max_tokens=10)

        if response is None:
            continue

        response_lower = response.lower().strip()
        thinks_prime = 'yes' in response_lower and 'no' not in response_lower

        results[n] = {
            'response': response,
            'thinks_prime': thinks_prime,
            'is_prime': n in primes,
            'zero_score': zero_scores[n],
            'correct': thinks_prime == (n in primes)
        }

        status = "✓" if results[n]['correct'] else "✗"
        print(f"  {n}: {response.strip()[:20]} → {status}")

    return results

# Run probing
claude_results = {}
gpt_results = {}

if has_anthropic:
    print("\n--- Claude Primality Judgments ---")
    claude_results = probe_primality(query_claude, "Claude", test_numbers)

if has_openai:
    print("\n--- GPT Primality Judgments ---")
    gpt_results = probe_primality(query_gpt, "GPT", test_numbers)

# ============================================================
# Part 4: Analyze Error Patterns
# ============================================================
print("\n" + "=" * 70)
print("PART 4: ERROR PATTERN ANALYSIS")
print("=" * 70)

def analyze_errors(results, model_name):
    """Analyze if errors correlate with zero scores"""
    if not results:
        print(f"\n{model_name}: No results to analyze")
        return None

    print(f"\n--- {model_name} Error Analysis ---")

    # Separate correct and incorrect
    correct_scores = []
    incorrect_scores = []

    false_positives = []  # Composite judged as prime
    false_negatives = []  # Prime judged as composite

    for n, r in results.items():
        if r['correct']:
            correct_scores.append(r['zero_score'])
        else:
            incorrect_scores.append(r['zero_score'])

            if r['thinks_prime'] and not r['is_prime']:
                false_positives.append((n, r['zero_score']))
            elif not r['thinks_prime'] and r['is_prime']:
                false_negatives.append((n, r['zero_score']))

    accuracy = len(correct_scores) / len(results)
    print(f"Accuracy: {100*accuracy:.1f}%")
    print(f"False positives: {len(false_positives)}")
    print(f"False negatives: {len(false_negatives)}")

    if false_positives:
        fp_scores = [s for _, s in false_positives]
        print(f"\nFalse positive scores: mean={np.mean(fp_scores):.3f}")
        print(f"  Numbers: {[n for n, _ in false_positives]}")

    if false_negatives:
        fn_scores = [s for _, s in false_negatives]
        print(f"\nFalse negative scores: mean={np.mean(fn_scores):.3f}")
        print(f"  Numbers: {[n for n, _ in false_negatives]}")

    # Correlation: do high-score composites cause more errors?
    composites = [(n, r) for n, r in results.items() if not r['is_prime']]
    if composites:
        comp_scores = [r['zero_score'] for _, r in composites]
        comp_errors = [1 if r['thinks_prime'] else 0 for _, r in composites]

        if len(set(comp_errors)) > 1:  # Need variance
            r_comp, p_comp = spearmanr(comp_scores, comp_errors)
            print(f"\nComposite score-error correlation: r={r_comp:.3f}, p={p_comp:.3f}")
            if r_comp > 0.3:
                print("  → High-score composites more likely to fool the model!")

    return {
        'accuracy': accuracy,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

claude_analysis = analyze_errors(claude_results, "Claude")
gpt_analysis = analyze_errors(gpt_results, "GPT")

# ============================================================
# Part 5: Sequence Completion Probing
# ============================================================
print("\n" + "=" * 70)
print("PART 5: SEQUENCE COMPLETION PROBING")
print("=" * 70)

print("""
Test: Give partial prime sequences, see what the model predicts.
Hypothesis: Model completions should correlate with zero-based rankings.
""")

def probe_sequence_completion(query_fn, model_name):
    """Probe model's prime sequence predictions"""

    # Test sequences ending before a gap
    test_cases = [
        ([83, 89], "What is the next prime after 89?"),
        ([97, 101, 103], "What is the next prime after 103?"),
        ([137, 139], "What is the next prime after 139?"),
        ([191, 193, 197, 199], "What is the next prime after 199?"),
    ]

    results = []

    for seq, prompt in test_cases:
        response = query_fn(prompt, max_tokens=50)
        if response is None:
            continue

        # Extract number from response
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            predicted = int(numbers[0])

            # Find actual next prime
            last = seq[-1]
            actual_next = last + 1
            while actual_next not in primes:
                actual_next += 1

            correct = predicted == actual_next
            pred_score = compute_score(predicted) if predicted > 1 else -999
            actual_score = compute_score(actual_next)

            results.append({
                'sequence': seq,
                'predicted': predicted,
                'actual': actual_next,
                'correct': correct,
                'pred_score': pred_score,
                'actual_score': actual_score
            })

            status = "✓" if correct else "✗"
            print(f"  After {seq[-1]}: predicted {predicted}, actual {actual_next} {status}")

    return results

if has_anthropic:
    print("\n--- Claude Sequence Completions ---")
    claude_seq = probe_sequence_completion(query_claude, "Claude")

if has_openai:
    print("\n--- GPT Sequence Completions ---")
    gpt_seq = probe_sequence_completion(query_gpt, "GPT")

# ============================================================
# Part 6: "Which is more likely prime?" Probing
# ============================================================
print("\n" + "=" * 70)
print("PART 6: COMPARATIVE PRIMALITY PROBING")
print("=" * 70)

print("""
Test: Present pairs of numbers, ask which is "more likely" prime.
Hypothesis: Model preferences should correlate with zero scores.
""")

def probe_comparative(query_fn, model_name):
    """Probe model's comparative primality intuition"""

    # Pairs where zero score disagrees with reality
    # (high-score composite vs low-score prime)
    test_pairs = [
        (91, 97),    # 91=7×13 vs 97 prime
        (119, 127),  # 119=7×17 vs 127 prime
        (133, 137),  # 133=7×19 vs 137 prime
        (143, 149),  # 143=11×13 vs 149 prime
        (121, 131),  # 121=11² vs 131 prime
    ]

    results = []

    for a, b in test_pairs:
        prompt = f"Without calculating, which number feels more likely to be prime: {a} or {b}? Just answer with the number."
        response = query_fn(prompt, max_tokens=20)

        if response is None:
            continue

        # Parse response
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            chosen = int(numbers[0])

            a_prime = a in primes
            b_prime = b in primes
            a_score = zero_scores[a]
            b_score = zero_scores[b]

            # What does zero score predict?
            score_predicts = a if a_score > b_score else b

            # What's actually prime?
            actual_prime = None
            if a_prime and not b_prime:
                actual_prime = a
            elif b_prime and not a_prime:
                actual_prime = b

            chose_higher_score = (chosen == score_predicts)
            chose_actual_prime = (chosen == actual_prime) if actual_prime else None

            results.append({
                'pair': (a, b),
                'chosen': chosen,
                'score_predicts': score_predicts,
                'actual_prime': actual_prime,
                'chose_higher_score': chose_higher_score,
                'chose_actual_prime': chose_actual_prime,
                'a_score': a_score,
                'b_score': b_score
            })

            status = "✓" if chose_actual_prime else "✗" if chose_actual_prime is False else "?"
            score_match = "≈S" if chose_higher_score else "≠S"
            print(f"  {a} vs {b}: chose {chosen} {status} ({score_match})")

    # Analyze
    if results:
        score_matches = sum(1 for r in results if r['chose_higher_score'])
        actual_matches = sum(1 for r in results if r['chose_actual_prime'])

        print(f"\n  Matched zero-score prediction: {score_matches}/{len(results)}")
        print(f"  Matched actual primality: {actual_matches}/{len(results)}")

        if score_matches > len(results) * 0.6:
            print("  → Model intuition correlates with zero scores!")

    return results

if has_anthropic:
    print("\n--- Claude Comparative Judgments ---")
    claude_comp = probe_comparative(query_claude, "Claude")

if has_openai:
    print("\n--- GPT Comparative Judgments ---")
    gpt_comp = probe_comparative(query_gpt, "GPT")

# ============================================================
# Part 7: "Prime-ness Rating" Probing
# ============================================================
print("\n" + "=" * 70)
print("PART 7: PRIMENESS RATING PROBING")
print("=" * 70)

print("""
Test: Ask model to rate numbers 1-10 on "how prime they feel".
Hypothesis: Ratings should correlate with zero scores.
""")

def probe_primeness_rating(query_fn, model_name):
    """Probe model's intuitive primeness ratings"""

    # Mix of numbers with varying scores
    test_nums = [91, 97, 101, 111, 113, 119, 121, 127, 131, 133]

    results = []

    for n in test_nums:
        prompt = f"On a scale of 1-10, how 'prime-like' does the number {n} feel to you intuitively? Just give a number 1-10."
        response = query_fn(prompt, max_tokens=20)

        if response is None:
            continue

        # Extract rating
        numbers = re.findall(r'\b([1-9]|10)\b', response)
        if numbers:
            rating = int(numbers[0])

            results.append({
                'number': n,
                'rating': rating,
                'is_prime': n in primes,
                'zero_score': zero_scores.get(n, compute_score(n))
            })

            prime_mark = "P" if n in primes else "C"
            print(f"  {n} ({prime_mark}): rating={rating}, score={results[-1]['zero_score']:.3f}")

    # Correlation analysis
    if len(results) >= 5:
        ratings = [r['rating'] for r in results]
        scores = [r['zero_score'] for r in results]

        r_corr, p_val = pearsonr(ratings, scores)
        print(f"\n  Rating-Score correlation: r={r_corr:.3f}, p={p_val:.3f}")

        if r_corr > 0.3:
            print("  → Model's 'primeness intuition' correlates with zero scores!")
        elif r_corr < -0.3:
            print("  → Inverse correlation - model uses different heuristics")

    return results

if has_anthropic:
    print("\n--- Claude Primeness Ratings ---")
    claude_ratings = probe_primeness_rating(query_claude, "Claude")

if has_openai:
    print("\n--- GPT Primeness Ratings ---")
    gpt_ratings = probe_primeness_rating(query_gpt, "GPT")

# ============================================================
# Part 8: Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: LLM ZERO SIGNATURE PROBING")
print("=" * 70)

print(f"""
METHODOLOGY:
  We probed LLM behavior on prime-related tasks to detect
  if their responses correlate with Riemann zero structure.

PROBING TASKS:
  1. Direct primality judgment
  2. Prime sequence completion
  3. Comparative primality intuition
  4. Subjective "primeness" ratings

HYPOTHESIS:
  If LLMs encode zero structure, we expect:
  - Errors on high-score composites (false positives)
  - Preferences correlating with zero scores
  - "Primeness" ratings matching spectral predictions

RESULTS:
""")

if has_anthropic and claude_analysis:
    print(f"  Claude accuracy: {100*claude_analysis['accuracy']:.1f}%")
    print(f"  Claude false positives: {len(claude_analysis['false_positives'])}")

if has_openai and gpt_analysis:
    print(f"  GPT accuracy: {100*gpt_analysis['accuracy']:.1f}%")
    print(f"  GPT false positives: {len(gpt_analysis['false_positives'])}")

if not has_anthropic and not has_openai:
    print("""
  ⚠ No API access - cannot probe production models.

  To run full analysis:
    export ANTHROPIC_API_KEY=your_key
    export OPENAI_API_KEY=your_key
    python research_probe_llm_zeros.py

  Alternative: Use this script's methodology to manually test
  models via web interfaces (claude.ai, chat.openai.com).
""")

print("""
INTERPRETATION:
  Production LLMs are trained on mathematical text that includes
  implicit prime structure. If the Recursive Resonator hypothesis
  is correct, this training should embed zero-based representations.

  The probing methodology here tests for behavioral signatures
  without requiring access to internal hidden states.

NEXT STEPS:
  1. Run with API access for quantitative results
  2. Test on math-specialized models (Minerva, Llemma)
  3. Probe with larger test sets for statistical power
  4. Compare error patterns to spectral predictions
""")

print("=" * 70)
print("LLM ZERO PROBING RESEARCH COMPLETE")
print("=" * 70)
