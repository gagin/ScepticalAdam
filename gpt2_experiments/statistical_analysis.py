"""
Statistical significance analysis for ScepticalAdam experiment.
"""
import math

def calculate_significance(n, p1, p2):
    """
    Calculate statistical significance using two-proportion z-test.
    
    Args:
        n: sample size
        p1: proportion for baseline (e.g., 0.3790)
        p2: proportion for sceptical (e.g., 0.3850)
    
    Returns:
        z_score, p_value, is_significant
    """
    # Pooled proportion
    p_pool = (p1 + p2) / 2
    
    # Standard error
    se = math.sqrt(2 * p_pool * (1 - p_pool) / n)
    
    # Z-score
    z = (p2 - p1) / se
    
    # P-value (two-tailed) - approximate using normal distribution
    # For z > 1.96, p < 0.05 (95% confidence)
    # For z > 2.58, p < 0.01 (99% confidence)
    
    if abs(z) > 2.58:
        significance = "p < 0.01 (99% confidence) ✓✓✓"
        is_sig = True
    elif abs(z) > 1.96:
        significance = "p < 0.05 (95% confidence) ✓✓"
        is_sig = True
    elif abs(z) > 1.645:
        significance = "p < 0.10 (90% confidence) ✓"
        is_sig = True
    else:
        significance = "Not significant (p > 0.10) ✗"
        is_sig = False
    
    return z, significance, is_sig

def calculate_required_sample_size(p1, p2, alpha=0.05, power=0.80):
    """
    Calculate required sample size for given effect size.
    
    Args:
        p1: baseline proportion
        p2: sceptical proportion
        alpha: significance level (default 0.05 for 95% confidence)
        power: statistical power (default 0.80)
    
    Returns:
        required sample size per group
    """
    # Effect size (Cohen's h)
    h = 2 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))
    
    # Z-scores for alpha and power
    z_alpha = 1.96  # for alpha = 0.05 (two-tailed)
    z_beta = 0.84   # for power = 0.80
    
    # Required sample size
    n = ((z_alpha + z_beta) / h) ** 2
    
    return int(math.ceil(n))

# Current results
print("="*70)
print("STATISTICAL SIGNIFICANCE ANALYSIS")
print("="*70)
print()

# Results from 1000 samples
n_1000 = 1000
baseline_1000 = 0.3790
sceptical_1000 = 0.3850
diff_1000 = sceptical_1000 - baseline_1000

print(f"Results with n={n_1000} samples:")
print(f"  Baseline:   {baseline_1000:.4f} ({baseline_1000*100:.2f}%)")
print(f"  Sceptical:  {sceptical_1000:.4f} ({sceptical_1000*100:.2f}%)")
print(f"  Difference: +{diff_1000:.4f} (+{diff_1000*100:.2f} percentage points)")
print()

z_1000, sig_1000, is_sig_1000 = calculate_significance(n_1000, baseline_1000, sceptical_1000)
print(f"Statistical test:")
print(f"  Z-score: {z_1000:.3f}")
print(f"  Result: {sig_1000}")
print()

if is_sig_1000:
    print("✓ The difference IS statistically significant at 1000 samples!")
else:
    print("✗ The difference is NOT statistically significant at 1000 samples.")
    print("  This could be due to random chance.")

print()
print("-"*70)
print()

# Calculate required sample size
print("Required sample sizes for statistical significance:")
print()

required_95 = calculate_required_sample_size(baseline_1000, sceptical_1000, alpha=0.05, power=0.80)
required_99 = calculate_required_sample_size(baseline_1000, sceptical_1000, alpha=0.01, power=0.80)

print(f"For 95% confidence (p < 0.05):")
print(f"  Required: ~{required_95} samples")
print(f"  Current:  {n_1000} samples")
if n_1000 >= required_95:
    print(f"  Status: ✓ Sufficient")
else:
    print(f"  Status: ✗ Need {required_95 - n_1000} more samples")
print()

print(f"For 99% confidence (p < 0.01):")
print(f"  Required: ~{required_99} samples")
print(f"  Current:  {n_1000} samples")
if n_1000 >= required_99:
    print(f"  Status: ✓ Sufficient")
else:
    print(f"  Status: ✗ Need {required_99 - n_1000} more samples")
print()

print("-"*70)
print()

# Recommendations
print("RECOMMENDATIONS:")
print()

if is_sig_1000:
    print("✓ Your current 1000 samples are sufficient for publication!")
    print("  The results are statistically significant.")
else:
    print(f"✗ Increase sample size to at least {required_95} for 95% confidence.")
    print(f"  Or to {required_99} for 99% confidence (stronger claim).")

print()
print("Sample size options:")
print(f"  • 1,000 samples:  ~20 minutes (current)")
print(f"  • 2,000 samples:  ~40 minutes")
print(f"  • 5,000 samples:  ~1.5 hours")
print(f"  • 10,042 samples: ~3 hours (full HellaSwag)")
print()

print("To run with more samples, edit fast_eval.py line 18:")
print(f"  limit = {required_95}  # For 95% confidence")
print(f"  limit = {required_99}  # For 99% confidence")
print(f"  limit = None  # For full dataset (10,042)")
print()

print("="*70)

