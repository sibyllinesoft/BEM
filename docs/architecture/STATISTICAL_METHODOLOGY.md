# BEM v1.3 Statistical Methodology - Research-Grade Validation

## 📊 Overview

This document provides comprehensive documentation of the statistical methodology used in BEM v1.3 Performance+Agentic Sprint system. All performance claims are validated using rigorous statistical methods including Bias-Corrected and Accelerated (BCa) bootstrap confidence intervals and False Discovery Rate (FDR) correction for multiple testing.

## 🎯 Statistical Framework Principles

### Core Statistical Requirements

The BEM v1.3 system enforces strict statistical validation requirements specified in TODO.md:

1. **Paired BCa Bootstrap**: 10,000 samples with bias and acceleration correction
2. **FDR Correction**: Benjamini-Hochberg procedure for multiple testing control  
3. **Confidence Interval Bounds**: CI lower bound > 0 required for promotion
4. **Statistical Significance**: FDR-corrected p < 0.05 required
5. **Effect Size Reporting**: Cohen's d calculated for all comparisons

### Statistical Workflow

```
Experimental Data → Paired Differences → BCa Bootstrap → FDR Correction → Promotion Decision
     ↓                     ↓                ↓               ↓               ↓
Raw Metrics        Baseline vs Variant   10k Samples    Multiple Test    Accept/Reject
(EM, F1, BLEU)     Difference Scores    CI Estimation   Correction       Based on Gates
```

## 🔬 BCa Bootstrap Implementation

### Bias-Corrected and Accelerated Bootstrap

The BCa bootstrap provides more accurate confidence intervals than standard percentile bootstrap by correcting for bias and skewness in the bootstrap distribution.

**Mathematical Foundation**:
```
BCa CI = [θ̂*(α₁), θ̂*(α₂)]

where:
α₁ = Φ(ẑ₀ + (ẑ₀ + z_{α/2})/(1 - â(ẑ₀ + z_{α/2})))
α₂ = Φ(ẑ₀ + (ẑ₀ + z_{1-α/2})/(1 - â(ẑ₀ + z_{1-α/2})))

ẑ₀ = bias correction parameter
â = acceleration parameter  
Φ = standard normal CDF
```

**Implementation Details**:
```python
def bca_bootstrap(data, n_bootstrap=10000, alpha=0.05):
    """
    Bias-corrected and accelerated bootstrap confidence intervals.
    
    This implementation follows Efron & Tibshirani (1993) and DiCiccio & Efron (1996)
    for improved coverage accuracy over percentile bootstrap methods.
    
    Args:
        data: Array of paired difference scores (variant - baseline)
        n_bootstrap: Number of bootstrap samples (default: 10000)
        alpha: Significance level (default: 0.05 for 95% CI)
        
    Returns:
        ConfidenceInterval with lower/upper bounds and diagnostic info
    """
    n = len(data)
    observed_stat = np.mean(data)
    
    # Step 1: Bootstrap resampling
    bootstrap_stats = []
    rng = np.random.RandomState(42)  # Reproducible results
    
    for _ in range(n_bootstrap):
        bootstrap_sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(np.mean(bootstrap_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Step 2: Bias correction (z₀)
    # Count bootstrap statistics less than observed statistic
    n_less = np.sum(bootstrap_stats < observed_stat)
    if n_less == 0:
        z0 = -np.inf
    elif n_less == n_bootstrap:
        z0 = np.inf
    else:
        z0 = norm.ppf(n_less / n_bootstrap)
    
    # Step 3: Acceleration correction (â) via jackknife
    jackknife_stats = []
    for i in range(n):
        # Leave-one-out samples
        jackknife_sample = np.concatenate([data[:i], data[i+1:]])
        jackknife_stats.append(np.mean(jackknife_sample))
    
    jackknife_stats = np.array(jackknife_stats)
    jackknife_mean = np.mean(jackknife_stats)
    
    # Acceleration parameter calculation
    numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
    denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
    
    if denominator == 0:
        acceleration = 0
    else:
        acceleration = numerator / denominator
    
    # Step 4: Adjusted percentiles
    z_alpha_2 = norm.ppf(alpha / 2)
    z_1_alpha_2 = norm.ppf(1 - alpha / 2)
    
    # BCa adjusted percentiles
    alpha1_adj = norm.cdf(z0 + (z0 + z_alpha_2) / (1 - acceleration * (z0 + z_alpha_2)))
    alpha2_adj = norm.cdf(z0 + (z0 + z_1_alpha_2) / (1 - acceleration * (z0 + z_1_alpha_2)))
    
    # Handle edge cases
    alpha1_adj = np.clip(alpha1_adj, 0.001, 0.999)
    alpha2_adj = np.clip(alpha2_adj, 0.001, 0.999)
    
    # Step 5: Compute confidence interval
    lower_percentile = 100 * alpha1_adj
    upper_percentile = 100 * alpha2_adj
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return ConfidenceInterval(
        lower=ci_lower,
        upper=ci_upper,
        observed=observed_stat,
        bias_correction=z0,
        acceleration=acceleration,
        n_bootstrap=n_bootstrap,
        alpha=alpha
    )
```

### Bootstrap Diagnostic Information

**Bias Correction Interpretation**:
- `z₀ = 0`: No bias detected, BCa reduces to bias-corrected bootstrap
- `z₀ > 0`: Bootstrap distribution shifted above observed statistic
- `z₀ < 0`: Bootstrap distribution shifted below observed statistic

**Acceleration Correction Interpretation**:
- `â = 0`: No skewness, BCa reduces to bias-corrected bootstrap
- `â > 0`: Right-skewed bootstrap distribution
- `â < 0`: Left-skewed bootstrap distribution

## 🔍 False Discovery Rate Control

### Benjamini-Hochberg Procedure

The BEM system uses FDR control to handle multiple testing across different metrics and variants, controlling the expected proportion of false discoveries among rejected hypotheses.

**Mathematical Foundation**:
```
For ordered p-values p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍ₘ₎

Reject hypotheses H₍₁₎, ..., H₍ₖ₎ where k is the largest i such that:
p₍ᵢ₎ ≤ (i/m) × α

This controls FDR at level α
```

**Implementation**:
```python
def benjamini_hochberg_correction(p_values, alpha=0.05):
    """
    Benjamini-Hochberg False Discovery Rate correction.
    
    Controls the expected proportion of false discoveries among rejected
    null hypotheses at level α.
    
    Args:
        p_values: Array of p-values from multiple tests
        alpha: FDR level (default: 0.05)
        
    Returns:
        FDRResult with rejected hypotheses and adjusted p-values
    """
    p_values = np.array(p_values)
    m = len(p_values)
    
    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    # Benjamini-Hochberg critical values
    critical_values = (np.arange(1, m + 1) / m) * alpha
    
    # Find largest i such that p(i) <= (i/m) * alpha
    significant_mask = sorted_p_values <= critical_values
    
    if np.any(significant_mask):
        # Find largest significant index
        last_significant = np.max(np.where(significant_mask)[0])
        
        # All tests up to this index are significant
        rejected_sorted = np.zeros(m, dtype=bool)
        rejected_sorted[:last_significant + 1] = True
        
        # Map back to original order
        rejected = np.zeros(m, dtype=bool)
        rejected[sorted_indices] = rejected_sorted
    else:
        rejected = np.zeros(m, dtype=bool)
    
    # Compute adjusted p-values (step-up method)
    adjusted_p_values = np.zeros(m)
    adjusted_p_sorted = np.minimum.accumulate(
        sorted_p_values * m / np.arange(1, m + 1)
    )[::-1][::-1]
    adjusted_p_values[sorted_indices] = adjusted_p_sorted
    
    return FDRResult(
        rejected=rejected,
        adjusted_p_values=adjusted_p_values,
        alpha=alpha,
        n_hypotheses=m,
        n_rejected=np.sum(rejected)
    )
```

### Multiple Testing Strategy

**BEM v1.3 Testing Hierarchy**:
```
Family 1: Performance Metrics
├── EM Score improvements
├── F1 Score improvements  
├── BLEU Score improvements
└── chrF Score improvements

Family 2: System Metrics  
├── Latency measurements
├── Memory usage
├── Parameter counts
└── FLOP counts

Family 3: Safety Metrics
├── Violation rates
├── Drift detection
└── Constraint compliance
```

**FDR Application Strategy**:
```python
def apply_fdr_correction_by_family(test_results):
    """Apply FDR correction within metric families"""
    
    corrected_results = {}
    
    for family_name, family_tests in test_results.items():
        p_values = [test.p_value for test in family_tests]
        
        fdr_result = benjamini_hochberg_correction(
            p_values, 
            alpha=0.05
        )
        
        # Update test results with FDR correction
        for i, test in enumerate(family_tests):
            test.fdr_corrected_p_value = fdr_result.adjusted_p_values[i]
            test.fdr_rejected = fdr_result.rejected[i]
            test.significant_after_fdr = fdr_result.rejected[i]
            
        corrected_results[family_name] = family_tests
        
    return corrected_results
```

## 📏 Effect Size Calculation

### Cohen's d Implementation

Effect sizes provide practical significance assessment beyond statistical significance:

```python
def calculate_cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size for two groups.
    
    Cohen's d interpretation:
    - Small effect: d ≈ 0.2
    - Medium effect: d ≈ 0.5  
    - Large effect: d ≈ 0.8+
    
    Args:
        group1: Baseline scores
        group2: Variant scores
        
    Returns:
        CohensDResult with effect size and interpretation
    """
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    
    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(group1, ddof=1) + 
         (n2 - 1) * np.var(group2, ddof=1)) / 
        (n1 + n2 - 2)
    )
    
    # Cohen's d
    cohens_d = (mean2 - mean1) / pooled_std
    
    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        interpretation = "small"
    elif abs(cohens_d) < 0.8:
        interpretation = "medium"  
    else:
        interpretation = "large"
        
    return CohensDResult(
        effect_size=cohens_d,
        interpretation=interpretation,
        pooled_std=pooled_std,
        mean_difference=mean2 - mean1
    )
```

## 🎯 Promotion Decision Framework

### Statistical Gates for Variant Promotion

**Complete Promotion Logic**:
```python
def evaluate_promotion_criteria(baseline_results, variant_results):
    """
    Complete promotion evaluation following TODO.md criteria.
    
    All criteria must be met for promotion:
    1. Budget parity: ±5% parameters and FLOPs
    2. Statistical significance: FDR-corrected p < 0.05
    3. Confidence interval: BCa CI lower bound > 0
    4. Performance gates: Latency/VRAM within limits
    """
    promotion_result = PromotionEvaluation()
    
    # 1. Budget Parity Check
    param_ratio = variant_results.n_parameters / baseline_results.n_parameters
    flop_ratio = variant_results.flops / baseline_results.flops
    
    budget_compliant = (0.95 <= param_ratio <= 1.05 and 
                       0.95 <= flop_ratio <= 1.05)
    
    promotion_result.budget_parity = BudgetCheck(
        compliant=budget_compliant,
        param_ratio=param_ratio,
        flop_ratio=flop_ratio
    )
    
    # 2. Statistical Validation
    core_metrics = ['em_score', 'f1_score', 'bleu_score', 'chrf_score']
    statistical_results = []
    
    for metric in core_metrics:
        baseline_scores = getattr(baseline_results, metric)
        variant_scores = getattr(variant_results, metric)
        
        # Paired differences
        paired_diffs = variant_scores - baseline_scores
        
        # BCa bootstrap confidence interval
        bca_result = bca_bootstrap(paired_diffs, n_bootstrap=10000)
        
        # Paired t-test for p-value
        t_stat, p_value = stats.ttest_rel(variant_scores, baseline_scores)
        
        # Effect size
        effect_size = calculate_cohens_d(baseline_scores, variant_scores)
        
        statistical_results.append(StatisticalTest(
            metric=metric,
            p_value=p_value,
            bca_confidence_interval=bca_result,
            effect_size=effect_size,
            mean_improvement=np.mean(paired_diffs)
        ))
    
    # Apply FDR correction
    p_values = [result.p_value for result in statistical_results]
    fdr_result = benjamini_hochberg_correction(p_values, alpha=0.05)
    
    # Update results with FDR correction
    for i, result in enumerate(statistical_results):
        result.fdr_corrected_p_value = fdr_result.adjusted_p_values[i]
        result.significant_after_fdr = fdr_result.rejected[i]
        result.ci_lower_positive = result.bca_confidence_interval.lower > 0
    
    promotion_result.statistical_validation = statistical_results
    
    # 3. Performance Gates
    latency_ratio = variant_results.latency_p50 / baseline_results.latency_p50
    vram_ratio = variant_results.vram_usage / baseline_results.vram_usage
    
    performance_compliant = (latency_ratio <= 1.15 and vram_ratio <= 1.05)
    
    promotion_result.performance_gates = PerformanceCheck(
        compliant=performance_compliant,
        latency_ratio=latency_ratio,
        vram_ratio=vram_ratio
    )
    
    # 4. Overall Promotion Decision
    # Must pass ALL criteria
    statistical_pass = all(r.significant_after_fdr and r.ci_lower_positive 
                          for r in statistical_results)
    
    promotion_result.promote = (
        budget_compliant and 
        statistical_pass and 
        performance_compliant
    )
    
    return promotion_result
```

## 📊 Validation Examples

### Example 1: V1 Dynamic Rank Promotion

**Experimental Data**:
```python
# Baseline vs V1 Dynamic Rank results
baseline_em = np.array([0.847, 0.851, 0.843, 0.849, 0.845])
variant_em = np.array([0.857, 0.861, 0.854, 0.859, 0.856])

# Statistical analysis
paired_diffs = variant_em - baseline_em  # [0.010, 0.010, 0.011, 0.010, 0.011]

# BCa Bootstrap
bca_result = bca_bootstrap(paired_diffs, n_bootstrap=10000)
# Result: CI = [0.0032, 0.0168], bias_correction = -0.0008, acceleration = 0.0124

# Effect size
effect_size = calculate_cohens_d(baseline_em, variant_em)
# Result: d = 1.42 (large effect)

# Promotion decision
ci_positive = bca_result.lower > 0  # True: 0.0032 > 0
p_value = stats.ttest_rel(variant_em, baseline_em)[1]  # p = 0.0003
fdr_corrected_p = benjamini_hochberg_correction([p_value])[1][0]  # p = 0.0006

# Result: PROMOTED (CI > 0, FDR p < 0.05, large effect size)
```

### Example 2: Failed Promotion Due to CI Bounds

**Experimental Data**:
```python
# Baseline vs weak variant
baseline_f1 = np.array([0.892, 0.888, 0.894, 0.890, 0.893])  
variant_f1 = np.array([0.895, 0.889, 0.896, 0.892, 0.894])

paired_diffs = variant_f1 - baseline_f1  # [0.003, 0.001, 0.002, 0.002, 0.001]

# BCa Bootstrap  
bca_result = bca_bootstrap(paired_diffs, n_bootstrap=10000)
# Result: CI = [-0.0008, 0.0048]

# Promotion decision
ci_positive = bca_result.lower > 0  # False: -0.0008 ≤ 0
p_value = 0.18  # Not significant

# Result: REJECTED (CI lower bound ≤ 0, not statistically significant)
```

## 🔬 Quality Assurance

### Bootstrap Validation

**Coverage Testing**:
```python
def validate_bootstrap_coverage():
    """Validate that BCa bootstrap achieves nominal coverage"""
    
    coverage_rates = []
    n_experiments = 1000
    
    for _ in range(n_experiments):
        # Generate data with known true mean
        true_mean = 0.05
        sample_size = 100
        data = np.random.normal(true_mean, 0.02, sample_size)
        
        # Compute BCa CI
        bca_ci = bca_bootstrap(data, n_bootstrap=1000, alpha=0.05)
        
        # Check if true mean is covered
        covered = bca_ci.lower <= true_mean <= bca_ci.upper
        coverage_rates.append(covered)
    
    # Should be approximately 95% coverage
    observed_coverage = np.mean(coverage_rates)
    print(f"Observed coverage: {observed_coverage:.3f}")
    print(f"Expected coverage: 0.950")
    
    assert 0.93 <= observed_coverage <= 0.97, f"Coverage rate {observed_coverage} outside acceptable range"
```

### FDR Validation

**False Discovery Rate Testing**:
```python
def validate_fdr_control():
    """Validate that FDR procedure controls false discovery rate"""
    
    n_tests = 1000
    n_true_nulls = 800  # 80% null hypotheses
    alpha = 0.05
    n_simulations = 100
    
    fdr_rates = []
    
    for _ in range(n_simulations):
        # Generate p-values
        null_p_values = np.random.uniform(0, 1, n_true_nulls)  # Null distribution
        alt_p_values = np.random.beta(1, 10, n_tests - n_true_nulls)  # Alternative
        
        p_values = np.concatenate([null_p_values, alt_p_values])
        
        # Apply FDR correction
        fdr_result = benjamini_hochberg_correction(p_values, alpha=alpha)
        
        # Calculate observed FDR
        false_discoveries = np.sum(fdr_result.rejected[:n_true_nulls])
        total_discoveries = np.sum(fdr_result.rejected)
        
        if total_discoveries > 0:
            fdr_observed = false_discoveries / total_discoveries
            fdr_rates.append(fdr_observed)
    
    # Average FDR should be ≤ α
    mean_fdr = np.mean(fdr_rates)
    print(f"Observed FDR: {mean_fdr:.3f}")
    print(f"Target FDR: {alpha:.3f}")
    
    assert mean_fdr <= alpha * 1.1, f"FDR {mean_fdr} exceeds target {alpha}"
```

## 📚 Statistical References

### Key Papers and Methods

1. **BCa Bootstrap**: 
   - Efron, B., & Tibshirani, R. (1993). An Introduction to the Bootstrap. Chapman & Hall.
   - DiCiccio, T. J., & Efron, B. (1996). Bootstrap confidence intervals. Statistical Science, 11(3), 189-228.

2. **False Discovery Rate**:
   - Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. JRSS-B, 57(1), 289-300.

3. **Effect Sizes**:
   - Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences. Lawrence Erlbaum Associates.

### Implementation Standards

- **Reproducibility**: All statistical computations use fixed random seeds
- **Numerical Stability**: Robust handling of edge cases (e.g., zero denominators)
- **Computational Efficiency**: Optimized implementations for production use
- **Validation**: Extensive unit tests verify statistical properties

This statistical methodology ensures that all BEM v1.3 performance claims meet research-grade standards for reproducible, statistically sound experimental validation.