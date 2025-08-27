# BEM Fleet Statistical Framework

## 📊 Statistical Validation Overview

The BEM Fleet employs a rigorous statistical validation framework designed to produce publication-quality results with high statistical confidence. Our approach combines bias-corrected bootstrap confidence intervals, multiple testing corrections, and comprehensive effect size analysis to ensure research findings meet the highest academic standards.

## 🎯 Statistical Design Principles

### Core Statistical Philosophy
1. **Paired Testing**: All comparisons use paired statistical tests to control for confounding variables
2. **Bootstrap Confidence**: BCa (Bias-Corrected and Accelerated) bootstrap provides robust confidence intervals
3. **Multiple Testing Control**: FDR correction prevents inflation of Type I error rates
4. **Effect Size Focus**: Statistical significance must be accompanied by practical significance
5. **Reproducibility**: All statistical procedures are deterministic and fully documented

### Statistical Rigor Standards
```yaml
statistical_standards:
  confidence_level: 0.95        # 95% confidence intervals
  significance_level: 0.05      # 5% Type I error rate
  fdr_correction: true          # Benjamini-Hochberg FDR control
  bootstrap_samples: 10000      # Bootstrap replicates
  minimum_effect_size:          # Minimum meaningful effect sizes
    mission_a: 0.5              # Cohen's d ≥ 0.5 
    mission_b: 0.8              # Cohen's d ≥ 0.8
    mission_c: 0.6              # Cohen's d ≥ 0.6
    mission_d: 0.4              # Cohen's d ≥ 0.4
    mission_e: 0.5              # Cohen's d ≥ 0.5
  random_seeds: [1, 2, 3, 4, 5]  # Independent replications
```

## 🔬 BCa Bootstrap Implementation

### Bias-Corrected and Accelerated Bootstrap

The BCa bootstrap provides robust confidence intervals that correct for bias and skewness in the bootstrap distribution:

```python
class BCABootstrap:
    """
    Bias-Corrected and Accelerated Bootstrap for BEM Fleet validation
    
    This implementation follows Efron & Tibshirani (1993) and provides
    accurate confidence intervals even for non-normal distributions.
    """
    
    def __init__(self, n_bootstrap=10000, confidence_level=0.95, random_state=42):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.random_state = random_state
        
    def bootstrap_confidence_interval(self, data1, data2, statistic_func):
        """
        Calculate BCa bootstrap confidence interval
        
        Args:
            data1: Treatment/experimental data
            data2: Control/baseline data  
            statistic_func: Function to calculate test statistic
            
        Returns:
            BCaResult with confidence interval and bias correction
        """
        # Original statistic
        original_stat = statistic_func(data1, data2)
        
        # Bootstrap resamples
        bootstrap_stats = self._generate_bootstrap_samples(
            data1, data2, statistic_func
        )
        
        # Bias correction
        z0 = self._calculate_bias_correction(bootstrap_stats, original_stat)
        
        # Acceleration factor
        acceleration = self._calculate_acceleration(data1, data2, statistic_func)
        
        # BCa confidence interval
        ci_lower, ci_upper = self._calculate_bca_interval(
            bootstrap_stats, z0, acceleration
        )
        
        return BCaResult(
            statistic=original_stat,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            bootstrap_mean=np.mean(bootstrap_stats),
            bias_correction=z0,
            acceleration=acceleration,
            p_value=self._calculate_p_value(bootstrap_stats, original_stat)
        )
        
    def _generate_bootstrap_samples(self, data1, data2, statistic_func):
        """Generate bootstrap distribution of test statistic"""
        np.random.seed(self.random_state)
        n = len(data1)
        bootstrap_stats = np.zeros(self.n_bootstrap)
        
        for i in range(self.n_bootstrap):
            # Paired bootstrap resampling
            idx = np.random.choice(n, size=n, replace=True)
            boot_data1 = data1[idx]
            boot_data2 = data2[idx]
            bootstrap_stats[i] = statistic_func(boot_data1, boot_data2)
            
        return bootstrap_stats
        
    def _calculate_bias_correction(self, bootstrap_stats, original_stat):
        """Calculate bias correction factor z0"""
        # Proportion of bootstrap stats ≤ original statistic
        prop_less = np.mean(bootstrap_stats <= original_stat)
        
        # Bias correction (inverse normal CDF)
        z0 = stats.norm.ppf(prop_less) if 0 < prop_less < 1 else 0
        
        return z0
        
    def _calculate_acceleration(self, data1, data2, statistic_func):
        """Calculate acceleration factor using jackknife"""
        n = len(data1)
        jackknife_stats = np.zeros(n)
        
        # Jackknife resampling (leave-one-out)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            jack_data1 = data1[mask]
            jack_data2 = data2[mask]
            jackknife_stats[i] = statistic_func(jack_data1, jack_data2)
            
        # Calculate acceleration
        mean_jack = np.mean(jackknife_stats)
        numerator = np.sum((mean_jack - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((mean_jack - jackknife_stats) ** 2) ** 1.5)
        
        acceleration = numerator / denominator if denominator != 0 else 0
        
        return acceleration
        
    def _calculate_bca_interval(self, bootstrap_stats, z0, acceleration):
        """Calculate BCa confidence interval bounds"""
        # Standard normal quantiles
        z_alpha_2 = stats.norm.ppf(self.alpha / 2)
        z_1_alpha_2 = stats.norm.ppf(1 - self.alpha / 2)
        
        # BCa adjusted quantiles
        alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha_2) / (1 - acceleration * (z0 + z_alpha_2)))
        alpha2 = stats.norm.cdf(z0 + (z0 + z_1_alpha_2) / (1 - acceleration * (z0 + z_1_alpha_2)))
        
        # Clip to valid probability range
        alpha1 = np.clip(alpha1, 0.001, 0.999)
        alpha2 = np.clip(alpha2, 0.001, 0.999)
        
        # Bootstrap percentiles
        ci_lower = np.percentile(bootstrap_stats, alpha1 * 100)
        ci_upper = np.percentile(bootstrap_stats, alpha2 * 100)
        
        return ci_lower, ci_upper
```

### Bootstrap Test Statistics

#### Mean Difference (Continuous Metrics)
```python
def mean_difference(treatment, control):
    """Test statistic: difference in means"""
    return np.mean(treatment) - np.mean(control)
```

#### Proportion Difference (Binary Metrics)  
```python
def proportion_difference(treatment, control):
    """Test statistic: difference in proportions"""
    return np.mean(treatment) - np.mean(control)
```

#### Effect Size (Cohen's d)
```python
def cohens_d(treatment, control):
    """Test statistic: Cohen's d effect size"""
    pooled_std = np.sqrt(((len(treatment) - 1) * np.var(treatment, ddof=1) + 
                         (len(control) - 1) * np.var(control, ddof=1)) / 
                        (len(treatment) + len(control) - 2))
    return (np.mean(treatment) - np.mean(control)) / pooled_std
```

## 🎛️ Multiple Testing Correction

### False Discovery Rate (FDR) Control

The Benjamini-Hochberg procedure controls the expected proportion of false discoveries:

```python
class FDRCorrection:
    """
    False Discovery Rate correction for multiple testing
    
    Implements Benjamini-Hochberg procedure to control FDR at level α
    """
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        
    def correct_p_values(self, p_values, test_names=None):
        """
        Apply Benjamini-Hochberg FDR correction
        
        Args:
            p_values: Array of p-values to correct
            test_names: Optional names for tests
            
        Returns:
            FDRResult with corrected p-values and rejection decisions
        """
        from statsmodels.stats.multitest import multipletests
        
        # Apply BH procedure
        rejected, corrected_p_values, alpha_sidak, alpha_bonf = multipletests(
            p_values, 
            alpha=self.alpha,
            method='fdr_bh',
            is_sorted=False,
            returnsorted=False
        )
        
        # Create results dictionary
        if test_names is None:
            test_names = [f'test_{i}' for i in range(len(p_values))]
            
        results = {}
        for i, name in enumerate(test_names):
            results[name] = TestResult(
                test_name=name,
                raw_p_value=p_values[i],
                corrected_p_value=corrected_p_values[i],
                significant=rejected[i],
                fdr_alpha=self.alpha,
                correction_method='benjamini_hochberg'
            )
            
        return FDRResult(
            results=results,
            n_tests=len(p_values),
            n_significant=sum(rejected),
            fdr_level=self.alpha,
            family_wise_error_rate=1 - (1 - self.alpha) ** len(p_values)
        )
        
    def stepwise_procedure(self, p_values, test_names=None):
        """
        Show step-by-step BH procedure
        
        Educational method to demonstrate the BH algorithm
        """
        if test_names is None:
            test_names = [f'test_{i}' for i in range(len(p_values))]
            
        # Create (p-value, test_name) pairs and sort
        p_test_pairs = list(zip(p_values, test_names))
        p_test_pairs.sort(key=lambda x: x[0])
        
        m = len(p_values)
        steps = []
        
        # Work backwards through sorted p-values
        for k in range(m, 0, -1):
            i = k - 1  # Convert to 0-based indexing
            p_value, test_name = p_test_pairs[i]
            threshold = (k / m) * self.alpha
            
            steps.append({
                'rank': k,
                'test_name': test_name,
                'p_value': p_value,
                'threshold': threshold,
                'significant': p_value <= threshold
            })
            
            # BH stopping rule: first significant test
            if p_value <= threshold:
                # All tests with smaller p-values are also significant
                significant_tests = p_test_pairs[:k]
                break
        else:
            # No significant tests found
            significant_tests = []
            
        return BHStepwiseResult(
            steps=steps,
            significant_tests=significant_tests,
            fdr_level=self.alpha
        )
```

### Mission-Specific Testing Strategy

Each mission requires different statistical approaches based on the type of data and research questions:

```yaml
mission_testing_strategies:
  mission_a:
    primary_tests:
      - test: "paired_t_test"
        metric: "em_f1_difference"
        statistic: "mean_difference"
      - test: "bootstrap_bca"
        metric: "em_f1_difference" 
        statistic: "cohens_d"
        
  mission_b:
    primary_tests:
      - test: "wilcoxon_signed_rank"
        metric: "correction_time"
        statistic: "median_difference"
      - test: "bootstrap_bca"
        metric: "correction_time"
        statistic: "cohens_d"
        
  mission_c:
    primary_tests:
      - test: "mcnemar_test"
        metric: "violation_reduction"
        statistic: "proportion_difference"
      - test: "bootstrap_bca"
        metric: "safety_score"
        statistic: "mean_difference"
        
  mission_d:
    primary_tests:
      - test: "paired_t_test"
        metric: "ood_performance"
        statistic: "mean_difference"
      - test: "bootstrap_bca"
        metric: "transfer_score"
        statistic: "cohens_d"
        
  mission_e:
    primary_tests:
      - test: "mixed_effects_model"
        metric: "context_scaling"
        statistic: "slope_difference"
      - test: "bootstrap_bca"
        metric: "long_context_performance"
        statistic: "mean_difference"
```

## 📏 Effect Size Analysis

### Effect Size Calculations

#### Cohen's d (Standardized Mean Difference)
```python
def cohens_d_with_ci(treatment, control, confidence_level=0.95):
    """
    Calculate Cohen's d with confidence interval
    
    Returns effect size with BCa bootstrap confidence interval
    """
    # Point estimate
    pooled_std = np.sqrt(((len(treatment) - 1) * np.var(treatment, ddof=1) + 
                         (len(control) - 1) * np.var(control, ddof=1)) / 
                        (len(treatment) + len(control) - 2))
    d = (np.mean(treatment) - np.mean(control)) / pooled_std
    
    # Bootstrap confidence interval for d
    bootstrap = BCABootstrap(confidence_level=confidence_level)
    result = bootstrap.bootstrap_confidence_interval(
        treatment, control, cohens_d
    )
    
    return EffectSizeResult(
        effect_size=d,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        magnitude=interpret_cohens_d(d),
        significant=result.ci_lower > 0 or result.ci_upper < 0
    )

def interpret_cohens_d(d):
    """Interpret Cohen's d magnitude"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
```

#### Glass's Δ (Control Group Standardizer)
```python
def glass_delta(treatment, control):
    """
    Glass's Δ using control group standard deviation
    
    Useful when control group represents a standard or reference
    """
    control_std = np.std(control, ddof=1)
    return (np.mean(treatment) - np.mean(control)) / control_std
```

#### Hedges' g (Bias-Corrected Effect Size)
```python  
def hedges_g(treatment, control):
    """
    Hedges' g with small sample bias correction
    
    Recommended for small samples (n < 20 per group)
    """
    d = cohens_d(treatment, control)
    n = len(treatment) + len(control)
    correction_factor = 1 - (3 / (4 * (n - 2) - 1))
    return d * correction_factor
```

## 🎯 Mission-Specific Validation

### Mission A: Agentic Planner Validation

```python
class MissionAValidator:
    """Statistical validation for Mission A: Agentic Planner"""
    
    def __init__(self):
        self.bootstrap = BCABootstrap(n_bootstrap=10000)
        self.fdr = FDRCorrection(alpha=0.05)
        self.acceptance_gates = {
            'em_f1_improvement': 1.5,  # % improvement required
            'min_effect_size': 0.5,    # Minimum Cohen's d
            'confidence_level': 0.95   # CI level
        }
        
    def validate_mission(self, treatment_results, baseline_results):
        """
        Complete statistical validation for Mission A
        
        Args:
            treatment_results: Agentic planner results
            baseline_results: Single fused BEM baseline
            
        Returns:
            MissionValidationResult with all statistical tests
        """
        tests = []
        
        # Primary metric: EM/F1 improvement
        em_f1_test = self._test_em_f1_improvement(
            treatment_results['em_f1'], 
            baseline_results['em_f1']
        )
        tests.append(em_f1_test)
        
        # Secondary metrics
        latency_test = self._test_latency_overhead(
            treatment_results['latency_p95'],
            baseline_results['latency_p95']
        )
        tests.append(latency_test)
        
        cache_test = self._test_cache_efficiency(
            treatment_results['kv_hit_ratio'],
            baseline_results['kv_hit_ratio'] 
        )
        tests.append(cache_test)
        
        # Apply FDR correction
        p_values = [test.p_value for test in tests]
        test_names = [test.test_name for test in tests]
        fdr_results = self.fdr.correct_p_values(p_values, test_names)
        
        # Check acceptance gates
        gates_passed = self._check_acceptance_gates(em_f1_test, fdr_results)
        
        return MissionValidationResult(
            mission='mission_a',
            tests=tests,
            fdr_results=fdr_results,
            gates_passed=gates_passed,
            overall_significant=gates_passed and fdr_results.n_significant > 0,
            recommendation=self._make_recommendation(gates_passed, fdr_results)
        )
        
    def _test_em_f1_improvement(self, treatment_em_f1, baseline_em_f1):
        """Test EM/F1 improvement with multiple statistics"""
        # Percentage improvement
        improvement = ((treatment_em_f1 - baseline_em_f1) / baseline_em_f1) * 100
        
        # Mean difference
        mean_diff_result = self.bootstrap.bootstrap_confidence_interval(
            treatment_em_f1, baseline_em_f1, mean_difference
        )
        
        # Effect size
        effect_size_result = self.bootstrap.bootstrap_confidence_interval(
            treatment_em_f1, baseline_em_f1, cohens_d
        )
        
        return StatisticalTest(
            test_name='em_f1_improvement',
            metric='em_f1',
            mean_improvement=np.mean(improvement),
            mean_difference=mean_diff_result,
            effect_size=effect_size_result,
            passes_gate=np.mean(improvement) >= self.acceptance_gates['em_f1_improvement']
        )
```

### Cross-Mission Statistical Integration

```python
class FleetStatisticalValidator:
    """Statistical validation across all missions"""
    
    def __init__(self):
        self.mission_validators = {
            'mission_a': MissionAValidator(),
            'mission_b': MissionBValidator(), 
            'mission_c': MissionCValidator(),
            'mission_d': MissionDValidator(),
            'mission_e': MissionEValidator()
        }
        self.fdr = FDRCorrection(alpha=0.05)
        
    def validate_fleet(self, mission_results):
        """
        Fleet-wide statistical validation with family-wise error control
        
        Performs individual mission validation then applies FDR correction
        across all tests from all missions
        """
        all_tests = []
        mission_validations = {}
        
        # Individual mission validation
        for mission_name, results in mission_results.items():
            validator = self.mission_validators[mission_name]
            validation = validator.validate_mission(
                results['treatment'], 
                results['baseline']
            )
            mission_validations[mission_name] = validation
            all_tests.extend(validation.tests)
            
        # Fleet-wide FDR correction
        all_p_values = [test.p_value for test in all_tests]
        all_test_names = [f"{test.mission}_{test.test_name}" for test in all_tests]
        
        fleet_fdr_results = self.fdr.correct_p_values(all_p_values, all_test_names)
        
        # Integration testing
        integration_tests = self._test_mission_integration(mission_results)
        
        return FleetValidationResult(
            mission_validations=mission_validations,
            fleet_fdr_results=fleet_fdr_results,
            integration_tests=integration_tests,
            overall_fleet_success=self._determine_fleet_success(
                mission_validations, fleet_fdr_results
            )
        )
        
    def _test_mission_integration(self, mission_results):
        """Test for interference between missions"""
        integration_tests = []
        
        # Test Mission A + Mission B integration
        ab_integration = self._test_router_online_integration(
            mission_results['mission_a'], 
            mission_results['mission_b']
        )
        integration_tests.append(ab_integration)
        
        # Test Mission C safety overlay
        safety_integration = self._test_safety_overlay_integration(
            mission_results
        )
        integration_tests.append(safety_integration)
        
        return integration_tests
```

## 📊 Power Analysis and Sample Size

### Statistical Power Calculations

```python
class PowerAnalysis:
    """Statistical power analysis for experiment planning"""
    
    def __init__(self):
        self.alpha = 0.05
        self.target_power = 0.8
        
    def calculate_required_sample_size(self, effect_size, test_type='paired_t'):
        """
        Calculate required sample size for desired power
        
        Args:
            effect_size: Expected Cohen's d
            test_type: Type of statistical test
            
        Returns:
            Required sample size per group
        """
        if test_type == 'paired_t':
            # Power calculation for paired t-test
            from scipy import stats
            
            # Approximate sample size calculation
            z_alpha = stats.norm.ppf(1 - self.alpha/2)
            z_beta = stats.norm.ppf(self.target_power)
            
            n_required = ((z_alpha + z_beta) / effect_size) ** 2
            
            return math.ceil(n_required)
            
    def post_hoc_power_analysis(self, observed_effect_size, sample_size, test_type='paired_t'):
        """
        Calculate achieved power given observed effect and sample size
        """
        if test_type == 'paired_t':
            # Calculate achieved power
            z_alpha = stats.norm.ppf(1 - self.alpha/2)
            
            # Non-centrality parameter
            ncp = observed_effect_size * math.sqrt(sample_size)
            
            # Power calculation
            power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
            
            return power
```

## 📈 Reporting and Visualization

### Statistical Results Reporting

```python
class StatisticalReporter:
    """Generate comprehensive statistical reports"""
    
    def __init__(self):
        self.formatter = StatisticalFormatter()
        
    def generate_mission_report(self, validation_result):
        """Generate detailed statistical report for a mission"""
        report = {
            'mission': validation_result.mission,
            'summary': {
                'significant_tests': len([t for t in validation_result.tests if t.significant]),
                'total_tests': len(validation_result.tests),
                'gates_passed': validation_result.gates_passed,
                'overall_recommendation': validation_result.recommendation
            },
            'primary_results': [],
            'secondary_results': [],
            'effect_sizes': [],
            'confidence_intervals': []
        }
        
        for test in validation_result.tests:
            test_report = {
                'test_name': test.test_name,
                'metric': test.metric,
                'statistic': test.statistic,
                'p_value': test.p_value,
                'corrected_p_value': test.corrected_p_value,
                'significant': test.significant,
                'effect_size': test.effect_size,
                'confidence_interval': [test.ci_lower, test.ci_upper],
                'interpretation': self._interpret_result(test)
            }
            
            if test.is_primary:
                report['primary_results'].append(test_report)
            else:
                report['secondary_results'].append(test_report)
                
        return report
        
    def generate_fleet_summary(self, fleet_validation):
        """Generate executive summary of fleet validation"""
        summary = {
            'fleet_status': 'SUCCESS' if fleet_validation.overall_fleet_success else 'FAILURE',
            'missions_successful': sum(1 for v in fleet_validation.mission_validations.values() 
                                     if v.overall_significant),
            'total_missions': len(fleet_validation.mission_validations),
            'significant_results': fleet_validation.fleet_fdr_results.n_significant,
            'total_tests': fleet_validation.fleet_fdr_results.n_tests,
            'fdr_controlled': True,
            'integration_tests_passed': all(t.significant for t in fleet_validation.integration_tests),
            'recommendations': self._generate_fleet_recommendations(fleet_validation)
        }
        
        return summary
```

### Visualization of Statistical Results

```python
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticalVisualizer:
    """Create visualizations of statistical results"""
    
    def __init__(self):
        self.style_config = {
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12
        }
        plt.rcParams.update(self.style_config)
        
    def plot_confidence_intervals(self, validation_results):
        """Plot confidence intervals for all missions"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (mission, results) in enumerate(validation_results.items()):
            ax = axes[i]
            
            # Extract primary test results
            primary_test = results.get_primary_test()
            
            # Plot confidence interval
            self._plot_single_ci(
                ax, 
                primary_test.effect_size,
                primary_test.ci_lower,
                primary_test.ci_upper,
                mission
            )
            
        plt.tight_layout()
        return fig
        
    def plot_effect_sizes(self, validation_results):
        """Forest plot of effect sizes across missions"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        missions = []
        effect_sizes = []
        ci_lowers = []
        ci_uppers = []
        
        for mission, results in validation_results.items():
            primary_test = results.get_primary_test()
            missions.append(mission.replace('mission_', 'Mission ').upper())
            effect_sizes.append(primary_test.effect_size)
            ci_lowers.append(primary_test.ci_lower)
            ci_uppers.append(primary_test.ci_upper)
            
        # Forest plot
        y_positions = range(len(missions))
        
        ax.errorbar(effect_sizes, y_positions, 
                   xerr=[np.array(effect_sizes) - np.array(ci_lowers),
                         np.array(ci_uppers) - np.array(effect_sizes)],
                   fmt='o', capsize=5, capthick=2, elinewidth=2)
                   
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(missions)
        ax.set_xlabel("Effect Size (Cohen's d)")
        ax.set_title("BEM Fleet Effect Sizes with 95% Confidence Intervals")
        ax.grid(True, alpha=0.3)
        
        return fig
        
    def plot_p_value_distribution(self, p_values, corrected_p_values):
        """Plot distribution of raw and corrected p-values"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Raw p-values
        ax1.hist(p_values, bins=20, alpha=0.7, color='blue')
        ax1.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
        ax1.set_xlabel('Raw p-values')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Raw p-values')
        ax1.legend()
        
        # Corrected p-values
        ax2.hist(corrected_p_values, bins=20, alpha=0.7, color='green')
        ax2.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
        ax2.set_xlabel('FDR-Corrected p-values')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of FDR-Corrected p-values')
        ax2.legend()
        
        plt.tight_layout()
        return fig
```

## 🎯 Quality Assurance and Validation

### Statistical Quality Checks

```python
class StatisticalQualityAssurance:
    """Quality assurance for statistical procedures"""
    
    def __init__(self):
        self.checks = [
            self.check_sample_sizes,
            self.check_data_quality,
            self.check_assumptions,
            self.check_effect_sizes,
            self.check_multiple_testing
        ]
        
    def validate_statistical_pipeline(self, mission_data, validation_results):
        """Run complete statistical quality assurance"""
        qa_results = {}
        
        for check in self.checks:
            try:
                check_result = check(mission_data, validation_results)
                qa_results[check.__name__] = check_result
            except Exception as e:
                qa_results[check.__name__] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                
        return StatisticalQAReport(
            qa_results=qa_results,
            overall_status=self._determine_overall_qa_status(qa_results),
            recommendations=self._generate_qa_recommendations(qa_results)
        )
        
    def check_sample_sizes(self, mission_data, validation_results):
        """Verify adequate sample sizes for statistical power"""
        power_analyzer = PowerAnalysis()
        
        sample_size_checks = {}
        for mission, data in mission_data.items():
            n_samples = len(data['treatment'])
            expected_effect_size = validation_results[mission].expected_effect_size
            
            required_n = power_analyzer.calculate_required_sample_size(expected_effect_size)
            achieved_power = power_analyzer.post_hoc_power_analysis(
                validation_results[mission].observed_effect_size, 
                n_samples
            )
            
            sample_size_checks[mission] = {
                'actual_n': n_samples,
                'required_n': required_n,
                'achieved_power': achieved_power,
                'adequate': n_samples >= required_n and achieved_power >= 0.8
            }
            
        return sample_size_checks
        
    def check_data_quality(self, mission_data, validation_results):
        """Check data quality and completeness"""
        quality_checks = {}
        
        for mission, data in mission_data.items():
            treatment_data = data['treatment']
            control_data = data['control']
            
            quality_checks[mission] = {
                'missing_values': {
                    'treatment': np.sum(np.isnan(treatment_data)),
                    'control': np.sum(np.isnan(control_data))
                },
                'outliers': {
                    'treatment': self._count_outliers(treatment_data),
                    'control': self._count_outliers(control_data)
                },
                'data_balance': abs(len(treatment_data) - len(control_data)) / len(treatment_data),
                'quality_score': self._calculate_quality_score(treatment_data, control_data)
            }
            
        return quality_checks
```

This comprehensive statistical framework ensures that the BEM Fleet produces research results that meet the highest standards of statistical rigor, appropriate for publication in top-tier academic venues while providing robust evidence for practical deployment decisions.