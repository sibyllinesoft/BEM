#!/usr/bin/env python3
"""
Automated Claim Promotion Engine for BEM Research Validation

Create promotion/demotion rules based on statistical evidence with honesty layer 
for transparent failure reporting. Only promotes statistically validated claims.
"""

import json
import logging
import yaml
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

import numpy as np
from statistical_validator import StatisticalValidationResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromotionStatus(Enum):
    """Status of claim promotion."""
    PROMOTED = "promoted"
    CONDITIONALLY_PROMOTED = "conditionally_promoted"  
    DEMOTED = "demoted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

class PromotionReason(Enum):
    """Reasons for promotion/demotion decisions."""
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    CONFIDENCE_INTERVAL = "confidence_interval"
    EFFECT_SIZE = "effect_size"
    PRODUCTION_SLO = "production_slo"
    REPLICATION_FAILURE = "replication_failure"
    MULTIPLE_TESTING_CORRECTION = "multiple_testing_correction"

@dataclass
class PromotionCriteria:
    """Criteria for claim promotion."""
    requires_statistical_significance: bool = True
    requires_ci_excludes_null: bool = True
    requires_minimum_effect_size: bool = True
    requires_production_slo: bool = False
    significance_threshold: float = 0.05
    minimum_effect_size: float = 0.5
    confidence_level: float = 0.95

@dataclass
class PromotionDecision:
    """Decision about claim promotion."""
    claim_id: str
    original_claim: str
    status: PromotionStatus
    promoted_claim: Optional[str]
    evidence_summary: Dict[str, Any]
    reasons: List[PromotionReason]
    confidence_score: float
    recommendations: List[str]

@dataclass
class HonestyReport:
    """Transparent reporting of failed claims."""
    failed_claims: List[PromotionDecision]
    partial_successes: List[PromotionDecision]
    methodology_limitations: List[str]
    data_quality_issues: List[str]
    recommendations_for_improvement: List[str]

class ClaimPromotionEngine:
    """Engine for promoting/demoting claims based on statistical evidence."""
    
    def __init__(self, 
                 claim_configs: Dict[str, Any],
                 promotion_criteria: Optional[PromotionCriteria] = None):
        
        self.claim_configs = claim_configs
        self.promotion_criteria = promotion_criteria or PromotionCriteria()
        
        # Templates for claim modifications
        self.claim_templates = {
            'accuracy_improvement': {
                'strong': "BEM achieves {improvement:.1f}% better accuracy than {baseline} (95% CI: {ci_lower:.1f}%-{ci_upper:.1f}%, p<{p_value:.3f})",
                'moderate': "BEM shows {improvement:.1f}% accuracy improvement over {baseline} (95% CI: {ci_lower:.1f}%-{ci_upper:.1f}%)",
                'weak': "BEM demonstrates potential accuracy benefits over {baseline}, though results require further validation"
            },
            'robustness_improvement': {
                'strong': "BEM reduces performance degradation by {improvement:.1f} percentage points compared to {baseline} under distribution shifts (95% CI: {ci_lower:.1f}-{ci_upper:.1f}pp)",
                'moderate': "BEM shows improved robustness with {improvement:.1f}pp less degradation than {baseline}",
                'weak': "BEM exhibits some robustness advantages over {baseline}, pending additional validation"
            },
            'production_slo': {
                'strong': "BEM maintains {performance:.1f}% of baseline performance while meeting production SLOs",
                'moderate': "BEM achieves acceptable production performance ({performance:.1f}% of baseline)",
                'weak': "BEM shows promise for production deployment with performance optimizations"
            }
        }
    
    def evaluate_claim_promotion(self, 
                                validation_results: Dict[str, StatisticalValidationResult]) -> List[PromotionDecision]:
        """Evaluate all claims for promotion based on validation results."""
        
        logger.info(f"Evaluating {len(validation_results)} claims for promotion")
        
        promotion_decisions = []
        
        for claim_id, validation_result in validation_results.items():
            decision = self._evaluate_single_claim(claim_id, validation_result)
            promotion_decisions.append(decision)
            
        # Log promotion summary
        promoted_count = sum(1 for d in promotion_decisions if d.status == PromotionStatus.PROMOTED)
        demoted_count = sum(1 for d in promotion_decisions if d.status == PromotionStatus.DEMOTED)
        
        logger.info(f"Promotion results: {promoted_count} promoted, {demoted_count} demoted")
        
        return promotion_decisions
    
    def _evaluate_single_claim(self, 
                              claim_id: str, 
                              validation_result: StatisticalValidationResult) -> PromotionDecision:
        """Evaluate a single claim for promotion."""
        
        # Get original claim configuration
        claim_config = self._find_claim_config(claim_id)
        original_claim = claim_config.get('claim', 'Unknown claim') if claim_config else 'Unknown claim'
        
        # Evaluate promotion criteria
        criteria_results = self._check_promotion_criteria(validation_result)
        
        # Determine promotion status
        status, reasons = self._determine_promotion_status(criteria_results, validation_result)
        
        # Generate promoted claim text if applicable
        promoted_claim = None
        if status in [PromotionStatus.PROMOTED, PromotionStatus.CONDITIONALLY_PROMOTED]:
            promoted_claim = self._generate_promoted_claim(claim_id, validation_result, status)
            
        # Compute confidence score
        confidence_score = self._compute_confidence_score(criteria_results, validation_result)
        
        # Generate recommendations
        recommendations = self._generate_claim_recommendations(
            claim_id, validation_result, criteria_results, status
        )
        
        # Compile evidence summary
        evidence_summary = {
            'p_value': validation_result.p_value,
            'effect_size': validation_result.effect_size,
            'confidence_interval': validation_result.bootstrap_result.confidence_interval,
            'statistical_significance': validation_result.passes_significance_test,
            'ci_excludes_null': validation_result.confidence_interval_excludes_null,
            'meets_effect_size': validation_result.meets_minimum_effect_size,
            'observed_statistic': validation_result.observed_statistic
        }
        
        return PromotionDecision(
            claim_id=claim_id,
            original_claim=original_claim,
            status=status,
            promoted_claim=promoted_claim,
            evidence_summary=evidence_summary,
            reasons=reasons,
            confidence_score=confidence_score,
            recommendations=recommendations
        )
    
    def _find_claim_config(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Find claim configuration by ID."""
        
        # Search through different claim categories
        for category in ['accuracy_claims', 'robustness_claims', 'production_claims', 'domain_shift_claims']:
            if category in self.claim_configs and claim_id in self.claim_configs[category]:
                return self.claim_configs[category][claim_id]
                
        return None
    
    def _check_promotion_criteria(self, 
                                 validation_result: StatisticalValidationResult) -> Dict[str, bool]:
        """Check all promotion criteria."""
        
        criteria_results = {}
        
        # Statistical significance
        if self.promotion_criteria.requires_statistical_significance:
            criteria_results['statistical_significance'] = validation_result.passes_significance_test
            
        # Confidence interval excludes null
        if self.promotion_criteria.requires_ci_excludes_null:
            criteria_results['ci_excludes_null'] = validation_result.confidence_interval_excludes_null
            
        # Minimum effect size
        if self.promotion_criteria.requires_minimum_effect_size:
            criteria_results['minimum_effect_size'] = validation_result.meets_minimum_effect_size
            
        # Production SLO (if applicable)
        if self.promotion_criteria.requires_production_slo:
            # Would check against production metrics
            criteria_results['production_slo'] = True  # Placeholder
            
        return criteria_results
    
    def _determine_promotion_status(self, 
                                   criteria_results: Dict[str, bool],
                                   validation_result: StatisticalValidationResult) -> Tuple[PromotionStatus, List[PromotionReason]]:
        """Determine promotion status based on criteria."""
        
        reasons = []
        
        # Count how many criteria are met
        total_criteria = len(criteria_results)
        met_criteria = sum(criteria_results.values())
        
        # Check individual criteria for reasons
        if not criteria_results.get('statistical_significance', True):
            reasons.append(PromotionReason.STATISTICAL_SIGNIFICANCE)
            
        if not criteria_results.get('ci_excludes_null', True):
            reasons.append(PromotionReason.CONFIDENCE_INTERVAL)
            
        if not criteria_results.get('minimum_effect_size', True):
            reasons.append(PromotionReason.EFFECT_SIZE)
            
        if not criteria_results.get('production_slo', True):
            reasons.append(PromotionReason.PRODUCTION_SLO)
            
        # Multiple testing correction failure
        corrected_p = validation_result.raw_data.get('corrected_p_value')
        if corrected_p and corrected_p >= 0.05:
            reasons.append(PromotionReason.MULTIPLE_TESTING_CORRECTION)
            
        # Determine status
        if met_criteria == total_criteria and len(reasons) == 0:
            return PromotionStatus.PROMOTED, []
        elif met_criteria >= total_criteria * 0.75:  # At least 75% of criteria met
            return PromotionStatus.CONDITIONALLY_PROMOTED, reasons
        elif met_criteria >= total_criteria * 0.25:  # At least 25% of criteria met
            return PromotionStatus.DEMOTED, reasons
        else:
            return PromotionStatus.INSUFFICIENT_EVIDENCE, reasons
    
    def _generate_promoted_claim(self, 
                                claim_id: str,
                                validation_result: StatisticalValidationResult,
                                status: PromotionStatus) -> str:
        """Generate promoted claim text based on evidence strength."""
        
        claim_config = self._find_claim_config(claim_id)
        if not claim_config:
            return "Evidence-based claim (details in statistical report)"
            
        # Determine claim type and template
        claim_type = self._determine_claim_type(claim_id, claim_config)
        template_strength = 'strong' if status == PromotionStatus.PROMOTED else 'moderate'
        
        if claim_type not in self.claim_templates:
            return f"Validated claim: {claim_config.get('claim', 'Statistical evidence available')}"
            
        template = self.claim_templates[claim_type][template_strength]
        
        # Extract parameters for template
        ci_lower, ci_upper = validation_result.bootstrap_result.confidence_interval
        
        try:
            if claim_type == 'accuracy_improvement':
                # Convert confidence interval to percentage improvement
                improvement = validation_result.observed_statistic * 100
                
                return template.format(
                    improvement=improvement,
                    baseline=claim_config.get('baseline_method', 'baseline'),
                    ci_lower=ci_lower * 100,
                    ci_upper=ci_upper * 100,
                    p_value=validation_result.p_value
                )
                
            elif claim_type == 'robustness_improvement':
                improvement = validation_result.observed_statistic
                
                return template.format(
                    improvement=improvement,
                    baseline=claim_config.get('baseline_method', 'baseline'),
                    ci_lower=ci_lower,
                    ci_upper=ci_upper
                )
                
            elif claim_type == 'production_slo':
                performance = validation_result.observed_statistic * 100
                
                return template.format(
                    performance=performance
                )
                
        except (KeyError, ValueError) as e:
            logger.warning(f"Could not format claim template: {e}")
            
        # Fallback to generic evidence-based claim
        return f"BEM demonstrates statistically significant improvement (p={validation_result.p_value:.3f}, effect size: {validation_result.effect_size:.2f})"
    
    def _determine_claim_type(self, claim_id: str, claim_config: Dict[str, Any]) -> str:
        """Determine the type of claim for template selection."""
        
        if 'accuracy' in claim_id.lower() or 'em_score' in claim_config.get('metric', ''):
            return 'accuracy_improvement'
        elif 'degradation' in claim_id.lower() or 'robustness' in claim_id.lower():
            return 'robustness_improvement'
        elif 'production' in claim_id.lower() or 'slo' in claim_id.lower():
            return 'production_slo'
        else:
            return 'accuracy_improvement'  # Default
    
    def _compute_confidence_score(self, 
                                 criteria_results: Dict[str, bool],
                                 validation_result: StatisticalValidationResult) -> float:
        """Compute overall confidence score (0-1)."""
        
        confidence_factors = []
        
        # Criteria satisfaction rate
        if criteria_results:
            criteria_score = sum(criteria_results.values()) / len(criteria_results)
            confidence_factors.append(criteria_score)
            
        # P-value strength (lower p-value = higher confidence)
        p_value = validation_result.p_value
        p_value_score = max(0.0, min(1.0, (0.05 - p_value) / 0.05)) if p_value <= 0.05 else 0.0
        confidence_factors.append(p_value_score)
        
        # Effect size magnitude
        effect_size = abs(validation_result.effect_size)
        effect_size_score = min(1.0, effect_size / 1.0)  # Normalize to Cohen's large effect (0.8)
        confidence_factors.append(effect_size_score)
        
        # Confidence interval precision (narrower CI = higher confidence)
        ci_lower, ci_upper = validation_result.bootstrap_result.confidence_interval
        ci_width = ci_upper - ci_lower
        
        if validation_result.observed_statistic != 0:
            ci_precision_score = max(0.0, 1.0 - (ci_width / abs(validation_result.observed_statistic)))
        else:
            ci_precision_score = 0.5  # Neutral score
            
        confidence_factors.append(ci_precision_score)
        
        # Overall confidence (weighted average)
        weights = [0.3, 0.3, 0.2, 0.2]  # Criteria, p-value, effect size, CI precision
        
        return sum(w * f for w, f in zip(weights, confidence_factors))
    
    def _generate_claim_recommendations(self, 
                                      claim_id: str,
                                      validation_result: StatisticalValidationResult,
                                      criteria_results: Dict[str, bool],
                                      status: PromotionStatus) -> List[str]:
        """Generate recommendations for claim improvement or usage."""
        
        recommendations = []
        
        # Status-specific recommendations
        if status == PromotionStatus.PROMOTED:
            recommendations.append("Claim has strong statistical support and can be promoted with confidence.")
            
        elif status == PromotionStatus.CONDITIONALLY_PROMOTED:
            recommendations.append("Claim has moderate support. Consider presenting with appropriate caveats.")
            
            # Specific improvement suggestions
            if not criteria_results.get('statistical_significance', True):
                recommendations.append("Consider increasing sample size or effect size to achieve statistical significance.")
                
            if not criteria_results.get('minimum_effect_size', True):
                recommendations.append("Effect size is small. Consider practical significance and replication studies.")
                
        elif status == PromotionStatus.DEMOTED:
            recommendations.append("Claim lacks sufficient statistical support and should be demoted or removed.")
            
            # Specific issues to address
            if validation_result.p_value > 0.05:
                recommendations.append(f"P-value ({validation_result.p_value:.3f}) exceeds significance threshold. Need stronger evidence.")
                
            if not validation_result.confidence_interval_excludes_null:
                ci_lower, ci_upper = validation_result.bootstrap_result.confidence_interval
                recommendations.append(f"Confidence interval [{ci_lower:.3f}, {ci_upper:.3f}] includes null hypothesis.")
                
        else:  # INSUFFICIENT_EVIDENCE
            recommendations.append("Insufficient evidence to make promotion decision. Additional data collection recommended.")
            recommendations.append("Consider pilot study or methodological improvements before full validation.")
            
        # General methodological recommendations
        if validation_result.effect_size < 0.2:
            recommendations.append("Very small effect size. Consider practical significance and power analysis.")
            
        if 'corrected_p_value' in validation_result.raw_data:
            corrected_p = validation_result.raw_data['corrected_p_value']
            if corrected_p > validation_result.p_value:
                recommendations.append(f"Multiple testing correction increased p-value to {corrected_p:.3f}. Consider focusing on fewer claims.")
                
        return recommendations

class HonestyEngine:
    """Engine for transparent failure reporting and methodology limitations."""
    
    def __init__(self):
        self.failure_categories = {
            'statistical_power': "Insufficient statistical power to detect claimed effects",
            'effect_size': "Effect sizes smaller than claimed or practically insignificant", 
            'replication': "Results not consistent across experimental conditions",
            'multiple_testing': "Claims fail multiple testing correction",
            'measurement': "Issues with measurement validity or reliability",
            'methodology': "Limitations in experimental design or analysis approach"
        }
        
    def generate_honesty_report(self, 
                              promotion_decisions: List[PromotionDecision],
                              validation_metadata: Dict[str, Any]) -> HonestyReport:
        """Generate comprehensive honesty report."""
        
        logger.info("Generating transparency and honesty report")
        
        # Categorize decisions
        failed_claims = [d for d in promotion_decisions if d.status == PromotionStatus.DEMOTED]
        partial_successes = [d for d in promotion_decisions if d.status == PromotionStatus.CONDITIONALLY_PROMOTED]
        
        # Analyze failure patterns
        methodology_limitations = self._identify_methodology_limitations(
            promotion_decisions, validation_metadata
        )
        
        # Identify data quality issues
        data_quality_issues = self._identify_data_quality_issues(
            promotion_decisions, validation_metadata
        )
        
        # Generate improvement recommendations
        improvement_recommendations = self._generate_improvement_recommendations(
            failed_claims, partial_successes, methodology_limitations
        )
        
        return HonestyReport(
            failed_claims=failed_claims,
            partial_successes=partial_successes,
            methodology_limitations=methodology_limitations,
            data_quality_issues=data_quality_issues,
            recommendations_for_improvement=improvement_recommendations
        )
    
    def _identify_methodology_limitations(self, 
                                        decisions: List[PromotionDecision],
                                        metadata: Dict[str, Any]) -> List[str]:
        """Identify methodological limitations."""
        
        limitations = []
        
        # Sample size limitations
        total_tests = metadata.get('multiple_testing_correction', {}).get('n_tests', 0)
        if total_tests > 10:
            limitations.append(
                f"Large number of statistical tests ({total_tests}) increases multiple comparison burden"
            )
            
        # Effect size patterns
        effect_sizes = [d.evidence_summary['effect_size'] for d in decisions]
        small_effects = sum(1 for es in effect_sizes if abs(es) < 0.3)
        
        if small_effects > len(effect_sizes) * 0.5:
            limitations.append(
                f"Many claims show small effect sizes ({small_effects}/{len(effect_sizes)}), limiting practical significance"
            )
            
        # Statistical power issues
        high_p_values = sum(1 for d in decisions if d.evidence_summary['p_value'] > 0.05)
        if high_p_values > len(decisions) * 0.3:
            limitations.append(
                f"High proportion of non-significant results ({high_p_values}/{len(decisions)}) suggests possible power issues"
            )
            
        return limitations
    
    def _identify_data_quality_issues(self, 
                                    decisions: List[PromotionDecision],
                                    metadata: Dict[str, Any]) -> List[str]:
        """Identify data quality issues."""
        
        issues = []
        
        # Wide confidence intervals suggest high variance
        wide_cis = 0
        for decision in decisions:
            ci_lower, ci_upper = decision.evidence_summary['confidence_interval']
            ci_width = ci_upper - ci_lower
            observed = abs(decision.evidence_summary['observed_statistic'])
            
            if observed > 0 and (ci_width / observed) > 1.0:  # CI width > effect size
                wide_cis += 1
                
        if wide_cis > len(decisions) * 0.3:
            issues.append(
                f"Wide confidence intervals in {wide_cis} claims suggest high measurement variance or small sample sizes"
            )
            
        # Inconsistent results across conditions
        failed_significance = sum(1 for d in decisions if not d.evidence_summary['statistical_significance'])
        if failed_significance > 0:
            issues.append(
                f"Statistical significance failures in {failed_significance} claims may indicate inconsistent effects"
            )
            
        return issues
    
    def _generate_improvement_recommendations(self, 
                                            failed_claims: List[PromotionDecision],
                                            partial_successes: List[PromotionDecision],
                                            limitations: List[str]) -> List[str]:
        """Generate recommendations for improving the research."""
        
        recommendations = []
        
        if failed_claims:
            recommendations.append(
                f"Consider removing or substantially revising {len(failed_claims)} failed claims from main results"
            )
            
            # Analyze common failure reasons
            common_reasons = {}
            for claim in failed_claims:
                for reason in claim.reasons:
                    common_reasons[reason] = common_reasons.get(reason, 0) + 1
                    
            if PromotionReason.STATISTICAL_SIGNIFICANCE in common_reasons:
                recommendations.append(
                    "Increase sample sizes or improve experimental design to achieve statistical significance"
                )
                
            if PromotionReason.EFFECT_SIZE in common_reasons:
                recommendations.append(
                    "Focus on conditions with larger effect sizes or investigate why effects are small"
                )
                
        if partial_successes:
            recommendations.append(
                f"Present {len(partial_successes)} partially supported claims with appropriate caveats and confidence intervals"
            )
            
        if limitations:
            recommendations.append(
                "Address methodological limitations through improved experimental design or additional data collection"
            )
            
        # General recommendations
        recommendations.append(
            "Consider pre-registration of hypotheses and analysis plans to improve research credibility"
        )
        
        recommendations.append(
            "Report all attempted validations, including null results, for complete transparency"
        )
        
        return recommendations

class PromotionOrchestrator:
    """Main orchestrator for claim promotion and honesty reporting."""
    
    def __init__(self, 
                 claim_configs: Dict[str, Any],
                 promotion_criteria: Optional[PromotionCriteria] = None,
                 output_dir: str = "experiments/promotions"):
        
        self.promotion_engine = ClaimPromotionEngine(claim_configs, promotion_criteria)
        self.honesty_engine = HonestyEngine()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_validation_results(self, 
                                 validation_results: Dict[str, StatisticalValidationResult],
                                 validation_metadata: Dict[str, Any]) -> Tuple[List[PromotionDecision], HonestyReport]:
        """Process validation results through promotion and honesty engines."""
        
        logger.info("Processing validation results for claim promotion")
        
        # Evaluate claims for promotion
        promotion_decisions = self.promotion_engine.evaluate_claim_promotion(validation_results)
        
        # Generate honesty report
        honesty_report = self.honesty_engine.generate_honesty_report(
            promotion_decisions, validation_metadata
        )
        
        # Save results
        self._save_promotion_results(promotion_decisions, honesty_report)
        
        return promotion_decisions, honesty_report
    
    def _save_promotion_results(self, 
                              decisions: List[PromotionDecision], 
                              honesty_report: HonestyReport) -> None:
        """Save promotion results and honesty report."""
        
        # Convert decisions to JSON-serializable format
        decisions_dict = []
        for decision in decisions:
            decision_dict = asdict(decision)
            # Convert enums to strings
            decision_dict['status'] = decision.status.value
            decision_dict['reasons'] = [r.value for r in decision.reasons]
            decisions_dict.append(decision_dict)
            
        # Save promotion decisions
        decisions_file = self.output_dir / "promotion_decisions.json"
        with open(decisions_file, "w") as f:
            json.dump(decisions_dict, f, indent=2)
            
        # Convert honesty report to JSON-serializable format
        honesty_dict = asdict(honesty_report)
        # Convert nested PromotionDecision objects
        for key in ['failed_claims', 'partial_successes']:
            honesty_dict[key] = [
                {**asdict(d), 'status': d.status.value, 'reasons': [r.value for r in d.reasons]}
                for d in honesty_dict[key]
            ]
            
        # Save honesty report
        honesty_file = self.output_dir / "honesty_report.json"
        with open(honesty_file, "w") as f:
            json.dump(honesty_dict, f, indent=2)
            
        # Generate human-readable summary
        self._generate_promotion_summary(decisions, honesty_report)
        
        logger.info(f"Saved promotion results to {self.output_dir}")
    
    def _generate_promotion_summary(self, 
                                  decisions: List[PromotionDecision], 
                                  honesty_report: HonestyReport) -> None:
        """Generate human-readable promotion summary."""
        
        promoted = [d for d in decisions if d.status == PromotionStatus.PROMOTED]
        conditionally_promoted = [d for d in decisions if d.status == PromotionStatus.CONDITIONALLY_PROMOTED]
        demoted = [d for d in decisions if d.status == PromotionStatus.DEMOTED]
        
        summary = f"""# BEM Claim Promotion Summary

## Overview
- **Total Claims Evaluated**: {len(decisions)}
- **Promoted**: {len(promoted)}
- **Conditionally Promoted**: {len(conditionally_promoted)}
- **Demoted**: {len(demoted)}

## Promoted Claims (Ready for Publication)
"""
        
        for decision in promoted:
            summary += f"\n### {decision.claim_id}\n"
            summary += f"**Original**: {decision.original_claim}\n\n"
            summary += f"**Promoted**: {decision.promoted_claim}\n\n"
            summary += f"**Confidence**: {decision.confidence_score:.3f}\n\n"
            
        summary += "\n## Conditionally Promoted Claims (Require Caveats)\n"
        
        for decision in conditionally_promoted:
            summary += f"\n### {decision.claim_id}\n"
            summary += f"**Claim**: {decision.promoted_claim or decision.original_claim}\n\n"
            summary += f"**Issues**: {', '.join([r.value for r in decision.reasons])}\n\n"
            
        summary += "\n## Demoted Claims (Removed from Main Results)\n"
        
        for decision in demoted:
            summary += f"\n### {decision.claim_id}\n"
            summary += f"**Original**: {decision.original_claim}\n\n"
            summary += f"**Reasons for Demotion**: {', '.join([r.value for r in decision.reasons])}\n\n"
            
        summary += f"\n## Honesty Report\n"
        summary += f"\n### Methodology Limitations\n"
        
        for limitation in honesty_report.methodology_limitations:
            summary += f"- {limitation}\n"
            
        summary += f"\n### Recommendations for Improvement\n"
        
        for recommendation in honesty_report.recommendations_for_improvement:
            summary += f"- {recommendation}\n"
            
        # Save summary
        summary_file = self.output_dir / "promotion_summary.md"
        with open(summary_file, "w") as f:
            f.write(summary)

def main():
    """Example usage of promotion engine."""
    
    # Load claim configurations
    claim_configs = {
        'accuracy_claims': {
            'static_lora_advantage': {
                'claim': 'BEM +41.7% better accuracy than Static LoRA',
                'metric': 'em_score',
                'baseline_method': 'static_lora',
                'expected_improvement_pct': 41.7
            }
        }
    }
    
    # Mock validation results
    from statistical_validator import BCaBootstrapValidator, BootstrapResult, StatisticalValidationResult
    
    mock_bootstrap_result = BootstrapResult(
        statistic=0.417,
        confidence_interval=(0.25, 0.58),
        confidence_level=0.95,
        bootstrap_distribution=np.random.normal(0.417, 0.1, 1000),
        bias_correction=0.01,
        acceleration=0.02
    )
    
    mock_validation = StatisticalValidationResult(
        claim_id='static_lora_advantage',
        metric_name='em_score',
        observed_statistic=0.417,
        bootstrap_result=mock_bootstrap_result,
        p_value=0.003,
        effect_size=1.2,
        effect_size_interpretation='large',
        passes_significance_test=True,
        confidence_interval_excludes_null=True,
        meets_minimum_effect_size=True,
        raw_data={'bem_scores': [0.81, 0.82, 0.80], 'baseline_scores': [0.78, 0.77, 0.79]}
    )
    
    validation_results = {'static_lora_advantage': mock_validation}
    validation_metadata = {'n_tests': 1, 'methodology': 'BCa_bootstrap'}
    
    # Initialize orchestrator
    orchestrator = PromotionOrchestrator(claim_configs)
    
    # Process results
    decisions, honesty_report = orchestrator.process_validation_results(
        validation_results, validation_metadata
    )
    
    # Print summary
    print("Promotion Results:")
    for decision in decisions:
        print(f"{decision.claim_id}: {decision.status.value} (confidence: {decision.confidence_score:.3f})")
        if decision.promoted_claim:
            print(f"  Promoted: {decision.promoted_claim}")

if __name__ == "__main__":
    main()