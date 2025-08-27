#!/usr/bin/env python3
"""Test script to validate robustness analysis."""

import json
from pathlib import Path
from analysis.robustness_analysis import RobustnessAnalyzer

def test_robustness_analysis():
    """Test robustness analysis with synthetic data."""
    
    # Set up analyzer
    analyzer = RobustnessAnalyzer(n_bootstrap=1000, alpha=0.05, min_effect_size=0.01)
    
    # Define paths
    s1_path = 'logs/S1_stack/eval.json'
    robustness_paths = {
        'R1_robustness_second_model': 'logs/R1_robustness_second_model/eval.json',
        'ADV_adversarial_retrieval': 'logs/ADV_adversarial_retrieval/eval.json'
    }
    
    # Campaign config
    campaign_config = {
        'campaign_id': 'v1.3-stack-phase4-robustness',
        'reference_config': 'S1_stack',
        'robustness_tests': {
            'R1_robustness_second_model': {
                'test_type': 'cross_model_generalization',
                'base_model': 'microsoft/DialoGPT-medium',
                'validation_type': 'same_direction_overlapping_ci'
            },
            'ADV_adversarial_retrieval': {
                'test_type': 'adversarial_stress_testing',
                'max_degradation_pct': 5.0,
                'monotonicity_required': True
            }
        }
    }
    
    try:
        print("🔬 Testing complete robustness campaign analysis...")
        campaign_result = analyzer.analyze_robustness_campaign(
            reference_results_path=s1_path,
            robustness_results_paths=robustness_paths,
            campaign_config=campaign_config
        )
        
        print("✅ Robustness campaign analysis completed!")
        print(f"Campaign ID: {campaign_result.campaign_id}")
        print(f"Overall robustness score: {campaign_result.overall_robustness_score:.1f}/100")
        print(f"Robustness evidence: {campaign_result.robustness_evidence}")
        print(f"Quality gates passed: {campaign_result.robustness_gates_passed}/{campaign_result.total_robustness_gates}")
        print(f"Promotion recommendation: {campaign_result.promotion_recommendation}")
        
        return campaign_result
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_robustness_analysis()