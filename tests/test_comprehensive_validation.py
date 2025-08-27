#!/usr/bin/env python3
"""Test script to run the comprehensive validation framework."""

import json
from pathlib import Path
from analysis.comprehensive_validation import ComprehensiveValidator

def test_comprehensive_validation():
    """Test the complete validation campaign framework."""
    
    # Set up validator
    validator = ComprehensiveValidator(n_bootstrap=1000, alpha=0.05)
    
    # Define all experiment paths
    experiment_paths = {
        # Reference experiments
        'S0_baseline': 'logs/S0_baseline/eval.json',
        'S1_stack': 'logs/S1_stack/eval.json',
        
        # Phase 3: Ablation experiments
        'A1_no_diagonal': 'logs/A1_no_diagonal/eval.json',
        'A2_no_hard_negatives': 'logs/A2_no_hard_negatives/eval.json',
        'A3_fp16_instead_fp8': 'logs/A3_fp16_instead_fp8/eval.json',
        
        # Phase 4: Robustness experiments
        'R1_robustness_second_model': 'logs/R1_robustness_second_model/eval.json',
        'ADV_adversarial_retrieval': 'logs/ADV_adversarial_retrieval/eval.json'
    }
    
    # Campaign configuration
    campaign_config = {
        'campaign_id': 'v1.3-stack-comprehensive-validation',
        'reference_experiment': 'S1_stack',
        'baseline_experiment': 'S0_baseline',
        
        # Phase 3: Composition ablations
        'ablation_experiments': {
            'A1_no_diagonal': {
                'ablation_type': 'F5.2_diagonal_head_removal',
                'ingredient': 'F5.2 Low-Rank + Diagonal',
                'expected_impact': 'Quality degradation (-1 to -2%)'
            },
            'A2_no_hard_negatives': {
                'ablation_type': 'F5.5_hard_negatives_removal',
                'ingredient': 'F5.5 Hard Negatives Training',
                'expected_impact': 'Quality degradation (-1 to -2%)'
            },
            'A3_fp16_instead_fp8': {
                'ablation_type': 'F5.4_fp8_quantization_removal',
                'ingredient': 'F5.4 FP8 Quantization-Aware Training',
                'expected_impact': 'Efficiency degradation (latency/memory)'
            }
        },
        
        # Phase 4: Robustness validation
        'robustness_experiments': {
            'R1_robustness_second_model': {
                'test_type': 'cross_model_generalization',
                'base_model': 'microsoft/DialoGPT-medium'
            },
            'ADV_adversarial_retrieval': {
                'test_type': 'adversarial_stress_testing',
                'max_degradation_pct': 5.0
            }
        },
        
        # Quality standards
        'promotion_criteria': {
            'min_causal_ingredients': 2,
            'min_robustness_score': 60,
            'max_quality_degradation': 2.0,
            'required_generalization': True
        }
    }
    
    try:
        print("🎯 Running comprehensive v1.3-Stack validation...")
        print(f"   Total experiments: {len(experiment_paths)}")
        print(f"   Ablation tests: {len(campaign_config['ablation_experiments'])}")
        print(f"   Robustness tests: {len(campaign_config['robustness_experiments'])}")
        
        # Run comprehensive validation
        validation_result = validator.run_comprehensive_validation(
            validation_config=campaign_config,
            results_paths=experiment_paths
        )
        
        print("\n✅ Comprehensive validation completed!")
        print(f"📊 VALIDATION SUMMARY:")
        print(f"   Overall validation score: {validation_result.overall_validation_score:.1f}/100")
        print(f"   Quality assessment: {validation_result.quality_assessment}")
        print(f"   Scientific rigor: {validation_result.scientific_rigor}")
        print(f"   Deployment readiness: {validation_result.deployment_readiness}")
        
        print(f"\n🎯 PROMOTION DECISION:")
        print(f"   Recommendation: {validation_result.promotion_recommendation}")
        print(f"   Confidence level: {validation_result.validation_confidence}")
        
        if validation_result.supporting_evidence:
            print(f"\n✅ Supporting Evidence:")
            for evidence in validation_result.supporting_evidence:
                print(f"   • {evidence}")
                
        if validation_result.critical_risks:
            print(f"\n⚠️ Critical Risks:")
            for risk in validation_result.critical_risks:
                print(f"   • {risk}")
        
        # Save summary results (avoiding complex object serialization)
        results_file = Path('analysis/comprehensive_validation_summary.json')
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            # Convert to dict for JSON serialization
            result_dict = {
                'campaign_id': validation_result.campaign_id,
                'overall_validation_score': validation_result.overall_validation_score,
                'deployment_readiness': validation_result.deployment_readiness,
                'promotion_recommendation': validation_result.promotion_recommendation,
                'validation_confidence': validation_result.validation_confidence,
                'supporting_evidence': validation_result.supporting_evidence,
                'critical_risks': validation_result.critical_risks
            }
            json.dump(result_dict, f, indent=2)
            
        print(f"\n📁 Detailed results saved: {results_file}")
        
        return validation_result
        
    except Exception as e:
        print(f"❌ Comprehensive validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_comprehensive_validation()