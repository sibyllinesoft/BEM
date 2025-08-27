#!/usr/bin/env python3
"""Test script to debug ablation analysis."""

import json
from pathlib import Path
from analysis.ablation_analysis import AblationAnalyzer

def test_ablation_analysis():
    """Test ablation analysis with synthetic data."""
    
    # Set up analyzer
    analyzer = AblationAnalyzer(n_bootstrap=1000, alpha=0.05, min_effect_size=0.01)
    
    # Define paths
    project_root = Path('.')
    s1_path = 'logs/S1_stack/eval.json'
    ablation_paths = {
        'A1_no_diagonal': 'logs/A1_no_diagonal/eval.json',
        'A2_no_hard_negatives': 'logs/A2_no_hard_negatives/eval.json', 
        'A3_fp16_instead_fp8': 'logs/A3_fp16_instead_fp8/eval.json'
    }
    
    # Campaign config
    campaign_config = {
        'campaign_id': 'v1.3-stack-phase3-ablations',
        'full_stack_config': 'S1_stack',
        'ablations': {
            'A1_no_diagonal': {
                'ablation_type': 'F5.2_diagonal_head_removal',
                'ingredient_removed': 'F5.2 Low-Rank + Diagonal',
                'hypothesis': 'Quality degradation (-1 to -2%)'
            },
            'A2_no_hard_negatives': {
                'ablation_type': 'F5.5_hard_negatives_removal',
                'ingredient_removed': 'F5.5 Hard Negatives Training',
                'hypothesis': 'Quality degradation (-1 to -2%)'
            },
            'A3_fp16_instead_fp8': {
                'ablation_type': 'F5.4_fp8_quantization_removal',
                'ingredient_removed': 'F5.4 FP8 Quantization-Aware Training',
                'hypothesis': 'Efficiency degradation (latency/memory)'
            }
        }
    }
    
    try:
        # Run analysis
        print("🔬 Running ablation campaign analysis...")
        result = analyzer.analyze_ablation_campaign(
            full_stack_results_path=s1_path,
            ablation_results_paths=ablation_paths,
            campaign_config=campaign_config
        )
        
        print("✅ Analysis completed successfully!")
        print(f"Campaign ID: {result.campaign_id}")
        print(f"Ablation results count: {len(result.ablation_results)}")
        print(f"All ingredients causal: {result.all_ingredients_causal}")
        print(f"Promotion recommendation: {result.promotion_recommendation}")
        
        return result
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_ablation_analysis()