#!/usr/bin/env python3
"""
Claims Preregistration Script for BEM Real Runs Campaign

Pre-registers claims with thresholds for statistical validation.
Implements B3 phase requirements from TODO.md XML workflow.

Usage:
    python scripts/init_claims.py --out paper/claims.yaml --templates paper/claims_templates
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any
import time

def create_claims_structure() -> Dict[str, Any]:
    """Create the claims structure for BEM Real Runs Campaign."""
    
    claims = {
        "metadata": {
            "preregistration_date": time.strftime("%Y-%m-%d"),
            "campaign": "BEM Real Runs Campaign",
            "version": "1.0.0",
            "description": "Pre-registered claims for BEM-v1.1-stable vs baselines with statistical thresholds"
        },
        
        "statistical_framework": {
            "bootstrap_samples": 10000,
            "confidence_interval": "BCa 95%",
            "multiple_testing_correction": "FDR (Benjamini-Hochberg)",
            "significance_threshold": "CI > 0 (relative improvements)",
            "effect_size_reporting": "Relative percentage improvements: Δ% = (BEM - Baseline) / Baseline"
        },
        
        "experimental_design": {
            "baseline": "L0_static_lora",
            "primary_method": "B1_bem_v11_stable", 
            "comparison_methods": ["V2_dual_path", "V7_film_lite", "V11_learned_cache_policy"],
            "seeds_per_method": 5,
            "evaluation_slices": ["Slice A (Retrieval-Strong)", "Slice B (Full)"],
            "budget_constraint": "±5% parameters/FLOPs"
        },
        
        # Primary Claims (Must ALL be true for publication)
        "primary_claims": {
            
            "claim_1_quality_improvement": {
                "statement": "BEM-v1.1-stable (B1) improves ALL four metrics (EM, F1, BLEU, chrF) on both Slice A and Slice B compared to static LoRA baseline (L0)",
                "hypothesis": "H1: μ(B1) > μ(L0) for all metrics on both slices",
                "null_hypothesis": "H0: μ(B1) ≤ μ(L0) for any metric on any slice", 
                "success_criteria": "BCa 95% CI > 0 for relative improvement on ALL metrics, BOTH slices",
                "statistical_test": "Paired bootstrap with FDR correction per metric family",
                "experiment_ids": ["L0_static_lora", "B1_bem_v11_stable"],
                "auto_downgrade_if_failed": True,
                "priority": "critical"
            },
            
            "claim_2_latency_budget": {
                "statement": "BEM-v1.1-stable maintains p50 latency within +15% of baseline",
                "hypothesis": "H1: p50_latency(B1) ≤ 1.15 × p50_latency(L0)",
                "success_criteria": "Relative latency increase ≤ 15%",
                "threshold": 0.15,
                "experiment_ids": ["L0_static_lora", "B1_bem_v11_stable"],
                "auto_downgrade_if_failed": True,
                "priority": "critical"
            },
            
            "claim_3_cache_efficiency": {
                "statement": "BEM-v1.1-stable achieves KV cache hit rate ≥ 80%",
                "hypothesis": "H1: KV_hit_rate(B1) ≥ 0.80",
                "success_criteria": "KV hit rate ≥ 80% across all evaluation runs",
                "threshold": 0.80,
                "experiment_ids": ["B1_bem_v11_stable"],
                "auto_downgrade_if_failed": True,
                "priority": "critical"
            },
            
            "claim_4_memory_efficiency": {
                "statement": "BEM-v1.1-stable maintains VRAM usage within ±5% of baseline",
                "hypothesis": "H1: |VRAM(B1) - VRAM(L0)| / VRAM(L0) ≤ 0.05",
                "success_criteria": "Relative VRAM change ≤ 5%",
                "threshold": 0.05,
                "experiment_ids": ["L0_static_lora", "B1_bem_v11_stable"],
                "auto_downgrade_if_failed": True,
                "priority": "critical"
            },
            
            "claim_5_no_leakage": {
                "statement": "Policy-over-memory property: routing decisions are based on semantic understanding, not data leakage",
                "hypothesis": "H1: Index-swap evaluation shows clean > shuffled > corrupt performance",
                "success_criteria": "Monotonic degradation with BCa 95% CI separation",
                "experiment_ids": ["B1_bem_v11_stable", "V11_learned_cache_policy"],
                "statistical_test": "Paired comparison with CI separation",
                "auto_downgrade_if_failed": True,
                "priority": "critical"
            }
        },
        
        # Secondary Claims (Nice to have, but not required for publication)
        "secondary_claims": {
            
            "claim_6_variant_performance": {
                "statement": "At least one variant (V2, V7, V11) improves ≥1 primary metric without violating latency/VRAM gates",
                "success_criteria": "BCa 95% CI > 0 for at least one metric, latency ≤ +15%, VRAM ≤ ±5%",
                "experiment_ids": ["V2_dual_path", "V7_film_lite", "V11_learned_cache_policy"],
                "auto_downgrade_if_failed": False,
                "priority": "secondary"
            },
            
            "claim_7_cache_policy_improvement": {
                "statement": "V11 learned cache policy achieves either higher quality than B1 OR lower latency with similar quality",
                "success_criteria": "Pareto improvement: (quality ≥ B1 AND latency < B1) OR (quality > B1 AND latency ≤ 1.15 × B1)",
                "experiment_ids": ["B1_bem_v11_stable", "V11_learned_cache_policy"],
                "auto_downgrade_if_failed": False,
                "priority": "secondary"
            },
            
            "claim_8_routing_stability": {
                "statement": "BEM routing is stable without flip thrashing (flip rate < 0.1 flips/token)",
                "success_criteria": "Mean flip rate < 0.1 across all BEM variants",
                "threshold": 0.1,
                "experiment_ids": ["B1_bem_v11_stable", "V2_dual_path", "V7_film_lite", "V11_learned_cache_policy"],
                "auto_downgrade_if_failed": False,
                "priority": "secondary"
            }
        },
        
        # Quality Gates (Must pass for any result to be reported)
        "quality_gates": {
            
            "data_quality": {
                "leak_detection": {
                    "threshold": 0.01,  # Max 1% leakage allowed
                    "description": "Test/train data leakage via similarity search"
                },
                "coverage_threshold": 0.8,
                "consistency_threshold": 0.8
            },
            
            "experimental_validity": {
                "seed_reproducibility": "All 5 seeds must produce valid results",
                "convergence_check": "Training must converge (loss decreasing)",
                "evaluation_completeness": "All metrics computed for both slices"
            },
            
            "statistical_validity": {
                "minimum_effect_size": 0.01,  # 1% minimum relative improvement
                "bootstrap_convergence": "BCa CI must converge (width < 0.1)",
                "multiple_testing": "FDR correction applied to all metric families"
            }
        },
        
        # Auto-downgrade policies
        "downgrade_policies": {
            "critical_claim_failure": {
                "policy": "If any primary claim fails, automatically downgrade language from 'superior' to 'competitive'",
                "affected_sections": ["abstract", "conclusion", "main_results"]
            },
            
            "latency_budget_exceeded": {
                "policy": "If latency > +15%, downgrade from 'efficient' to 'effective'",
                "affected_sections": ["abstract", "systems_analysis"]
            },
            
            "cache_hit_below_threshold": {
                "policy": "If KV hit < 80%, remove cache efficiency claims",
                "affected_sections": ["systems_analysis", "cache_metrics"]
            },
            
            "no_significant_improvements": {
                "policy": "If BCa CI ≤ 0 for any metric, remove significance stars and qualifying language",
                "affected_sections": ["hero_table", "main_results"]
            }
        },
        
        # Reporting standards
        "reporting_standards": {
            "significance_marking": {
                "rule": "Stars ONLY when BCa 95% CI > 0 for relative improvements",
                "symbols": {"*": "p < 0.05 (BCa CI > 0)", "ns": "not significant"}
            },
            
            "effect_size_reporting": {
                "format": "Relative percentage: Δ% = (BEM - Baseline) / Baseline",
                "precision": 2,  # 2 decimal places
                "always_include_ci": True
            },
            
            "honest_reporting": {
                "negative_results": "All negative/neutral results must be reported",
                "failed_experiments": "Failed runs documented with failure reasons",
                "multiple_testing_disclosure": "FDR correction methodology fully described"
            }
        }
    }
    
    return claims

def create_claims_templates_dir(templates_dir: Path):
    """Create template files for different claim outcomes."""
    
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    # Success template
    success_template = """
# Successful Claim Template

## Hero Results
- **Quality Improvement**: BEM-v1.1-stable improves ALL metrics on BOTH slices
- **Efficiency**: Maintains p50 latency ≤ +15% and VRAM ≤ ±5%
- **Cache Performance**: Achieves KV hit rate ≥ 80%
- **No Leakage**: Policy-over-memory validated via index-swap

## Statistical Evidence
- All comparisons use BCa 95% confidence intervals
- FDR correction applied to control family-wise error rate
- Effect sizes reported as relative improvements: Δ% = (BEM - Baseline) / Baseline
- Significance stars (*) ONLY when CI > 0

## Abstract Language Options
- **Full Success**: "BEM-v1.1-stable demonstrates superior performance..."
- **Partial Success**: "BEM-v1.1-stable shows competitive performance..."
    """
    
    # Failure template  
    failure_template = """
# Failed Claim Template

## Downgraded Claims
- Remove "superior" → use "competitive" 
- Remove efficiency claims if latency/VRAM exceeded
- Remove cache claims if hit rate < 80%
- Remove significance stars if CI ≤ 0

## Honest Reporting
- Document all failed claims transparently
- Report negative results with full context
- Explain statistical thresholds and why they weren't met
- Suggest future work to address limitations

## Abstract Language (Downgraded)
- "BEM-v1.1-stable shows promise but requires further optimization..."
- "While not achieving all performance targets, BEM demonstrates..."
    """
    
    with open(templates_dir / "success_template.md", 'w') as f:
        f.write(success_template)
    
    with open(templates_dir / "failure_template.md", 'w') as f:
        f.write(failure_template)

def main():
    parser = argparse.ArgumentParser(description="Initialize BEM claims preregistration")
    parser.add_argument("--out", required=True,
                       help="Output path for claims.yaml")
    parser.add_argument("--templates", 
                       help="Directory to create claim templates")
    
    args = parser.parse_args()
    
    print("📋 BEM Claims Preregistration")
    print("=" * 50)
    
    # Create claims structure
    claims = create_claims_structure()
    
    # Save claims YAML
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(claims, f, default_flow_style=False, sort_keys=False)
    
    print(f"📝 Claims preregistered: {output_path}")
    
    # Create templates if requested
    if args.templates:
        create_claims_templates_dir(Path(args.templates))
        print(f"📄 Templates created: {args.templates}")
    
    # Summary
    primary_claims = len(claims["primary_claims"])
    secondary_claims = len(claims["secondary_claims"])
    
    print("✅ Claims preregistration completed!")
    print(f"🎯 Primary claims: {primary_claims} (must ALL pass)")
    print(f"🔄 Secondary claims: {secondary_claims} (nice to have)")
    print(f"📊 Statistical framework: BCa 95% CI with FDR correction")
    print(f"⚖️ Auto-downgrade policies: Enabled for honest reporting")
    
    print("\n🔬 Experimental Validation Required:")
    for claim_id, claim in claims["primary_claims"].items():
        print(f"  • {claim['statement']}")
    
    return 0

if __name__ == "__main__":
    exit(main())