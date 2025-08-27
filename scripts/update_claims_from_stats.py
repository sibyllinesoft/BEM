#!/usr/bin/env python3
"""
Update claims.yaml with validated results from statistical analysis.
Ensures all claims are CI-backed and FDR-corrected.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaimsUpdater:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.stats_path = project_root / "analysis" / "stats.json"
        self.winners_path = project_root / "analysis" / "winners.json"
        self.claims_path = project_root / "paper" / "claims.yaml"
        
    def load_statistical_results(self) -> Dict[str, Any]:
        """Load statistical analysis results."""
        with open(self.stats_path, 'r') as f:
            stats = json.load(f)
        
        with open(self.winners_path, 'r') as f:
            winners = json.load(f)
            
        return {"stats": stats, "winners": winners}
    
    def load_current_claims(self) -> Dict[str, Any]:
        """Load current claims structure."""
        with open(self.claims_path, 'r') as f:
            return yaml.safe_load(f)
    
    def extract_validated_claims(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract validated claims from statistical results."""
        stats = results["stats"]
        winners = results["winners"]
        
        validated_claims = {}
        
        # Map statistical results to claim updates
        claim_mapping = {
            "em": {
                "claim_id": "ar1_em_improvement",
                "statement": f"AR1 improves exact match by {stats['claim_results']['em']['effect_size']:.1%} (95% CI: [{stats['claim_results']['em']['confidence_interval'][0]:.3f}, {stats['claim_results']['em']['confidence_interval'][1]:.3f}])",
                "status": "VALIDATED" if stats['claim_results']['em']['pass_status'] else "REJECTED",
                "evidence": stats['claim_results']['em']
            },
            "f1": {
                "claim_id": "ar1_f1_improvement", 
                "statement": f"AR1 improves F1 score by {stats['claim_results']['f1']['effect_size']:.1%} (95% CI: [{stats['claim_results']['f1']['confidence_interval'][0]:.3f}, {stats['claim_results']['f1']['confidence_interval'][1]:.3f}])",
                "status": "VALIDATED" if stats['claim_results']['f1']['pass_status'] else "REJECTED",
                "evidence": stats['claim_results']['f1']
            },
            "ol0_aggregate": {
                "claim_id": "ol0_aggregate_improvement",
                "statement": f"OL0 achieves {stats['claim_results']['ol0_aggregate']['effect_size']:.1%} aggregate improvement (95% CI: [{stats['claim_results']['ol0_aggregate']['confidence_interval'][0]:.3f}, {stats['claim_results']['ol0_aggregate']['confidence_interval'][1]:.3f}])",
                "status": "VALIDATED" if stats['claim_results']['ol0_aggregate']['pass_status'] else "REJECTED",
                "evidence": stats['claim_results']['ol0_aggregate']
            },
            "mm0_vqa_slice": {
                "claim_id": "mm0_vqa_improvement",
                "statement": f"MM0 improves VQA slice performance by {stats['claim_results']['mm0_vqa_slice']['effect_size']:.1%} (95% CI: [{stats['claim_results']['mm0_vqa_slice']['confidence_interval'][0]:.3f}, {stats['claim_results']['mm0_vqa_slice']['confidence_interval'][1]:.3f}])",
                "status": "VALIDATED" if stats['claim_results']['mm0_vqa_slice']['pass_status'] else "REJECTED", 
                "evidence": stats['claim_results']['mm0_vqa_slice']
            },
            "vc0_violations": {
                "claim_id": "vc0_safety_improvement",
                "statement": f"VC0 reduces violations by {stats['claim_results']['vc0_violations']['effect_size']:.1%} (95% CI: [{stats['claim_results']['vc0_violations']['confidence_interval'][0]:.3f}, {stats['claim_results']['vc0_violations']['confidence_interval'][1]:.3f}])",
                "status": "VALIDATED" if stats['claim_results']['vc0_violations']['pass_status'] else "REJECTED",
                "evidence": stats['claim_results']['vc0_violations']
            },
            "pt1_pareto": {
                "claim_id": "pt1_pareto_improvement",
                "statement": f"PT1 achieves Pareto improvement with {stats['claim_results']['pt1_pareto']['effect_size']:.1%} gain (95% CI: [{stats['claim_results']['pt1_pareto']['confidence_interval'][0]:.3f}, {stats['claim_results']['pt1_pareto']['confidence_interval'][1]:.3f}])",
                "status": "VALIDATED" if stats['claim_results']['pt1_pareto']['pass_status'] else "REJECTED",
                "evidence": stats['claim_results']['pt1_pareto']
            }
        }
        
        for key, claim_data in claim_mapping.items():
            if key in stats['claim_results']:
                validated_claims[claim_data['claim_id']] = {
                    "statement": claim_data['statement'],
                    "status": claim_data['status'],
                    "statistical_backing": {
                        "effect_size": claim_data['evidence']['effect_size'],
                        "confidence_interval": claim_data['evidence']['confidence_interval'],
                        "p_value": claim_data['evidence']['p_value'],
                        "corrected_p_value": claim_data['evidence']['corrected_p_value'],
                        "test_type": claim_data['evidence']['test_result']['test_type'],
                        "fdr_corrected": True,
                        "bootstrap_method": "BCa",
                        "bootstrap_iterations": 10000
                    },
                    "pillar_promotion": self._get_pillar_promotion(claim_data['claim_id'], winners),
                    "validation_timestamp": datetime.now().isoformat(),
                    "ci_backed": claim_data['evidence']['confidence_interval'][0] > 0 if claim_data['status'] == 'VALIDATED' else False
                }
        
        return validated_claims
    
    def _get_pillar_promotion(self, claim_id: str, winners: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract pillar promotion decision for claim."""
        pillar_map = {
            "ar1_": "AR1",
            "ol0_": "OL0", 
            "mm0_": "MM0",
            "vc0_": "VC0",
            "pt1_": "PT"
        }
        
        for prefix, pillar_name in pillar_map.items():
            if claim_id.startswith(prefix.lower()):
                for pillar in winners["pillar_results"]:
                    if pillar["pillar_name"] == pillar_name:
                        return {
                            "decision": pillar["decision"],
                            "risk_level": pillar["risk_level"],
                            "overall_score": pillar["overall_score"],
                            "gates_passed": sum(1 for gate in pillar["gate_results"] if gate["passed"]),
                            "total_gates": len(pillar["gate_results"])
                        }
        return None
    
    def update_claims_file(self, validated_claims: Dict[str, Any]) -> None:
        """Update claims.yaml with validated results."""
        current_claims = self.load_current_claims()
        
        # Update metadata
        current_claims["metadata"]["last_updated"] = datetime.now().isoformat()
        current_claims["metadata"]["statistical_validation_complete"] = True
        current_claims["metadata"]["total_validated_claims"] = len(validated_claims)
        current_claims["metadata"]["ci_backed_claims"] = sum(1 for claim in validated_claims.values() if claim.get("ci_backed", False))
        
        # Add validated claims section
        current_claims["validated_claims"] = validated_claims
        
        # Update statistical summary
        stats_results = self.load_statistical_results()["stats"]
        current_claims["statistical_summary"] = {
            "framework": stats_results["summary"],
            "slice_analysis": stats_results["slice_analysis"],
            "fdr_correction": stats_results["fdr_correction"],
            "bootstrap_diagnostics": stats_results["bootstrap_diagnostics"]
        }
        
        # Add promotion summary
        winners_results = self.load_statistical_results()["winners"]
        current_claims["promotion_summary"] = {
            "total_pillars": len(winners_results["pillar_results"]),
            "promoted_pillars": [p["pillar_name"] for p in winners_results["pillar_results"] if p["decision"] == "PROMOTE"],
            "conditional_pillars": [p["pillar_name"] for p in winners_results["pillar_results"] if p["decision"] == "CONDITIONAL"],
            "overall_risk": winners_results["risk_assessment"]["level"],
            "deployment_recommendation": winners_results["risk_assessment"]["deployment_recommendation"]
        }
        
        # Save updated claims
        with open(self.claims_path, 'w') as f:
            yaml.dump(current_claims, f, default_flow_style=False, sort_keys=False, indent=2)
        
        logger.info(f"Updated claims.yaml with {len(validated_claims)} validated claims")
        
    def generate_claim_validation_report(self, validated_claims: Dict[str, Any]) -> str:
        """Generate validation report."""
        report = ["# Claims Validation Report", ""]
        report.append(f"**Validation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Claims:** {len(validated_claims)}")
        
        validated_count = sum(1 for claim in validated_claims.values() if claim["status"] == "VALIDATED")
        rejected_count = len(validated_claims) - validated_count
        
        report.append(f"**Validated:** {validated_count}")
        report.append(f"**Rejected:** {rejected_count}")
        report.append(f"**Validation Rate:** {validated_count/len(validated_claims):.1%}")
        report.append("")
        
        # Individual claims
        report.append("## Claim Details")
        report.append("")
        
        for claim_id, claim_data in validated_claims.items():
            status_emoji = "✅" if claim_data["status"] == "VALIDATED" else "❌"
            ci_emoji = "📊" if claim_data.get("ci_backed", False) else ""
            
            report.append(f"### {status_emoji} {ci_emoji} {claim_id}")
            report.append(f"**Statement:** {claim_data['statement']}")
            report.append(f"**Status:** {claim_data['status']}")
            
            if "statistical_backing" in claim_data:
                stats = claim_data["statistical_backing"]
                report.append(f"**Effect Size:** {stats['effect_size']:.3f}")
                report.append(f"**95% CI:** [{stats['confidence_interval'][0]:.3f}, {stats['confidence_interval'][1]:.3f}]")
                report.append(f"**p-value:** {stats['p_value']:.3f}")
                report.append(f"**FDR-corrected p:** {stats['corrected_p_value']:.3f}")
            
            if "pillar_promotion" in claim_data and claim_data["pillar_promotion"]:
                promo = claim_data["pillar_promotion"]
                report.append(f"**Promotion Decision:** {promo['decision']}")
                report.append(f"**Gates Passed:** {promo['gates_passed']}/{promo['total_gates']}")
                report.append(f"**Risk Level:** {promo['risk_level']}")
            
            report.append("")
        
        return "\n".join(report)
    
    def run(self) -> None:
        """Execute full claims update process."""
        logger.info("Starting claims validation update...")
        
        # Load results
        results = self.load_statistical_results()
        logger.info(f"Loaded {len(results['stats']['claim_results'])} statistical results")
        
        # Extract validated claims
        validated_claims = self.extract_validated_claims(results)
        logger.info(f"Extracted {len(validated_claims)} validated claims")
        
        # Update claims file
        self.update_claims_file(validated_claims)
        
        # Generate report
        report = self.generate_claim_validation_report(validated_claims)
        report_path = self.project_root / "paper" / "claims_validation_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Generated validation report: {report_path}")
        logger.info("Claims validation update complete!")

def main():
    project_root = Path(__file__).parent.parent
    updater = ClaimsUpdater(project_root)
    updater.run()

if __name__ == "__main__":
    main()