#!/usr/bin/env python3
"""
Master orchestration script for BEM 2.0 publication bundle.
Runs the complete pipeline from statistical validation to paper compilation and reproducibility pack.
"""

import subprocess
import sys
from pathlib import Path
import logging
import time
from typing import List, Tuple, Optional
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PublicationPipeline:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.scripts_dir = project_root / "scripts"
        self.analysis_dir = project_root / "analysis"
        self.dist_dir = project_root / "dist"
        self.dist_dir.mkdir(exist_ok=True)
        
        # Pipeline stages
        self.stages = [
            ("P1.1", "update_claims", "Update claims from statistical analysis"),
            ("P1.2", "build_tables", "Generate publication tables"),
            ("P1.3", "build_figures", "Generate publication figures"),
            ("P1.4", "render_paper", "Compile LaTeX paper"),
            ("P2.1", "make_repro_pack", "Create reproducibility pack"),
        ]
        
    def run_stage(self, stage_id: str, script_name: str, description: str) -> Tuple[bool, float, str]:
        """Run a single pipeline stage."""
        logger.info(f"🔄 {stage_id}: {description}")
        start_time = time.time()
        
        # Determine script path and command
        if script_name in ["update_claims"]:
            script_path = self.scripts_dir / f"{script_name}_from_stats.py"
        elif script_name in ["build_tables", "build_figures"]:
            script_path = self.analysis_dir / f"{script_name.replace('build_', '').replace('s', '')}.py"
        else:
            script_path = self.scripts_dir / f"{script_name}.py"
        
        if not script_path.exists():
            return False, 0, f"Script not found: {script_path}"
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout per stage
            )
            
            runtime = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ {stage_id} completed in {runtime:.1f}s")
                return True, runtime, result.stdout
            else:
                logger.error(f"❌ {stage_id} failed after {runtime:.1f}s")
                logger.error(f"STDERR: {result.stderr}")
                return False, runtime, result.stderr
                
        except subprocess.TimeoutExpired:
            runtime = time.time() - start_time
            logger.error(f"⏰ {stage_id} timed out after {runtime:.1f}s")
            return False, runtime, "Stage timed out"
        except Exception as e:
            runtime = time.time() - start_time
            logger.error(f"💥 {stage_id} crashed after {runtime:.1f}s: {e}")
            return False, runtime, str(e)
    
    def validate_prerequisites(self) -> bool:
        """Validate that required input files exist."""
        required_files = [
            "analysis/stats.json",
            "analysis/winners.json", 
            "paper/main.tex",
            "paper/references.bib"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            return False
        
        # Check statistical results are valid
        try:
            with open(self.project_root / "analysis/stats.json", 'r') as f:
                stats = json.load(f)
            
            if "claim_results" not in stats:
                logger.error("❌ Invalid stats.json: missing claim_results")
                return False
                
            if len(stats["claim_results"]) == 0:
                logger.error("❌ No statistical results found")
                return False
                
            logger.info(f"✅ Found {len(stats['claim_results'])} statistical results")
            
        except Exception as e:
            logger.error(f"❌ Error validating stats.json: {e}")
            return False
        
        logger.info("✅ Prerequisites validated")
        return True
    
    def generate_pipeline_report(self, results: List[Tuple[str, str, str, bool, float, str]]) -> str:
        """Generate comprehensive pipeline execution report."""
        total_time = sum(runtime for _, _, _, success, runtime, _ in results)
        successful_stages = sum(1 for _, _, _, success, _, _ in results if success)
        total_stages = len(results)
        
        report = [
            "# BEM 2.0 Publication Pipeline Report",
            "",
            f"**Execution Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Runtime:** {total_time:.1f} seconds ({total_time/60:.1f} minutes)",
            f"**Success Rate:** {successful_stages}/{total_stages} stages ({successful_stages/total_stages:.1%})",
            "",
            "## Stage Results",
            ""
        ]
        
        for stage_id, script_name, description, success, runtime, output in results:
            status_emoji = "✅" if success else "❌"
            report.extend([
                f"### {status_emoji} {stage_id}: {description}",
                f"**Runtime:** {runtime:.1f}s",
                f"**Status:** {'SUCCESS' if success else 'FAILED'}",
                ""
            ])
            
            if not success and output:
                report.extend([
                    "**Error Output:**",
                    "```",
                    output[:1000] + ("..." if len(output) > 1000 else ""),
                    "```",
                    ""
                ])
        
        # Summary and next steps
        report.extend([
            "## Summary",
            ""
        ])
        
        if successful_stages == total_stages:
            report.extend([
                "🎉 **Publication pipeline completed successfully!**",
                "",
                "Generated outputs:",
                "- `paper/claims_validation_report.md` - Updated claims with statistical backing",
                "- `paper/tables/` - Publication-ready LaTeX tables",
                "- `paper/figs/` - High-quality figures",
                "- `dist/main.pdf` - Compiled paper", 
                "- `dist/reproducibility_pack/` - Complete reproduction package",
                "",
                "## Next Steps",
                "",
                "1. **Review the compiled paper**: `dist/main.pdf`",
                "2. **Validate tables and figures**: Check `paper/tables/` and `paper/figs/`",
                "3. **Test reproducibility**: Run `dist/reproducibility_pack/run.sh`",
                "4. **Final proofreading**: Review all claims against statistical evidence",
                "5. **Submission preparation**: Package for conference submission"
            ])
        else:
            failed_stages = [stage_id for stage_id, _, _, success, _, _ in results if not success]
            report.extend([
                f"⚠️ **Pipeline partially failed** ({len(failed_stages)} stages failed)",
                "",
                f"**Failed stages:** {', '.join(failed_stages)}",
                "",
                "## Recovery Actions",
                "",
                "1. **Review error outputs above** for each failed stage",
                "2. **Fix underlying issues** (missing files, LaTeX errors, etc.)",
                "3. **Re-run individual stages** using their respective scripts:",
            ])
            
            for stage_id, script_name, description, success, _, _ in results:
                if not success:
                    if script_name in ["update_claims"]:
                        script_path = f"scripts/{script_name}_from_stats.py"
                    elif script_name in ["build_tables", "build_figures"]:
                        script_path = f"analysis/{script_name.replace('build_', '').replace('s', '')}.py"
                    else:
                        script_path = f"scripts/{script_name}.py"
                    report.append(f"   - `python3 {script_path}` ({stage_id}: {description})")
            
            report.extend([
                "",
                "4. **Re-run the full pipeline** once issues are resolved"
            ])
        
        return "\n".join(report)
    
    def run_pipeline(self) -> bool:
        """Execute the complete publication pipeline."""
        logger.info("🚀 Starting BEM 2.0 publication pipeline...")
        start_time = time.time()
        
        # Validate prerequisites
        if not self.validate_prerequisites():
            logger.error("❌ Prerequisites validation failed")
            return False
        
        # Execute stages
        results = []
        pipeline_success = True
        
        for stage_id, script_name, description in self.stages:
            success, runtime, output = self.run_stage(stage_id, script_name, description)
            results.append((stage_id, script_name, description, success, runtime, output))
            
            if not success:
                pipeline_success = False
                logger.error(f"❌ Stage {stage_id} failed, continuing with remaining stages...")
        
        total_time = time.time() - start_time
        
        # Generate report
        report = self.generate_pipeline_report(results)
        report_path = self.dist_dir / "publication_pipeline_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Final status
        if pipeline_success:
            logger.info(f"🎉 Publication pipeline completed successfully in {total_time:.1f}s!")
            logger.info(f"📊 Generated {sum(1 for _, _, _, success, _, _ in results if success)}/{len(results)} outputs")
            logger.info(f"📄 Pipeline report: {report_path}")
        else:
            failed_count = sum(1 for _, _, _, success, _, _ in results if not success)
            logger.error(f"⚠️  Pipeline completed with {failed_count} failures in {total_time:.1f}s")
            logger.error(f"📄 See detailed report: {report_path}")
        
        return pipeline_success

def main():
    project_root = Path(__file__).parent.parent
    pipeline = PublicationPipeline(project_root)
    
    success = pipeline.run_pipeline()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()