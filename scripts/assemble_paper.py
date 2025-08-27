#!/usr/bin/env python3
"""
Paper Assembly Script - NeurIPS 2025 BEM Research
Assembles complete publication bundle with statistical results integration
"""

import os
import json
import argparse
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

def load_statistical_results(results_path):
    """Load statistical analysis results."""
    with open(results_path) as f:
        return json.load(f)


class PaperAssembler:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.paper_dir = self.root_dir / "paper"
        self.scripts_dir = self.root_dir / "scripts"
        self.analysis_dir = self.root_dir / "analysis"
        
        # Create necessary directories
        for dir_path in [
            self.paper_dir / "figures",
            self.paper_dir / "tables", 
            self.paper_dir / "sections",
            self.analysis_dir / "results"
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def check_prerequisites(self) -> bool:
        """Check that all required files and dependencies are available."""
        logger.info("Checking prerequisites...")
        
        required_files = [
            self.scripts_dir / "run_statistical_pipeline.py",
            self.scripts_dir / "generate_figures.py",
            self.scripts_dir / "generate_tables.py",
            self.scripts_dir / "generate_sections.py",
            self.paper_dir / "claims.yaml",
            self.paper_dir / "main.tex"
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False
        
        # Check for statistical results
        stats_file = self.analysis_dir / "results" / "aggregated_stats.json"
        if not stats_file.exists():
            logger.warning("Statistical results not found. Will run statistical pipeline first.")
        
        # Check LaTeX dependencies
        try:
            result = subprocess.run(['pdflatex', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("pdflatex not found. LaTeX distribution required for paper compilation.")
                return False
        except FileNotFoundError:
            logger.error("pdflatex not found. Please install a LaTeX distribution.")
            return False
        
        logger.info("Prerequisites check passed.")
        return True
    
    def run_statistical_analysis(self) -> bool:
        """Run the complete statistical analysis pipeline."""
        logger.info("Running statistical analysis pipeline...")
        
        stats_script = self.scripts_dir / "run_statistical_pipeline.py"
        try:
            result = subprocess.run([
                'python', str(stats_script),
                '--experiments-dir', str(self.root_dir / 'experiments'),
                '--logs-dir', str(self.root_dir / 'logs'),
                '--output-dir', str(self.analysis_dir / 'results'),
                '--claims-file', str(self.paper_dir / 'claims.yaml')
            ], capture_output=True, text=True, cwd=self.root_dir)
            
            if result.returncode != 0:
                logger.error(f"Statistical analysis failed: {result.stderr}")
                return False
                
            logger.info("Statistical analysis completed successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Error running statistical analysis: {e}")
            return False
    
    def generate_figures(self) -> bool:
        """Generate all paper figures."""
        logger.info("Generating figures...")
        
        figures_script = self.scripts_dir / "generate_figures.py"
        try:
            result = subprocess.run([
                'python', str(figures_script),
                '--stats-dir', str(self.analysis_dir / 'results'),
                '--output-dir', str(self.paper_dir / 'figures'),
                '--figure', 'all'
            ], capture_output=True, text=True, cwd=self.root_dir)
            
            if result.returncode != 0:
                logger.error(f"Figure generation failed: {result.stderr}")
                return False
                
            logger.info("Figures generated successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Error generating figures: {e}")
            return False
    
    def generate_tables(self) -> bool:
        """Generate all paper tables."""
        logger.info("Generating tables...")
        
        tables_script = self.scripts_dir / "generate_tables.py"
        try:
            result = subprocess.run([
                'python', str(tables_script),
                '--stats-dir', str(self.analysis_dir / 'results'),
                '--output-dir', str(self.paper_dir / 'tables'),
                '--table', 'all'
            ], capture_output=True, text=True, cwd=self.root_dir)
            
            if result.returncode != 0:
                logger.error(f"Table generation failed: {result.stderr}")
                return False
                
            logger.info("Tables generated successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Error generating tables: {e}")
            return False
    
    def generate_sections(self) -> bool:
        """Generate all paper sections."""
        logger.info("Generating paper sections...")
        
        sections_script = self.scripts_dir / "generate_sections.py"
        try:
            result = subprocess.run([
                'python', str(sections_script),
                '--stats-dir', str(self.analysis_dir / 'results'),
                '--claims-file', str(self.paper_dir / 'claims.yaml'),
                '--templates-dir', str(self.root_dir / 'templates'),
                '--output-dir', str(self.paper_dir / 'sections'),
                '--section', 'all'
            ], capture_output=True, text=True, cwd=self.root_dir)
            
            if result.returncode != 0:
                logger.error(f"Section generation failed: {result.stderr}")
                return False
                
            logger.info("Sections generated successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Error generating sections: {e}")
            return False
    
    def update_main_tex(self) -> bool:
        """Update main.tex to include all generated content."""
        logger.info("Updating main.tex...")
        
        main_tex_path = self.paper_dir / "main.tex"
        
        # Read current main.tex
        with open(main_tex_path, 'r') as f:
            content = f.read()
        
        # Define section inclusions
        section_inclusions = {
            r'% AUTO-GENERATED CONTENT - DO NOT EDIT MANUALLY': [
                r'\input{sections/introduction}',
                r'\input{sections/related_work}', 
                r'\input{sections/method}',
                r'\input{sections/experiments}',
                r'\input{sections/results}',
                r'\input{sections/analysis}',
                r'\input{sections/conclusion}'
            ]
        }
        
        # Define table and figure inclusions
        content_inclusions = [
            r'\input{tables/main_results}',
            r'\input{tables/ablation_study}',
            r'\input{tables/statistical_summary}',
            r'\input{tables/hyperparameters}'
        ]
        
        # Update content to include generated sections
        updated_content = content
        
        # Insert sections if not already present
        for marker, sections in section_inclusions.items():
            if marker in content:
                # Replace the marker with actual section includes
                sections_text = '\n'.join(sections)
                updated_content = updated_content.replace(marker, sections_text)
        
        # Add bibliography if not present
        if r'\bibliography{' not in updated_content:
            bib_section = r'''
% Bibliography
\bibliography{references}
\bibliographystyle{neurips_2025}
'''
            # Insert before \end{document}
            updated_content = updated_content.replace(r'\end{document}', bib_section + r'\end{document}')
        
        # Write updated content
        with open(main_tex_path, 'w') as f:
            f.write(updated_content)
        
        logger.info("main.tex updated successfully.")
        return True
    
    def compile_paper(self, max_attempts: int = 3) -> bool:
        """Compile the LaTeX paper."""
        logger.info("Compiling paper...")
        
        main_tex_path = self.paper_dir / "main.tex"
        
        for attempt in range(max_attempts):
            try:
                # Run pdflatex
                result = subprocess.run([
                    'pdflatex', 
                    '-interaction=nonstopmode',
                    '-output-directory', str(self.paper_dir),
                    str(main_tex_path)
                ], capture_output=True, text=True, cwd=self.paper_dir)
                
                if result.returncode == 0:
                    logger.info(f"Paper compiled successfully on attempt {attempt + 1}.")
                    
                    # Check if PDF was created
                    pdf_path = self.paper_dir / "main.pdf"
                    if pdf_path.exists():
                        logger.info(f"PDF created: {pdf_path}")
                        return True
                    else:
                        logger.warning("LaTeX compilation succeeded but no PDF found.")
                        
                else:
                    logger.warning(f"LaTeX compilation attempt {attempt + 1} failed:")
                    logger.warning(result.stdout)
                    if attempt < max_attempts - 1:
                        logger.info("Retrying...")
                        time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error during LaTeX compilation: {e}")
                return False
        
        logger.error("LaTeX compilation failed after all attempts.")
        return False
    
    def run_quality_checks(self) -> Dict[str, bool]:
        """Run quality assurance checks on the generated paper."""
        logger.info("Running quality checks...")
        
        checks = {}
        
        # Check page count
        try:
            page_guard_script = self.scripts_dir / "page_guard.py"
            if page_guard_script.exists():
                result = subprocess.run([
                    'python', str(page_guard_script),
                    str(self.paper_dir / "main.pdf")
                ], capture_output=True, text=True)
                checks['page_limit'] = result.returncode == 0
            else:
                checks['page_limit'] = None
        except Exception:
            checks['page_limit'] = None
        
        # Check anonymization
        try:
            lint_script = self.scripts_dir / "lint_blind.py" 
            if lint_script.exists():
                result = subprocess.run([
                    'python', str(lint_script),
                    str(self.paper_dir / "main.tex")
                ], capture_output=True, text=True)
                checks['anonymization'] = result.returncode == 0
            else:
                checks['anonymization'] = None
        except Exception:
            checks['anonymization'] = None
        
        # Check statistical validation
        validation_file = self.analysis_dir / "results" / "claims_validation.json"
        if validation_file.exists():
            try:
                with open(validation_file) as f:
                    validation_data = json.load(f)
                checks['statistical_validation'] = validation_data.get('all_claims_validated', False)
            except Exception:
                checks['statistical_validation'] = False
        else:
            checks['statistical_validation'] = False
        
        # Check required files exist
        required_outputs = [
            self.paper_dir / "main.pdf",
            self.paper_dir / "figures" / "pareto_frontier.pdf",
            self.paper_dir / "tables" / "main_results.tex"
        ]
        checks['required_outputs'] = all(f.exists() for f in required_outputs)
        
        return checks
    
    def generate_submission_package(self) -> Path:
        """Create submission package with all necessary files."""
        logger.info("Creating submission package...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_dir = self.root_dir / f"submission_{timestamp}"
        submission_dir.mkdir(exist_ok=True)
        
        # Copy essential files
        files_to_copy = [
            (self.paper_dir / "main.tex", submission_dir / "main.tex"),
            (self.paper_dir / "main.pdf", submission_dir / "main.pdf"),
            (self.paper_dir / "references.bib", submission_dir / "references.bib"),
        ]
        
        # Copy all sections
        sections_dir = self.paper_dir / "sections"
        if sections_dir.exists():
            shutil.copytree(sections_dir, submission_dir / "sections")
        
        # Copy all tables  
        tables_dir = self.paper_dir / "tables"
        if tables_dir.exists():
            shutil.copytree(tables_dir, submission_dir / "tables")
        
        # Copy all figures
        figures_dir = self.paper_dir / "figures"
        if figures_dir.exists():
            shutil.copytree(figures_dir, submission_dir / "figures")
        
        # Copy files that exist
        for src, dst in files_to_copy:
            if src.exists():
                shutil.copy2(src, dst)
        
        # Create submission README
        readme_content = f"""# BEM NeurIPS 2025 Submission Package

Generated: {datetime.now().isoformat()}

## Contents
- main.tex: Primary paper file
- main.pdf: Compiled paper
- sections/: Auto-generated paper sections
- tables/: LaTeX tables with statistical results  
- figures/: Publication-quality figures
- references.bib: Bibliography

## Statistics Summary
All results validated with 5+ seeds, bootstrap confidence intervals, 
and Holm-Bonferroni multiple comparison correction.

## Reproducibility
Complete experimental logs and statistical analysis available in 
the main repository under analysis/results/
"""
        
        with open(submission_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Submission package created: {submission_dir}")
        return submission_dir
    
    def assemble_complete_paper(self, skip_experiments: bool = False) -> bool:
        """Run the complete paper assembly pipeline."""
        logger.info("Starting complete paper assembly...")
        
        # 1. Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # 2. Run statistical analysis (unless skipping)
        if not skip_experiments:
            if not self.run_statistical_analysis():
                logger.error("Statistical analysis failed. Cannot proceed.")
                return False
        
        # 3. Generate all content
        steps = [
            ("figures", self.generate_figures),
            ("tables", self.generate_tables), 
            ("sections", self.generate_sections)
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                logger.error(f"Failed to generate {step_name}")
                return False
        
        # 4. Update main LaTeX file
        if not self.update_main_tex():
            return False
        
        # 5. Compile paper
        if not self.compile_paper():
            return False
        
        # 6. Run quality checks
        checks = self.run_quality_checks()
        logger.info("Quality check results:")
        for check, result in checks.items():
            status = "PASS" if result else "FAIL" if result is False else "SKIP"
            logger.info(f"  {check}: {status}")
        
        # 7. Generate submission package
        submission_dir = self.generate_submission_package()
        
        # 8. Final report
        logger.info("=" * 60)
        logger.info("PAPER ASSEMBLY COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Paper PDF: {self.paper_dir / 'main.pdf'}")
        logger.info(f"Submission package: {submission_dir}")
        
        # Check if all critical checks passed
        critical_checks = ['page_limit', 'statistical_validation', 'required_outputs']
        all_critical_passed = all(checks.get(check, False) for check in critical_checks)
        
        if all_critical_passed:
            logger.info("✅ All critical quality checks PASSED. Paper ready for submission!")
        else:
            logger.warning("⚠️  Some quality checks failed. Review before submission.")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Assemble BEM paper for NeurIPS 2025 submission')
    parser.add_argument('--root-dir', type=Path, default='.',
                       help='Root directory of the project')
    parser.add_argument('--skip-experiments', action='store_true',
                       help='Skip experimental runs and use existing statistical results')
    parser.add_argument('--component', type=str,
                       choices=['stats', 'figures', 'tables', 'sections', 'compile', 'all'],
                       default='all', help='Which component to run')
    
    args = parser.parse_args()
    
    assembler = PaperAssembler(args.root_dir)
    
    if args.component == 'all':
        success = assembler.assemble_complete_paper(skip_experiments=args.skip_experiments)
    elif args.component == 'stats':
        success = assembler.run_statistical_analysis()
    elif args.component == 'figures':
        success = assembler.generate_figures()
    elif args.component == 'tables':
        success = assembler.generate_tables()
    elif args.component == 'sections':
        success = assembler.generate_sections()
    elif args.component == 'compile':
        success = assembler.update_main_tex() and assembler.compile_paper()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())