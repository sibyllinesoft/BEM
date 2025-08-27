#!/usr/bin/env python3
"""
LaTeX compilation pipeline for BEM 2.0 paper.
Compiles main paper and supplement with all figures and tables.
"""

import subprocess
import shutil
from pathlib import Path
import logging
import sys
import os
from typing import List, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperRenderer:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.paper_dir = project_root / "paper"
        self.output_dir = project_root / "dist"
        self.output_dir.mkdir(exist_ok=True)
        
        # LaTeX files to compile
        self.main_tex = self.paper_dir / "main.tex"
        self.supplement_tex = self.paper_dir / "supplement.tex"
        
    def check_latex_installation(self) -> bool:
        """Check if required LaTeX tools are available."""
        required_tools = ["pdflatex", "bibtex", "makeindex"]
        
        for tool in required_tools:
            if not shutil.which(tool):
                logger.error(f"Required tool '{tool}' not found in PATH")
                return False
        
        logger.info("LaTeX installation verified")
        return True
    
    def clean_latex_artifacts(self, tex_file: Path) -> None:
        """Clean LaTeX auxiliary files."""
        base_name = tex_file.stem
        artifacts = [".aux", ".log", ".bbl", ".blg", ".toc", ".out", ".fls", 
                    ".fdb_latexmk", ".synctex.gz", ".nav", ".snm", ".vrb"]
        
        for ext in artifacts:
            artifact_file = tex_file.parent / f"{base_name}{ext}"
            if artifact_file.exists():
                artifact_file.unlink()
        
        logger.info(f"Cleaned LaTeX artifacts for {base_name}")
    
    def run_latex_command(self, command: List[str], cwd: Path, 
                         description: str) -> bool:
        """Run a LaTeX command and handle output."""
        logger.info(f"Running: {description}")
        
        try:
            result = subprocess.run(
                command, 
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Command failed: {' '.join(command)}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False
            
            logger.info(f"✓ {description} completed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(command)}")
            return False
        except Exception as e:
            logger.error(f"Command failed with exception: {e}")
            return False
    
    def compile_pdf(self, tex_file: Path, passes: int = 3) -> bool:
        """Compile LaTeX to PDF with multiple passes for references."""
        if not tex_file.exists():
            logger.error(f"LaTeX file not found: {tex_file}")
            return False
        
        base_name = tex_file.stem
        logger.info(f"Compiling {base_name}.tex to PDF...")
        
        # Clean old artifacts first
        self.clean_latex_artifacts(tex_file)
        
        # First pass: initial compilation
        if not self.run_latex_command(
            ["pdflatex", "-interaction=nonstopmode", f"{base_name}.tex"],
            tex_file.parent,
            f"PDFLaTeX pass 1 for {base_name}"
        ):
            return False
        
        # Run bibtex if .bib file exists
        bib_file = tex_file.parent / "references.bib"
        if bib_file.exists():
            if not self.run_latex_command(
                ["bibtex", base_name],
                tex_file.parent,
                f"BibTeX for {base_name}"
            ):
                logger.warning("BibTeX failed, continuing without bibliography")
        
        # Additional passes for cross-references
        for i in range(2, passes + 1):
            if not self.run_latex_command(
                ["pdflatex", "-interaction=nonstopmode", f"{base_name}.tex"],
                tex_file.parent,
                f"PDFLaTeX pass {i} for {base_name}"
            ):
                return False
        
        # Check if PDF was generated
        pdf_file = tex_file.parent / f"{base_name}.pdf"
        if not pdf_file.exists():
            logger.error(f"PDF not generated: {pdf_file}")
            return False
        
        # Copy to output directory
        output_pdf = self.output_dir / f"{base_name}.pdf"
        shutil.copy2(pdf_file, output_pdf)
        
        logger.info(f"✓ Successfully compiled {base_name}.pdf -> {output_pdf}")
        return True
    
    def validate_latex_structure(self) -> bool:
        """Validate that required LaTeX structure exists."""
        required_files = [
            self.main_tex,
            self.paper_dir / "references.bib",
        ]
        
        required_dirs = [
            self.paper_dir / "sections",
            self.paper_dir / "tables", 
            self.paper_dir / "figs"
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        missing_dirs = [d for d in required_dirs if not d.exists()]
        
        if missing_files:
            logger.error(f"Missing required files: {[str(f) for f in missing_files]}")
        
        if missing_dirs:
            logger.error(f"Missing required directories: {[str(d) for d in missing_dirs]}")
        
        if missing_files or missing_dirs:
            return False
        
        logger.info("LaTeX structure validation passed")
        return True
    
    def generate_compilation_report(self, success: bool, 
                                  compilation_time: float) -> str:
        """Generate compilation report."""
        report = [
            "# BEM 2.0 Paper Compilation Report",
            "",
            f"**Compilation Time:** {compilation_time:.1f} seconds",
            f"**Status:** {'✓ SUCCESS' if success else '✗ FAILED'}",
            f"**Output Directory:** {self.output_dir}",
            "",
            "## Generated Files",
        ]
        
        if success:
            pdf_files = list(self.output_dir.glob("*.pdf"))
            if pdf_files:
                report.append("")
                for pdf in pdf_files:
                    size_mb = pdf.stat().st_size / (1024 * 1024)
                    report.append(f"- `{pdf.name}` ({size_mb:.1f} MB)")
            else:
                report.append("- No PDF files generated")
        else:
            report.append("- Compilation failed, no outputs generated")
        
        report.extend([
            "",
            "## LaTeX Environment", 
            f"- Working Directory: {self.paper_dir}",
            f"- Main Document: {self.main_tex.name}",
            f"- Supplement: {self.supplement_tex.name if self.supplement_tex.exists() else 'Not found'}",
            "",
            "## Next Steps",
        ])
        
        if success:
            report.extend([
                "- Review generated PDFs in `dist/` directory",
                "- Check figure quality and table formatting",
                "- Validate all cross-references and citations",
                "- Run spell-check and grammar review"
            ])
        else:
            report.extend([
                "- Check LaTeX log files for compilation errors", 
                "- Ensure all required packages are installed",
                "- Verify figure and table file paths",
                "- Review bibliography formatting"
            ])
        
        return "\n".join(report)
    
    def run(self) -> bool:
        """Execute full paper rendering process."""
        start_time = time.time()
        logger.info("Starting paper compilation...")
        
        # Pre-flight checks
        if not self.check_latex_installation():
            logger.error("LaTeX installation check failed")
            return False
        
        if not self.validate_latex_structure():
            logger.error("LaTeX structure validation failed")
            return False
        
        success = True
        
        # Compile main paper
        if self.main_tex.exists():
            logger.info("Compiling main paper...")
            if not self.compile_pdf(self.main_tex):
                logger.error("Main paper compilation failed")
                success = False
        else:
            logger.error(f"Main LaTeX file not found: {self.main_tex}")
            success = False
        
        # Compile supplement if it exists
        if self.supplement_tex.exists():
            logger.info("Compiling supplement...")
            if not self.compile_pdf(self.supplement_tex):
                logger.error("Supplement compilation failed")
                success = False
        else:
            logger.warning(f"Supplement not found: {self.supplement_tex}")
        
        # Clean up artifacts
        if self.main_tex.exists():
            self.clean_latex_artifacts(self.main_tex)
        if self.supplement_tex.exists():
            self.clean_latex_artifacts(self.supplement_tex)
        
        compilation_time = time.time() - start_time
        
        # Generate report
        report = self.generate_compilation_report(success, compilation_time)
        report_path = self.output_dir / "compilation_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        if success:
            logger.info(f"✓ Paper compilation completed successfully in {compilation_time:.1f}s")
            logger.info(f"Output files available in: {self.output_dir}")
        else:
            logger.error(f"✗ Paper compilation failed after {compilation_time:.1f}s")
        
        logger.info(f"Compilation report: {report_path}")
        return success

def main():
    project_root = Path(__file__).parent.parent
    renderer = PaperRenderer(project_root)
    
    success = renderer.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()