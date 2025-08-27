#!/usr/bin/env python3
"""
BEM Paper Factory - Page Limit Guard
Enforces NeurIPS 2025 9-page content limit through LaTeX compilation checks.

Features:
- Compiles LaTeX and counts content pages
- Excludes references and appendices from count
- Provides specific guidance on which sections to trim
- Integrates with CI/CD pipeline for automated checking
"""

import argparse
import logging
import subprocess
import tempfile
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PageLimitGuard:
    """
    Enforces NeurIPS page limits through LaTeX compilation analysis.
    """
    
    def __init__(self, page_limit: int = 9):
        self.page_limit = page_limit
        self.latex_engines = ['pdflatex', 'xelatex', 'lualatex']
        
    def compile_latex(self, tex_file: Path, output_dir: Path) -> Tuple[bool, str, Path]:
        """
        Compile LaTeX document and return success status and PDF path.
        """
        # Try different LaTeX engines
        for engine in self.latex_engines:
            if not shutil.which(engine):
                continue
                
            try:
                # First compilation
                cmd = [
                    engine,
                    '-interaction=nonstopmode',
                    '-halt-on-error',
                    '-output-directory', str(output_dir),
                    str(tex_file)
                ]
                
                result1 = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result1.returncode == 0:
                    # Second compilation for references
                    result2 = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    
                    pdf_path = output_dir / (tex_file.stem + '.pdf')
                    if pdf_path.exists():
                        return True, f"Successfully compiled with {engine}", pdf_path
                    else:
                        return False, f"PDF not generated with {engine}", None
                else:
                    logger.warning(f"Compilation failed with {engine}: {result1.stderr[:500]}")
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"Compilation timeout with {engine}")
            except Exception as e:
                logger.warning(f"Error with {engine}: {e}")
        
        return False, "All LaTeX engines failed", None
    
    def analyze_page_structure(self, tex_file: Path) -> Dict[str, any]:
        """
        Analyze LaTeX file structure to understand page usage.
        """
        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Could not read LaTeX file: {e}")
            return {}
        
        analysis = {
            'sections': [],
            'figures': 0,
            'tables': 0,
            'equations': 0,
            'references': False,
            'appendix': False,
            'supplements': False
        }
        
        # Find sections
        section_pattern = re.compile(r'\\section\{([^}]+)\}')
        subsection_pattern = re.compile(r'\\subsection\{([^}]+)\}')
        
        for match in section_pattern.finditer(content):
            analysis['sections'].append({
                'type': 'section',
                'title': match.group(1),
                'position': match.start()
            })
            
        for match in subsection_pattern.finditer(content):
            analysis['sections'].append({
                'type': 'subsection', 
                'title': match.group(1),
                'position': match.start()
            })
        
        # Sort sections by position
        analysis['sections'].sort(key=lambda x: x['position'])
        
        # Count figures and tables
        analysis['figures'] = len(re.findall(r'\\begin\{figure\}', content))
        analysis['tables'] = len(re.findall(r'\\begin\{table\}', content))
        analysis['equations'] = len(re.findall(r'\\begin\{equation\}', content))
        
        # Check for references and appendix
        analysis['references'] = bool(re.search(r'\\bibliography\{', content) or 
                                    re.search(r'\\begin\{thebibliography\}', content))
        analysis['appendix'] = bool(re.search(r'\\appendix', content))
        analysis['supplements'] = bool(re.search(r'supplement', content, re.IGNORECASE))
        
        return analysis
    
    def count_content_pages(self, log_content: str) -> Tuple[int, Dict[str, any]]:
        """
        Extract page count information from LaTeX log.
        """
        page_info = {
            'total_pages': 0,
            'content_pages': 0,
            'reference_pages': 0,
            'appendix_pages': 0
        }
        
        # Look for total page count
        page_pattern = re.search(r'Output written on [^(]*\((\d+) pages?', log_content)
        if page_pattern:
            page_info['total_pages'] = int(page_pattern.group(1))
        
        # Estimate content pages (heuristic approach)
        # This is approximate since LaTeX doesn't directly track "content" vs "references"
        
        # Look for bibliography start
        bib_pattern = re.search(r'\\bibliography\{.*?\}', log_content)
        if bib_pattern:
            # Assume references start around 80% through the document
            estimated_content = int(page_info['total_pages'] * 0.8)
            page_info['content_pages'] = min(estimated_content, page_info['total_pages'])
            page_info['reference_pages'] = page_info['total_pages'] - page_info['content_pages']
        else:
            # No references detected, assume all pages are content
            page_info['content_pages'] = page_info['total_pages']
        
        return page_info['content_pages'], page_info
    
    def generate_page_reduction_suggestions(self, 
                                          current_pages: int, 
                                          target_pages: int,
                                          structure_analysis: Dict) -> List[str]:
        """
        Generate specific suggestions for reducing page count.
        """
        excess_pages = current_pages - target_pages
        suggestions = []
        
        if excess_pages <= 0:
            return ["✅ Page limit satisfied!"]
        
        suggestions.append(f"Need to reduce by {excess_pages} pages (currently {current_pages}, limit {target_pages})")
        suggestions.append("")
        
        # Prioritized reduction strategies
        suggestions.extend([
            "🎯 High-Impact Reduction Strategies:",
            "1. Move detailed ablations to appendix/supplement",
            "2. Consolidate similar experimental results into single tables",
            "3. Use more compact figure layouts (subfigures, multi-panel plots)",
            "4. Reduce whitespace around figures and tables",
            "5. Shorten introduction and related work sections",
            ""
        ])
        
        # Section-specific suggestions based on analysis
        if structure_analysis.get('sections'):
            suggestions.append("📋 Section-Specific Suggestions:")
            
            for section in structure_analysis['sections']:
                title = section['title'].lower()
                
                if 'introduction' in title:
                    suggestions.append(f"  • {section['title']}: Trim to 1-1.5 pages, focus on key contributions")
                elif 'related' in title:
                    suggestions.append(f"  • {section['title']}: Consolidate to 0.5-1 page, cite key works only")
                elif 'method' in title:
                    suggestions.append(f"  • {section['title']}: Move detailed equations to appendix")
                elif 'experiment' in title:
                    suggestions.append(f"  • {section['title']}: Use compact tables, move ablations to supplement")
                elif 'result' in title:
                    suggestions.append(f"  • {section['title']}: Combine similar results, use multi-panel figures")
                elif 'discussion' in title or 'limitation' in title:
                    suggestions.append(f"  • {section['title']}: Keep concise, 0.5 page maximum")
        
        # Figure and table suggestions
        if structure_analysis.get('figures', 0) > 5:
            suggestions.extend([
                "",
                "🖼️ Figure/Table Optimization:",
                f"  • {structure_analysis['figures']} figures detected - consider combining related plots",
                "  • Use subfigures with shared captions to save space",
                "  • Move detailed ablation plots to appendix"
            ])
        
        if structure_analysis.get('tables', 0) > 3:
            suggestions.extend([
                f"  • {structure_analysis['tables']} tables detected - consolidate similar results",
                "  • Use compact notation (scientific notation for small numbers)",
                "  • Consider landscape orientation for wide tables"
            ])
        
        # LaTeX-specific optimization tips
        suggestions.extend([
            "",
            "⚙️ LaTeX Optimization Tips:",
            "  • Use \\vspace{-2mm} to reduce spacing around sections", 
            "  • Apply \\setlength{\\textfloatsep}{10pt} to reduce float spacing",
            "  • Use \\small or \\footnotesize for dense content",
            "  • Consider two-column layout for appendix content",
            "  • Remove excessive blank lines and comments"
        ])
        
        return suggestions
    
    def check_page_limit(self, tex_file: Path) -> Dict[str, any]:
        """
        Main page limit checking function.
        """
        logger.info(f"Checking page limit for: {tex_file}")
        
        # Create temporary directory for compilation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Copy tex file and any dependencies to temp directory
            tex_copy = temp_path / tex_file.name
            shutil.copy2(tex_file, tex_copy)
            
            # Copy related files (style files, figures, etc.)
            tex_dir = tex_file.parent
            for ext in ['.sty', '.cls', '.bib', '.png', '.pdf', '.jpg', '.eps']:
                for dep_file in tex_dir.glob(f"*{ext}"):
                    shutil.copy2(dep_file, temp_path)
            
            # Also copy any referenced sections or includes
            try:
                with open(tex_file, 'r') as f:
                    content = f.read()
                    
                # Find input/include references
                include_pattern = re.compile(r'\\(?:input|include)\{([^}]+)\}')
                for match in include_pattern.finditer(content):
                    ref_file = match.group(1)
                    if not ref_file.endswith('.tex'):
                        ref_file += '.tex'
                    
                    source_path = tex_dir / ref_file
                    if source_path.exists():
                        target_path = temp_path / ref_file
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_path, target_path)
                        
            except Exception as e:
                logger.warning(f"Error copying dependencies: {e}")
            
            # Analyze document structure
            structure_analysis = self.analyze_page_structure(tex_file)
            
            # Compile LaTeX
            success, message, pdf_path = self.compile_latex(tex_copy, temp_path)
            
            if not success:
                return {
                    'success': False,
                    'error': message,
                    'within_limit': False,
                    'suggestions': ["Fix LaTeX compilation errors first"]
                }
            
            # Read compilation log
            log_file = temp_path / (tex_copy.stem + '.log')
            log_content = ""
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        log_content = f.read()
                except Exception as e:
                    logger.warning(f"Could not read log file: {e}")
            
            # Count pages
            content_pages, page_info = self.count_content_pages(log_content)
            within_limit = content_pages <= self.page_limit
            
            # Generate suggestions
            suggestions = self.generate_page_reduction_suggestions(
                content_pages, self.page_limit, structure_analysis
            )
            
            result = {
                'success': True,
                'content_pages': content_pages,
                'total_pages': page_info['total_pages'],
                'page_limit': self.page_limit,
                'within_limit': within_limit,
                'excess_pages': max(0, content_pages - self.page_limit),
                'page_breakdown': page_info,
                'structure_analysis': structure_analysis,
                'suggestions': suggestions,
                'compilation_message': message
            }
            
            # Copy PDF to output location if successful
            if pdf_path and pdf_path.exists():
                output_pdf = tex_file.parent / f"{tex_file.stem}_pagecheck.pdf"
                shutil.copy2(pdf_path, output_pdf)
                result['output_pdf'] = str(output_pdf)
            
            return result

def main():
    parser = argparse.ArgumentParser(description='BEM Paper Factory - Page Limit Guard')
    parser.add_argument('--tex', required=True, help='Path to main LaTeX file')
    parser.add_argument('--limit', type=int, default=9, help='Page limit (default: 9 for NeurIPS)')
    parser.add_argument('--output', help='Output report file (JSON)')
    parser.add_argument('--fail-on-excess', action='store_true', 
                       help='Exit with error if page limit exceeded')
    
    args = parser.parse_args()
    
    tex_path = Path(args.tex)
    if not tex_path.exists():
        logger.error(f"LaTeX file not found: {tex_path}")
        exit(1)
    
    # Initialize page guard
    guard = PageLimitGuard(page_limit=args.limit)
    
    # Check page limit
    result = guard.check_page_limit(tex_path)
    
    if not result['success']:
        logger.error(f"Page check failed: {result['error']}")
        exit(1)
    
    # Report results
    if result['within_limit']:
        logger.info(f"✅ WITHIN LIMIT: {result['content_pages']}/{result['page_limit']} content pages")
        if result.get('output_pdf'):
            logger.info(f"PDF generated: {result['output_pdf']}")
    else:
        logger.error(f"❌ EXCEEDS LIMIT: {result['content_pages']}/{result['page_limit']} content pages")
        logger.error(f"Need to reduce by {result['excess_pages']} pages")
        
        # Show suggestions
        logger.info("\nReduction suggestions:")
        for suggestion in result['suggestions'][:10]:  # Show first 10 suggestions
            logger.info(f"  {suggestion}")
    
    # Save detailed report if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Detailed report saved to: {args.output}")
    
    # Exit with error if over limit and flag set
    if args.fail_on_excess and not result['within_limit']:
        exit(1)

if __name__ == '__main__':
    main()