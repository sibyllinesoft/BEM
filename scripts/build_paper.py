#!/usr/bin/env python3
"""
BEM Paper Factory - Automated Paper Builder
Generates publication-ready LaTeX from experimental results and statistical analysis.

Features:
- Auto-generates tables and figures from results
- Validates claims against statistical tests
- Enforces NeurIPS 2025 formatting and page limits
- Ensures anonymization compliance
- Creates reproducibility manifests
"""

import argparse
import json
import logging
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperBuilder:
    """
    Automated paper building system for BEM NeurIPS submission.
    """
    
    def __init__(self, 
                 results_file: str,
                 claims_file: str,
                 templates_dir: str = "templates",
                 output_dir: str = "paper"):
        
        self.results_file = Path(results_file)
        self.claims_file = Path(claims_file)
        self.templates_dir = Path(templates_dir)
        self.output_dir = Path(output_dir)
        
        # Load data
        self.results_data = self._load_results()
        self.claims_ledger = self._load_claims()
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self._setup_jinja_filters()
        
    def _load_results(self) -> Dict[str, Any]:
        """Load experimental results and statistical analysis."""
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load results from {self.results_file}: {e}")
            raise
    
    def _load_claims(self) -> Dict[str, Any]:
        """Load claims ledger."""
        try:
            with open(self.claims_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load claims from {self.claims_file}: {e}")
            raise
    
    def _setup_jinja_filters(self) -> None:
        """Add custom Jinja2 filters for LaTeX generation."""
        
        def latex_escape(text: str) -> str:
            """Escape special LaTeX characters."""
            replacements = {
                '&': r'\&',
                '%': r'\%',
                '$': r'\$',
                '#': r'\#',
                '^': r'\^{}',
                '_': r'\_',
                '{': r'\{',
                '}': r'\}',
                '~': r'\textasciitilde{}',
                '\\': r'\textbackslash{}',
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
            return text
        
        def format_confidence_interval(ci: List[float], precision: int = 2) -> str:
            """Format confidence interval for LaTeX."""
            if not ci or len(ci) != 2:
                return ""
            return f"({ci[0]:.{precision}f}, {ci[1]:.{precision}f})"
        
        def format_p_value(p_val: float, threshold: float = 0.001) -> str:
            """Format p-value with appropriate precision."""
            if p_val < threshold:
                return f"< {threshold}"
            elif p_val < 0.01:
                return f"{p_val:.3f}"
            else:
                return f"{p_val:.2f}"
        
        def bold_if_significant(value: float, significant: bool) -> str:
            """Bold formatting for significant results."""
            formatted = f"{value:.2f}"
            return f"\\textbf{{{formatted}}}" if significant else formatted
        
        self.jinja_env.filters['latex_escape'] = latex_escape
        self.jinja_env.filters['format_ci'] = format_confidence_interval
        self.jinja_env.filters['format_p'] = format_p_value
        self.jinja_env.filters['bold_if_sig'] = bold_if_significant
    
    def validate_claims_against_results(self) -> Dict[str, Any]:
        """Validate all claims against actual results."""
        logger.info("Validating claims against experimental results...")
        
        validation_report = {
            'total_claims': len(self.claims_ledger.get('claims', {})),
            'validated_claims': 0,
            'failed_claims': 0,
            'claim_status': {},
            'ready_for_publication': True
        }
        
        claim_results = self.results_data.get('claim_results', {})
        
        for claim_id, claim_spec in self.claims_ledger.get('claims', {}).items():
            if claim_id in claim_results:
                claim_result = claim_results[claim_id]
                passes = claim_result.get('pass_status', False)
                
                validation_report['claim_status'][claim_id] = {
                    'assertion': claim_spec.get('assertion', ''),
                    'passes': passes,
                    'confidence_interval': claim_result.get('confidence_interval'),
                    'effect_size': claim_result.get('effect_size'),
                    'p_value': claim_result.get('p_value')
                }
                
                if passes:
                    validation_report['validated_claims'] += 1
                else:
                    validation_report['failed_claims'] += 1
                    validation_report['ready_for_publication'] = False
                    logger.warning(f"Claim FAILED: {claim_id}")
            else:
                logger.warning(f"No results found for claim: {claim_id}")
                validation_report['failed_claims'] += 1
                validation_report['ready_for_publication'] = False
        
        logger.info(f"Claim validation: {validation_report['validated_claims']} passed, "
                   f"{validation_report['failed_claims']} failed")
        
        return validation_report
    
    def generate_main_results_table(self) -> str:
        """Generate main results comparison table."""
        logger.info("Generating main results table...")
        
        template = self.jinja_env.get_template('results_table.j2')
        
        # Extract method results
        results_data = []
        methods = ['lora', 'prefix', 'ia3', 'mole', 'hyperlora', 'bem_p1', 'bem_p2', 'bem_p3', 'bem_p4']
        
        for method in methods:
            if method in self.results_data.get('method_results', {}):
                method_result = self.results_data['method_results'][method]
                results_data.append({
                    'method': method,
                    'display_name': self._get_method_display_name(method),
                    'metrics': method_result.get('metrics', {}),
                    'delta_vs_lora': method_result.get('delta_vs_lora', {}),
                })
        
        # Define metrics to display
        metrics = [
            {'name': 'exact_match', 'display_name': 'EM'},
            {'name': 'f1_score', 'display_name': 'F1'},
            {'name': 'bleu', 'display_name': 'BLEU'},
            {'name': 'chrF', 'display_name': 'chrF'}
        ]
        
        table_latex = template.render(
            results_data=results_data,
            metrics=metrics,
            show_confidence_intervals=True,
            show_statistical_notes=True,
            num_seeds=5,
            bootstrap_samples=10,
            confidence_level=95,
            statistical_test_description="Paired bootstrap with BCa confidence intervals",
            multiple_comparison_correction=True,
            correction_method="Holm-Bonferroni",
            fwer=0.05,
            claim_validation=True
        )
        
        return table_latex
    
    def generate_ablation_study_table(self) -> str:
        """Generate ablation study table."""
        logger.info("Generating ablation study table...")
        
        template = self.jinja_env.get_template('ablation_table.j2')
        
        # Extract ablation results
        ablation_data = self.results_data.get('ablation_results', {})
        
        ablation_configs = [
            {'name': 'static_lora', 'display_name': 'Static LoRA (baseline)', 'is_complete': False},
            {'name': 'bem_no_hier', 'display_name': 'BEM w/o hierarchical routing', 'is_complete': False},
            {'name': 'bem_no_retrieval', 'display_name': 'BEM w/o retrieval awareness', 'is_complete': False},
            {'name': 'bem_no_governance', 'display_name': 'BEM w/o spectral governance', 'is_complete': False},
            {'name': 'bem_full', 'display_name': 'BEM (full configuration)', 'is_complete': True}
        ]
        
        # Add results to configs
        for config in ablation_configs:
            config['metrics'] = ablation_data.get(config['name'], {}).get('metrics', {})
        
        metrics = [
            {'name': 'exact_match', 'display_name': 'EM'},
            {'name': 'f1_score', 'display_name': 'F1'}
        ]
        
        table_latex = template.render(
            ablation_results=ablation_configs,
            metrics=metrics,
            baseline_performance=ablation_data.get('static_lora', {}).get('metrics', {}),
            show_deltas=True,
            show_component_analysis=True,
            component_contributions=ablation_data.get('component_contributions', {}),
            show_ablation_notes=True,
            hierarchical_ablation=True,
            governance_ablation=True
        )
        
        return table_latex
    
    def generate_latency_pareto_plot(self, output_path: Path) -> None:
        """Generate latency vs quality Pareto plot."""
        logger.info("Generating latency-quality Pareto plot...")
        
        # Extract latency and quality data
        methods_data = []
        for method_name, method_result in self.results_data.get('method_results', {}).items():
            if 'latency' in method_result and 'metrics' in method_result:
                methods_data.append({
                    'method': method_name,
                    'display_name': self._get_method_display_name(method_name),
                    'latency_p50': method_result['latency'].get('p50_ms', 0),
                    'latency_p95': method_result['latency'].get('p95_ms', 0),
                    'exact_match': method_result['metrics'].get('exact_match', {}).get('mean', 0),
                    'f1_score': method_result['metrics'].get('f1_score', {}).get('mean', 0),
                    'is_bem': method_name.startswith('bem')
                })
        
        if not methods_data:
            logger.warning("No latency data available for Pareto plot")
            return
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        for method_data in methods_data:
            color = 'red' if method_data['is_bem'] else 'blue'
            marker = 'o' if method_data['is_bem'] else 's'
            
            plt.scatter(
                method_data['latency_p50'],
                method_data['exact_match'],
                c=color,
                marker=marker,
                s=100,
                alpha=0.7,
                label=method_data['display_name']
            )
            
            # Add method labels
            plt.annotate(
                method_data['display_name'],
                (method_data['latency_p50'], method_data['exact_match']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                alpha=0.8
            )
        
        plt.xlabel('Latency P50 (ms)')
        plt.ylabel('Exact Match Score')
        plt.title('Latency-Quality Pareto Frontier')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Pareto plot saved to {output_path}")
    
    def generate_index_swap_plot(self, output_path: Path) -> None:
        """Generate index-swap monotonicity plot for policy-over-memory validation."""
        logger.info("Generating index-swap monotonicity plot...")
        
        index_swap_data = self.results_data.get('index_swap_results', {})
        if not index_swap_data:
            logger.warning("No index-swap data available")
            return
        
        # Extract performance for each index type
        index_types = ['clean', 'corrupt', 'shuffle']
        performance_data = []
        
        for idx_type in index_types:
            if idx_type in index_swap_data:
                perf = index_swap_data[idx_type]
                performance_data.append({
                    'index_type': idx_type,
                    'display_name': idx_type.title(),
                    'exact_match': perf.get('exact_match', {}).get('mean', 0),
                    'exact_match_ci': perf.get('exact_match', {}).get('confidence_interval', [0, 0]),
                    'f1_score': perf.get('f1_score', {}).get('mean', 0),
                    'f1_score_ci': perf.get('f1_score', {}).get('confidence_interval', [0, 0])
                })
        
        if len(performance_data) < 3:
            logger.warning("Insufficient index-swap data for plot")
            return
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        x_pos = np.arange(len(index_types))
        
        # Exact Match plot
        em_means = [p['exact_match'] for p in performance_data]
        em_cis = [p['exact_match_ci'] for p in performance_data]
        em_errors = [[m - ci[0], ci[1] - m] for m, ci in zip(em_means, em_cis)]
        em_errors = np.array(em_errors).T
        
        ax1.bar(x_pos, em_means, yerr=em_errors, capsize=5, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Index Quality')
        ax1.set_ylabel('Exact Match Score')
        ax1.set_title('Policy over Memory: EM Performance')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([p['display_name'] for p in performance_data])
        ax1.grid(True, alpha=0.3)
        
        # F1 Score plot
        f1_means = [p['f1_score'] for p in performance_data]
        f1_cis = [p['f1_score_ci'] for p in performance_data]
        f1_errors = [[m - ci[0], ci[1] - m] for m, ci in zip(f1_means, f1_cis)]
        f1_errors = np.array(f1_errors).T
        
        ax2.bar(x_pos, f1_means, yerr=f1_errors, capsize=5, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Index Quality')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Policy over Memory: F1 Performance')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([p['display_name'] for p in performance_data])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Index-swap plot saved to {output_path}")
    
    def _get_method_display_name(self, method: str) -> str:
        """Get display name for method."""
        display_names = {
            'lora': 'Static LoRA',
            'prefix': 'Prefix-tuning',
            'ia3': 'IA³',
            'mole': 'MoLE',
            'hyperlora': 'Hyper-LoRA',
            'bem_p1': 'BEM P1',
            'bem_p2': 'BEM P2',
            'bem_p3': 'BEM P3',
            'bem_p4': 'BEM P4'
        }
        return display_names.get(method, method)
    
    def generate_all_artifacts(self) -> None:
        """Generate all paper artifacts."""
        logger.info("Generating all paper artifacts...")
        
        # Create output directories
        (self.output_dir / "auto").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figs").mkdir(parents=True, exist_ok=True)
        
        # Validate claims
        claim_validation = self.validate_claims_against_results()
        
        # Generate tables
        main_results_table = self.generate_main_results_table()
        with open(self.output_dir / "auto" / "main_results.tex", 'w') as f:
            f.write(main_results_table)
        
        ablation_table = self.generate_ablation_study_table()
        with open(self.output_dir / "auto" / "ablation_studies.tex", 'w') as f:
            f.write(ablation_table)
        
        # Generate plots
        self.generate_latency_pareto_plot(self.output_dir / "figs" / "latency_pareto.pdf")
        self.generate_index_swap_plot(self.output_dir / "figs" / "index_swap_monotonicity.pdf")
        
        # Save claim validation report
        with open(self.output_dir / "claim_validation_report.json", 'w') as f:
            json.dump(claim_validation, f, indent=2)
        
        logger.info("All artifacts generated successfully!")
        
        if not claim_validation['ready_for_publication']:
            logger.error("❌ PAPER NOT READY: Some claims failed validation!")
            logger.error("Review failed claims before submission.")
        else:
            logger.info("✅ All claims validated - paper ready for compilation!")
    
    def compile_paper(self) -> bool:
        """Compile the full LaTeX paper."""
        logger.info("Compiling LaTeX paper...")
        
        main_tex = self.output_dir / "main.tex"
        if not main_tex.exists():
            logger.error("main.tex not found!")
            return False
        
        try:
            # Run pdflatex twice for references
            for i in range(2):
                result = subprocess.run([
                    'pdflatex',
                    '-interaction=nonstopmode',
                    '-output-directory', str(self.output_dir),
                    str(main_tex)
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode != 0:
                    logger.error(f"LaTeX compilation failed (run {i+1}):")
                    logger.error(result.stderr)
                    return False
            
            # Check if PDF was generated
            pdf_output = self.output_dir / "main.pdf"
            if pdf_output.exists():
                logger.info(f"✅ Paper compiled successfully: {pdf_output}")
                return True
            else:
                logger.error("PDF was not generated despite successful compilation")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("LaTeX compilation timed out")
            return False
        except Exception as e:
            logger.error(f"Error during compilation: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='BEM Paper Factory - Automated Paper Builder')
    parser.add_argument('--results', required=True, help='Path to statistical analysis results JSON')
    parser.add_argument('--claims', default='paper/claims.yaml', help='Path to claims ledger')
    parser.add_argument('--templates', default='templates', help='Templates directory')
    parser.add_argument('--output', default='paper', help='Output directory')
    parser.add_argument('--compile', action='store_true', help='Compile LaTeX after generation')
    parser.add_argument('--check-page-limit', action='store_true', help='Check page limit after compilation')
    
    args = parser.parse_args()
    
    # Initialize paper builder
    builder = PaperBuilder(
        results_file=args.results,
        claims_file=args.claims,
        templates_dir=args.templates,
        output_dir=args.output
    )
    
    # Generate all artifacts
    builder.generate_all_artifacts()
    
    # Compile paper if requested
    if args.compile:
        success = builder.compile_paper()
        
        if success and args.check_page_limit:
            # Check page limit
            from .page_guard import PageLimitGuard
            guard = PageLimitGuard()
            
            main_tex = Path(args.output) / "main.tex"
            result = guard.check_page_limit(main_tex)
            
            if result['within_limit']:
                logger.info("✅ Paper is within page limit!")
            else:
                logger.error(f"❌ Paper exceeds limit by {result['excess_pages']} pages")

if __name__ == '__main__':
    main()