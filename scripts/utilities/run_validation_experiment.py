#!/usr/bin/env python3
"""
BEM Validation Experiment - Executive Runner

This script orchestrates the complete BEM validation experiment pipeline:
1. Data preparation (synthetic JSON and summarization tasks)
2. Static LoRA training (separate JSON and summary experts)
3. BEM controller training (interpolation learning)
4. Comprehensive evaluation (statistical analysis and benchmarking)
5. Report generation and visualization

Execute this to run the full "dirt simple" validation experiment.
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

# Rich imports for beautiful console output
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn
from rich.table import Table
from rich.text import Text
from rich.prompt import Confirm

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Check that all required dependencies are available."""
    
    console.print("[bold blue]Checking system dependencies...")
    
    required_packages = [
        'torch',
        'transformers',
        'peft',
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
        'pandas',
        'rich',
        'wandb'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        console.print(f"[red]❌ Missing required packages: {', '.join(missing)}")
        console.print("[yellow]Install with: pip install " + " ".join(missing))
        return False
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            console.print(f"[green]✓[/green] CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            console.print("[yellow]⚠️[/yellow] CUDA not available - will use CPU (slower)")
    except Exception as e:
        console.print(f"[yellow]⚠️[/yellow] Could not check CUDA: {e}")
    
    console.print("[green]✓[/green] All dependencies satisfied")
    return True


def run_data_preparation(args: argparse.Namespace) -> bool:
    """Run the data preparation phase."""
    
    console.print("\n" + "="*60)
    console.print("[bold green]🔧 PHASE 1: DATA PREPARATION")
    console.print("="*60)
    
    data_script = Path(__file__).parent / "scripts" / "prepare_validation_data.py"
    
    cmd = [
        sys.executable, str(data_script),
        "--output-dir", args.data_dir,
        "--num-json", str(args.num_samples),
        "--num-summary", str(args.num_samples),
        "--seed", str(args.seed)
    ]
    
    if args.no_viz:
        cmd.append("--no-viz")
    
    try:
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Preparing validation data...", total=None)
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            progress.update(task, completed=True)
        
        console.print("[green]✅ Data preparation completed successfully")
        
        if args.verbose and result.stdout:
            console.print("\n[dim]Data preparation output:[/dim]")
            console.print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Data preparation failed: {e}")
        if e.stderr:
            console.print(f"[red]Error output:[/red] {e.stderr}")
        return False


def run_validation_training(args: argparse.Namespace) -> bool:
    """Run the main validation experiment training."""
    
    console.print("\n" + "="*60)
    console.print("[bold green]🚀 PHASE 2: BEM VALIDATION TRAINING")
    console.print("="*60)
    
    training_script = Path(__file__).parent / "experiments" / "validation_experiment.py"
    
    cmd = [
        sys.executable, str(training_script),
        "--output-dir", args.output_dir
    ]
    
    if args.no_wandb:
        cmd.append("--no-wandb")
    
    if args.quick:
        cmd.append("--quick")
    
    try:
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Training BEM validation experiment...", total=None)
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            progress.update(task, completed=True)
        
        console.print("[green]✅ BEM validation training completed successfully")
        
        if args.verbose and result.stdout:
            console.print("\n[dim]Training output:[/dim]")
            console.print(result.stdout[-2000:])  # Show last 2000 chars to avoid spam
        
        return True
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ BEM validation training failed: {e}")
        if e.stderr:
            console.print(f"[red]Error output:[/red] {e.stderr}")
        return False


def run_comprehensive_evaluation(args: argparse.Namespace) -> bool:
    """Run the comprehensive evaluation phase."""
    
    console.print("\n" + "="*60)
    console.print("[bold green]🧪 PHASE 3: COMPREHENSIVE EVALUATION")
    console.print("="*60)
    
    eval_script = Path(__file__).parent / "eval" / "bem_evaluator.py"
    
    cmd = [
        sys.executable, str(eval_script),
        "--bem-model", str(Path(args.output_dir) / "bem_model.pt"),
        "--experiment-dir", args.output_dir,
        "--output-dir", args.eval_dir,
        "--num-samples", str(args.num_eval_samples)
    ]
    
    if args.no_viz:
        cmd.append("--no-visualizations")
    
    try:
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Running comprehensive evaluation...", total=None)
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            progress.update(task, completed=True)
        
        console.print("[green]✅ Comprehensive evaluation completed successfully")
        
        if args.verbose and result.stdout:
            console.print("\n[dim]Evaluation output:[/dim]")
            console.print(result.stdout[-1500:])  # Show last 1500 chars
        
        return True
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Comprehensive evaluation failed: {e}")
        if e.stderr:
            console.print(f"[red]Error output:[/red] {e.stderr}")
        return False


def generate_executive_summary(args: argparse.Namespace) -> Dict[str, Any]:
    """Generate an executive summary of the validation experiment results."""
    
    console.print("\n" + "="*60)
    console.print("[bold green]📊 PHASE 4: EXECUTIVE SUMMARY")
    console.print("="*60)
    
    summary = {
        'experiment_info': {
            'run_date': datetime.now().isoformat(),
            'total_samples': args.num_samples * 2,  # JSON + Summary
            'eval_samples': args.num_eval_samples,
            'seed': args.seed,
            'quick_mode': args.quick
        },
        'results': {},
        'status': 'unknown'
    }
    
    try:
        # Try to load evaluation results
        eval_results_path = Path(args.eval_dir) / "evaluation_results.json"
        if eval_results_path.exists():
            with open(eval_results_path) as f:
                eval_results = json.load(f)
                summary['results'] = eval_results
        
        # Try to load training metrics
        training_metrics_path = Path(args.output_dir) / "controller_training_metrics.json"
        if training_metrics_path.exists():
            with open(training_metrics_path) as f:
                training_metrics = json.load(f)
                summary['training_metrics'] = training_metrics
        
        # Extract key metrics for display
        if 'task_specialization' in summary.get('results', {}):
            task_spec = summary['results']['task_specialization']
            if 'accuracy_scores' in task_spec and 'bem' in task_spec['accuracy_scores']:
                bem_acc = task_spec['accuracy_scores']['bem']['overall_accuracy']
                summary['key_metrics'] = {
                    'bem_accuracy': bem_acc,
                    'validation_status': 'SUCCESS' if bem_acc > 0.7 else 'NEEDS_WORK'
                }
                summary['status'] = 'success' if bem_acc > 0.7 else 'partial'
        
        # Save executive summary
        summary_path = Path(args.output_dir) / "executive_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        console.print(f"[green]✓[/green] Executive summary saved to {summary_path}")
        
    except Exception as e:
        console.print(f"[yellow]⚠️[/yellow] Could not generate complete summary: {e}")
        summary['status'] = 'error'
        summary['error'] = str(e)
    
    return summary


def display_results(summary: Dict[str, Any], args: argparse.Namespace):
    """Display a beautiful results summary."""
    
    console.print("\n" + "="*60)
    console.print("[bold green]🎉 BEM VALIDATION EXPERIMENT COMPLETED")
    console.print("="*60)
    
    # Create results table
    table = Table(title="Experiment Results Summary", show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_column("Status", justify="center")
    
    # Basic info
    table.add_row(
        "Run Date", 
        summary['experiment_info']['run_date'][:19], 
        "📅"
    )
    table.add_row(
        "Total Training Samples", 
        f"{summary['experiment_info']['total_samples']:,}", 
        "📊"
    )
    table.add_row(
        "Evaluation Samples", 
        f"{summary['experiment_info']['eval_samples']:,}", 
        "🧪"
    )
    
    # Key results
    if 'key_metrics' in summary:
        accuracy = summary['key_metrics']['bem_accuracy']
        status = summary['key_metrics']['validation_status']
        
        table.add_row(
            "BEM Controller Accuracy", 
            f"{accuracy:.1%}", 
            "✅" if status == "SUCCESS" else "⚠️"
        )
        table.add_row(
            "Validation Status", 
            status, 
            "✅" if status == "SUCCESS" else "⚠️"
        )
        
        # Overall assessment
        if accuracy > 0.8:
            assessment = "EXCELLENT - Ready for full implementation"
            emoji = "🚀"
        elif accuracy > 0.7:
            assessment = "GOOD - Hypothesis validated"
            emoji = "✅"
        elif accuracy > 0.6:
            assessment = "PARTIAL - Needs improvement"
            emoji = "⚠️"
        else:
            assessment = "POOR - Requires investigation"
            emoji = "❌"
        
        table.add_row("Overall Assessment", assessment, emoji)
    
    console.print(table)
    
    # File locations
    console.print("\n[bold blue]📁 Output Files:[/bold blue]")
    files_table = Table(show_header=False, box=None)
    files_table.add_column("", style="dim")
    files_table.add_column("", style="cyan")
    
    key_files = [
        ("Training Data", Path(args.data_dir) / "combined_tasks.json"),
        ("BEM Model", Path(args.output_dir) / "bem_model.pt"),
        ("Training Report", Path(args.output_dir) / "validation_report.md"),
        ("Evaluation Results", Path(args.eval_dir) / "evaluation_results.json"),
        ("Final Report", Path(args.eval_dir) / "comprehensive_report.md"),
        ("Build Profile", "logs/profile_build.md"),
        ("Executive Summary", Path(args.output_dir) / "executive_summary.json")
    ]
    
    for name, path in key_files:
        exists = "✅" if Path(path).exists() else "❌"
        files_table.add_row(f"{exists} {name}:", str(path))
    
    console.print(files_table)
    
    # Next steps
    if summary.get('status') == 'success':
        console.print("\n[bold green]🚀 Next Steps - Proceed to Full BEM Implementation:[/bold green]")
        console.print("• Phase 2: Hierarchical routing (prefix → chunk → token)")
        console.print("• Phase 3: Custom CUDA kernels for performance")
        console.print("• Phase 4: Retrieval-aware controller features")
        console.print("• Phase 5: Multi-BEM composition")
    else:
        console.print("\n[bold yellow]⚠️ Next Steps - Investigation Required:[/bold yellow]")
        console.print("• Review training logs for convergence issues")
        console.print("• Adjust hyperparameters (learning rate, architecture)")
        console.print("• Verify data quality and task differentiation")
        console.print("• Consider alternative controller architectures")


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="Run the complete BEM validation experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    parser.add_argument("--data-dir", default="data/validation_experiment",
                       help="Directory for training data")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of samples per task type")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Experiment configuration
    parser.add_argument("--output-dir", default="outputs/validation_experiment",
                       help="Output directory for experiment results")
    parser.add_argument("--eval-dir", default="outputs/evaluation_results",
                       help="Output directory for evaluation results")
    parser.add_argument("--num-eval-samples", type=int, default=500,
                       help="Number of evaluation samples")
    
    # Execution options
    parser.add_argument("--quick", action="store_true",
                       help="Quick test run with reduced data and epochs")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--no-viz", action="store_true",
                       help="Skip visualization generation")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed subprocess output")
    
    # Phase control
    parser.add_argument("--skip-data", action="store_true",
                       help="Skip data preparation phase")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training phase")
    parser.add_argument("--skip-eval", action="store_true",
                       help="Skip evaluation phase")
    parser.add_argument("--data-only", action="store_true",
                       help="Only run data preparation")
    
    args = parser.parse_args()
    
    # Display header
    console.print(Panel.fit(
        Text("BEM Validation Experiment\n'Dirt Simple' Controller Learning", 
             justify="center", style="bold white"),
        style="blue",
        title="🧪 Research Validation Pipeline"
    ))
    
    if args.quick:
        console.print("[yellow]⚡ Quick mode enabled - reduced data and training time[/yellow]")
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Create output directories
    for directory in [args.data_dir, args.output_dir, args.eval_dir, "logs"]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    success = True
    
    try:
        # Phase 1: Data Preparation
        if not args.skip_data:
            if not run_data_preparation(args):
                success = False
                return 1
        
        if args.data_only:
            console.print("\n[green]✅ Data-only run completed successfully!")
            return 0
        
        # Phase 2: Training
        if success and not args.skip_training:
            if not run_validation_training(args):
                success = False
        
        # Phase 3: Evaluation
        if success and not args.skip_eval:
            if not run_comprehensive_evaluation(args):
                success = False
        
        # Phase 4: Summary
        if success:
            summary = generate_executive_summary(args)
            display_results(summary, args)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Experiment interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]❌ Unexpected error: {e}[/red]")
        logger.exception("Unexpected error in main execution")
        return 1
    
    # Final summary
    total_time = time.time() - start_time
    
    if success:
        console.print(f"\n[bold green]🎉 EXPERIMENT COMPLETED SUCCESSFULLY![/bold green]")
        console.print(f"Total execution time: {total_time:.1f} seconds")
        
        # Check if we can make a recommendation
        summary_file = Path(args.output_dir) / "executive_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
                    if summary.get('status') == 'success':
                        console.print("\n[bold green]✅ HYPOTHESIS VALIDATED - PROCEED WITH FULL BEM IMPLEMENTATION[/bold green]")
                    else:
                        console.print("\n[bold yellow]⚠️ RESULTS MIXED - REVIEW REQUIRED BEFORE PROCEEDING[/bold yellow]")
            except Exception:
                pass
        
        return 0
    else:
        console.print(f"\n[bold red]❌ EXPERIMENT FAILED[/bold red]")
        console.print(f"Partial execution time: {total_time:.1f} seconds")
        console.print("Check logs and error messages above for details.")
        return 1


if __name__ == "__main__":
    exit(main())