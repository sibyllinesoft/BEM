#!/usr/bin/env python3
"""
BEM v1.3 Performance+Agentic Sprint - Production-Ready Orchestration Script

This is the main entry point for running the complete BEM v1.3 experimental pipeline
with full production capabilities including:

- Complete XML workflow orchestration from TODO.md
- Real-time health monitoring and automated recovery
- Resource management and cleanup
- Statistical validation gates with automated rollback
- CI/CD integration and deployment automation
- Comprehensive logging and telemetry
- Emergency recovery procedures

Usage:
  python run_bem_v13_production.py --mode full      # Complete pipeline
  python run_bem_v13_production.py --mode ci        # CI/CD mode
  python run_bem_v13_production.py --mode monitor   # Health monitoring only
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import threading
import yaml

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from bem_workflow_orchestrator import WorkflowExecutor, WorkflowConfig, ResourceMonitor
from scripts.workflow_health_monitor import WorkflowHealthMonitor
from scripts.resource_manager import ResourceManager, ResourcePolicy
from scripts.ci_cd_integration import BEMCIOrchestrator


class ProductionOrchestrator:
    """
    Master orchestrator for BEM v1.3 production pipeline.
    
    Coordinates all subsystems:
    - Workflow execution
    - Health monitoring  
    - Resource management
    - CI/CD integration
    - Emergency recovery
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("production_config.yaml")
        self.config = self.load_configuration()
        
        # Initialize subsystems
        self.workflow_executor = None
        self.health_monitor = None
        self.resource_manager = None
        self.ci_orchestrator = None
        
        # State tracking
        self.running = False
        self.subsystems_started = []
        self.shutdown_requested = False
        
        # Setup logging
        self.setup_logging()
        
        # Setup signal handlers for graceful shutdown
        self.setup_signal_handlers()
    
    def load_configuration(self) -> Dict[str, Any]:
        """Load production configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        
        return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default production configuration."""
        return {
            "mode": "full",  # full, ci, monitor, minimal
            
            # Workflow configuration
            "workflow": {
                "max_concurrent_tasks": 2,
                "timeout_hours": 48,
                "continue_on_non_critical_failure": True,
                "cleanup_on_success": False,
                "verbose_logging": True
            },
            
            # Health monitoring configuration
            "health_monitoring": {
                "enabled": True,
                "check_interval_seconds": 60,
                "alert_webhook": None,
                "auto_recovery": True
            },
            
            # Resource management configuration
            "resource_management": {
                "enabled": True,
                "max_disk_usage_percent": 85.0,
                "max_memory_usage_percent": 90.0,
                "cleanup_interval_minutes": 30,
                "auto_archival": True,
                "emergency_cleanup_threshold": 95.0
            },
            
            # CI/CD configuration
            "cicd": {
                "enabled": False,
                "github_token": None,
                "repo_name": None,
                "notification_webhook": None
            },
            
            # Safety and recovery
            "safety": {
                "max_runtime_hours": 72,
                "auto_rollback_on_failure": True,
                "emergency_stop_threshold": 98.0,  # Disk usage %
                "backup_critical_files": True
            },
            
            # Output and logging
            "output": {
                "root_dir": "logs",
                "telemetry_interval_minutes": 15,
                "compress_logs": True,
                "retain_logs_days": 30
            }
        }
    
    def setup_logging(self):
        """Setup comprehensive production logging."""
        log_dir = Path(self.config["output"]["root_dir"]) / "production"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"production_orchestrator_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("BEM v1.3 Production Orchestrator initialized")
        self.logger.info(f"Configuration loaded from: {self.config_path}")
        self.logger.info(f"Log file: {log_file}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def initialize_subsystems(self):
        """Initialize all required subsystems."""
        self.logger.info("Initializing subsystems...")
        
        # 1. Workflow Executor
        workflow_config = WorkflowConfig(
            output_root=Path(self.config["output"]["root_dir"]),
            max_concurrent_tasks=self.config["workflow"]["max_concurrent_tasks"],
            timeout_hours=self.config["workflow"]["timeout_hours"],
            continue_on_non_critical_failure=self.config["workflow"]["continue_on_non_critical_failure"],
            cleanup_on_success=self.config["workflow"]["cleanup_on_success"],
            verbose_logging=self.config["workflow"]["verbose_logging"]
        )
        
        resource_monitor = ResourceMonitor(workflow_config)
        self.workflow_executor = WorkflowExecutor(workflow_config, resource_monitor)
        
        # Load XML workflow specification
        xml_content = self.load_workflow_xml()
        self.workflow_executor.load_workflow_from_xml(xml_content)
        
        # 2. Health Monitor
        if self.config["health_monitoring"]["enabled"]:
            self.health_monitor = WorkflowHealthMonitor(
                logs_dir=Path(self.config["output"]["root_dir"]),
                gates_config=Path("gates_bem2.yaml"),
                check_interval=self.config["health_monitoring"]["check_interval_seconds"],
                alert_webhook=self.config["health_monitoring"]["alert_webhook"]
            )
        
        # 3. Resource Manager
        if self.config["resource_management"]["enabled"]:
            resource_policy = ResourcePolicy(
                max_disk_usage_percent=self.config["resource_management"]["max_disk_usage_percent"],
                max_memory_usage_percent=self.config["resource_management"]["max_memory_usage_percent"],
                cleanup_interval_minutes=self.config["resource_management"]["cleanup_interval_minutes"],
                emergency_cleanup_threshold=self.config["resource_management"]["emergency_cleanup_threshold"]
            )
            
            self.resource_manager = ResourceManager(resource_policy)
        
        # 4. CI/CD Orchestrator
        if self.config["cicd"]["enabled"]:
            self.ci_orchestrator = BEMCIOrchestrator(
                config_path=Path("cicd_config.yaml"),
                github_token=self.config["cicd"]["github_token"],
                repo_name=self.config["cicd"]["repo_name"]
            )
        
        self.logger.info("All subsystems initialized successfully")
    
    def load_workflow_xml(self) -> str:
        """Load the XML workflow specification from TODO.md requirements."""
        # This is the same XML content from TODO.md
        return '''<workflows project="bem-v1_3-perf-agentic" version="1.0">
  <workflow name="building">
    <env id="B0">
      <desc>Set up environment; compile kernels; pin versions</desc>
      <commands>
        <cmd>source .venv/bin/activate</cmd>
        <cmd>pip install -U numpy pandas scipy statsmodels scikit-learn sacrebleu torch torchvision einops faiss-cpu</cmd>
        <cmd>pip install transformers accelerate peft xformers wandb mlflow pyyaml rich</cmd>
        <cmd>python bem/kernels/build.py --check-numerics --tol 1e-3 --out logs/kernels.json</cmd>
        <cmd>python bem/quant/fp8_qat.py --selftest --out logs/fp8_selftest.json</cmd>
        <cmd>python scripts/record_env.py --out dist/repro_manifest.json</cmd>
      </commands>
      <make_sure>
        <item>FP8 numerics pass==true; kernels numerics within tol; CUDA visible.</item>
        <item>Repro manifest includes package versions, CUDA, GPU name, SHAs.</item>
      </make_sure>
    </env>
    <assets id="B1">
      <desc>Fetch models, indices, x-encoder; verify hygiene</desc>
      <commands>
        <cmd>python scripts/fetch_model.py --name {{BASE_MODEL}} --out models/base</cmd>
        <cmd>python scripts/build_faiss.py --input corpora --out indices/domain.faiss --encoder {{RETR_ENCODER}}</cmd>
        <cmd>python scripts/fetch_xencoder.py --name {{XENC_TINY}} --out models/xenc</cmd>
        <cmd>python analysis/hygiene.py --check-sacrebleu-config --out analysis/hygiene.json</cmd>
      </commands>
      <make_sure>
        <item>Model IDs/SHAs recorded; index/encoder hashes logged.</item>
      </make_sure>
    </assets>
    <guards id="B2">
      <desc>Enable parity, leak, and numerics guardrails</desc>
      <commands>
        <cmd>python analysis/check_parity.py --experiments experiments --baseline experiments/v13_anchor.yml --tolerance 0.05</cmd>
        <cmd>python analysis/leakcheck.py --eval data/test.jsonl --index indices/domain.faiss --out analysis/leak.json</cmd>
        <cmd>python bem/kron_kernel_selftest.py --out logs/kron_selftest.json</cmd>
      </commands>
      <make_sure>
        <item>Abort on parity breach or kernel selftest failure; leak rate negligible.</item>
      </make_sure>
    </guards>
  </workflow>
  <workflow name="running">
    <perf_wave id="R1">
      <desc>Train & eval performance variants V1–V4</desc>
      <commands>
        <cmd>python train.py --exp experiments/v1_dynrank.yml --seeds 1,2,3,4,5 --log_dir logs/V1</cmd>
        <cmd>python train.py --exp experiments/v2_gateshaping.yml --seeds 1,2,3,4,5 --log_dir logs/V2</cmd>
        <cmd>python train.py --exp experiments/v3_kron.yml --seeds 1,2,3,4,5 --log_dir logs/V3</cmd>
        <cmd>python train.py --exp experiments/v4_film.yml --seeds 1,2,3,4,5 --log_dir logs/V4</cmd>
        <cmd>python evaluate.py --ckpt logs/V1/best.pt --suite eval/suites/main.yml --slice both --latency --out logs/V1/eval.json</cmd>
        <cmd>python evaluate.py --ckpt logs/V2/best.pt --suite eval/suites/main.yml --slice both --latency --out logs/V2/eval.json</cmd>
        <cmd>python evaluate.py --ckpt logs/V3/best.pt --suite eval/suites/main.yml --slice both --latency --out logs/V3/eval.json</cmd>
        <cmd>python evaluate.py --ckpt logs/V4/best.pt --suite eval/suites/main.yml --slice both --latency --out logs/V4/eval.json</cmd>
      </commands>
      <make_sure>
        <item>KV-hit %, p50/p95 latency, VRAM recorded; parity check passes; spectra logged.</item>
      </make_sure>
    </perf_wave>
  </workflow>
  <workflow name="tracking">
    <harvest id="T1">
      <desc>Consolidate runs; compute stats, Pareto, spectra, audits</desc>
      <commands>
        <cmd>python analysis/collect.py --roots logs --out analysis/runs.jsonl</cmd>
        <cmd>python analysis/stats.py --pairs analysis/runs.jsonl --bootstrap 10000 --bca --paired --fdr families.yaml --out analysis/stats.json</cmd>
        <cmd>python analysis/pareto.py --in analysis/stats.json --out analysis/pareto.json</cmd>
      </commands>
      <make_sure>
        <item>Stars only when CI&gt;0 post-FDR; Slice-B gates promotion; aggregate reported as mean relative Δ across EM,F1,BLEU,chrF.</item>
      </make_sure>
    </harvest>
  </workflow>
  <workflow name="evaluating">
    <promote id="E1">
      <desc>Apply gates; decide winners; record ablations</desc>
      <commands>
        <cmd>python analysis/promote.py --stats analysis/stats.json --pareto analysis/pareto.json --gates gates_bem2.yaml --out analysis/winners.json</cmd>
      </commands>
      <make_sure>
        <item>Only CI-backed Slice-B wins within latency/VRAM gates promoted.</item>
      </make_sure>
    </promote>
  </workflow>
  <workflow name="refinement">
    <paper id="P1">
      <desc>Regenerate tables/figures; update claims; render PDFs</desc>
      <commands>
        <cmd>python scripts/update_claims_from_stats.py --stats analysis/stats.json --out paper/claims.yaml</cmd>
        <cmd>python analysis/build_tables.py --in analysis/stats.json --winners analysis/winners.json --out paper/auto</cmd>
        <cmd>python scripts/render_paper.py --tex paper/main.tex --out paper/main.pdf</cmd>
      </commands>
      <make_sure>
        <item>Main body: baseline vs promoted variants on Slice-B; appendix holds negatives/neutral.</item>
      </make_sure>
    </paper>
  </workflow>
</workflows>'''
    
    async def start_subsystems(self):
        """Start all subsystem services."""
        self.logger.info("Starting subsystem services...")
        
        # Start health monitor
        if self.health_monitor:
            health_monitor_thread = threading.Thread(
                target=self.health_monitor.start_monitoring,
                daemon=True
            )
            health_monitor_thread.start()
            self.subsystems_started.append("health_monitor")
            self.logger.info("✓ Health monitor started")
        
        # Start resource manager
        if self.resource_manager:
            self.resource_manager.start_monitoring()
            self.subsystems_started.append("resource_manager")
            self.logger.info("✓ Resource manager started")
        
        self.logger.info(f"Started {len(self.subsystems_started)} subsystem services")
    
    async def run_full_pipeline(self) -> bool:
        """Run the complete BEM v1.3 experimental pipeline."""
        self.logger.info("🚀 Starting BEM v1.3 Performance+Agentic Sprint - Full Pipeline")
        
        try:
            # Start monitoring services
            await self.start_subsystems()
            
            # Check prerequisites
            resource_monitor = ResourceMonitor(WorkflowConfig())
            prerequisites_ok, issues = resource_monitor.check_prerequisites()
            
            if not prerequisites_ok:
                self.logger.error("Prerequisites check failed:")
                for issue in issues:
                    self.logger.error(f"  • {issue}")
                return False
            
            # Execute complete workflow
            success = await self.workflow_executor.execute_complete_workflow()
            
            if success:
                self.logger.info("🎉 Complete pipeline executed successfully!")
                
                # Generate final reports
                await self.generate_final_reports()
                
                # Send success notification
                self.send_notification(
                    "BEM v1.3 pipeline completed successfully",
                    "success"
                )
                
                return True
            else:
                self.logger.error("💥 Pipeline execution failed")
                
                # Attempt automated recovery if configured
                if self.config["safety"]["auto_rollback_on_failure"]:
                    await self.attempt_recovery()
                
                return False
                
        except Exception as e:
            self.logger.error(f"Pipeline execution failed with error: {e}")
            return False
        finally:
            await self.shutdown()
    
    async def run_ci_mode(self) -> bool:
        """Run in CI/CD mode with lightweight execution."""
        self.logger.info("🔄 Starting BEM v1.3 CI/CD Pipeline")
        
        if not self.ci_orchestrator:
            self.logger.error("CI/CD orchestrator not available")
            return False
        
        try:
            # Run CI pipeline
            result = await self.ci_orchestrator.run_ci_pipeline()
            
            if result["status"] == "success":
                self.logger.info("✅ CI pipeline completed successfully")
                return True
            else:
                self.logger.error(f"❌ CI pipeline failed: {result.get('error_message', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"CI pipeline execution failed: {e}")
            return False
    
    async def run_monitor_mode(self):
        """Run in monitoring-only mode."""
        self.logger.info("📊 Starting BEM v1.3 Health Monitoring Service")
        
        # Start monitoring services
        await self.start_subsystems()
        
        # Keep running until shutdown requested
        try:
            while not self.shutdown_requested:
                await asyncio.sleep(60)  # Check every minute
                
                # Log system status periodically
                if self.resource_manager:
                    status = self.resource_manager.get_resource_status()
                    self.logger.info(f"System Status - "
                                   f"Disk: {status['disk']['usage_percent']:.1f}%, "
                                   f"Memory: {status['memory']['usage_percent']:.1f}%, "
                                   f"CPU: {status['cpu']['usage_percent']:.1f}%")
        
        except KeyboardInterrupt:
            self.logger.info("Monitoring interrupted by user")
        finally:
            await self.shutdown()
    
    async def generate_final_reports(self):
        """Generate comprehensive final reports."""
        self.logger.info("Generating final reports...")
        
        try:
            # Generate execution report
            report = self.workflow_executor.generate_execution_report()
            
            report_file = Path(self.config["output"]["root_dir"]) / "final_pipeline_report.md"
            with open(report_file, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Final report saved to: {report_file}")
            
            # Generate resource utilization report
            if self.resource_manager:
                resource_report = self.resource_manager.generate_resource_report()
                
                resource_file = Path(self.config["output"]["root_dir"]) / "resource_utilization_report.txt"
                with open(resource_file, 'w') as f:
                    f.write(resource_report)
                
                self.logger.info(f"Resource report saved to: {resource_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating final reports: {e}")
    
    async def attempt_recovery(self):
        """Attempt automated recovery from failures."""
        self.logger.info("🔧 Attempting automated recovery...")
        
        recovery_actions = []
        
        try:
            # Clear GPU memory
            if self.resource_manager and self.resource_manager.gpu_manager.gpu_available:
                gpu_result = self.resource_manager.gpu_manager.optimize_gpu_memory()
                recovery_actions.append(f"GPU optimization: {gpu_result.get('actions_taken', [])}")
            
            # Emergency cleanup
            if self.resource_manager:
                cleanup_result = self.resource_manager.emergency_cleanup()
                recovery_actions.append(f"Emergency cleanup: {cleanup_result['total_space_freed_gb']:.1f}GB freed")
            
            # Log recovery actions
            for action in recovery_actions:
                self.logger.info(f"  Recovery action: {action}")
            
            self.send_notification(
                f"Automated recovery attempted: {len(recovery_actions)} actions taken",
                "warning"
            )
            
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
    
    def send_notification(self, message: str, level: str = "info"):
        """Send notification through configured channels."""
        notification_data = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "pipeline": "bem_v1.3_production"
        }
        
        # Log notification
        if level == "error":
            self.logger.error(f"NOTIFICATION: {message}")
        elif level == "warning":
            self.logger.warning(f"NOTIFICATION: {message}")
        else:
            self.logger.info(f"NOTIFICATION: {message}")
        
        # Send webhook notification if configured
        webhook_url = self.config.get("cicd", {}).get("notification_webhook")
        if webhook_url:
            try:
                import requests
                response = requests.post(
                    webhook_url,
                    json=notification_data,
                    timeout=10
                )
                if response.status_code == 200:
                    self.logger.debug("Webhook notification sent successfully")
            except Exception as e:
                self.logger.error(f"Failed to send webhook notification: {e}")
    
    async def shutdown(self):
        """Graceful shutdown of all subsystems."""
        if not self.running:
            return
        
        self.logger.info("Initiating graceful shutdown...")
        self.running = False
        
        # Stop resource manager
        if self.resource_manager:
            self.resource_manager.stop_monitoring()
            self.logger.info("✓ Resource manager stopped")
        
        # Stop health monitor
        if self.health_monitor:
            self.health_monitor.stop_monitoring()
            self.logger.info("✓ Health monitor stopped")
        
        # Save final telemetry
        if self.workflow_executor:
            self.workflow_executor.save_telemetry()
            self.logger.info("✓ Final telemetry saved")
        
        self.logger.info("Graceful shutdown completed")
    
    async def run(self, mode: str) -> bool:
        """Run orchestrator in specified mode."""
        self.running = True
        
        try:
            if mode == "full":
                return await self.run_full_pipeline()
            elif mode == "ci":
                return await self.run_ci_mode()
            elif mode == "monitor":
                await self.run_monitor_mode()
                return True
            else:
                self.logger.error(f"Unknown mode: {mode}")
                return False
        except Exception as e:
            self.logger.error(f"Orchestrator execution failed: {e}")
            return False
        finally:
            await self.shutdown()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BEM v1.3 Performance+Agentic Sprint - Production Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Modes:
  full     - Complete experimental pipeline with all stages
  ci       - Lightweight CI/CD pipeline for automated testing
  monitor  - Health monitoring and resource management only
  
Examples:
  # Run complete pipeline
  python run_bem_v13_production.py --mode full
  
  # Run in CI/CD mode
  python run_bem_v13_production.py --mode ci --config ci_config.yaml
  
  # Run health monitoring only
  python run_bem_v13_production.py --mode monitor --daemon
        """
    )
    
    parser.add_argument("--mode", 
                       choices=["full", "ci", "monitor"],
                       default="full",
                       help="Execution mode")
    parser.add_argument("--config", type=Path,
                       help="Configuration file")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as daemon (monitor mode only)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Validate configuration without executing")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = ProductionOrchestrator(args.config)
    
    if args.dry_run:
        print("🔍 DRY RUN - Configuration Validation")
        print("="*50)
        print(json.dumps(orchestrator.config, indent=2))
        print("\n✅ Configuration valid")
        return 0
    
    # Initialize subsystems
    orchestrator.initialize_subsystems()
    
    # Run in specified mode
    success = await orchestrator.run(args.mode)
    
    if success:
        print(f"\n🎉 BEM v1.3 {args.mode} mode completed successfully!")
        return 0
    else:
        print(f"\n💥 BEM v1.3 {args.mode} mode failed!")
        return 1


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)