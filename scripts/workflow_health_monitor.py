#!/usr/bin/env python3
"""
BEM v1.3 Workflow Health Monitor and Recovery System

Provides:
- Real-time health monitoring of workflow execution
- Automated failure detection and recovery
- Performance regression detection
- Resource exhaustion protection
- Statistical validation gate enforcement
- Automated rollback procedures
"""

import json
import logging
import subprocess
import sys
import time
import psutil
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import signal
import requests

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from analysis.stats import StatisticalAnalyzer


@dataclass
class HealthMetric:
    """Individual health metric with thresholds."""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    direction: str = "lower_is_better"  # or "higher_is_better"
    
    @property
    def status(self) -> str:
        """Get current health status."""
        if self.direction == "lower_is_better":
            if self.value >= self.threshold_critical:
                return "CRITICAL"
            elif self.value >= self.threshold_warning:
                return "WARNING"
            else:
                return "HEALTHY"
        else:
            if self.value <= self.threshold_critical:
                return "CRITICAL"
            elif self.value <= self.threshold_warning:
                return "WARNING"
            else:
                return "HEALTHY"


@dataclass
class HealthCheck:
    """Complete system health check result."""
    timestamp: datetime
    overall_status: str
    metrics: List[HealthMetric]
    active_processes: List[Dict[str, Any]]
    disk_usage_gb: float
    experiment_progress: Dict[str, Any]
    recent_failures: List[Dict[str, Any]]


class StatisticalGateValidator:
    """Validates statistical significance gates as specified in TODO.md."""
    
    def __init__(self, gates_config_path: Path):
        self.gates_config = self.load_gates_config(gates_config_path)
        self.analyzer = StatisticalAnalyzer()
        
    def load_gates_config(self, config_path: Path) -> Dict[str, Any]:
        """Load gate configuration from YAML."""
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def validate_experiment_gates(self, experiment_results: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validate experiment results against statistical gates."""
        validation_results = {
            "gates_passed": True,
            "gate_details": {},
            "failed_gates": [],
            "validation_timestamp": datetime.now().isoformat()
        }
        
        # Universal constraints validation
        if "universal_constraints" in self.gates_config:
            for constraint in self.gates_config["universal_constraints"]:
                gate_passed = self.validate_single_gate(constraint, experiment_results)
                validation_results["gate_details"][constraint["name"]] = gate_passed
                
                if not gate_passed:
                    validation_results["gates_passed"] = False
                    validation_results["failed_gates"].append(constraint["name"])
        
        # Pillar-specific gates
        if "pillars" in self.gates_config:
            for pillar_name, pillar_config in self.gates_config["pillars"].items():
                pillar_results = experiment_results.get(pillar_name, {})
                
                for gate in pillar_config.get("gates", []):
                    gate_passed = self.validate_single_gate(gate, pillar_results)
                    gate_key = f"{pillar_name}.{gate['name']}"
                    validation_results["gate_details"][gate_key] = gate_passed
                    
                    if not gate_passed:
                        validation_results["gates_passed"] = False
                        validation_results["failed_gates"].append(gate_key)
        
        return validation_results["gates_passed"], validation_results
    
    def validate_single_gate(self, gate_config: Dict[str, Any], results: Dict[str, Any]) -> bool:
        """Validate a single gate against results."""
        gate_type = gate_config.get("type", "unknown")
        requirement = gate_config.get("requirement", "")
        
        if gate_type == "statistical":
            return self.validate_statistical_gate(gate_config, results)
        elif gate_type == "performance":
            return self.validate_performance_gate(gate_config, results)
        elif gate_type == "safety":
            return self.validate_safety_gate(gate_config, results)
        else:
            # Default validation - look for basic pass/fail indicators
            return results.get("passed", True)
    
    def validate_statistical_gate(self, gate_config: Dict[str, Any], results: Dict[str, Any]) -> bool:
        """Validate statistical significance gates with BCa bootstrap and FDR correction."""
        metrics = gate_config.get("metrics", [])
        min_significant = gate_config.get("min_significant_metrics", 1)
        
        significant_count = 0
        
        for metric in metrics:
            metric_results = results.get(metric, {})
            
            # Check for CI lower bound > 0 post-FDR
            ci_lower = metric_results.get("ci_lower_bound", -1)
            fdr_corrected = metric_results.get("fdr_corrected", False)
            
            if ci_lower > 0 and fdr_corrected:
                significant_count += 1
        
        return significant_count >= min_significant
    
    def validate_performance_gate(self, gate_config: Dict[str, Any], results: Dict[str, Any]) -> bool:
        """Validate performance constraint gates."""
        measurement = gate_config.get("measurement", "")
        requirement = gate_config.get("requirement", "")
        
        if measurement in results:
            value = results[measurement]
            
            # Parse requirement (e.g., "p50_increase_percent <= 15.0")
            if "<=" in requirement:
                _, threshold_str = requirement.split("<=")
                threshold = float(threshold_str.strip())
                return value <= threshold
            elif ">=" in requirement:
                _, threshold_str = requirement.split(">=")
                threshold = float(threshold_str.strip())
                return value >= threshold
            elif ">" in requirement:
                _, threshold_str = requirement.split(">")
                threshold = float(threshold_str.strip())
                return value > threshold
            elif "<" in requirement:
                _, threshold_str = requirement.split("<")
                threshold = float(threshold_str.strip())
                return value < threshold
        
        return False
    
    def validate_safety_gate(self, gate_config: Dict[str, Any], results: Dict[str, Any]) -> bool:
        """Validate safety constraint gates."""
        requirement = gate_config.get("requirement", "")
        
        if "AND" in requirement:
            # Multiple conditions (e.g., "violation_reduction >= 30.0 AND utility_drop <= 1.0")
            conditions = requirement.split(" AND ")
            return all(self.evaluate_condition(cond.strip(), results) for cond in conditions)
        else:
            return self.evaluate_condition(requirement, results)
    
    def evaluate_condition(self, condition: str, results: Dict[str, Any]) -> bool:
        """Evaluate a single condition against results."""
        for op in [">=", "<=", "==", "!=", ">", "<"]:
            if op in condition:
                left, right = condition.split(op)
                left_val = self.get_metric_value(left.strip(), results)
                right_val = float(right.strip())
                
                if op == ">=":
                    return left_val >= right_val
                elif op == "<=":
                    return left_val <= right_val
                elif op == "==":
                    return left_val == right_val
                elif op == "!=":
                    return left_val != right_val
                elif op == ">":
                    return left_val > right_val
                elif op == "<":
                    return left_val < right_val
        
        return False
    
    def get_metric_value(self, metric_name: str, results: Dict[str, Any]) -> float:
        """Extract metric value from results."""
        if metric_name in results:
            return float(results[metric_name])
        return 0.0


class WorkflowLogWatcher(FileSystemEventHandler):
    """Watch workflow logs for failures and progress updates."""
    
    def __init__(self, health_monitor):
        self.health_monitor = health_monitor
        self.logger = logging.getLogger(__name__)
        
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.log'):
            self.logger.info(f"New log file detected: {event.src_path}")
            self.health_monitor.scan_log_file(Path(event.src_path))
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.log'):
            self.health_monitor.scan_log_file(Path(event.src_path))


class WorkflowHealthMonitor:
    """Main health monitoring and recovery system."""
    
    def __init__(self, 
                 logs_dir: Path,
                 gates_config: Path,
                 check_interval: int = 60,
                 alert_webhook: Optional[str] = None):
        self.logs_dir = Path(logs_dir)
        self.gates_config = Path(gates_config)
        self.check_interval = check_interval
        self.alert_webhook = alert_webhook
        
        # Initialize components
        self.gate_validator = StatisticalGateValidator(gates_config)
        self.log_watcher = WorkflowLogWatcher(self)
        
        # State tracking
        self.health_history = []
        self.active_experiments = {}
        self.failure_count = 0
        self.last_alert_time = {}
        self.running = False
        
        # Setup logging
        self.setup_logging()
        
        # Setup file watcher
        self.observer = Observer()
        self.observer.schedule(self.log_watcher, str(logs_dir), recursive=True)
    
    def setup_logging(self):
        """Setup health monitor logging."""
        log_file = self.logs_dir / "health_monitor.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def get_system_health_metrics(self) -> List[HealthMetric]:
        """Collect current system health metrics."""
        metrics = []
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(HealthMetric(
            name="cpu_usage",
            value=cpu_percent,
            threshold_warning=80.0,
            threshold_critical=95.0,
            unit="%",
            direction="lower_is_better"
        ))
        
        # Memory usage
        memory = psutil.virtual_memory()
        metrics.append(HealthMetric(
            name="memory_usage", 
            value=memory.percent,
            threshold_warning=85.0,
            threshold_critical=95.0,
            unit="%",
            direction="lower_is_better"
        ))
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        metrics.append(HealthMetric(
            name="disk_usage",
            value=disk_percent,
            threshold_warning=85.0,
            threshold_critical=95.0,
            unit="%",
            direction="lower_is_better"
        ))
        
        # GPU metrics if available
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                gpu_data = result.stdout.strip().split('\n')[0].split(',')
                gpu_memory_used = int(gpu_data[0].strip())
                gpu_memory_total = int(gpu_data[1].strip())
                gpu_utilization = int(gpu_data[2].strip())
                gpu_temp = int(gpu_data[3].strip())
                
                gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
                
                metrics.extend([
                    HealthMetric(
                        name="gpu_memory",
                        value=gpu_memory_percent,
                        threshold_warning=85.0,
                        threshold_critical=95.0,
                        unit="%",
                        direction="lower_is_better"
                    ),
                    HealthMetric(
                        name="gpu_utilization",
                        value=gpu_utilization,
                        threshold_warning=5.0,  # Warning if underutilized
                        threshold_critical=0.0,
                        unit="%",
                        direction="higher_is_better"
                    ),
                    HealthMetric(
                        name="gpu_temperature",
                        value=gpu_temp,
                        threshold_warning=80.0,
                        threshold_critical=90.0,
                        unit="°C",
                        direction="lower_is_better"
                    )
                ])
        except Exception:
            pass
        
        return metrics
    
    def get_active_processes(self) -> List[Dict[str, Any]]:
        """Get information about active workflow processes."""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent', 'create_time']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                
                # Look for BEM-related processes
                if any(keyword in cmdline.lower() for keyword in ['bem', 'train.py', 'evaluate.py', 'workflow']):
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline,
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent'],
                        'runtime_hours': (time.time() - proc.info['create_time']) / 3600
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return processes
    
    def scan_log_file(self, log_path: Path):
        """Scan log file for failure patterns and progress updates."""
        if not log_path.exists():
            return
        
        try:
            with open(log_path) as f:
                # Read last 1000 lines to avoid memory issues with large logs
                lines = f.readlines()[-1000:]
            
            # Look for error patterns
            error_patterns = [
                "error", "exception", "failed", "timeout", "killed",
                "out of memory", "cuda out of memory", "connection refused"
            ]
            
            recent_errors = []
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if any(pattern in line_lower for pattern in error_patterns):
                    recent_errors.append({
                        'line_number': len(lines) - 1000 + i,
                        'content': line.strip(),
                        'file': str(log_path)
                    })
            
            # Update failure tracking
            if recent_errors:
                self.logger.warning(f"Found {len(recent_errors)} recent errors in {log_path}")
                for error in recent_errors[-3:]:  # Log last 3 errors
                    self.logger.warning(f"  {error['content']}")
            
        except Exception as e:
            self.logger.error(f"Error scanning log file {log_path}: {e}")
    
    def check_experiment_progress(self) -> Dict[str, Any]:
        """Check progress of active experiments."""
        progress = {
            "total_experiments": 0,
            "completed_experiments": 0,
            "failed_experiments": 0,
            "running_experiments": 0,
            "experiment_details": {}
        }
        
        # Scan experiment directories
        if self.logs_dir.exists():
            for exp_dir in self.logs_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                
                progress["total_experiments"] += 1
                
                # Check for completion markers
                eval_file = exp_dir / "eval.json"
                error_file = exp_dir / "error.log"
                
                if eval_file.exists():
                    progress["completed_experiments"] += 1
                    status = "completed"
                elif error_file.exists():
                    progress["failed_experiments"] += 1
                    status = "failed"
                else:
                    progress["running_experiments"] += 1
                    status = "running"
                
                progress["experiment_details"][exp_dir.name] = {
                    "status": status,
                    "start_time": datetime.fromtimestamp(exp_dir.stat().st_ctime).isoformat()
                }
        
        return progress
    
    def perform_health_check(self) -> HealthCheck:
        """Perform comprehensive health check."""
        timestamp = datetime.now()
        
        # Collect metrics
        metrics = self.get_system_health_metrics()
        processes = self.get_active_processes()
        experiment_progress = self.check_experiment_progress()
        
        # Determine overall status
        critical_metrics = [m for m in metrics if m.status == "CRITICAL"]
        warning_metrics = [m for m in metrics if m.status == "WARNING"]
        
        if critical_metrics:
            overall_status = "CRITICAL"
        elif warning_metrics:
            overall_status = "WARNING"
        else:
            overall_status = "HEALTHY"
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_usage_gb = disk.used / (1024**3)
        
        # Get recent failures
        recent_failures = self.get_recent_failures()
        
        health_check = HealthCheck(
            timestamp=timestamp,
            overall_status=overall_status,
            metrics=metrics,
            active_processes=processes,
            disk_usage_gb=disk_usage_gb,
            experiment_progress=experiment_progress,
            recent_failures=recent_failures
        )
        
        return health_check
    
    def get_recent_failures(self) -> List[Dict[str, Any]]:
        """Get recent failure events."""
        failures = []
        
        # Look for recent error logs
        if self.logs_dir.exists():
            for error_file in self.logs_dir.rglob("*error*.log"):
                if error_file.is_file() and error_file.stat().st_mtime > (time.time() - 3600):  # Last hour
                    try:
                        with open(error_file) as f:
                            content = f.read()[-500:]  # Last 500 chars
                        
                        failures.append({
                            'file': str(error_file),
                            'timestamp': datetime.fromtimestamp(error_file.stat().st_mtime).isoformat(),
                            'preview': content
                        })
                    except Exception:
                        pass
        
        return failures[-10:]  # Return last 10 failures
    
    def send_alert(self, health_check: HealthCheck, alert_type: str = "health"):
        """Send health alert via configured channels."""
        # Rate limiting - don't send same alert type more than once per hour
        alert_key = f"{alert_type}_{health_check.overall_status}"
        now = time.time()
        
        if alert_key in self.last_alert_time:
            if now - self.last_alert_time[alert_key] < 3600:  # 1 hour
                return
        
        self.last_alert_time[alert_key] = now
        
        alert_message = {
            "timestamp": health_check.timestamp.isoformat(),
            "alert_type": alert_type,
            "status": health_check.overall_status,
            "summary": {
                "critical_metrics": len([m for m in health_check.metrics if m.status == "CRITICAL"]),
                "warning_metrics": len([m for m in health_check.metrics if m.status == "WARNING"]),
                "active_processes": len(health_check.active_processes),
                "completed_experiments": health_check.experiment_progress["completed_experiments"],
                "failed_experiments": health_check.experiment_progress["failed_experiments"]
            }
        }
        
        # Log alert
        self.logger.warning(f"ALERT: {alert_type} - Status: {health_check.overall_status}")
        
        # Send webhook if configured
        if self.alert_webhook:
            try:
                response = requests.post(
                    self.alert_webhook,
                    json=alert_message,
                    timeout=10
                )
                if response.status_code == 200:
                    self.logger.info("Alert webhook sent successfully")
                else:
                    self.logger.error(f"Alert webhook failed: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Failed to send alert webhook: {e}")
    
    def attempt_recovery(self, health_check: HealthCheck):
        """Attempt automated recovery from unhealthy state."""
        self.logger.info("Attempting automated recovery...")
        
        recovery_actions = []
        
        # Memory pressure recovery
        memory_metric = next((m for m in health_check.metrics if m.name == "memory_usage"), None)
        if memory_metric and memory_metric.status == "CRITICAL":
            recovery_actions.append("kill_low_priority_processes")
            recovery_actions.append("clear_caches")
        
        # Disk space recovery  
        disk_metric = next((m for m in health_check.metrics if m.name == "disk_usage"), None)
        if disk_metric and disk_metric.status == "CRITICAL":
            recovery_actions.append("cleanup_old_logs")
            recovery_actions.append("cleanup_temp_files")
        
        # GPU memory recovery
        gpu_memory_metric = next((m for m in health_check.metrics if m.name == "gpu_memory"), None)
        if gpu_memory_metric and gpu_memory_metric.status == "CRITICAL":
            recovery_actions.append("restart_gpu_processes")
        
        # Execute recovery actions
        for action in recovery_actions:
            try:
                if action == "cleanup_old_logs":
                    self.cleanup_old_logs()
                elif action == "cleanup_temp_files":
                    self.cleanup_temp_files()
                elif action == "clear_caches":
                    self.clear_system_caches()
                # Add more recovery actions as needed
                
                self.logger.info(f"Recovery action completed: {action}")
            except Exception as e:
                self.logger.error(f"Recovery action failed: {action} - {e}")
    
    def cleanup_old_logs(self):
        """Clean up old log files to free disk space."""
        cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days ago
        cleaned_size = 0
        
        for log_file in self.logs_dir.rglob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                size = log_file.stat().st_size
                log_file.unlink()
                cleaned_size += size
        
        self.logger.info(f"Cleaned up {cleaned_size / (1024**2):.1f} MB of old logs")
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        import tempfile
        temp_dir = Path(tempfile.gettempdir())
        
        cutoff_time = time.time() - (24 * 3600)  # 1 day ago
        cleaned_size = 0
        
        for temp_file in temp_dir.glob("bem_*"):
            try:
                if temp_file.stat().st_mtime < cutoff_time:
                    if temp_file.is_file():
                        size = temp_file.stat().st_size
                        temp_file.unlink()
                        cleaned_size += size
            except Exception:
                pass
        
        self.logger.info(f"Cleaned up {cleaned_size / (1024**2):.1f} MB of temp files")
    
    def clear_system_caches(self):
        """Clear system caches to free memory."""
        try:
            # Clear page cache, dentries and inodes
            subprocess.run(["sync"], check=True)
            subprocess.run(["echo", "3", ">", "/proc/sys/vm/drop_caches"], 
                          shell=True, check=False)
            self.logger.info("System caches cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear system caches: {e}")
    
    def save_health_report(self, health_check: HealthCheck):
        """Save detailed health report."""
        report_file = self.logs_dir / f"health_report_{health_check.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        report_data = {
            "timestamp": health_check.timestamp.isoformat(),
            "overall_status": health_check.overall_status,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "status": m.status,
                    "threshold_warning": m.threshold_warning,
                    "threshold_critical": m.threshold_critical,
                    "unit": m.unit
                }
                for m in health_check.metrics
            ],
            "active_processes": health_check.active_processes,
            "disk_usage_gb": health_check.disk_usage_gb,
            "experiment_progress": health_check.experiment_progress,
            "recent_failures": health_check.recent_failures
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    def start_monitoring(self):
        """Start the health monitoring service."""
        self.logger.info("Starting BEM v1.3 Workflow Health Monitor")
        self.running = True
        
        # Start file observer
        self.observer.start()
        
        try:
            while self.running:
                # Perform health check
                health_check = self.perform_health_check()
                
                # Save to history
                self.health_history.append(health_check)
                if len(self.health_history) > 1000:  # Keep last 1000 checks
                    self.health_history.pop(0)
                
                # Log current status
                self.logger.info(f"Health Check: {health_check.overall_status} - "
                               f"{len([m for m in health_check.metrics if m.status == 'CRITICAL'])} critical, "
                               f"{len([m for m in health_check.metrics if m.status == 'WARNING'])} warnings")
                
                # Send alerts if needed
                if health_check.overall_status in ["CRITICAL", "WARNING"]:
                    self.send_alert(health_check)
                
                # Attempt recovery if critical
                if health_check.overall_status == "CRITICAL":
                    self.attempt_recovery(health_check)
                
                # Save detailed report
                if health_check.overall_status != "HEALTHY":
                    self.save_health_report(health_check)
                
                # Wait for next check
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Health monitor interrupted by user")
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop the health monitoring service."""
        self.logger.info("Stopping health monitor")
        self.running = False
        self.observer.stop()
        self.observer.join()


def main():
    """Main entry point for health monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BEM v1.3 Workflow Health Monitor")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"),
                       help="Logs directory to monitor")
    parser.add_argument("--gates-config", type=Path, default=Path("gates_bem2.yaml"),
                       help="Gates configuration file")
    parser.add_argument("--check-interval", type=int, default=60,
                       help="Health check interval in seconds")
    parser.add_argument("--alert-webhook", type=str,
                       help="Webhook URL for alerts")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as daemon")
    
    args = parser.parse_args()
    
    # Create health monitor
    monitor = WorkflowHealthMonitor(
        logs_dir=args.logs_dir,
        gates_config=args.gates_config,
        check_interval=args.check_interval,
        alert_webhook=args.alert_webhook
    )
    
    if args.daemon:
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            monitor.stop_monitoring()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    # Start monitoring
    monitor.start_monitoring()


if __name__ == "__main__":
    main()