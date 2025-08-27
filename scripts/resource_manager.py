#!/usr/bin/env python3
"""
BEM v1.3 Resource Manager and Cleanup System

Provides comprehensive resource management including:
- Automatic cleanup of temporary files and logs
- GPU memory management and optimization
- Disk space monitoring and management
- Process monitoring and resource limits
- Automated archival of experiment results
- Recovery from resource exhaustion scenarios
"""

import argparse
import json
import logging
import os
import psutil
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import tempfile
import tarfile
import gzip
import threading
import signal

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class ResourcePolicy:
    """Resource management policy configuration."""
    max_disk_usage_percent: float = 85.0
    max_memory_usage_percent: float = 90.0
    max_gpu_memory_percent: float = 95.0
    log_retention_days: int = 7
    temp_file_retention_hours: int = 24
    experiment_retention_days: int = 30
    cleanup_interval_minutes: int = 30
    archive_threshold_gb: float = 10.0
    emergency_cleanup_threshold: float = 95.0


@dataclass
class CleanupResult:
    """Result of cleanup operation."""
    category: str
    files_removed: int
    bytes_freed: int
    errors: List[str]
    duration_seconds: float


class GPUMemoryManager:
    """GPU memory management and optimization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gpu_available = self.check_gpu_availability()
    
    def check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_gpu_memory_info(self) -> List[Dict[str, Any]]:
        """Get detailed GPU memory information."""
        if not self.gpu_available:
            return []
        
        gpu_info = []
        try:
            result = subprocess.run([
                "nvidia-smi", 
                "--query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        gpu_info.append({
                            'index': int(parts[0]),
                            'memory_used_mb': int(parts[1]),
                            'memory_total_mb': int(parts[2]),
                            'memory_free_mb': int(parts[3]),
                            'utilization_percent': int(parts[4]),
                            'temperature_c': int(parts[5]),
                            'memory_usage_percent': (int(parts[1]) / int(parts[2])) * 100
                        })
            
        except Exception as e:
            self.logger.error(f"Error getting GPU memory info: {e}")
        
        return gpu_info
    
    def clear_gpu_memory(self) -> bool:
        """Clear GPU memory cache."""
        if not self.gpu_available:
            return False
        
        try:
            import torch
            torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("GPU memory cache cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing GPU memory: {e}")
            return False
    
    def kill_gpu_processes(self, exclude_current: bool = True) -> int:
        """Kill GPU processes to free memory."""
        killed_count = 0
        current_pid = os.getpid()
        
        try:
            result = subprocess.run([
                "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(',')
                        pid = int(parts[0].strip())
                        
                        if exclude_current and pid == current_pid:
                            continue
                        
                        try:
                            process = psutil.Process(pid)
                            
                            # Only kill Python processes that look like training jobs
                            if ('python' in process.name().lower() and 
                                any(keyword in ' '.join(process.cmdline()) 
                                    for keyword in ['train', 'bem', 'experiment'])):
                                
                                self.logger.warning(f"Killing GPU process {pid}: {process.name()}")
                                process.terminate()
                                killed_count += 1
                                
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                        
        except Exception as e:
            self.logger.error(f"Error killing GPU processes: {e}")
        
        return killed_count
    
    def optimize_gpu_memory(self) -> Dict[str, Any]:
        """Optimize GPU memory usage."""
        optimization_result = {
            "timestamp": datetime.now().isoformat(),
            "actions_taken": [],
            "memory_freed_mb": 0,
            "gpu_info_before": self.get_gpu_memory_info(),
            "gpu_info_after": []
        }
        
        # Clear cache
        if self.clear_gpu_memory():
            optimization_result["actions_taken"].append("cleared_cache")
        
        # Get updated info
        optimization_result["gpu_info_after"] = self.get_gpu_memory_info()
        
        # Calculate memory freed
        if (optimization_result["gpu_info_before"] and 
            optimization_result["gpu_info_after"]):
            before_used = sum(gpu["memory_used_mb"] for gpu in optimization_result["gpu_info_before"])
            after_used = sum(gpu["memory_used_mb"] for gpu in optimization_result["gpu_info_after"])
            optimization_result["memory_freed_mb"] = before_used - after_used
        
        return optimization_result


class DiskSpaceManager:
    """Disk space monitoring and management."""
    
    def __init__(self, root_path: Path = Path(".")):
        self.root_path = Path(root_path)
        self.logger = logging.getLogger(__name__)
    
    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage statistics."""
        usage = shutil.disk_usage(self.root_path)
        
        return {
            "total_gb": usage.total / (1024**3),
            "used_gb": usage.used / (1024**3), 
            "free_gb": usage.free / (1024**3),
            "usage_percent": (usage.used / usage.total) * 100
        }
    
    def get_directory_sizes(self) -> List[Dict[str, Any]]:
        """Get sizes of major directories."""
        directories = []
        
        for dir_path in ["logs", "analysis", "models", "data", "dist", "paper"]:
            full_path = self.root_path / dir_path
            if full_path.exists() and full_path.is_dir():
                size_bytes = self.get_directory_size(full_path)
                directories.append({
                    "path": str(full_path),
                    "size_gb": size_bytes / (1024**3),
                    "size_mb": size_bytes / (1024**2)
                })
        
        directories.sort(key=lambda x: x["size_gb"], reverse=True)
        return directories
    
    def get_directory_size(self, path: Path) -> int:
        """Get total size of directory."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    file_path = Path(dirpath) / filename
                    try:
                        total_size += file_path.stat().st_size
                    except (OSError, FileNotFoundError):
                        pass
        except (OSError, PermissionError):
            pass
        
        return total_size
    
    def find_large_files(self, min_size_mb: float = 100) -> List[Dict[str, Any]]:
        """Find large files that could be cleaned up."""
        large_files = []
        min_size_bytes = min_size_mb * 1024 * 1024
        
        try:
            for root, dirs, files in os.walk(self.root_path):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        stat = file_path.stat()
                        if stat.st_size > min_size_bytes:
                            large_files.append({
                                "path": str(file_path),
                                "size_mb": stat.st_size / (1024**2),
                                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                "extension": file_path.suffix.lower()
                            })
                    except (OSError, FileNotFoundError):
                        pass
        except Exception as e:
            self.logger.error(f"Error finding large files: {e}")
        
        large_files.sort(key=lambda x: x["size_mb"], reverse=True)
        return large_files[:50]  # Return top 50


class ExperimentArchiver:
    """Archive completed experiments to save space."""
    
    def __init__(self, logs_dir: Path, archive_dir: Path):
        self.logs_dir = Path(logs_dir)
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def find_archivable_experiments(self, retention_days: int = 30) -> List[Path]:
        """Find experiments that can be archived."""
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        archivable = []
        
        if not self.logs_dir.exists():
            return archivable
        
        for exp_dir in self.logs_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Check if experiment is complete (has eval.json or similar)
            completion_markers = ["eval.json", "results.json", "completed.flag"]
            is_complete = any((exp_dir / marker).exists() for marker in completion_markers)
            
            if is_complete and exp_dir.stat().st_mtime < cutoff_time:
                archivable.append(exp_dir)
        
        return archivable
    
    def archive_experiment(self, exp_dir: Path) -> bool:
        """Archive a single experiment directory."""
        try:
            archive_name = f"{exp_dir.name}_{datetime.now().strftime('%Y%m%d')}.tar.gz"
            archive_path = self.archive_dir / archive_name
            
            # Create compressed archive
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(exp_dir, arcname=exp_dir.name)
            
            # Verify archive was created successfully
            if archive_path.exists() and archive_path.stat().st_size > 0:
                # Remove original directory
                shutil.rmtree(exp_dir)
                
                self.logger.info(f"Archived experiment: {exp_dir.name} -> {archive_name}")
                return True
            else:
                self.logger.error(f"Archive creation failed: {archive_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error archiving {exp_dir}: {e}")
            return False
    
    def archive_experiments(self, retention_days: int = 30) -> Dict[str, Any]:
        """Archive multiple experiments."""
        archivable_experiments = self.find_archivable_experiments(retention_days)
        
        archive_result = {
            "timestamp": datetime.now().isoformat(),
            "total_found": len(archivable_experiments),
            "archived": 0,
            "failed": 0,
            "space_freed_gb": 0,
            "errors": []
        }
        
        total_size_before = sum(
            self.get_directory_size(exp_dir) for exp_dir in archivable_experiments
        )
        
        for exp_dir in archivable_experiments:
            if self.archive_experiment(exp_dir):
                archive_result["archived"] += 1
            else:
                archive_result["failed"] += 1
        
        archive_result["space_freed_gb"] = total_size_before / (1024**3)
        
        return archive_result
    
    def get_directory_size(self, path: Path) -> int:
        """Get total size of directory."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    file_path = Path(dirpath) / filename
                    try:
                        total_size += file_path.stat().st_size
                    except (OSError, FileNotFoundError):
                        pass
        except (OSError, PermissionError):
            pass
        
        return total_size


class ResourceManager:
    """Main resource manager coordinating all resource management activities."""
    
    def __init__(self, policy: ResourcePolicy, root_path: Path = Path(".")):
        self.policy = policy
        self.root_path = Path(root_path)
        
        # Initialize components
        self.gpu_manager = GPUMemoryManager()
        self.disk_manager = DiskSpaceManager(root_path)
        self.archiver = ExperimentArchiver(
            logs_dir=root_path / "logs",
            archive_dir=root_path / "archives"
        )
        
        # Setup logging
        self.setup_logging()
        
        # State tracking
        self.cleanup_history = []
        self.running = False
        self.cleanup_thread = None
    
    def setup_logging(self):
        """Setup resource manager logging."""
        log_dir = self.root_path / "logs" / "resource_manager"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"resource_manager_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "disk": self.disk_manager.get_disk_usage(),
            "directories": self.disk_manager.get_directory_sizes(),
            "gpu": self.gpu_manager.get_gpu_memory_info(),
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "used_gb": psutil.virtual_memory().used / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "usage_percent": psutil.virtual_memory().percent
            },
            "cpu": {
                "usage_percent": psutil.cpu_percent(interval=1),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
        }
        
        return status
    
    def cleanup_old_logs(self) -> CleanupResult:
        """Clean up old log files."""
        start_time = time.time()
        files_removed = 0
        bytes_freed = 0
        errors = []
        
        cutoff_time = time.time() - (self.policy.log_retention_days * 24 * 3600)
        
        log_directories = ["logs", "analysis", "dist"]
        log_extensions = [".log", ".out", ".err"]
        
        for log_dir_name in log_directories:
            log_dir = self.root_path / log_dir_name
            if not log_dir.exists():
                continue
            
            try:
                for log_file in log_dir.rglob("*"):
                    if (log_file.is_file() and 
                        log_file.suffix in log_extensions and
                        log_file.stat().st_mtime < cutoff_time):
                        
                        try:
                            file_size = log_file.stat().st_size
                            log_file.unlink()
                            files_removed += 1
                            bytes_freed += file_size
                        except Exception as e:
                            errors.append(f"Failed to remove {log_file}: {e}")
                            
            except Exception as e:
                errors.append(f"Error scanning {log_dir}: {e}")
        
        duration = time.time() - start_time
        
        self.logger.info(f"Cleaned up {files_removed} old log files, freed {bytes_freed / (1024**2):.1f} MB")
        
        return CleanupResult(
            category="old_logs",
            files_removed=files_removed,
            bytes_freed=bytes_freed,
            errors=errors,
            duration_seconds=duration
        )
    
    def cleanup_temp_files(self) -> CleanupResult:
        """Clean up temporary files."""
        start_time = time.time()
        files_removed = 0
        bytes_freed = 0
        errors = []
        
        cutoff_time = time.time() - (self.policy.temp_file_retention_hours * 3600)
        
        temp_patterns = [
            "tmp*", "temp*", "*.tmp", "*.temp", ".cache*", "__pycache__",
            "*.pyc", "*.pyo", "*~", ".DS_Store", "Thumbs.db"
        ]
        
        temp_directories = [
            tempfile.gettempdir(),
            str(self.root_path / ".cache"),
            str(self.root_path / "tmp"),
            str(self.root_path / "__pycache__")
        ]
        
        for temp_dir in temp_directories:
            temp_path = Path(temp_dir)
            if not temp_path.exists():
                continue
            
            try:
                for pattern in temp_patterns:
                    for temp_file in temp_path.rglob(pattern):
                        try:
                            if temp_file.is_file() and temp_file.stat().st_mtime < cutoff_time:
                                file_size = temp_file.stat().st_size
                                temp_file.unlink()
                                files_removed += 1
                                bytes_freed += file_size
                            elif temp_file.is_dir() and not any(temp_file.iterdir()):
                                # Remove empty directories
                                temp_file.rmdir()
                                files_removed += 1
                        except Exception as e:
                            errors.append(f"Failed to remove {temp_file}: {e}")
                            
            except Exception as e:
                errors.append(f"Error scanning {temp_dir}: {e}")
        
        duration = time.time() - start_time
        
        self.logger.info(f"Cleaned up {files_removed} temp files, freed {bytes_freed / (1024**2):.1f} MB")
        
        return CleanupResult(
            category="temp_files",
            files_removed=files_removed,
            bytes_freed=bytes_freed,
            errors=errors,
            duration_seconds=duration
        )
    
    def cleanup_large_files(self) -> CleanupResult:
        """Clean up large files that are no longer needed."""
        start_time = time.time()
        files_removed = 0
        bytes_freed = 0
        errors = []
        
        # Find large files
        large_files = self.disk_manager.find_large_files(min_size_mb=500)  # 500MB+
        
        # Patterns for files that can be safely removed
        removable_patterns = [
            "*.core",  # Core dumps
            "*.dump",  # Memory dumps
            "*.backup.*",  # Backup files older than retention period
            "wandb-*",  # Old wandb run directories
        ]
        
        cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days
        
        for file_info in large_files:
            file_path = Path(file_info["path"])
            
            try:
                # Check if file matches removable patterns and is old enough
                if (any(file_path.match(pattern) for pattern in removable_patterns) and
                    file_path.stat().st_mtime < cutoff_time):
                    
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    files_removed += 1
                    bytes_freed += file_size
                    
            except Exception as e:
                errors.append(f"Failed to remove {file_path}: {e}")
        
        duration = time.time() - start_time
        
        self.logger.info(f"Cleaned up {files_removed} large files, freed {bytes_freed / (1024**3):.1f} GB")
        
        return CleanupResult(
            category="large_files",
            files_removed=files_removed,
            bytes_freed=bytes_freed,
            errors=errors,
            duration_seconds=duration
        )
    
    def emergency_cleanup(self) -> Dict[str, Any]:
        """Perform emergency cleanup when resources are critically low."""
        self.logger.warning("Performing emergency cleanup - resources critically low")
        
        emergency_result = {
            "timestamp": datetime.now().isoformat(),
            "trigger": "emergency",
            "actions": [],
            "total_space_freed_gb": 0,
            "success": False
        }
        
        # 1. Clear GPU memory aggressively
        if self.gpu_manager.gpu_available:
            gpu_result = self.gpu_manager.optimize_gpu_memory()
            emergency_result["actions"].append({
                "type": "gpu_optimization",
                "result": gpu_result
            })
        
        # 2. Archive experiments immediately
        archive_result = self.archiver.archive_experiments(retention_days=1)  # Very aggressive
        emergency_result["actions"].append({
            "type": "emergency_archival",
            "result": archive_result
        })
        emergency_result["total_space_freed_gb"] += archive_result["space_freed_gb"]
        
        # 3. Clean up all temporary files regardless of age
        temp_cleanup = self.cleanup_temp_files()
        emergency_result["actions"].append({
            "type": "temp_cleanup",
            "result": temp_cleanup
        })
        emergency_result["total_space_freed_gb"] += temp_cleanup.bytes_freed / (1024**3)
        
        # 4. Remove large non-essential files
        large_cleanup = self.cleanup_large_files()
        emergency_result["actions"].append({
            "type": "large_file_cleanup", 
            "result": large_cleanup
        })
        emergency_result["total_space_freed_gb"] += large_cleanup.bytes_freed / (1024**3)
        
        # 5. Check if emergency cleanup was successful
        disk_usage = self.disk_manager.get_disk_usage()
        emergency_result["success"] = disk_usage["usage_percent"] < self.policy.emergency_cleanup_threshold
        emergency_result["final_disk_usage_percent"] = disk_usage["usage_percent"]
        
        self.logger.info(f"Emergency cleanup completed: {emergency_result['total_space_freed_gb']:.1f} GB freed")
        
        return emergency_result
    
    def perform_routine_cleanup(self) -> Dict[str, Any]:
        """Perform routine resource cleanup."""
        cleanup_start = time.time()
        
        routine_result = {
            "timestamp": datetime.now().isoformat(),
            "type": "routine",
            "cleanup_results": [],
            "total_files_removed": 0,
            "total_bytes_freed": 0,
            "duration_seconds": 0,
            "errors": []
        }
        
        # Get initial resource status
        initial_status = self.get_resource_status()
        
        # 1. Clean up old logs
        log_cleanup = self.cleanup_old_logs()
        routine_result["cleanup_results"].append(log_cleanup)
        routine_result["total_files_removed"] += log_cleanup.files_removed
        routine_result["total_bytes_freed"] += log_cleanup.bytes_freed
        routine_result["errors"].extend(log_cleanup.errors)
        
        # 2. Clean up temporary files
        temp_cleanup = self.cleanup_temp_files()
        routine_result["cleanup_results"].append(temp_cleanup)
        routine_result["total_files_removed"] += temp_cleanup.files_removed
        routine_result["total_bytes_freed"] += temp_cleanup.bytes_freed
        routine_result["errors"].extend(temp_cleanup.errors)
        
        # 3. Archive old experiments
        if initial_status["disk"]["usage_percent"] > self.policy.max_disk_usage_percent * 0.8:
            archive_result = self.archiver.archive_experiments(self.policy.experiment_retention_days)
            routine_result["cleanup_results"].append({
                "category": "experiment_archival",
                "files_removed": archive_result["archived"],
                "bytes_freed": archive_result["space_freed_gb"] * (1024**3),
                "errors": archive_result.get("errors", []),
                "duration_seconds": 0
            })
            routine_result["total_bytes_freed"] += archive_result["space_freed_gb"] * (1024**3)
        
        # 4. GPU memory optimization
        if self.gpu_manager.gpu_available:
            gpu_optimization = self.gpu_manager.optimize_gpu_memory()
            routine_result["gpu_optimization"] = gpu_optimization
        
        routine_result["duration_seconds"] = time.time() - cleanup_start
        routine_result["total_gb_freed"] = routine_result["total_bytes_freed"] / (1024**3)
        
        # Get final resource status
        routine_result["final_status"] = self.get_resource_status()
        
        self.logger.info(f"Routine cleanup completed: {routine_result['total_files_removed']} files, "
                        f"{routine_result['total_gb_freed']:.1f} GB freed in {routine_result['duration_seconds']:.1f}s")
        
        # Save cleanup history
        self.cleanup_history.append(routine_result)
        if len(self.cleanup_history) > 100:  # Keep last 100 cleanup records
            self.cleanup_history.pop(0)
        
        return routine_result
    
    def monitor_resources(self):
        """Continuous resource monitoring with automatic cleanup."""
        self.logger.info("Starting resource monitoring")
        
        while self.running:
            try:
                status = self.get_resource_status()
                
                # Check for emergency conditions
                emergency_needed = (
                    status["disk"]["usage_percent"] >= self.policy.emergency_cleanup_threshold or
                    status["memory"]["usage_percent"] >= self.policy.emergency_cleanup_threshold
                )
                
                if emergency_needed:
                    self.emergency_cleanup()
                else:
                    # Check for routine cleanup conditions
                    routine_needed = (
                        status["disk"]["usage_percent"] >= self.policy.max_disk_usage_percent or
                        status["memory"]["usage_percent"] >= self.policy.max_memory_usage_percent
                    )
                    
                    if routine_needed:
                        self.perform_routine_cleanup()
                
                # Sleep until next check
                time.sleep(self.policy.cleanup_interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(60)  # Brief pause before retrying
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if self.running:
            self.logger.warning("Resource monitoring already running")
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self.monitor_resources, daemon=True)
        self.cleanup_thread.start()
        
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.running = False
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=10)
        
        self.logger.info("Resource monitoring stopped")
    
    def generate_resource_report(self) -> str:
        """Generate comprehensive resource utilization report."""
        status = self.get_resource_status()
        
        report_lines = [
            "="*80,
            "BEM v1.3 Resource Management Report",
            "="*80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "DISK USAGE:",
            f"  Total: {status['disk']['total_gb']:.1f} GB",
            f"  Used: {status['disk']['used_gb']:.1f} GB ({status['disk']['usage_percent']:.1f}%)",
            f"  Free: {status['disk']['free_gb']:.1f} GB",
            "",
            "MEMORY USAGE:",
            f"  Total: {status['memory']['total_gb']:.1f} GB", 
            f"  Used: {status['memory']['used_gb']:.1f} GB ({status['memory']['usage_percent']:.1f}%)",
            f"  Available: {status['memory']['available_gb']:.1f} GB",
            "",
            "CPU USAGE:",
            f"  Current: {status['cpu']['usage_percent']:.1f}%",
            f"  Load Average: {', '.join(f'{load:.2f}' for load in status['cpu']['load_average'])}",
            ""
        ]
        
        if status["gpu"]:
            report_lines.extend([
                "GPU USAGE:",
                *[f"  GPU {gpu['index']}: {gpu['memory_usage_percent']:.1f}% "
                  f"({gpu['memory_used_mb']} / {gpu['memory_total_mb']} MB), "
                  f"{gpu['utilization_percent']}% util, {gpu['temperature_c']}°C"
                  for gpu in status["gpu"]],
                ""
            ])
        
        report_lines.extend([
            "TOP DIRECTORIES BY SIZE:",
            *[f"  {dir_info['path']}: {dir_info['size_gb']:.1f} GB"
              for dir_info in status["directories"][:10]],
            ""
        ])
        
        if self.cleanup_history:
            last_cleanup = self.cleanup_history[-1]
            report_lines.extend([
                "LAST CLEANUP:",
                f"  Date: {last_cleanup['timestamp']}",
                f"  Files Removed: {last_cleanup['total_files_removed']}",
                f"  Space Freed: {last_cleanup.get('total_gb_freed', 0):.1f} GB",
                f"  Duration: {last_cleanup.get('duration_seconds', 0):.1f} seconds",
                ""
            ])
        
        return "\n".join(report_lines)


def main():
    """Main entry point for resource manager."""
    parser = argparse.ArgumentParser(description="BEM v1.3 Resource Manager")
    parser.add_argument("--action", 
                       choices=["monitor", "cleanup", "status", "report", "emergency"],
                       default="status",
                       help="Action to perform")
    parser.add_argument("--root-path", type=Path, default=Path("."),
                       help="Root path for resource management")
    parser.add_argument("--config", type=Path,
                       help="Configuration file for resource policies")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as daemon")
    
    args = parser.parse_args()
    
    # Load policy (use defaults if no config provided)
    policy = ResourcePolicy()
    if args.config and args.config.exists():
        with open(args.config) as f:
            import yaml
            config_data = yaml.safe_load(f)
            # Update policy with config values
            for key, value in config_data.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
    
    # Create resource manager
    manager = ResourceManager(policy, args.root_path)
    
    if args.action == "status":
        status = manager.get_resource_status()
        print(json.dumps(status, indent=2))
    
    elif args.action == "report":
        report = manager.generate_resource_report()
        print(report)
    
    elif args.action == "cleanup":
        result = manager.perform_routine_cleanup()
        print(json.dumps(result, indent=2))
    
    elif args.action == "emergency":
        result = manager.emergency_cleanup()
        print(json.dumps(result, indent=2))
    
    elif args.action == "monitor":
        if args.daemon:
            # Setup signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                manager.stop_monitoring()
                sys.exit(0)
            
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        
        manager.start_monitoring()
        
        if not args.daemon:
            # Interactive monitoring - print status every minute
            try:
                while manager.running:
                    status = manager.get_resource_status()
                    print(f"{datetime.now().strftime('%H:%M:%S')} - "
                          f"Disk: {status['disk']['usage_percent']:.1f}%, "
                          f"Memory: {status['memory']['usage_percent']:.1f}%, "
                          f"CPU: {status['cpu']['usage_percent']:.1f}%")
                    time.sleep(60)
            except KeyboardInterrupt:
                manager.stop_monitoring()


if __name__ == "__main__":
    main()