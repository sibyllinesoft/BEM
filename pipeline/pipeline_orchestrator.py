#!/usr/bin/env python3
"""
Pipeline Orchestrator for BEM Validation

Master orchestrator that coordinates the entire BEM validation pipeline
from data generation through statistical validation to paper generation.
Manages parallel execution, resource allocation, and failure recovery.

Classes:
    ResourceManager: Manages computational resources and scheduling
    TaskScheduler: Schedules and tracks pipeline tasks
    PipelineOrchestrator: Main orchestrator coordinating all components
    ValidationPipeline: High-level pipeline interface
    PipelineMonitor: Real-time monitoring and alerting

Usage:
    orchestrator = PipelineOrchestrator(
        config_path="configs/pipeline_config.yaml",
        output_dir="pipeline_runs"
    )
    
    results = orchestrator.run_full_validation(
        models=["bem_model"],
        baselines=["static_lora", "adalora", "moelora"],
        shifts=["domain", "temporal", "adversarial"],
        seeds=[42, 43, 44]
    )
"""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import resource
import signal
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import yaml

# Pipeline component imports
from baseline_evaluators import BaselineOrchestrator
from evaluation_orchestrator import EvaluationOrchestrator
from paper_generator import PaperGenerator
import sys
sys.path.append(str(Path(__file__).parent.parent))
try:
    from generate_fixed_final_paper import create_fixed_final_paper, compile_paper_with_multiple_passes as compile_fixed_paper
    from generate_final_submission_paper import create_final_submission_paper, compile_paper_with_enhanced_error_handling as compile_submission_paper
    from generate_final_polished_paper import create_final_polished_paper, compile_paper as compile_polished_paper
    from generate_improved_paper import create_improved_paper, compile_paper
    ENHANCED_GENERATORS_AVAILABLE = True
except ImportError:
    ENHANCED_GENERATORS_AVAILABLE = False
    try:
        from generate_improved_paper import create_improved_paper, compile_paper
        logger.info("Using improved paper generator")
    except ImportError:
        logger.warning("No enhanced paper generators found, falling back to standard generator")
from promotion_engine import PromotionOrchestrator
from retrieval_ablator import RetrievalAblator
from routing_auditor import RoutingAuditor
from shift_generator import ShiftGeneratorOrchestrator
from spectral_monitor import SpectralMonitor
from statistical_validator import StatisticalValidationOrchestrator
from versioning_system import ArtifactVersioner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    # Resource limits
    max_parallel_processes: int = mp.cpu_count()
    max_memory_gb: int = 32
    max_gpu_memory_gb: int = 12
    timeout_hours: int = 24
    
    # Execution settings
    enable_checkpointing: bool = True
    checkpoint_interval: int = 300  # seconds
    retry_failed_tasks: bool = True
    max_retries: int = 2
    
    # Output settings
    output_dir: Path = Path("pipeline_runs")
    log_level: str = "INFO"
    save_intermediate_results: bool = True
    cleanup_temp_files: bool = True
    
    # Validation settings
    statistical_significance_level: float = 0.05
    effect_size_threshold: float = 0.5
    bootstrap_samples: int = 10000
    enable_fdr_correction: bool = True
    
    # Paper generation
    generate_paper: bool = True
    paper_template: str = "default"
    include_figures: bool = True


@dataclass
class TaskStatus:
    """Status of a pipeline task."""
    task_id: str
    task_type: str
    status: str  # pending, running, completed, failed, cancelled
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    resource_usage: Optional[Dict[str, Any]] = None
    dependencies: List[str] = None
    output_path: Optional[str] = None


@dataclass
class PipelineRun:
    """Represents a complete pipeline execution run."""
    run_id: str
    config: PipelineConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, cancelled
    tasks: Dict[str, TaskStatus] = None
    results_summary: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = {}


class ResourceManager:
    """Manages computational resources and scheduling."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize resource manager.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.current_processes = 0
        self.current_memory_gb = 0.0
        self.current_gpu_memory_gb = 0.0
        self.process_registry: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"ResourceManager initialized with:")
        logger.info(f"  Max processes: {config.max_parallel_processes}")
        logger.info(f"  Max memory: {config.max_memory_gb}GB")
        logger.info(f"  Max GPU memory: {config.max_gpu_memory_gb}GB")
    
    @contextmanager
    def acquire_resources(
        self, 
        task_id: str, 
        memory_gb: float = 1.0, 
        gpu_memory_gb: float = 0.0
    ):
        """Context manager for acquiring computational resources.
        
        Args:
            task_id: Task identifier
            memory_gb: Required memory in GB
            gpu_memory_gb: Required GPU memory in GB
        """
        # Wait for resources to become available
        while not self._can_allocate(memory_gb, gpu_memory_gb):
            time.sleep(1.0)
        
        # Allocate resources
        self._allocate_resources(task_id, memory_gb, gpu_memory_gb)
        
        try:
            yield
        finally:
            # Release resources
            self._release_resources(task_id)
    
    def _can_allocate(self, memory_gb: float, gpu_memory_gb: float) -> bool:
        """Check if resources can be allocated.
        
        Args:
            memory_gb: Required memory in GB
            gpu_memory_gb: Required GPU memory in GB
            
        Returns:
            True if resources are available
        """
        return (
            self.current_processes < self.config.max_parallel_processes and
            (self.current_memory_gb + memory_gb) <= self.config.max_memory_gb and
            (self.current_gpu_memory_gb + gpu_memory_gb) <= self.config.max_gpu_memory_gb
        )
    
    def _allocate_resources(self, task_id: str, memory_gb: float, gpu_memory_gb: float):
        """Allocate resources to a task.
        
        Args:
            task_id: Task identifier
            memory_gb: Memory to allocate in GB
            gpu_memory_gb: GPU memory to allocate in GB
        """
        self.current_processes += 1
        self.current_memory_gb += memory_gb
        self.current_gpu_memory_gb += gpu_memory_gb
        
        self.process_registry[task_id] = {
            'memory_gb': memory_gb,
            'gpu_memory_gb': gpu_memory_gb,
            'start_time': datetime.now()
        }
        
        logger.debug(f"Allocated resources to {task_id}: {memory_gb}GB RAM, {gpu_memory_gb}GB GPU")
    
    def _release_resources(self, task_id: str):
        """Release resources from a task.
        
        Args:
            task_id: Task identifier
        """
        if task_id in self.process_registry:
            allocation = self.process_registry[task_id]
            
            self.current_processes -= 1
            self.current_memory_gb -= allocation['memory_gb']
            self.current_gpu_memory_gb -= allocation['gpu_memory_gb']
            
            del self.process_registry[task_id]
            
            logger.debug(f"Released resources from {task_id}")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource utilization status.
        
        Returns:
            Resource status dictionary
        """
        return {
            'current_processes': self.current_processes,
            'max_processes': self.config.max_parallel_processes,
            'current_memory_gb': self.current_memory_gb,
            'max_memory_gb': self.config.max_memory_gb,
            'current_gpu_memory_gb': self.current_gpu_memory_gb,
            'max_gpu_memory_gb': self.config.max_gpu_memory_gb,
            'process_utilization': self.current_processes / self.config.max_parallel_processes,
            'memory_utilization': self.current_memory_gb / self.config.max_memory_gb,
            'gpu_utilization': self.current_gpu_memory_gb / max(1, self.config.max_gpu_memory_gb)
        }


class TaskScheduler:
    """Schedules and tracks pipeline tasks with dependency management."""
    
    def __init__(self, resource_manager: ResourceManager):
        """Initialize task scheduler.
        
        Args:
            resource_manager: Resource manager instance
        """
        self.resource_manager = resource_manager
        self.tasks: Dict[str, TaskStatus] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_dependencies: Dict[str, Set[str]] = {}
        
    def add_task(
        self,
        task_id: str,
        task_type: str,
        dependencies: Optional[List[str]] = None
    ) -> TaskStatus:
        """Add a task to the scheduler.
        
        Args:
            task_id: Unique task identifier
            task_type: Type of task
            dependencies: List of task IDs this task depends on
            
        Returns:
            Created task status
        """
        if task_id in self.tasks:
            raise ValueError(f"Task {task_id} already exists")
        
        dependencies = dependencies or []
        
        # Create task status
        task_status = TaskStatus(
            task_id=task_id,
            task_type=task_type,
            status="pending",
            dependencies=dependencies
        )
        self.tasks[task_id] = task_status
        
        # Update dependency graphs
        self.dependency_graph[task_id] = set(dependencies)
        for dep_id in dependencies:
            if dep_id not in self.reverse_dependencies:
                self.reverse_dependencies[dep_id] = set()
            self.reverse_dependencies[dep_id].add(task_id)
        
        logger.info(f"Added task {task_id} ({task_type}) with {len(dependencies)} dependencies")
        return task_status
    
    def get_ready_tasks(self) -> List[str]:
        """Get tasks that are ready to run (all dependencies completed).
        
        Returns:
            List of task IDs ready to run
        """
        ready_tasks = []
        
        for task_id, task_status in self.tasks.items():
            if task_status.status != "pending":
                continue
            
            # Check if all dependencies are completed
            dependencies = self.dependency_graph[task_id]
            if all(self.tasks[dep_id].status == "completed" for dep_id in dependencies):
                ready_tasks.append(task_id)
        
        return ready_tasks
    
    def update_task_status(
        self,
        task_id: str,
        status: str,
        progress: Optional[float] = None,
        error_message: Optional[str] = None,
        output_path: Optional[str] = None
    ):
        """Update task status.
        
        Args:
            task_id: Task identifier
            status: New status
            progress: Task progress (0.0 to 1.0)
            error_message: Error message if failed
            output_path: Path to task output
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        old_status = task.status
        task.status = status
        
        if progress is not None:
            task.progress = progress
        
        if error_message is not None:
            task.error_message = error_message
        
        if output_path is not None:
            task.output_path = output_path
        
        # Update timestamps
        if status == "running" and task.start_time is None:
            task.start_time = datetime.now()
        elif status in ["completed", "failed", "cancelled"]:
            task.end_time = datetime.now()
        
        if old_status != status:
            logger.info(f"Task {task_id} status: {old_status} → {status}")
    
    def get_pipeline_progress(self) -> Dict[str, Any]:
        """Get overall pipeline progress.
        
        Returns:
            Pipeline progress information
        """
        total_tasks = len(self.tasks)
        if total_tasks == 0:
            return {'progress': 0.0, 'tasks_by_status': {}}
        
        tasks_by_status = {}
        total_progress = 0.0
        
        for task in self.tasks.values():
            status = task.status
            if status not in tasks_by_status:
                tasks_by_status[status] = 0
            tasks_by_status[status] += 1
            
            # Add task progress to total
            if status == "completed":
                total_progress += 1.0
            elif status == "running":
                total_progress += task.progress
        
        return {
            'progress': total_progress / total_tasks,
            'total_tasks': total_tasks,
            'tasks_by_status': tasks_by_status,
            'completed_tasks': tasks_by_status.get('completed', 0),
            'running_tasks': tasks_by_status.get('running', 0),
            'failed_tasks': tasks_by_status.get('failed', 0)
        }


class PipelineOrchestrator:
    """Main orchestrator coordinating all pipeline components."""
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[PipelineConfig] = None,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize pipeline orchestrator.
        
        Args:
            config_path: Path to configuration file
            config: Pipeline configuration object
            output_dir: Output directory override
        """
        # Load configuration
        if config_path is not None:
            self.config = self._load_config(config_path)
        elif config is not None:
            self.config = config
        else:
            self.config = PipelineConfig()
        
        # Override output directory if specified
        if output_dir is not None:
            self.config.output_dir = Path(output_dir)
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.resource_manager = ResourceManager(self.config)
        self.task_scheduler = TaskScheduler(self.resource_manager)
        self.artifact_versioner = ArtifactVersioner(
            registry_dir=self.config.output_dir / "artifacts",
            storage_dir=self.config.output_dir / "artifact_storage"
        )
        
        # Pipeline components (initialized lazily)
        self._shift_generator = None
        self._baseline_orchestrator = None
        self._evaluation_orchestrator = None
        self._statistical_validator = None
        self._promotion_orchestrator = None
        self._paper_generator = None
        
        # Current run tracking
        self.current_run: Optional[PipelineRun] = None
        
        logger.info(f"Initialized PipelineOrchestrator")
        logger.info(f"  Output directory: {self.config.output_dir}")
        logger.info(f"  Max parallel processes: {self.config.max_parallel_processes}")
    
    def _load_config(self, config_path: Union[str, Path]) -> PipelineConfig:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Pipeline configuration
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return PipelineConfig()
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Convert nested dictionaries to appropriate types
            if 'output_dir' in config_data:
                config_data['output_dir'] = Path(config_data['output_dir'])
            
            return PipelineConfig(**config_data)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return PipelineConfig()
    
    @property
    def shift_generator(self):
        """Lazy initialization of shift generator."""
        if self._shift_generator is None:
            self._shift_generator = ShiftGeneratorOrchestrator()
        return self._shift_generator
    
    @property
    def baseline_orchestrator(self):
        """Lazy initialization of baseline orchestrator."""
        if self._baseline_orchestrator is None:
            self._baseline_orchestrator = BaselineOrchestrator()
        return self._baseline_orchestrator
    
    @property
    def evaluation_orchestrator(self):
        """Lazy initialization of evaluation orchestrator."""
        if self._evaluation_orchestrator is None:
            self._evaluation_orchestrator = EvaluationOrchestrator(
                output_dir=self.config.output_dir / "evaluations"
            )
        return self._evaluation_orchestrator
    
    @property
    def statistical_validator(self):
        """Lazy initialization of statistical validator."""
        if self._statistical_validator is None:
            self._statistical_validator = StatisticalValidationOrchestrator(
                bootstrap_samples=self.config.bootstrap_samples,
                significance_level=self.config.statistical_significance_level
            )
        return self._statistical_validator
    
    @property
    def promotion_orchestrator(self):
        """Lazy initialization of promotion orchestrator."""
        if self._promotion_orchestrator is None:
            self._promotion_orchestrator = PromotionOrchestrator()
        return self._promotion_orchestrator
    
    @property
    def paper_generator(self):
        """Lazy initialization of paper generator."""
        if self._paper_generator is None:
            self._paper_generator = PaperGenerator(
                output_dir=self.config.output_dir / "papers",
                figures_dir=self.config.output_dir / "figures"
            )
        return self._paper_generator
    
    def run_full_validation(
        self,
        models: List[str],
        baselines: List[str],
        shifts: List[str],
        seeds: List[int],
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run complete BEM validation pipeline.
        
        Args:
            models: List of model identifiers
            baselines: List of baseline model types
            shifts: List of distribution shift types
            seeds: List of random seeds
            run_id: Optional run identifier
            
        Returns:
            Pipeline execution results
        """
        if run_id is None:
            run_id = f"bem_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting full validation pipeline: {run_id}")
        
        # Initialize pipeline run
        self.current_run = PipelineRun(
            run_id=run_id,
            config=self.config,
            start_time=datetime.now()
        )
        
        try:
            # Create run directory
            run_dir = self.config.output_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Save run configuration
            self._save_run_config(run_dir)
            
            # Schedule all pipeline tasks
            self._schedule_pipeline_tasks(models, baselines, shifts, seeds, run_dir)
            
            # Execute pipeline
            results = self._execute_pipeline(run_dir)
            
            # Update run status
            self.current_run.status = "completed"
            self.current_run.end_time = datetime.now()
            self.current_run.results_summary = results
            
            # Save final results
            self._save_run_results(run_dir, results)
            
            logger.info(f"Pipeline completed successfully: {run_id}")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            
            # Update run status
            self.current_run.status = "failed"
            self.current_run.end_time = datetime.now()
            self.current_run.error_message = str(e)
            
            # Save error information
            self._save_run_results(run_dir, {'error': str(e), 'traceback': traceback.format_exc()})
            
            raise
    
    def _schedule_pipeline_tasks(
        self,
        models: List[str],
        baselines: List[str], 
        shifts: List[str],
        seeds: List[int],
        run_dir: Path
    ):
        """Schedule all tasks in the validation pipeline.
        
        Args:
            models: Model identifiers
            baselines: Baseline types
            shifts: Shift types
            seeds: Random seeds
            run_dir: Run directory
        """
        logger.info("Scheduling pipeline tasks...")
        
        # Task 1: Generate distribution shifts
        shift_task_id = "generate_shifts"
        self.task_scheduler.add_task(shift_task_id, "shift_generation")
        self.current_run.tasks[shift_task_id] = self.task_scheduler.tasks[shift_task_id]
        
        # Task 2: Prepare baseline evaluators
        baseline_task_id = "prepare_baselines"
        self.task_scheduler.add_task(baseline_task_id, "baseline_preparation")
        self.current_run.tasks[baseline_task_id] = self.task_scheduler.tasks[baseline_task_id]
        
        # Tasks 3-N: Evaluation tasks (model × baseline × shift × seed combinations)
        evaluation_tasks = []
        for model in models:
            for baseline in baselines:
                for shift in shifts:
                    for seed in seeds:
                        task_id = f"eval_{model}_{baseline}_{shift}_{seed}"
                        self.task_scheduler.add_task(
                            task_id,
                            "evaluation",
                            dependencies=[shift_task_id, baseline_task_id]
                        )
                        evaluation_tasks.append(task_id)
                        self.current_run.tasks[task_id] = self.task_scheduler.tasks[task_id]
        
        # Task N+1: Statistical validation
        validation_task_id = "statistical_validation"
        self.task_scheduler.add_task(
            validation_task_id,
            "statistical_validation",
            dependencies=evaluation_tasks
        )
        self.current_run.tasks[validation_task_id] = self.task_scheduler.tasks[validation_task_id]
        
        # Task N+2: Claim promotion
        promotion_task_id = "claim_promotion"
        self.task_scheduler.add_task(
            promotion_task_id,
            "claim_promotion", 
            dependencies=[validation_task_id]
        )
        self.current_run.tasks[promotion_task_id] = self.task_scheduler.tasks[promotion_task_id]
        
        # Task N+3: Paper generation (if enabled)
        if self.config.generate_paper:
            paper_task_id = "paper_generation"
            self.task_scheduler.add_task(
                paper_task_id,
                "paper_generation",
                dependencies=[promotion_task_id]
            )
            self.current_run.tasks[paper_task_id] = self.task_scheduler.tasks[paper_task_id]
        
        logger.info(f"Scheduled {len(self.current_run.tasks)} pipeline tasks")
    
    def _execute_pipeline(self, run_dir: Path) -> Dict[str, Any]:
        """Execute the scheduled pipeline tasks.
        
        Args:
            run_dir: Run directory
            
        Returns:
            Pipeline execution results
        """
        logger.info("Starting pipeline execution...")
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Use ThreadPoolExecutor for task coordination
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_processes) as executor:
            active_futures = {}
            completed_tasks = set()
            
            # Main execution loop
            start_time = time.time()
            while len(completed_tasks) < len(self.current_run.tasks):
                # Check for timeout
                if time.time() - start_time > self.config.timeout_hours * 3600:
                    logger.error("Pipeline execution timed out")
                    raise TimeoutError("Pipeline execution exceeded timeout")
                
                # Get ready tasks
                ready_tasks = [
                    task_id for task_id in self.task_scheduler.get_ready_tasks()
                    if task_id not in active_futures and task_id not in completed_tasks
                ]
                
                # Submit ready tasks
                for task_id in ready_tasks:
                    future = executor.submit(self._execute_task, task_id, run_dir)
                    active_futures[future] = task_id
                    self.task_scheduler.update_task_status(task_id, "running")
                
                # Check completed tasks
                for future in as_completed(active_futures.keys(), timeout=1.0):
                    task_id = active_futures[future]
                    
                    try:
                        result = future.result()
                        self.task_scheduler.update_task_status(
                            task_id, "completed", progress=1.0, output_path=result.get('output_path')
                        )
                        completed_tasks.add(task_id)
                        logger.info(f"Task completed: {task_id}")
                        
                    except Exception as e:
                        self.task_scheduler.update_task_status(
                            task_id, "failed", error_message=str(e)
                        )
                        completed_tasks.add(task_id)  # Mark as processed to avoid infinite loop
                        logger.error(f"Task failed: {task_id} - {e}")
                        
                        # Handle failure based on configuration
                        if not self.config.retry_failed_tasks:
                            raise
                    
                    # Remove from active futures
                    del active_futures[future]
                
                # Brief pause to avoid busy waiting
                time.sleep(0.1)
        
        # Collect final results
        results = self._collect_pipeline_results(run_dir)
        
        logger.info("Pipeline execution completed")
        return results
    
    def _execute_task(self, task_id: str, run_dir: Path) -> Dict[str, Any]:
        """Execute a single pipeline task.
        
        Args:
            task_id: Task identifier
            run_dir: Run directory
            
        Returns:
            Task execution results
        """
        task = self.task_scheduler.tasks[task_id]
        logger.info(f"Executing task: {task_id} ({task.task_type})")
        
        # Determine resource requirements based on task type
        memory_gb, gpu_memory_gb = self._get_task_resource_requirements(task.task_type)
        
        # Acquire resources and execute
        with self.resource_manager.acquire_resources(task_id, memory_gb, gpu_memory_gb):
            try:
                if task.task_type == "shift_generation":
                    return self._execute_shift_generation(task_id, run_dir)
                elif task.task_type == "baseline_preparation":
                    return self._execute_baseline_preparation(task_id, run_dir)
                elif task.task_type == "evaluation":
                    return self._execute_evaluation(task_id, run_dir)
                elif task.task_type == "statistical_validation":
                    return self._execute_statistical_validation(task_id, run_dir)
                elif task.task_type == "claim_promotion":
                    return self._execute_claim_promotion(task_id, run_dir)
                elif task.task_type == "paper_generation":
                    return self._execute_paper_generation(task_id, run_dir)
                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")
                    
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                raise
    
    def _get_task_resource_requirements(self, task_type: str) -> Tuple[float, float]:
        """Get resource requirements for task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            Tuple of (memory_gb, gpu_memory_gb)
        """
        requirements = {
            "shift_generation": (2.0, 0.0),
            "baseline_preparation": (4.0, 2.0),
            "evaluation": (8.0, 8.0),
            "statistical_validation": (16.0, 0.0),
            "claim_promotion": (1.0, 0.0),
            "paper_generation": (2.0, 0.0)
        }
        return requirements.get(task_type, (1.0, 0.0))
    
    def _execute_shift_generation(self, task_id: str, run_dir: Path) -> Dict[str, Any]:
        """Execute shift generation task."""
        output_dir = run_dir / "shifts"
        output_dir.mkdir(exist_ok=True)
        
        # Generate shifts using shift generator
        shift_results = self.shift_generator.generate_comprehensive_shifts(
            output_dir=str(output_dir),
            shift_types=["domain", "temporal", "adversarial"],
            num_samples_per_shift=1000
        )
        
        # Version the generated shifts
        artifact_id = self.artifact_versioner.version_artifact(
            artifact_path=output_dir,
            artifact_type="distribution_shifts",
            description="Generated distribution shifts for BEM validation",
            tags=["shifts", "validation_data"]
        )
        
        return {
            'output_path': str(output_dir),
            'artifact_id': artifact_id,
            'shift_results': shift_results
        }
    
    def _execute_baseline_preparation(self, task_id: str, run_dir: Path) -> Dict[str, Any]:
        """Execute baseline preparation task."""
        output_dir = run_dir / "baselines"
        output_dir.mkdir(exist_ok=True)
        
        # Prepare baseline evaluators
        baseline_results = self.baseline_orchestrator.prepare_all_baselines(
            output_dir=str(output_dir)
        )
        
        return {
            'output_path': str(output_dir),
            'baseline_results': baseline_results
        }
    
    def _execute_evaluation(self, task_id: str, run_dir: Path) -> Dict[str, Any]:
        """Execute evaluation task."""
        # Parse evaluation parameters from task ID
        parts = task_id.split('_')  # eval_{model}_{baseline}_{shift}_{seed}
        model, baseline, shift, seed = parts[1], parts[2], parts[3], parts[4]
        
        output_dir = run_dir / "evaluations" / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run evaluation
        eval_results = self.evaluation_orchestrator.run_evaluation(
            model_id=model,
            baseline_type=baseline,
            shift_type=shift,
            seed=int(seed),
            output_dir=str(output_dir)
        )
        
        return {
            'output_path': str(output_dir),
            'evaluation_results': eval_results
        }
    
    def _execute_statistical_validation(self, task_id: str, run_dir: Path) -> Dict[str, Any]:
        """Execute statistical validation task."""
        output_dir = run_dir / "statistical_validation"
        output_dir.mkdir(exist_ok=True)
        
        # Collect all evaluation results
        eval_results_dir = run_dir / "evaluations"
        
        # Run statistical validation
        validation_results = self.statistical_validator.validate_all_claims(
            results_directory=str(eval_results_dir),
            output_path=str(output_dir / "validation_results.json")
        )
        
        # Version validation results
        artifact_id = self.artifact_versioner.version_artifact(
            artifact_path=output_dir / "validation_results.json",
            artifact_type="statistical_validation",
            description="Statistical validation results for BEM claims",
            tags=["validation", "statistics"]
        )
        
        return {
            'output_path': str(output_dir),
            'artifact_id': artifact_id,
            'validation_results': validation_results
        }
    
    def _execute_claim_promotion(self, task_id: str, run_dir: Path) -> Dict[str, Any]:
        """Execute claim promotion task."""
        output_dir = run_dir / "promotion"
        output_dir.mkdir(exist_ok=True)
        
        # Load validation results
        validation_results_path = run_dir / "statistical_validation" / "validation_results.json"
        
        # Run claim promotion
        promotion_results = self.promotion_orchestrator.process_validation_results(
            validation_results_path=str(validation_results_path),
            output_path=str(output_dir / "promotion_results.json")
        )
        
        # Version promotion results  
        artifact_id = self.artifact_versioner.version_artifact(
            artifact_path=output_dir / "promotion_results.json",
            artifact_type="claim_promotion",
            description="Claim promotion results from statistical validation",
            tags=["promotion", "claims"]
        )
        
        return {
            'output_path': str(output_dir),
            'artifact_id': artifact_id,
            'promotion_results': promotion_results
        }
    
    def _execute_paper_generation(self, task_id: str, run_dir: Path) -> Dict[str, Any]:
        """Execute paper generation task using the newest fixed generator with fallback priority."""
        output_dir = run_dir / "paper"
        output_dir.mkdir(exist_ok=True)
        
        # Load promotion results
        promotion_results_path = run_dir / "promotion" / "promotion_results.json"
        paper_path = None
        generation_method = "unknown"
        
        if ENHANCED_GENERATORS_AVAILABLE:
            # Priority 1: Fixed final paper generator (newest, with all fixes)
            try:
                logger.info("🎯 Generating paper with FIXED generator (table references + BEM terminology fixes)")
                paper_dir = create_fixed_final_paper()
                paper_path = compile_fixed_paper(paper_dir)
                
                if paper_path:
                    logger.info(f"✅ Successfully generated FIXED paper: {paper_path}")
                    import shutil
                    run_paper_path = output_dir / "fixed_main.pdf"
                    shutil.copy2(paper_path, run_paper_path)
                    paper_path = str(run_paper_path)
                    generation_method = "fixed_final_paper"
                else:
                    raise Exception("Fixed paper generation failed")
                    
            except Exception as e:
                logger.warning(f"Fixed paper generator failed: {e}. Trying fallback #1...")
                
                # Priority 2: Final submission paper generator (fallback)
                try:
                    logger.info("📄 Falling back to final submission paper generator")
                    paper_dir = create_final_submission_paper()
                    paper_path = compile_submission_paper(paper_dir)
                    
                    if paper_path:
                        logger.info(f"✅ Successfully generated submission paper: {paper_path}")
                        import shutil
                        run_paper_path = output_dir / "submission_main.pdf"
                        shutil.copy2(paper_path, run_paper_path)
                        paper_path = str(run_paper_path)
                        generation_method = "final_submission_paper"
                    else:
                        raise Exception("Submission paper generation failed")
                        
                except Exception as e2:
                    logger.warning(f"Submission paper generator failed: {e2}. Trying fallback #2...")
                    
                    # Priority 3: Final polished paper generator (fallback)
                    try:
                        logger.info("📝 Falling back to final polished paper generator")
                        paper_dir = create_final_polished_paper()
                        paper_path = compile_polished_paper(paper_dir)
                        
                        if paper_path:
                            logger.info(f"✅ Successfully generated polished paper: {paper_path}")
                            import shutil
                            run_paper_path = output_dir / "polished_main.pdf"
                            shutil.copy2(paper_path, run_paper_path)
                            paper_path = str(run_paper_path)
                            generation_method = "final_polished_paper"
                        else:
                            raise Exception("Polished paper generation failed")
                            
                    except Exception as e3:
                        logger.warning(f"Polished paper generator failed: {e3}. Trying fallback #3...")
                        
                        # Priority 4: Improved paper generator (fallback)
                        try:
                            logger.info("📊 Falling back to improved paper generator") 
                            paper_dir = create_improved_paper()
                            paper_path = compile_paper(paper_dir)
                            
                            if paper_path:
                                logger.info(f"✅ Successfully generated improved paper: {paper_path}")
                                import shutil
                                run_paper_path = output_dir / "improved_main.pdf"
                                shutil.copy2(paper_path, run_paper_path)
                                paper_path = str(run_paper_path)
                                generation_method = "improved_paper"
                            else:
                                raise Exception("Improved paper generation failed")
                                
                        except Exception as e4:
                            logger.error(f"All enhanced generators failed. Falling back to standard generator.")
                            generation_method = "standard_fallback"
        
        # Final fallback to standard paper generator
        if not paper_path:
            logger.info("🔧 Using standard paper generator as final fallback")
            paper_path = self.paper_generator.generate_paper(
                promotion_results=str(promotion_results_path),
                metadata={
                    'title': 'BEM Validation Results',
                    'authors': ['BEM Research Team'],
                    'institution': 'Research Institution'
                }
            )
            generation_method = "standard_generator"
        
        # Version the paper
        artifact_id = self.artifact_versioner.version_artifact(
            artifact_path=paper_path,
            artifact_type="research_paper",
            description=f"Generated research paper from BEM validation (method: {generation_method})",
            tags=["paper", "publication", generation_method, "bem_validation"]
        )
        
        return {
            'output_path': paper_path,
            'artifact_id': artifact_id,
            'generation_method': generation_method
        }
    
    def _collect_pipeline_results(self, run_dir: Path) -> Dict[str, Any]:
        """Collect final pipeline results.
        
        Args:
            run_dir: Run directory
            
        Returns:
            Aggregated pipeline results
        """
        progress = self.task_scheduler.get_pipeline_progress()
        resource_status = self.resource_manager.get_resource_status()
        
        # Collect task results
        task_results = {}
        for task_id, task_status in self.current_run.tasks.items():
            task_results[task_id] = {
                'status': task_status.status,
                'progress': task_status.progress,
                'start_time': task_status.start_time.isoformat() if task_status.start_time else None,
                'end_time': task_status.end_time.isoformat() if task_status.end_time else None,
                'output_path': task_status.output_path,
                'error_message': task_status.error_message
            }
        
        results = {
            'run_id': self.current_run.run_id,
            'start_time': self.current_run.start_time.isoformat(),
            'end_time': self.current_run.end_time.isoformat() if self.current_run.end_time else None,
            'status': self.current_run.status,
            'pipeline_progress': progress,
            'resource_utilization': resource_status,
            'task_results': task_results
        }
        
        # Add promotion summary if available
        promotion_results_path = run_dir / "promotion" / "promotion_results.json"
        if promotion_results_path.exists():
            with open(promotion_results_path, 'r') as f:
                promotion_data = json.load(f)
                results['promotion_summary'] = promotion_data.get('summary', {})
        
        return results
    
    def _save_run_config(self, run_dir: Path):
        """Save run configuration to disk."""
        config_path = run_dir / "pipeline_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
    
    def _save_run_results(self, run_dir: Path, results: Dict[str, Any]):
        """Save run results to disk."""
        results_path = run_dir / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, attempting graceful shutdown...")
            if self.current_run:
                self.current_run.status = "cancelled"
                self.current_run.end_time = datetime.now()
            # Additional cleanup could be added here
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def get_run_status(self) -> Optional[Dict[str, Any]]:
        """Get current run status.
        
        Returns:
            Current run status or None if no active run
        """
        if self.current_run is None:
            return None
        
        progress = self.task_scheduler.get_pipeline_progress()
        resource_status = self.resource_manager.get_resource_status()
        
        return {
            'run_id': self.current_run.run_id,
            'status': self.current_run.status,
            'start_time': self.current_run.start_time.isoformat(),
            'pipeline_progress': progress,
            'resource_utilization': resource_status
        }


class ValidationPipeline:
    """High-level interface for BEM validation pipeline."""
    
    def __init__(self, config_path: Optional[str] = None, output_dir: Optional[str] = None):
        """Initialize validation pipeline.
        
        Args:
            config_path: Path to configuration file
            output_dir: Output directory
        """
        self.orchestrator = PipelineOrchestrator(
            config_path=config_path,
            output_dir=output_dir
        )
    
    def validate_bem_claims(
        self,
        model_path: str,
        baseline_types: Optional[List[str]] = None,
        shift_types: Optional[List[str]] = None,
        num_seeds: int = 3,
        generate_paper: bool = True
    ) -> Dict[str, Any]:
        """Validate BEM performance claims.
        
        Args:
            model_path: Path to BEM model
            baseline_types: List of baseline types to compare against
            shift_types: List of distribution shift types
            num_seeds: Number of random seeds for evaluation
            generate_paper: Whether to generate research paper
            
        Returns:
            Validation results
        """
        if baseline_types is None:
            baseline_types = ["static_lora", "adalora", "moelora", "qlora"]
        
        if shift_types is None:
            shift_types = ["domain", "temporal", "adversarial"]
        
        seeds = list(range(42, 42 + num_seeds))
        
        # Update configuration
        self.orchestrator.config.generate_paper = generate_paper
        
        # Run pipeline
        return self.orchestrator.run_full_validation(
            models=[model_path],
            baselines=baseline_types,
            shifts=shift_types,
            seeds=seeds
        )
    
    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get current validation status."""
        return self.orchestrator.get_run_status()


class PipelineMonitor:
    """Real-time monitoring and alerting for pipeline execution."""
    
    def __init__(self, orchestrator: PipelineOrchestrator, update_interval: int = 30):
        """Initialize pipeline monitor.
        
        Args:
            orchestrator: Pipeline orchestrator to monitor
            update_interval: Update interval in seconds
        """
        self.orchestrator = orchestrator
        self.update_interval = update_interval
        self.monitoring = False
        
    async def start_monitoring(self):
        """Start real-time monitoring."""
        self.monitoring = True
        logger.info("Started pipeline monitoring")
        
        while self.monitoring:
            status = self.orchestrator.get_run_status()
            if status:
                self._log_status_update(status)
                self._check_alerts(status)
            
            await asyncio.sleep(self.update_interval)
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False
        logger.info("Stopped pipeline monitoring")
    
    def _log_status_update(self, status: Dict[str, Any]):
        """Log status update."""
        progress = status['pipeline_progress']
        resource_util = status['resource_utilization']
        
        logger.info(f"Pipeline Progress: {progress['progress']:.1%}")
        logger.info(f"Completed Tasks: {progress['completed_tasks']}/{progress['total_tasks']}")
        logger.info(f"Resource Utilization: CPU {resource_util['process_utilization']:.1%}, "
                   f"Memory {resource_util['memory_utilization']:.1%}")
    
    def _check_alerts(self, status: Dict[str, Any]):
        """Check for alert conditions."""
        resource_util = status['resource_utilization']
        progress = status['pipeline_progress']
        
        # High resource utilization alert
        if resource_util['memory_utilization'] > 0.9:
            logger.warning("High memory utilization detected!")
        
        # Failed tasks alert
        if progress['failed_tasks'] > 0:
            logger.warning(f"Failed tasks detected: {progress['failed_tasks']}")


# Example usage and testing
if __name__ == "__main__":
    # Create sample configuration
    config = PipelineConfig(
        max_parallel_processes=4,
        max_memory_gb=16,
        timeout_hours=2,
        generate_paper=True,
        output_dir=Path("test_pipeline_run")
    )
    
    # Initialize pipeline
    pipeline = ValidationPipeline()
    
    # Run validation (with mock data for testing)
    try:
        results = pipeline.validate_bem_claims(
            model_path="test_bem_model",
            baseline_types=["static_lora", "adalora"],
            shift_types=["domain", "temporal"],
            num_seeds=2,
            generate_paper=True
        )
        
        print("Pipeline completed successfully!")
        print(f"Results: {json.dumps(results, indent=2, default=str)}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        print(traceback.format_exc())
    
    # Clean up test directory
    import shutil
    shutil.rmtree("test_pipeline_run", ignore_errors=True)