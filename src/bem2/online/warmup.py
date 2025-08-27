"""
Warmup Manager for BEM 2.0 Online Learning.

Handles warmup from AR1 best checkpoint as specified in TODO.md.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import logging
import time
from pathlib import Path

from .interfaces import LearningPhase
from .ewc_regularizer import EWCRegularizer, FisherConfig
from .canary_gate import CanaryGate
from .drift_monitor import DriftMonitor, DriftThresholds


@dataclass
class WarmupConfig:
    """Configuration for warmup process."""
    
    # Source checkpoint
    checkpoint_path: str = ""
    
    # Fisher information computation
    fisher_samples: int = 1000
    fisher_batch_size: int = 32
    compute_fisher: bool = True
    
    # Baseline establishment
    establish_baselines: bool = True
    baseline_samples: int = 500
    
    # Canary validation
    validate_with_canaries: bool = True
    required_canary_pass_rate: float = 0.9
    
    # Performance validation
    validate_performance: bool = True
    min_performance_threshold: float = 0.8
    
    # Safety checks
    run_safety_checks: bool = True
    max_drift_tolerance: float = 0.1
    
    # Output
    warmup_checkpoint_path: str = "warmup_checkpoint.pt"
    save_warmup_metadata: bool = True
    
    # Logging
    verbose: bool = True


@dataclass
class WarmupMetrics:
    """Metrics collected during warmup process."""
    
    # Timing
    total_time: float = 0.0
    checkpoint_load_time: float = 0.0
    fisher_compute_time: float = 0.0
    baseline_compute_time: float = 0.0
    canary_validation_time: float = 0.0
    
    # Performance
    initial_performance: float = 0.0
    baseline_performance: float = 0.0
    performance_drop: float = 0.0
    
    # Fisher information
    fisher_samples_used: int = 0
    fisher_mean: float = 0.0
    fisher_std: float = 0.0
    
    # Canary results
    canary_pass_rate: float = 0.0
    canary_tests_run: int = 0
    canary_tests_passed: int = 0
    
    # Safety
    initial_drift_score: float = 0.0
    safety_checks_passed: bool = True
    
    # Validation
    warmup_successful: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timing': {
                'total_time': self.total_time,
                'checkpoint_load_time': self.checkpoint_load_time,
                'fisher_compute_time': self.fisher_compute_time,
                'baseline_compute_time': self.baseline_compute_time,
                'canary_validation_time': self.canary_validation_time
            },
            'performance': {
                'initial_performance': self.initial_performance,
                'baseline_performance': self.baseline_performance,
                'performance_drop': self.performance_drop
            },
            'fisher': {
                'samples_used': self.fisher_samples_used,
                'mean': self.fisher_mean,
                'std': self.fisher_std
            },
            'canaries': {
                'pass_rate': self.canary_pass_rate,
                'tests_run': self.canary_tests_run,
                'tests_passed': self.canary_tests_passed
            },
            'safety': {
                'initial_drift_score': self.initial_drift_score,
                'safety_checks_passed': self.safety_checks_passed
            },
            'validation': {
                'warmup_successful': self.warmup_successful,
                'validation_errors': self.validation_errors
            }
        }


class WarmupManager:
    """
    Manages warmup process for online learning from AR1 checkpoint.
    
    Responsibilities:
    1. Load AR1 best checkpoint
    2. Compute Fisher information for EWC
    3. Establish canary baselines
    4. Validate model performance
    5. Create warmup checkpoint for online learning
    
    As specified in TODO.md workflow:
    ```
    python bem2/online/warmup.py --in logs/AR1/best.pt --out logs/OL0/warm.ckpt
    ```
    """
    
    def __init__(self, config: WarmupConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Components (will be initialized during warmup)
        self.ewc_regularizer: Optional[EWCRegularizer] = None
        self.canary_gate: Optional[CanaryGate] = None
        self.drift_monitor: Optional[DriftMonitor] = None
        
        # Metrics
        self.metrics = WarmupMetrics()
        
        self.logger.info("WarmupManager initialized")
    
    def warmup_from_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: str,
        data_loader: torch.utils.data.DataLoader,
        canary_gate: Optional[CanaryGate] = None,
        ewc_regularizer: Optional[EWCRegularizer] = None,
        drift_monitor: Optional[DriftMonitor] = None
    ) -> Tuple[bool, WarmupMetrics]:
        """
        Perform complete warmup process.
        
        Args:
            model: Model to warmup
            checkpoint_path: Path to AR1 checkpoint
            data_loader: Data for Fisher computation and validation
            canary_gate: Optional canary gate for validation
            ewc_regularizer: Optional EWC regularizer
            drift_monitor: Optional drift monitor
            
        Returns:
            (success, metrics) tuple
        """
        self.logger.info("Starting warmup process...")
        start_time = time.time()
        
        try:
            # Store components
            self.canary_gate = canary_gate
            self.ewc_regularizer = ewc_regularizer
            self.drift_monitor = drift_monitor
            
            # Phase 1: Load checkpoint
            success = self._load_checkpoint(model, checkpoint_path)
            if not success:
                return False, self.metrics
            
            # Phase 2: Compute Fisher information
            if self.config.compute_fisher and self.ewc_regularizer:
                success = self._compute_fisher_information(model, data_loader)
                if not success:
                    return False, self.metrics
            
            # Phase 3: Establish baselines
            if self.config.establish_baselines:
                success = self._establish_baselines(model, data_loader)
                if not success:
                    return False, self.metrics
            
            # Phase 4: Validate with canaries
            if self.config.validate_with_canaries and self.canary_gate:
                success = self._validate_canaries(model, data_loader)
                if not success:
                    return False, self.metrics
            
            # Phase 5: Performance validation
            if self.config.validate_performance:
                success = self._validate_performance(model, data_loader)
                if not success:
                    return False, self.metrics
            
            # Phase 6: Safety checks
            if self.config.run_safety_checks:
                success = self._run_safety_checks(model)
                if not success:
                    return False, self.metrics
            
            # Phase 7: Save warmup checkpoint
            success = self._save_warmup_checkpoint(model)
            if not success:
                return False, self.metrics
            
            # Finalize metrics
            self.metrics.total_time = time.time() - start_time
            self.metrics.warmup_successful = True
            
            self.logger.info(f"Warmup completed successfully in {self.metrics.total_time:.2f}s")
            return True, self.metrics
            
        except Exception as e:
            self.logger.error(f"Warmup failed: {e}")
            self.metrics.validation_errors.append(str(e))
            self.metrics.total_time = time.time() - start_time
            return False, self.metrics
    
    def _load_checkpoint(self, model: nn.Module, checkpoint_path: str) -> bool:
        """Load AR1 checkpoint."""
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        start_time = time.time()
        
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                error_msg = f"Checkpoint file not found: {checkpoint_path}"
                self.logger.error(error_msg)
                self.metrics.validation_errors.append(error_msg)
                return False
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                # Assume the checkpoint is the state dict itself
                state_dict = checkpoint
            
            # Load state dict into model
            model.load_state_dict(state_dict, strict=False)
            
            # Set base model for drift monitoring
            if self.drift_monitor:
                self.drift_monitor.set_base_model(model)
            
            self.metrics.checkpoint_load_time = time.time() - start_time
            self.logger.info(f"Checkpoint loaded successfully in {self.metrics.checkpoint_load_time:.2f}s")
            return True
            
        except Exception as e:
            error_msg = f"Failed to load checkpoint: {e}"
            self.logger.error(error_msg)
            self.metrics.validation_errors.append(error_msg)
            return False
    
    def _compute_fisher_information(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader
    ) -> bool:
        """Compute Fisher information for EWC."""
        self.logger.info("Computing Fisher information...")
        start_time = time.time()
        
        try:
            # Create subset data loader for Fisher computation
            fisher_data = self._create_subset_dataloader(
                data_loader, self.config.fisher_samples
            )
            
            # Compute Fisher information
            fisher_matrix = self.ewc_regularizer.compute_fisher_information(
                model, fisher_data
            )
            
            # Store metrics
            self.metrics.fisher_samples_used = fisher_matrix.num_samples
            self.metrics.fisher_mean = fisher_matrix.fisher_mean
            self.metrics.fisher_std = fisher_matrix.fisher_std
            self.metrics.fisher_compute_time = time.time() - start_time
            
            self.logger.info(f"Fisher information computed: "
                           f"{fisher_matrix.num_samples} samples, "
                           f"mean={fisher_matrix.fisher_mean:.6f}, "
                           f"std={fisher_matrix.fisher_std:.6f}")
            return True
            
        except Exception as e:
            error_msg = f"Fisher information computation failed: {e}"
            self.logger.error(error_msg)
            self.metrics.validation_errors.append(error_msg)
            return False
    
    def _establish_baselines(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader
    ) -> bool:
        """Establish performance and safety baselines."""
        self.logger.info("Establishing baselines...")
        start_time = time.time()
        
        try:
            # Create subset data loader for baseline computation
            baseline_data = self._create_subset_dataloader(
                data_loader, self.config.baseline_samples
            )
            
            # Compute baseline performance
            baseline_performance = self._compute_performance(model, baseline_data)
            self.metrics.baseline_performance = baseline_performance
            
            # Set baselines in components
            if self.canary_gate:
                self.canary_gate.set_baseline(model, baseline_data)
            
            if self.drift_monitor:
                self.drift_monitor.set_baseline_performance(baseline_performance)
            
            self.metrics.baseline_compute_time = time.time() - start_time
            self.logger.info(f"Baselines established: performance={baseline_performance:.4f}")
            return True
            
        except Exception as e:
            error_msg = f"Baseline establishment failed: {e}"
            self.logger.error(error_msg)
            self.metrics.validation_errors.append(error_msg)
            return False
    
    def _validate_canaries(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader
    ) -> bool:
        """Validate model with canary tests."""
        self.logger.info("Running canary validation...")
        start_time = time.time()
        
        try:
            # Run canary tests
            all_passed, results = self.canary_gate.run_canaries(model, data_loader)
            
            # Update metrics
            self.metrics.canary_tests_run = len(results)
            self.metrics.canary_tests_passed = sum(1 for r in results if r.passed)
            self.metrics.canary_pass_rate = self.metrics.canary_tests_passed / max(1, self.metrics.canary_tests_run)
            self.metrics.canary_validation_time = time.time() - start_time
            
            # Check pass rate
            if self.metrics.canary_pass_rate < self.config.required_canary_pass_rate:
                error_msg = (f"Canary pass rate too low: "
                           f"{self.metrics.canary_pass_rate:.2%} < "
                           f"{self.config.required_canary_pass_rate:.2%}")
                self.logger.error(error_msg)
                self.metrics.validation_errors.append(error_msg)
                return False
            
            self.logger.info(f"Canary validation passed: "
                           f"{self.metrics.canary_tests_passed}/{self.metrics.canary_tests_run} "
                           f"({self.metrics.canary_pass_rate:.1%})")
            return True
            
        except Exception as e:
            error_msg = f"Canary validation failed: {e}"
            self.logger.error(error_msg)
            self.metrics.validation_errors.append(error_msg)
            return False
    
    def _validate_performance(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader
    ) -> bool:
        """Validate model performance."""
        self.logger.info("Validating performance...")
        
        try:
            # Compute current performance
            current_performance = self._compute_performance(model, data_loader)
            self.metrics.initial_performance = current_performance
            
            # Check against threshold
            if current_performance < self.config.min_performance_threshold:
                error_msg = (f"Performance too low: "
                           f"{current_performance:.4f} < "
                           f"{self.config.min_performance_threshold:.4f}")
                self.logger.error(error_msg)
                self.metrics.validation_errors.append(error_msg)
                return False
            
            # Compute performance drop from baseline
            if self.metrics.baseline_performance > 0:
                self.metrics.performance_drop = (
                    self.metrics.baseline_performance - current_performance
                )
            
            self.logger.info(f"Performance validation passed: {current_performance:.4f}")
            return True
            
        except Exception as e:
            error_msg = f"Performance validation failed: {e}"
            self.logger.error(error_msg)
            self.metrics.validation_errors.append(error_msg)
            return False
    
    def _run_safety_checks(self, model: nn.Module) -> bool:
        """Run safety checks."""
        self.logger.info("Running safety checks...")
        
        try:
            if self.drift_monitor:
                # Run initial drift check
                drift_metrics = self.drift_monitor.check_drift(model, step=0)
                self.metrics.initial_drift_score = drift_metrics.drift_score
                
                # Check drift tolerance
                if drift_metrics.drift_score > self.config.max_drift_tolerance:
                    error_msg = (f"Initial drift too high: "
                               f"{drift_metrics.drift_score:.4f} > "
                               f"{self.config.max_drift_tolerance:.4f}")
                    self.logger.error(error_msg)
                    self.metrics.validation_errors.append(error_msg)
                    self.metrics.safety_checks_passed = False
                    return False
            
            self.logger.info("Safety checks passed")
            return True
            
        except Exception as e:
            error_msg = f"Safety checks failed: {e}"
            self.logger.error(error_msg)
            self.metrics.validation_errors.append(error_msg)
            self.metrics.safety_checks_passed = False
            return False
    
    def _save_warmup_checkpoint(self, model: nn.Module) -> bool:
        """Save warmup checkpoint."""
        self.logger.info("Saving warmup checkpoint...")
        
        try:
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'warmup_metrics': self.metrics.to_dict(),
                'warmup_config': {
                    'checkpoint_path': self.config.checkpoint_path,
                    'fisher_samples': self.config.fisher_samples,
                    'baseline_samples': self.config.baseline_samples,
                },
                'warmup_timestamp': time.time(),
                'phase': LearningPhase.STREAMING.value
            }
            
            # Add Fisher matrix if available
            if self.ewc_regularizer and self.ewc_regularizer.fisher_matrix:
                checkpoint_data['fisher_matrix'] = {
                    'fisher_diag': self.ewc_regularizer.fisher_matrix.fisher_diag,
                    'theta_star': self.ewc_regularizer.fisher_matrix.theta_star
                }
            
            # Save checkpoint
            checkpoint_path = Path(self.config.warmup_checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_data, checkpoint_path)
            
            self.logger.info(f"Warmup checkpoint saved: {checkpoint_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to save warmup checkpoint: {e}"
            self.logger.error(error_msg)
            self.metrics.validation_errors.append(error_msg)
            return False
    
    def _create_subset_dataloader(
        self,
        data_loader: torch.utils.data.DataLoader,
        num_samples: int
    ) -> torch.utils.data.DataLoader:
        """Create subset data loader with specified number of samples."""
        
        # Extract samples from original data loader
        samples = []
        total_extracted = 0
        
        for batch in data_loader:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                batch_size = batch[0].shape[0] if torch.is_tensor(batch[0]) else len(batch[0])
            elif isinstance(batch, dict):
                first_key = next(iter(batch.keys()))
                batch_size = batch[first_key].shape[0] if torch.is_tensor(batch[first_key]) else len(batch[first_key])
            else:
                batch_size = batch.shape[0] if torch.is_tensor(batch) else len(batch)
            
            samples.append(batch)
            total_extracted += batch_size
            
            if total_extracted >= num_samples:
                break
        
        # Create new dataset from samples
        if isinstance(samples[0], (list, tuple)):
            # Concatenate tensors
            combined = []
            for i in range(len(samples[0])):
                combined.append(torch.cat([batch[i] for batch in samples]))
            subset_dataset = data.TensorDataset(*combined)
        else:
            # Single tensor dataset
            combined = torch.cat(samples)
            subset_dataset = data.TensorDataset(combined)
        
        # Create new data loader
        return data.DataLoader(
            subset_dataset,
            batch_size=data_loader.batch_size,
            shuffle=False,  # Don't shuffle for reproducible baselines
            num_workers=0  # Avoid multiprocessing issues
        )
    
    def _compute_performance(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader
    ) -> float:
        """Compute model performance (accuracy) on dataset."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                    else:
                        continue  # Skip malformed batches
                elif isinstance(batch, dict):
                    inputs = batch.get('input_ids', batch.get('inputs'))
                    targets = batch.get('labels', batch.get('targets'))
                else:
                    continue  # Skip unknown formats
                
                if inputs is None or targets is None:
                    continue
                
                # Move to device if needed
                device = next(model.parameters()).device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Get predictions
                if len(outputs.shape) > 1 and outputs.shape[-1] > 1:
                    _, predicted = torch.max(outputs.data, -1)
                else:
                    predicted = (outputs > 0).long()
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        model.train()
        return correct / total if total > 0 else 0.0


# Utility functions
def create_warmup_manager(
    checkpoint_path: str,
    warmup_checkpoint_path: str = "warmup_checkpoint.pt",
    fisher_samples: int = 1000,
    validate_canaries: bool = True
) -> WarmupManager:
    """Create warmup manager with specified configuration."""
    config = WarmupConfig(
        checkpoint_path=checkpoint_path,
        warmup_checkpoint_path=warmup_checkpoint_path,
        fisher_samples=fisher_samples,
        validate_with_canaries=validate_canaries
    )
    return WarmupManager(config)


def load_warmup_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load warmup checkpoint and return contents."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Warmup BEM 2.0 for online learning")
    parser.add_argument("--in", dest="input_checkpoint", required=True,
                       help="Path to AR1 checkpoint")
    parser.add_argument("--out", dest="output_checkpoint", required=True,
                       help="Path for warmup checkpoint")
    parser.add_argument("--fisher-samples", type=int, default=1000,
                       help="Number of samples for Fisher computation")
    parser.add_argument("--no-canaries", action="store_true",
                       help="Skip canary validation")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print(f"Starting warmup from {args.input_checkpoint}")
    
    # Create dummy model and data for demonstration
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Create dummy dataset
    X = torch.randn(1000, 100)
    y = torch.randint(0, 10, (1000,))
    dataset = data.TensorDataset(X, y)
    data_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create warmup manager
    manager = create_warmup_manager(
        checkpoint_path=args.input_checkpoint,
        warmup_checkpoint_path=args.output_checkpoint,
        fisher_samples=args.fisher_samples,
        validate_canaries=not args.no_canaries
    )
    
    # Create components
    from .ewc_regularizer import create_ewc_regularizer
    from .canary_gate import create_default_canary_gate
    from .drift_monitor import create_drift_monitor
    
    ewc = create_ewc_regularizer(num_samples=args.fisher_samples)
    canaries = None if args.no_canaries else create_default_canary_gate()
    monitor = create_drift_monitor()
    
    # Run warmup
    success, metrics = manager.warmup_from_checkpoint(
        model, args.input_checkpoint, data_loader, canaries, ewc, monitor
    )
    
    if success:
        print(f"✅ Warmup completed successfully!")
        print(f"   Time: {metrics.total_time:.2f}s")
        print(f"   Performance: {metrics.initial_performance:.4f}")
        print(f"   Canary pass rate: {metrics.canary_pass_rate:.1%}")
        print(f"   Warmup checkpoint: {args.output_checkpoint}")
    else:
        print(f"❌ Warmup failed!")
        for error in metrics.validation_errors:
            print(f"   Error: {error}")
        exit(1)