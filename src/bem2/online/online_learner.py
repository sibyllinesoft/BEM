"""
Online Learner - Main orchestrator for BEM 2.0 safe online learning.

Implements controller-only updates with EWC/Prox regularization, replay buffer,
canary gates, and drift monitoring as specified in TODO.md requirements.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import time
import copy
from pathlib import Path

from .interfaces import (
    OnlineUpdateResult, SafetyStatus, UpdateDecision, LearningPhase,
    CanaryStatus, LearningState, UpdateMetrics, SafetyLimits, LearningRates
)
from .ewc_regularizer import EWCRegularizer, FisherConfig
from .replay_buffer import ReplayBuffer, ReplayConfig, Experience
from .canary_gate import CanaryGate, CanaryResult
from .drift_monitor import DriftMonitor, DriftThresholds, DriftMetrics
from .checkpointing import CheckpointManager, CheckpointConfig


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning system."""
    
    # Learning rates and regularization
    learning_rates: LearningRates = field(default_factory=LearningRates)
    
    # Safety limits
    safety_limits: SafetyLimits = field(default_factory=SafetyLimits)
    
    # Component configurations
    ewc_config: FisherConfig = field(default_factory=FisherConfig)
    replay_config: ReplayConfig = field(default_factory=ReplayConfig)
    drift_thresholds: DriftThresholds = field(default_factory=DriftThresholds)
    checkpoint_config: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Update behavior
    controller_only: bool = True  # Only update controller parameters
    between_prompts_only: bool = True  # Only apply updates between prompts
    
    # Frequencies
    drift_check_frequency: int = 10  # Check drift every N steps
    canary_frequency: int = 50  # Run canaries every N updates
    replay_frequency: int = 5  # Use replay every N updates
    
    # Rollback configuration
    enable_auto_rollback: bool = True
    rollback_steps: int = 10  # Steps to rollback on failure
    max_consecutive_failures: int = 3
    
    # Performance requirements
    min_canary_pass_rate: float = 0.9  # Minimum canary success rate
    max_update_time: float = 60.0  # Maximum time per update (seconds)
    
    # Logging and monitoring
    verbose: bool = True
    log_frequency: int = 10


class OnlineLearner:
    """
    Main controller for BEM 2.0 online learning system.
    
    Orchestrates safe controller-only updates using:
    - EWC regularization to prevent catastrophic forgetting
    - Replay buffer for knowledge retention (10k samples)  
    - Canary gates for safety validation
    - Drift monitoring with automatic rollback
    - Between-prompts-only updates
    
    Implements the pseudocode from TODO.md:
    ```python
    def online_step(theta, batch, fisher_diag, theta0):
        loss_online = ctrl_loss(batch)
        ewc = ((theta - theta0)**2 * fisher_diag).sum()
        prox = ((theta - theta_prev)**2).sum()
        total = loss_online + lam_ewc*ewc + lam_prox*prox
        theta_new = opt_step(theta, grad(total))
        if run_canaries(theta_new):  # must pass before activation
            return theta_new
        else:
            rollback_checkpoint()
            return theta
    ```
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: OnlineLearningConfig,
        canary_gate: CanaryGate,
        data_loader: Optional[torch.utils.data.DataLoader] = None
    ):
        self.model = model
        self.config = config
        self.canary_gate = canary_gate
        self.data_loader = data_loader
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Learning state
        self.learning_state = LearningState(
            phase=LearningPhase.WARMUP,
            step=0,
            updates_applied=0,
            rollbacks_triggered=0,
            last_update_time=time.time(),
            last_checkpoint_time=time.time()
        )
        
        # Optimizer and scheduler
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        
        # Performance tracking
        self.performance_history = []
        self.update_history = []
        
        # Safety state
        self.consecutive_failures = 0
        self.last_successful_update = 0
        self.prompt_boundary = True  # Start at prompt boundary
        
        # Statistics
        self.total_update_attempts = 0
        self.successful_updates = 0
        self.canary_failures = 0
        self.drift_rollbacks = 0
        
        self.logger.info("OnlineLearner initialized")
    
    def _initialize_components(self):
        """Initialize online learning components."""
        # EWC Regularizer
        self.ewc_regularizer = EWCRegularizer(self.config.ewc_config)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(self.config.replay_config)
        
        # Drift Monitor  
        self.drift_monitor = DriftMonitor(
            self.config.drift_thresholds,
            base_model=self.model
        )
        
        # Checkpoint Manager
        self.checkpoint_manager = CheckpointManager(self.config.checkpoint_config)
        
        # Store base model state
        self.base_model_state = copy.deepcopy(self.model.state_dict())
        
        self.logger.info("Online learning components initialized")
    
    def setup_optimizer(
        self,
        optimizer_class: type = optim.AdamW,
        **optimizer_kwargs
    ):
        """Setup optimizer for online learning."""
        # Filter parameters if controller-only mode
        if self.config.controller_only:
            # In real BEM implementation, this would filter for controller parameters
            # For now, we'll use all parameters with a flag
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.logger.info(f"Controller-only mode: optimizing {len(params)} parameter groups")
        else:
            params = self.model.parameters()
        
        self.optimizer = optimizer_class(
            params,
            lr=self.config.learning_rates.controller_lr,
            **optimizer_kwargs
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.learning_rates.decay_steps,
            gamma=self.config.learning_rates.lr_decay_factor
        )
        
        self.logger.info(f"Optimizer setup complete: {optimizer_class.__name__}")
    
    def warmup_from_checkpoint(self, checkpoint_path: str):
        """Warmup online learner from AR1 best checkpoint as specified in TODO.md."""
        self.learning_state.phase = LearningPhase.WARMUP
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # Update base model state
            self.base_model_state = copy.deepcopy(self.model.state_dict())
            self.drift_monitor.set_base_model(self.model)
            
            # Establish canary baselines
            if self.data_loader is not None:
                self.canary_gate.set_baseline(self.model, self.data_loader)
            
            # Compute initial Fisher information if data available
            if self.data_loader is not None:
                self.logger.info("Computing initial Fisher information...")
                self.ewc_regularizer.compute_fisher_information(
                    self.model, self.data_loader
                )
            
            # Create initial checkpoint
            self.checkpoint_manager.create_checkpoint(
                self.model, self.optimizer, 
                step=0, metrics={'warmup': True}
            )
            
            self.learning_state.phase = LearningPhase.STREAMING
            self.logger.info(f"Warmup complete from: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Warmup failed: {e}")
            raise
    
    def online_update_step(
        self,
        batch: Dict[str, Any],
        feedback_score: Optional[float] = None,
        force_update: bool = False
    ) -> OnlineUpdateResult:
        """
        Perform one online update step following TODO.md pseudocode.
        
        Args:
            batch: Training batch
            feedback_score: Optional feedback signal  
            force_update: Force update even if not at prompt boundary
            
        Returns:
            OnlineUpdateResult with full details
        """
        if not force_update and not self.prompt_boundary:
            return OnlineUpdateResult(
                decision=UpdateDecision.DEFER,
                safety_status=SafetyStatus.SAFE,
                canary_status=CanaryStatus.PASSED,
                kl_divergence=0.0,
                parameter_norm=0.0,
                ewc_loss=0.0,
                replay_loss=0.0,
                update_applied=False,
                rollback_triggered=False,
                checkpoint_created=False,
                drift_detected=False,
                performance_delta=0.0,
                time_elapsed=0.0,
                memory_usage=0.0,
                message="Deferred: not at prompt boundary"
            )
        
        start_time = time.time()
        self.total_update_attempts += 1
        self.learning_state.step += 1
        
        try:
            # Phase 1: Prepare update
            update_result = self._prepare_update(batch, feedback_score)
            if update_result.decision != UpdateDecision.APPLY:
                return update_result
            
            # Phase 2: Compute losses (EWC + Prox + Online)
            losses = self._compute_losses(batch)
            
            # Phase 3: Apply gradient update
            self._apply_gradient_update(losses)
            
            # Phase 4: Safety checks
            safety_result = self._run_safety_checks(batch)
            
            # Phase 5: Decision - Apply or rollback
            final_result = self._make_update_decision(
                update_result, losses, safety_result, start_time
            )
            
            # Phase 6: Post-processing
            self._post_process_update(final_result, batch)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Online update failed: {e}")
            return self._create_error_result(str(e), start_time)
    
    def _prepare_update(
        self,
        batch: Dict[str, Any],
        feedback_score: Optional[float]
    ) -> OnlineUpdateResult:
        """Prepare for online update - checks and setup."""
        
        # Check consecutive failures
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            return OnlineUpdateResult(
                decision=UpdateDecision.REJECT,
                safety_status=SafetyStatus.CRITICAL,
                canary_status=CanaryStatus.FAILED,
                kl_divergence=0.0, parameter_norm=0.0, ewc_loss=0.0, replay_loss=0.0,
                update_applied=False, rollback_triggered=False,
                checkpoint_created=False, drift_detected=False,
                performance_delta=0.0, time_elapsed=0.0, memory_usage=0.0,
                message=f"Too many consecutive failures: {self.consecutive_failures}"
            )
        
        # Add experience to replay buffer
        if self._should_add_to_replay(batch, feedback_score):
            experience = self._create_experience(batch, feedback_score)
            self.replay_buffer.add(experience)
        
        return OnlineUpdateResult(
            decision=UpdateDecision.APPLY,
            safety_status=SafetyStatus.SAFE,
            canary_status=CanaryStatus.PASSED,
            kl_divergence=0.0, parameter_norm=0.0, ewc_loss=0.0, replay_loss=0.0,
            update_applied=False, rollback_triggered=False,
            checkpoint_created=False, drift_detected=False,
            performance_delta=0.0, time_elapsed=0.0, memory_usage=0.0,
            message="Ready for update"
        )
    
    def _compute_losses(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute all losses: online + EWC + replay."""
        losses = {}
        
        # 1. Online loss from current batch
        online_loss = self._compute_online_loss(batch)
        losses['online'] = online_loss
        
        # 2. EWC regularization loss
        ewc_loss = self.ewc_regularizer.compute_ewc_loss(
            self.model, 
            lambda_ewc=self.config.learning_rates.ewc_lambda
        )
        losses['ewc'] = ewc_loss
        
        # 3. Proximal regularization loss (TODO: implement properly)
        prox_loss = self._compute_proximal_loss()
        losses['prox'] = prox_loss
        
        # 4. Replay loss (if replay is enabled)
        replay_loss = torch.tensor(0.0, device=online_loss.device)
        if (self.learning_state.step % self.config.replay_frequency == 0 and
            len(self.replay_buffer.buffer) >= self.config.replay_config.min_replay_size):
            
            replay_batch = self.replay_buffer.sample()
            if replay_batch:
                replay_loss = self._compute_replay_loss(replay_batch)
                losses['replay'] = replay_loss
        
        # 5. Total loss
        total_loss = (online_loss + ewc_loss + prox_loss + replay_loss)
        losses['total'] = total_loss
        
        return losses
    
    def _compute_online_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute online loss from current batch."""
        # Extract inputs and targets
        if 'input_ids' in batch and 'labels' in batch:
            inputs = batch['input_ids']
            targets = batch['labels']
        elif 'inputs' in batch and 'targets' in batch:
            inputs = batch['inputs']
            targets = batch['targets']
        else:
            # Fallback for different batch formats
            inputs = batch.get('input_ids', batch.get('inputs'))
            targets = batch.get('labels', batch.get('targets'))
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Compute loss
        if targets is not None:
            loss = F.cross_entropy(outputs, targets)
        else:
            # Unsupervised loss (e.g., language modeling)
            loss = F.cross_entropy(outputs[..., :-1, :].contiguous().view(-1, outputs.size(-1)),
                                 inputs[..., 1:].contiguous().view(-1))
        
        return loss
    
    def _compute_proximal_loss(self) -> torch.Tensor:
        """Compute proximal regularization loss."""
        # For now, simple L2 regularization to previous parameters
        # In full implementation, this would be to the last checkpoint
        prox_loss = torch.tensor(0.0)
        
        if hasattr(self, '_prev_params'):
            device = next(self.model.parameters()).device
            prox_loss = prox_loss.to(device)
            
            for (name, param), (prev_name, prev_param) in zip(
                self.model.named_parameters(), self._prev_params
            ):
                if param.requires_grad and name == prev_name:
                    prox_loss += torch.norm(param - prev_param.to(device)) ** 2
            
            prox_loss *= self.config.learning_rates.prox_lambda
        
        return prox_loss
    
    def _compute_replay_loss(self, replay_batch: List[Experience]) -> torch.Tensor:
        """Compute loss on replay batch."""
        # Convert replay experiences to tensors
        input_ids = torch.stack([exp.input_ids for exp in replay_batch])
        
        if replay_batch[0].targets is not None:
            targets = torch.stack([exp.targets for exp in replay_batch])
        else:
            targets = None
        
        # Move to device
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if targets is not None:
            targets = targets.to(device)
        
        # Forward pass
        outputs = self.model(input_ids)
        
        # Compute loss
        if targets is not None:
            loss = F.cross_entropy(outputs, targets.squeeze())
        else:
            loss = F.cross_entropy(
                outputs[..., :-1, :].contiguous().view(-1, outputs.size(-1)),
                input_ids[..., 1:].contiguous().view(-1)
            )
        
        return loss
    
    def _apply_gradient_update(self, losses: Dict[str, torch.Tensor]):
        """Apply gradient update with total loss."""
        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized")
        
        # Store previous parameters for proximal loss
        self._prev_params = [(name, param.data.clone()) 
                            for name, param in self.model.named_parameters()]
        
        # Backward pass
        self.optimizer.zero_grad()
        losses['total'].backward()
        
        # Gradient clipping
        if hasattr(self.config.safety_limits, 'max_gradient_norm'):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.safety_limits.max_gradient_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        
        # Update learning rate scheduler
        if self.scheduler is not None:
            self.scheduler.step()
    
    def _run_safety_checks(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run safety checks: drift monitoring and canaries."""
        safety_result = {
            'drift_metrics': None,
            'canary_passed': True,
            'canary_results': [],
            'safety_status': SafetyStatus.SAFE,
            'rollback_needed': False
        }
        
        # 1. Drift monitoring
        if self.learning_state.step % self.config.drift_check_frequency == 0:
            drift_metrics = self.drift_monitor.check_drift(
                self.model,
                step=self.learning_state.step
            )
            safety_result['drift_metrics'] = drift_metrics
            
            # Check for rollback trigger
            should_rollback, rollback_reason = self.drift_monitor.should_rollback()
            if should_rollback:
                safety_result['rollback_needed'] = True
                safety_result['safety_status'] = SafetyStatus.CRITICAL
                self.logger.error(f"Drift rollback triggered: {rollback_reason}")
        
        # 2. Canary tests
        if (self.learning_state.step % self.config.canary_frequency == 0 and 
            self.data_loader is not None):
            
            canary_passed, canary_results = self.canary_gate.run_canaries(
                self.model, self.data_loader
            )
            
            safety_result['canary_passed'] = canary_passed
            safety_result['canary_results'] = canary_results
            
            if not canary_passed:
                safety_result['safety_status'] = SafetyStatus.DANGER
                self.canary_failures += 1
                self.logger.warning("Canary tests failed")
        
        return safety_result
    
    def _make_update_decision(
        self,
        update_result: OnlineUpdateResult,
        losses: Dict[str, torch.Tensor],
        safety_result: Dict[str, Any],
        start_time: float
    ) -> OnlineUpdateResult:
        """Make final decision on whether to keep update or rollback."""
        
        # Extract metrics
        drift_metrics = safety_result.get('drift_metrics')
        canary_passed = safety_result.get('canary_passed', True)
        rollback_needed = safety_result.get('rollback_needed', False)
        
        # Determine decision
        if rollback_needed:
            decision = UpdateDecision.ROLLBACK
            update_applied = False
            self._rollback_model()
            self.drift_rollbacks += 1
        elif not canary_passed:
            decision = UpdateDecision.ROLLBACK  
            update_applied = False
            self._rollback_model()
            self.canary_failures += 1
        else:
            decision = UpdateDecision.APPLY
            update_applied = True
            self.successful_updates += 1
            self.consecutive_failures = 0
        
        # Update failure counter
        if not update_applied:
            self.consecutive_failures += 1
        
        # Create checkpoint if successful
        checkpoint_created = False
        if update_applied and self.learning_state.step % 100 == 0:
            self.checkpoint_manager.create_checkpoint(
                self.model, self.optimizer,
                step=self.learning_state.step,
                metrics={
                    'online_loss': losses['online'].item(),
                    'ewc_loss': losses['ewc'].item(),
                    'total_loss': losses['total'].item()
                }
            )
            checkpoint_created = True
        
        # Update learning state
        if update_applied:
            self.learning_state.updates_applied += 1
            self.learning_state.last_update_time = time.time()
            
            if drift_metrics:
                self.learning_state.kl_divergence = drift_metrics.kl_divergence
                self.learning_state.parameter_norm = drift_metrics.parameter_norm
        
        if decision == UpdateDecision.ROLLBACK:
            self.learning_state.rollbacks_triggered += 1
        
        # Compute elapsed time
        time_elapsed = time.time() - start_time
        
        # Memory usage
        memory_usage = 0.0
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        # Create result
        return OnlineUpdateResult(
            decision=decision,
            safety_status=safety_result['safety_status'],
            canary_status=CanaryStatus.PASSED if canary_passed else CanaryStatus.FAILED,
            kl_divergence=drift_metrics.kl_divergence if drift_metrics else 0.0,
            parameter_norm=drift_metrics.parameter_norm if drift_metrics else 0.0,
            ewc_loss=losses['ewc'].item(),
            replay_loss=losses.get('replay', torch.tensor(0.0)).item(),
            update_applied=update_applied,
            rollback_triggered=(decision == UpdateDecision.ROLLBACK),
            checkpoint_created=checkpoint_created,
            drift_detected=(drift_metrics and drift_metrics.drift_level.value != 'normal'),
            performance_delta=0.0,  # TODO: implement performance tracking
            time_elapsed=time_elapsed,
            memory_usage=memory_usage,
            message=f"Online step completed: {decision.value}"
        )
    
    def _post_process_update(self, result: OnlineUpdateResult, batch: Dict[str, Any]):
        """Post-process update results and logging."""
        # Update statistics
        self.update_history.append(result)
        if len(self.update_history) > 1000:  # Keep last 1000 updates
            self.update_history.pop(0)
        
        # Log results
        if (self.config.verbose and 
            self.learning_state.step % self.config.log_frequency == 0):
            
            self._log_update_result(result)
        
        # Reset prompt boundary flag
        self.prompt_boundary = False
    
    def _rollback_model(self):
        """Rollback model to last safe checkpoint."""
        try:
            # Get last safe checkpoint
            checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            if checkpoint is not None:
                # Load model state
                self.model.load_state_dict(checkpoint.model_state)
                if self.optimizer is not None and checkpoint.optimizer_state is not None:
                    self.optimizer.load_state_dict(checkpoint.optimizer_state)
                
                self.logger.info(f"Rolled back to checkpoint from step {checkpoint.step}")
            else:
                # Fallback to base model
                self.model.load_state_dict(self.base_model_state)
                self.logger.warning("No checkpoints available, rolled back to base model")
            
            # Reset drift monitor
            self.drift_monitor.reset_rollback()
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
    
    def set_prompt_boundary(self, at_boundary: bool = True):
        """Set prompt boundary flag for between-prompt updates."""
        self.prompt_boundary = at_boundary
        if at_boundary:
            self.logger.debug("At prompt boundary - updates enabled")
    
    def get_learning_state(self) -> LearningState:
        """Get current learning state."""
        return self.learning_state
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        canary_stats = self.canary_gate.get_statistics()
        drift_stats = self.drift_monitor.get_drift_summary()
        replay_stats = self.replay_buffer.get_statistics()
        
        return {
            'learning_state': {
                'phase': self.learning_state.phase.value,
                'step': self.learning_state.step,
                'updates_applied': self.learning_state.updates_applied,
                'rollbacks_triggered': self.learning_state.rollbacks_triggered
            },
            'performance': {
                'total_update_attempts': self.total_update_attempts,
                'successful_updates': self.successful_updates,
                'success_rate': self.successful_updates / max(1, self.total_update_attempts),
                'canary_failures': self.canary_failures,
                'drift_rollbacks': self.drift_rollbacks,
                'consecutive_failures': self.consecutive_failures
            },
            'components': {
                'canary_gate': canary_stats,
                'drift_monitor': drift_stats,
                'replay_buffer': replay_stats,
                'ewc_regularizer': self.ewc_regularizer.get_fisher_statistics()
            },
            'recent_updates': [
                {
                    'decision': r.decision.value,
                    'safety_status': r.safety_status.value,
                    'update_applied': r.update_applied,
                    'time_elapsed': r.time_elapsed
                }
                for r in self.update_history[-10:]
            ]
        }
    
    def _should_add_to_replay(
        self, 
        batch: Dict[str, Any], 
        feedback_score: Optional[float]
    ) -> bool:
        """Determine if batch should be added to replay buffer."""
        # Add all batches for now
        # In practice, might filter based on feedback, difficulty, etc.
        return True
    
    def _create_experience(
        self,
        batch: Dict[str, Any],
        feedback_score: Optional[float]
    ) -> Experience:
        """Create experience from batch for replay buffer."""
        # Extract first sample from batch
        input_ids = batch.get('input_ids', batch.get('inputs'))[0]
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask[0]
        
        targets = batch.get('labels', batch.get('targets'))
        if targets is not None:
            targets = targets[0]
        
        return Experience(
            input_ids=input_ids,
            attention_mask=attention_mask,
            targets=targets,
            feedback_score=feedback_score,
            context_length=input_ids.shape[-1]
        )
    
    def _create_error_result(self, error_msg: str, start_time: float) -> OnlineUpdateResult:
        """Create error result."""
        return OnlineUpdateResult(
            decision=UpdateDecision.REJECT,
            safety_status=SafetyStatus.CRITICAL,
            canary_status=CanaryStatus.ERROR,
            kl_divergence=0.0, parameter_norm=0.0, ewc_loss=0.0, replay_loss=0.0,
            update_applied=False, rollback_triggered=False,
            checkpoint_created=False, drift_detected=False,
            performance_delta=0.0, 
            time_elapsed=time.time() - start_time,
            memory_usage=0.0,
            message=f"Error: {error_msg}"
        )
    
    def _log_update_result(self, result: OnlineUpdateResult):
        """Log update result."""
        self.logger.info(
            f"Step {self.learning_state.step}: "
            f"{result.decision.value} "
            f"(safety={result.safety_status.value}, "
            f"kl={result.kl_divergence:.4f}, "
            f"norm={result.parameter_norm:.4f}, "
            f"time={result.time_elapsed:.2f}s)"
        )


# Utility functions
def create_online_learner(
    model: nn.Module,
    canary_gate: CanaryGate,
    data_loader: Optional[torch.utils.data.DataLoader] = None,
    config: Optional[OnlineLearningConfig] = None
) -> OnlineLearner:
    """Create online learner with default configuration."""
    if config is None:
        config = OnlineLearningConfig()
    
    return OnlineLearner(model, config, canary_gate, data_loader)


# Example usage
if __name__ == "__main__":
    # Create test setup
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Create data loader
    import torch.utils.data as data
    X = torch.randn(1000, 100)
    y = torch.randint(0, 10, (1000,))
    dataset = data.TensorDataset(X, y)
    data_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create canary gate
    from .canary_gate import create_default_canary_gate
    canary_gate = create_default_canary_gate()
    
    # Create online learner
    learner = create_online_learner(model, canary_gate, data_loader)
    learner.setup_optimizer()
    
    # Initialize from base model (simulate warmup)
    learner.learning_state.phase = LearningPhase.STREAMING
    learner.drift_monitor.set_base_model(model)
    
    # Simulate online learning steps
    for step in range(50):
        # Create batch
        batch = {
            'inputs': torch.randn(16, 100),
            'targets': torch.randint(0, 10, (16,))
        }
        
        # Set prompt boundary (simulate between-prompts updates)
        learner.set_prompt_boundary(True)
        
        # Online update step
        result = learner.online_update_step(batch, feedback_score=0.8)
        
        if step % 10 == 0:
            print(f"Step {step}: {result.decision.value} "
                  f"(applied={result.update_applied})")
    
    # Print final statistics
    stats = learner.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"Success rate: {stats['performance']['success_rate']:.1%}")
    print(f"Updates applied: {stats['learning_state']['updates_applied']}")
    print(f"Rollbacks: {stats['learning_state']['rollbacks_triggered']}")