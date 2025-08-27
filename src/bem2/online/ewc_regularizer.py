"""
Elastic Weight Consolidation (EWC) Regularizer for BEM 2.0 Online Learning.

Implements EWC regularization to prevent catastrophic forgetting during online updates.
Uses diagonal Fisher information matrix approximation as specified in TODO.md.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from collections import defaultdict
import copy


@dataclass
class FisherConfig:
    """Configuration for Fisher information matrix computation."""
    
    # Sample size for Fisher estimation
    num_samples: int = 1000
    
    # Batch size for Fisher computation
    batch_size: int = 32
    
    # Minimum Fisher value (for numerical stability)
    min_fisher_value: float = 1e-6
    
    # Maximum Fisher value (for clipping)
    max_fisher_value: float = 1e3
    
    # Smoothing factor for Fisher updates
    smoothing_factor: float = 0.9
    
    # Only compute Fisher for parameters with gradients
    only_trainable: bool = True


@dataclass 
class FisherInformationMatrix:
    """Diagonal Fisher Information Matrix for EWC."""
    
    # Fisher information for each parameter
    fisher_diag: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Parameter values at consolidation point
    theta_star: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Metadata
    num_samples: int = 0
    compute_time: float = 0.0
    total_parameters: int = 0
    fisher_mean: float = 0.0
    fisher_std: float = 0.0
    
    def is_valid(self) -> bool:
        """Check if Fisher matrix is valid for use."""
        return (len(self.fisher_diag) > 0 and 
                len(self.theta_star) > 0 and 
                self.num_samples > 0)
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB."""
        total_bytes = 0
        for tensor in self.fisher_diag.values():
            total_bytes += tensor.numel() * tensor.element_size()
        for tensor in self.theta_star.values():
            total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes / (1024 * 1024)


class EWCRegularizer:
    """
    Elastic Weight Consolidation regularizer for preventing catastrophic forgetting.
    
    Implements diagonal Fisher information matrix approximation for controller-only
    online updates as specified in BEM 2.0 requirements.
    """
    
    def __init__(self, config: FisherConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Current Fisher information matrix
        self.fisher_matrix: Optional[FisherInformationMatrix] = None
        
        # Historical Fisher matrices (for multi-task scenarios)
        self.fisher_history: List[FisherInformationMatrix] = []
        
        # Computation state
        self.is_computing = False
        self.last_compute_time = 0.0
        
        # Statistics
        self.total_computations = 0
        self.total_regularization_calls = 0
    
    def compute_fisher_information(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        criterion: Optional[nn.Module] = None
    ) -> FisherInformationMatrix:
        """
        Compute diagonal Fisher information matrix.
        
        Args:
            model: The model to compute Fisher for
            data_loader: DataLoader with representative samples
            criterion: Loss function (defaults to CrossEntropyLoss)
            
        Returns:
            FisherInformationMatrix object
        """
        if self.is_computing:
            self.logger.warning("Fisher computation already in progress")
            return self.fisher_matrix
        
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        else:
            import time
            cpu_start = time.time()
        
        self.is_computing = True
        self.logger.info("Starting Fisher information computation...")
        
        try:
            # Initialize Fisher matrix
            fisher_diag = {}
            theta_star = {}
            
            # Get reference parameters
            for name, param in model.named_parameters():
                if param.requires_grad or not self.config.only_trainable:
                    theta_star[name] = param.data.clone().detach()
                    fisher_diag[name] = torch.zeros_like(param.data)
            
            # Default criterion
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            
            model.eval()
            total_samples = 0
            samples_processed = 0
            
            # Compute Fisher information using gradient squares
            for batch_idx, batch in enumerate(data_loader):
                if samples_processed >= self.config.num_samples:
                    break
                
                # Extract inputs and targets from batch
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                    else:
                        inputs = batch[0]
                        targets = None
                elif isinstance(batch, dict):
                    inputs = batch.get('input', batch.get('inputs'))
                    targets = batch.get('target', batch.get('targets'))
                else:
                    inputs = batch
                    targets = None
                
                # Move to device
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    if targets is not None:
                        targets = targets.cuda()
                
                batch_size = inputs.shape[0]
                actual_batch_size = min(batch_size, self.config.num_samples - samples_processed)
                
                if actual_batch_size < batch_size:
                    inputs = inputs[:actual_batch_size]
                    if targets is not None:
                        targets = targets[:actual_batch_size]
                
                # Forward pass
                model.zero_grad()
                outputs = model(inputs)
                
                # Compute loss
                if targets is not None:
                    loss = criterion(outputs, targets)
                else:
                    # For unsupervised case, use negative log-likelihood
                    loss = -torch.mean(torch.log_softmax(outputs, dim=-1))
                
                # Backward pass to get gradients
                loss.backward()
                
                # Accumulate squared gradients (Fisher information)
                for name, param in model.named_parameters():
                    if name in fisher_diag and param.grad is not None:
                        fisher_diag[name] += param.grad.data ** 2
                
                samples_processed += actual_batch_size
                total_samples += actual_batch_size
                
                if batch_idx % 100 == 0:
                    self.logger.debug(f"Processed {samples_processed}/{self.config.num_samples} samples")
            
            # Normalize Fisher information by number of samples
            for name in fisher_diag:
                fisher_diag[name] /= total_samples
                
                # Apply clipping for numerical stability
                fisher_diag[name] = torch.clamp(
                    fisher_diag[name],
                    min=self.config.min_fisher_value,
                    max=self.config.max_fisher_value
                )
            
            # Compute statistics
            all_fisher_values = torch.cat([f.flatten() for f in fisher_diag.values()])
            fisher_mean = torch.mean(all_fisher_values).item()
            fisher_std = torch.std(all_fisher_values).item()
            
            # Record timing
            if start_time and end_time:
                end_time.record()
                torch.cuda.synchronize()
                compute_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                compute_time = time.time() - cpu_start
            
            # Create Fisher matrix object
            fisher_matrix = FisherInformationMatrix(
                fisher_diag=fisher_diag,
                theta_star=theta_star,
                num_samples=total_samples,
                compute_time=compute_time,
                total_parameters=sum(p.numel() for p in fisher_diag.values()),
                fisher_mean=fisher_mean,
                fisher_std=fisher_std
            )
            
            # Update internal state
            self.fisher_matrix = fisher_matrix
            self.total_computations += 1
            self.last_compute_time = compute_time
            
            self.logger.info(f"Fisher computation complete: {total_samples} samples, "
                           f"{compute_time:.2f}s, mean={fisher_mean:.6f}, std={fisher_std:.6f}")
            
            return fisher_matrix
            
        except Exception as e:
            self.logger.error(f"Error computing Fisher information: {e}")
            raise
        finally:
            self.is_computing = False
            model.train()
    
    def update_fisher_smoothed(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        criterion: Optional[nn.Module] = None
    ) -> FisherInformationMatrix:
        """
        Update Fisher information with exponential smoothing.
        
        Useful for continuous online learning where Fisher matrix
        should adapt slowly to new data.
        """
        new_fisher = self.compute_fisher_information(model, data_loader, criterion)
        
        if self.fisher_matrix is None:
            self.fisher_matrix = new_fisher
            return new_fisher
        
        # Apply exponential smoothing
        alpha = self.config.smoothing_factor
        for name in self.fisher_matrix.fisher_diag:
            if name in new_fisher.fisher_diag:
                self.fisher_matrix.fisher_diag[name] = (
                    alpha * self.fisher_matrix.fisher_diag[name] + 
                    (1 - alpha) * new_fisher.fisher_diag[name]
                )
                self.fisher_matrix.theta_star[name] = new_fisher.theta_star[name]
        
        # Update metadata
        self.fisher_matrix.num_samples = new_fisher.num_samples
        self.fisher_matrix.compute_time = new_fisher.compute_time
        
        self.logger.info("Fisher matrix updated with exponential smoothing")
        return self.fisher_matrix
    
    def compute_ewc_loss(
        self,
        model: nn.Module,
        lambda_ewc: float = 1.0
    ) -> torch.Tensor:
        """
        Compute EWC regularization loss.
        
        Args:
            model: Current model
            lambda_ewc: EWC regularization strength
            
        Returns:
            EWC loss tensor
        """
        if self.fisher_matrix is None or not self.fisher_matrix.is_valid():
            self.logger.warning("No valid Fisher matrix available for EWC loss")
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        self.total_regularization_calls += 1
        ewc_loss = 0.0
        device = next(model.parameters()).device
        
        for name, param in model.named_parameters():
            if name in self.fisher_matrix.fisher_diag and name in self.fisher_matrix.theta_star:
                fisher_diag = self.fisher_matrix.fisher_diag[name].to(device)
                theta_star = self.fisher_matrix.theta_star[name].to(device)
                
                # EWC loss: F * (theta - theta_star)^2
                param_diff = param - theta_star
                ewc_term = fisher_diag * (param_diff ** 2)
                ewc_loss += torch.sum(ewc_term)
        
        # Scale by lambda and normalize by number of parameters
        total_params = sum(1 for _ in model.parameters() if _.requires_grad)
        ewc_loss = lambda_ewc * ewc_loss / total_params
        
        return ewc_loss
    
    def get_fisher_statistics(self) -> Dict[str, Any]:
        """Get statistics about current Fisher matrix."""
        if self.fisher_matrix is None:
            return {
                'is_valid': False,
                'total_computations': self.total_computations,
                'total_regularization_calls': self.total_regularization_calls
            }
        
        return {
            'is_valid': self.fisher_matrix.is_valid(),
            'num_samples': self.fisher_matrix.num_samples,
            'total_parameters': self.fisher_matrix.total_parameters,
            'fisher_mean': self.fisher_matrix.fisher_mean,
            'fisher_std': self.fisher_matrix.fisher_std,
            'memory_usage_mb': self.fisher_matrix.get_memory_usage(),
            'compute_time': self.fisher_matrix.compute_time,
            'total_computations': self.total_computations,
            'total_regularization_calls': self.total_regularization_calls,
            'last_compute_time': self.last_compute_time
        }
    
    def save_fisher_matrix(self, filepath: str):
        """Save Fisher matrix to disk."""
        if self.fisher_matrix is None:
            raise ValueError("No Fisher matrix to save")
        
        torch.save({
            'fisher_diag': self.fisher_matrix.fisher_diag,
            'theta_star': self.fisher_matrix.theta_star,
            'metadata': {
                'num_samples': self.fisher_matrix.num_samples,
                'compute_time': self.fisher_matrix.compute_time,
                'total_parameters': self.fisher_matrix.total_parameters,
                'fisher_mean': self.fisher_matrix.fisher_mean,
                'fisher_std': self.fisher_matrix.fisher_std
            },
            'config': self.config
        }, filepath)
        
        self.logger.info(f"Fisher matrix saved to {filepath}")
    
    def load_fisher_matrix(self, filepath: str):
        """Load Fisher matrix from disk."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.fisher_matrix = FisherInformationMatrix(
            fisher_diag=checkpoint['fisher_diag'],
            theta_star=checkpoint['theta_star'],
            **checkpoint['metadata']
        )
        
        # Update config if saved
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        
        self.logger.info(f"Fisher matrix loaded from {filepath}")
    
    def reset(self):
        """Reset Fisher matrix and statistics."""
        self.fisher_matrix = None
        self.fisher_history.clear()
        self.total_computations = 0
        self.total_regularization_calls = 0
        self.last_compute_time = 0.0
        self.logger.info("EWC regularizer reset")


# Utility functions
def create_ewc_regularizer(
    num_samples: int = 1000,
    batch_size: int = 32,
    min_fisher_value: float = 1e-6,
    max_fisher_value: float = 1e3,
    smoothing_factor: float = 0.9
) -> EWCRegularizer:
    """Create EWC regularizer with specified configuration."""
    config = FisherConfig(
        num_samples=num_samples,
        batch_size=batch_size,
        min_fisher_value=min_fisher_value,
        max_fisher_value=max_fisher_value,
        smoothing_factor=smoothing_factor
    )
    return EWCRegularizer(config)


# Example usage for testing
if __name__ == "__main__":
    import torch.utils.data as data
    
    # Create dummy model and data
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
    
    # Create EWC regularizer
    ewc = create_ewc_regularizer()
    
    # Compute Fisher information
    fisher_matrix = ewc.compute_fisher_information(model, data_loader)
    
    print("Fisher matrix computed:")
    print(ewc.get_fisher_statistics())
    
    # Compute EWC loss
    ewc_loss = ewc.compute_ewc_loss(model, lambda_ewc=1.0)
    print(f"EWC loss: {ewc_loss.item():.6f}")