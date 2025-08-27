#!/usr/bin/env python3
"""
Simple BEM Demonstration

This script provides a minimal demonstration of the BEM concept:
1. Create a simple base model (linear layer)
2. Train two static LoRAs for different "tasks" 
3. Train a controller to interpolate between them
4. Show that the controller learns meaningful routing

This is a simplified version to demonstrate the core concept quickly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SimpleLoRA(nn.Module):
    """A simple LoRA implementation for demonstration."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        self.rank = rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.lora_A.T @ self.lora_B.T


class SimpleController(nn.Module):
    """Simple controller that predicts interpolation weights."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.mean(dim=1))  # Pool sequence dimension


class SimpleBEMDemo(nn.Module):
    """Demonstration BEM that interpolates between two LoRAs."""
    
    def __init__(self, input_dim: int, output_dim: int, rank: int = 4):
        super().__init__()
        
        # Base layer (frozen)
        self.base = nn.Linear(input_dim, output_dim)
        for param in self.base.parameters():
            param.requires_grad = False
            
        # Two task-specific LoRAs (frozen after training)
        self.lora_task_a = SimpleLoRA(input_dim, output_dim, rank)
        self.lora_task_b = SimpleLoRA(input_dim, output_dim, rank)
        
        # Controller (trainable)
        self.controller = SimpleController(input_dim)
        
    def forward(self, x: torch.Tensor, task_features: torch.Tensor) -> torch.Tensor:
        # Base output
        base_out = self.base(x)
        
        # Get interpolation weights
        weights = self.controller(task_features)
        
        # Apply LoRAs
        lora_a_out = self.lora_task_a(x)
        lora_b_out = self.lora_task_b(x)
        
        # Interpolate
        lora_out = weights[:, 0:1].unsqueeze(-1) * lora_a_out + \
                   weights[:, 1:2].unsqueeze(-1) * lora_b_out
        
        return base_out + lora_out, weights


def generate_demo_data(num_samples: int = 200) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate simple demonstration data for two tasks."""
    
    input_dim = 16
    seq_len = 10
    
    # Task A: inputs should produce positive outputs
    task_a_inputs = torch.randn(num_samples, seq_len, input_dim)
    task_a_targets = torch.abs(task_a_inputs.sum(dim=2, keepdim=True))  # Positive values
    task_a_features = torch.ones(num_samples, seq_len, input_dim) * 0.1  # Task identifier
    
    # Task B: inputs should produce negative outputs  
    task_b_inputs = torch.randn(num_samples, seq_len, input_dim)
    task_b_targets = -torch.abs(task_b_inputs.sum(dim=2, keepdim=True))  # Negative values
    task_b_features = torch.ones(num_samples, seq_len, input_dim) * -0.1  # Task identifier
    
    # Combine data
    inputs = torch.cat([task_a_inputs, task_b_inputs], dim=0)
    targets = torch.cat([task_a_targets, task_b_targets], dim=0)
    features = torch.cat([task_a_features, task_b_features], dim=0)
    
    # Task labels (0 = task A, 1 = task B)
    task_labels = torch.cat([
        torch.zeros(num_samples), 
        torch.ones(num_samples)
    ])
    
    return inputs, targets, features, task_labels


def train_static_loras(model: SimpleBEMDemo, inputs: torch.Tensor, targets: torch.Tensor, 
                      task_labels: torch.Tensor, epochs: int = 100):
    """Train the static LoRAs separately for each task."""
    
    print("Training static LoRAs...")
    
    # Train Task A LoRA
    task_a_mask = task_labels == 0
    task_a_inputs = inputs[task_a_mask]
    task_a_targets = targets[task_a_mask]
    
    optimizer_a = torch.optim.Adam(model.lora_task_a.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        optimizer_a.zero_grad()
        
        # Forward pass with only task A LoRA
        base_out = model.base(task_a_inputs)
        lora_out = model.lora_task_a(task_a_inputs)
        output = base_out + lora_out
        
        loss = F.mse_loss(output, task_a_targets)
        loss.backward()
        optimizer_a.step()
        
        if epoch % 20 == 0:
            print(f"  Task A LoRA - Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Train Task B LoRA
    task_b_mask = task_labels == 1
    task_b_inputs = inputs[task_b_mask]
    task_b_targets = targets[task_b_mask]
    
    optimizer_b = torch.optim.Adam(model.lora_task_b.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        optimizer_b.zero_grad()
        
        # Forward pass with only task B LoRA
        base_out = model.base(task_b_inputs)
        lora_out = model.lora_task_b(task_b_inputs)
        output = base_out + lora_out
        
        loss = F.mse_loss(output, task_b_targets)
        loss.backward()
        optimizer_b.step()
        
        if epoch % 20 == 0:
            print(f"  Task B LoRA - Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Freeze LoRA parameters
    for param in model.lora_task_a.parameters():
        param.requires_grad = False
    for param in model.lora_task_b.parameters():
        param.requires_grad = False
    
    print("✓ Static LoRAs trained and frozen")


def train_controller(model: SimpleBEMDemo, inputs: torch.Tensor, targets: torch.Tensor,
                    features: torch.Tensor, task_labels: torch.Tensor, epochs: int = 200):
    """Train the controller to predict correct interpolation weights."""
    
    print("\nTraining BEM controller...")
    
    optimizer = torch.optim.Adam(model.controller.parameters(), lr=0.001)
    
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs, weights = model(inputs, features)
        
        # Task loss: minimize prediction error
        task_loss = F.mse_loss(outputs, targets)
        
        # Controller loss: learn correct routing
        target_weights = torch.zeros_like(weights)
        task_a_mask = task_labels == 0
        task_b_mask = task_labels == 1
        
        target_weights[task_a_mask, 0] = 1.0  # Task A should use LoRA A
        target_weights[task_b_mask, 1] = 1.0  # Task B should use LoRA B
        
        controller_loss = F.mse_loss(weights, target_weights)
        
        # Combine losses
        total_loss = task_loss + controller_loss
        total_loss.backward()
        optimizer.step()
        
        # Track metrics
        losses.append(total_loss.item())
        
        # Accuracy: correct task routing
        predicted_tasks = torch.argmax(weights, dim=1)
        accuracy = (predicted_tasks == task_labels).float().mean().item()
        accuracies.append(accuracy)
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}, Total Loss: {total_loss.item():.4f}, "
                  f"Task Loss: {task_loss.item():.4f}, "
                  f"Controller Loss: {controller_loss.item():.4f}, "
                  f"Accuracy: {accuracy:.3f}")
    
    print(f"✓ Controller training completed - Final accuracy: {accuracies[-1]:.3f}")
    
    return losses, accuracies


def evaluate_bem(model: SimpleBEMDemo, inputs: torch.Tensor, features: torch.Tensor, 
                task_labels: torch.Tensor) -> Dict:
    """Evaluate the trained BEM model."""
    
    print("\nEvaluating BEM performance...")
    
    model.eval()
    with torch.no_grad():
        outputs, weights = model(inputs, features)
        
        # Task specialization analysis
        task_a_mask = task_labels == 0
        task_b_mask = task_labels == 1
        
        task_a_weights = weights[task_a_mask].numpy()
        task_b_weights = weights[task_b_mask].numpy()
        
        # Accuracy metrics
        predicted_tasks = torch.argmax(weights, dim=1)
        overall_accuracy = (predicted_tasks == task_labels).float().mean().item()
        
        task_a_accuracy = (predicted_tasks[task_a_mask] == 0).float().mean().item()
        task_b_accuracy = (predicted_tasks[task_b_mask] == 1).float().mean().item()
        
        results = {
            'overall_accuracy': overall_accuracy,
            'task_a_accuracy': task_a_accuracy,
            'task_b_accuracy': task_b_accuracy,
            'task_a_weights_mean': task_a_weights.mean(axis=0),
            'task_b_weights_mean': task_b_weights.mean(axis=0),
            'task_a_weights': task_a_weights,
            'task_b_weights': task_b_weights
        }
        
        print(f"Overall Accuracy: {overall_accuracy:.3f}")
        print(f"Task A Accuracy: {task_a_accuracy:.3f}")
        print(f"Task B Accuracy: {task_b_accuracy:.3f}")
        print(f"Task A Weights (mean): {results['task_a_weights_mean']}")
        print(f"Task B Weights (mean): {results['task_b_weights_mean']}")
        
    return results


def visualize_results(losses: List[float], accuracies: List[float], eval_results: Dict):
    """Create visualizations of the training and evaluation results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Simple BEM Demonstration Results', fontsize=16)
    
    # Training loss
    axes[0, 0].plot(losses)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Training accuracy
    axes[0, 1].plot(accuracies)
    axes[0, 1].set_title('Controller Accuracy During Training')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True)
    
    # Weight distribution
    task_a_weights = eval_results['task_a_weights']
    task_b_weights = eval_results['task_b_weights']
    
    axes[1, 0].hist(task_a_weights[:, 0], alpha=0.7, label='Task A → LoRA A', bins=20)
    axes[1, 0].hist(task_b_weights[:, 0], alpha=0.7, label='Task B → LoRA A', bins=20)
    axes[1, 0].set_title('LoRA A Weight Distribution')
    axes[1, 0].set_xlabel('Weight')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Specialization visualization
    means_a = eval_results['task_a_weights_mean']
    means_b = eval_results['task_b_weights_mean']
    
    x = ['LoRA A Weight', 'LoRA B Weight']
    width = 0.35
    x_pos = np.arange(len(x))
    
    axes[1, 1].bar(x_pos - width/2, means_a, width, label='Task A', alpha=0.8)
    axes[1, 1].bar(x_pos + width/2, means_b, width, label='Task B', alpha=0.8)
    axes[1, 1].set_title('Mean Interpolation Weights by Task')
    axes[1, 1].set_ylabel('Weight')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(x)
    axes[1, 1].legend()
    axes[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('simple_bem_demo_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Results visualization saved as 'simple_bem_demo_results.png'")


def main():
    """Run the simple BEM demonstration."""
    
    print("=" * 60)
    print("Simple BEM Demonstration")
    print("Proving controller-based LoRA interpolation")
    print("=" * 60)
    
    # Parameters
    input_dim = 16
    output_dim = 1
    rank = 4
    num_samples = 200
    
    # Generate data
    print("\n1. Generating demonstration data...")
    inputs, targets, features, task_labels = generate_demo_data(num_samples)
    print(f"   Generated {len(inputs)} samples ({num_samples} per task)")
    
    # Create model
    print("\n2. Creating BEM model...")
    model = SimpleBEMDemo(input_dim, output_dim, rank)
    print(f"   Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train static LoRAs
    print("\n3. Training static LoRAs...")
    train_static_loras(model, inputs, targets, task_labels, epochs=100)
    
    # Train controller
    print("\n4. Training BEM controller...")
    losses, accuracies = train_controller(model, inputs, targets, features, task_labels, epochs=200)
    
    # Evaluate
    print("\n5. Evaluating trained BEM...")
    eval_results = evaluate_bem(model, inputs, features, task_labels)
    
    # Visualize
    print("\n6. Creating visualizations...")
    visualize_results(losses, accuracies, eval_results)
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMONSTRATION SUMMARY")
    print("=" * 60)
    
    success = eval_results['overall_accuracy'] > 0.8
    
    print(f"Controller Accuracy: {eval_results['overall_accuracy']:.1%}")
    print(f"Task A Specialization: {eval_results['task_a_accuracy']:.1%}")  
    print(f"Task B Specialization: {eval_results['task_b_accuracy']:.1%}")
    
    if success:
        print("\n✅ DEMONSTRATION SUCCESSFUL!")
        print("The controller successfully learned to route between LoRAs based on task context.")
        print("This validates the core BEM hypothesis with a simple example.")
    else:
        print("\n❌ DEMONSTRATION FAILED!")
        print("The controller did not learn effective routing. This might indicate:")
        print("- Need for different architecture or hyperparameters")
        print("- Insufficient task differentiation in the synthetic data")
        print("- Training instability or convergence issues")
    
    print(f"\nCore concept validated: {'YES' if success else 'NO'}")
    print("Ready for full validation experiment: run_validation_experiment.py")


if __name__ == "__main__":
    main()