"""
Training Components for Agentic Router

Implements both Behavior Cloning (BC) and Policy Gradient (PG) training:
- BCTrainer: Supervised learning from synthetic traces
- PGTrainer: Policy gradient with trust-region constraints (TRPO/PPO-style)
- Training loops with proper evaluation and checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, NamedTuple
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from .macro_policy import MacroPolicy, MacroAction, MacroPolicyState
from .agentic_router import AgenticRouter
from .trace_generator import SyntheticTrace, TraceGenerator

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for router training."""
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-5
    eval_every: int = 500
    save_every: int = 1000
    log_every: int = 100


class BCDataset(Dataset):
    """Dataset for Behavior Cloning training."""
    
    def __init__(self, traces: List[SyntheticTrace]):
        self.traces = traces
        self.samples = []
        
        # Flatten traces into individual (state, action) pairs
        for trace in traces:
            for i, (state_dict, action_dict) in enumerate(zip(trace.states, trace.actions)):
                self.samples.append((state_dict, action_dict, trace.rewards[i]))
        
        logger.info(f"Created BC dataset with {len(self.samples)} samples from {len(traces)} traces")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        state_dict, action_dict, reward = self.samples[idx]
        
        # Convert state to tensors
        state_tensors = {}
        for key, value in state_dict.items():
            if value is not None:
                state_tensors[key] = torch.tensor(value, dtype=torch.float32)
            else:
                state_tensors[key] = None
        
        # Convert action to tensor
        action_tensor = torch.tensor([
            action_dict['expert_id'],
            1.0 if action_dict['scope'] == 'global' else 0.0,
            action_dict['span'],
            action_dict['rank_budget'],
            action_dict['bias_scale']
        ], dtype=torch.float32)
        
        return {
            'state': state_tensors,
            'action': action_tensor,
            'reward': torch.tensor(reward, dtype=torch.float32)
        }


class BCTrainer:
    """
    Behavior Cloning trainer for macro-policy.
    
    Trains the policy to imitate synthetic expert traces using supervised learning.
    """
    
    def __init__(
        self,
        policy: MacroPolicy,
        config: TrainingConfig,
        device: torch.device = None
    ):
        self.policy = policy
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move policy to device
        self.policy.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=config.warmup_steps
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        logger.info(f"Initialized BCTrainer on device {self.device}")
    
    def train(
        self,
        train_traces: List[SyntheticTrace],
        eval_traces: Optional[List[SyntheticTrace]] = None,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Train the policy using behavior cloning.
        
        Args:
            train_traces: Training traces
            eval_traces: Evaluation traces
            output_dir: Directory to save checkpoints and logs
            
        Returns:
            Training metrics
        """
        # Create datasets
        train_dataset = BCDataset(train_traces)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # No multiprocessing for simplicity
        )
        
        eval_loader = None
        if eval_traces:
            eval_dataset = BCDataset(eval_traces)
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        
        # Training loop
        training_losses = []
        eval_losses = []
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Training epoch
            epoch_losses = self._train_epoch(train_loader)
            training_losses.extend(epoch_losses)
            
            # Evaluation
            if eval_loader is not None:
                eval_loss = self._evaluate(eval_loader)
                eval_losses.append(eval_loss)
                
                # Save best model
                if eval_loss < self.best_eval_loss and output_dir:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint(output_dir, 'best_model.pt')
                
                logger.info(f"Epoch {epoch}: train_loss={np.mean(epoch_losses):.4f}, "
                          f"eval_loss={eval_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch}: train_loss={np.mean(epoch_losses):.4f}")
            
            # Save regular checkpoint
            if output_dir and epoch % 5 == 0:
                self._save_checkpoint(output_dir, f'checkpoint_epoch_{epoch}.pt')
        
        # Final checkpoint
        if output_dir:
            self._save_checkpoint(output_dir, 'final_model.pt')
        
        return {
            'training_losses': training_losses,
            'eval_losses': eval_losses,
            'best_eval_loss': self.best_eval_loss,
            'total_steps': self.step
        }
    
    def _train_epoch(self, data_loader: DataLoader) -> List[float]:
        """Train for one epoch."""
        self.policy.train()
        epoch_losses = []
        
        pbar = tqdm(data_loader, desc=f"Training Epoch {self.epoch}")
        for batch in pbar:
            loss = self._train_step(batch)
            epoch_losses.append(loss)
            
            if self.step % self.config.log_every == 0:
                pbar.set_postfix({'loss': f'{loss:.4f}', 'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'})
            
            self.step += 1
        
        return epoch_losses
    
    def _train_step(self, batch: Dict) -> float:
        """Single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Prepare state
        state = self._prepare_state(batch['state'])
        target_actions = batch['action']  # [batch_size, 5]
        
        # Forward pass
        policy_output = self.policy(state, sample=False)
        predicted_actions = policy_output['actions']  # [batch_size, 5]
        
        # Compute loss (MSE for continuous actions, CE for discrete)
        loss = self._compute_bc_loss(predicted_actions, target_actions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        # Update scheduler
        if self.step < self.config.warmup_steps:
            self.scheduler.step()
        
        return loss.item()
    
    def _prepare_state(self, state_dict: Dict) -> MacroPolicyState:
        """Prepare MacroPolicyState from batch dictionary."""
        return MacroPolicyState(
            chunk_summary=state_dict['chunk_summary'],
            retrieval_features=state_dict['retrieval_features'],
            vision_features=state_dict['vision_features'],
            value_features=state_dict['value_features'],
            prev_action=state_dict['prev_action'],
            chunk_index=state_dict['chunk_index']
        )
    
    def _compute_bc_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute behavior cloning loss."""
        # Separate discrete and continuous components
        # [0] = expert_id (discrete), [1] = scope (discrete), [2] = span (discrete)  
        # [3] = rank_budget (discrete), [4] = bias_scale (continuous)
        
        # Discrete losses (cross-entropy style)
        expert_loss = F.mse_loss(predicted[:, 0], target[:, 0])  # Treat as regression for simplicity
        scope_loss = F.mse_loss(predicted[:, 1], target[:, 1])
        span_loss = F.mse_loss(predicted[:, 2], target[:, 2])
        rank_loss = F.mse_loss(predicted[:, 3], target[:, 3])
        
        # Continuous loss
        bias_loss = F.mse_loss(predicted[:, 4], target[:, 4])
        
        # Weighted combination
        total_loss = (
            2.0 * expert_loss +  # Expert choice most important
            1.0 * scope_loss +
            1.0 * span_loss +
            1.0 * rank_loss +
            1.0 * bias_loss
        )
        
        return total_loss
    
    def _evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate the model."""
        self.policy.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                state = self._prepare_state(batch['state'])
                target_actions = batch['action']
                
                policy_output = self.policy(state, sample=False)
                predicted_actions = policy_output['actions']
                
                loss = self._compute_bc_loss(predicted_actions, target_actions)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def _save_checkpoint(self, output_dir: str, filename: str):
        """Save model checkpoint."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'best_eval_loss': self.best_eval_loss,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, output_path / filename)
        logger.info(f"Saved checkpoint: {output_path / filename}")


class PGExperience(NamedTuple):
    """Experience tuple for policy gradient training."""
    states: torch.Tensor
    actions: torch.Tensor  
    log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor


class PGTrainer:
    """
    Policy Gradient trainer with trust-region constraints.
    
    Implements PPO-style policy gradient with KL divergence constraints
    and trust-region projection for safe updates.
    """
    
    def __init__(
        self,
        router: AgenticRouter,
        config: TrainingConfig,
        device: torch.device = None,
        kl_target: float = 0.01,
        clip_ratio: float = 0.2
    ):
        self.router = router
        self.policy = router.macro_policy
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kl_target = kl_target
        self.clip_ratio = clip_ratio
        
        # Move to device
        self.router.to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Training state
        self.step = 0
        self.episode = 0
        self.kl_divergences = []
        self.policy_losses = []
        self.value_losses = []
        
        logger.info(f"Initialized PGTrainer on device {self.device}")
    
    def train(
        self,
        environment: 'TaskEnvironment',  # Would be actual task environment
        num_episodes: int = 1000,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Train using policy gradient.
        
        Args:
            environment: Task environment for generating rewards
            num_episodes: Number of training episodes
            output_dir: Directory for checkpoints
            
        Returns:
            Training metrics
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            self.episode = episode
            
            # Collect experience
            experience = self._collect_experience(environment)
            
            # Update policy
            metrics = self._update_policy(experience)
            
            # Track metrics
            episode_reward = experience.rewards.sum().item()
            episode_rewards.append(episode_reward)
            
            if episode % self.config.log_every == 0:
                logger.info(f"Episode {episode}: reward={episode_reward:.3f}, "
                          f"policy_loss={metrics.get('policy_loss', 0):.4f}, "
                          f"kl_div={metrics.get('kl_divergence', 0):.4f}")
            
            # Save checkpoints
            if output_dir and episode % self.config.save_every == 0:
                self._save_checkpoint(output_dir, f'checkpoint_episode_{episode}.pt')
        
        # Final checkpoint
        if output_dir:
            self._save_checkpoint(output_dir, 'final_policy.pt')
        
        return {
            'episode_rewards': episode_rewards,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'kl_divergences': self.kl_divergences,
            'total_steps': self.step
        }
    
    def _collect_experience(self, environment: 'TaskEnvironment') -> PGExperience:
        """Collect experience from environment."""
        # This would interact with actual task environment
        # For now, create dummy experience
        
        batch_size = self.config.batch_size
        seq_len = 10  # Number of routing steps
        
        # Generate dummy input
        input_ids = torch.randint(0, 1000, (batch_size, 128 * seq_len), device=self.device)
        
        # Run router to get experience
        self.router.train()
        states_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        values_list = []
        
        # Simulate experience collection
        for step in range(seq_len):
            # Generate dummy state
            state = MacroPolicyState(
                chunk_summary=torch.randn(batch_size, 512, device=self.device),
                retrieval_features=torch.randn(batch_size, 64, device=self.device),
                vision_features=torch.zeros(batch_size, 768, device=self.device),
                value_features=torch.randn(batch_size, 32, device=self.device),
                prev_action=None,
                chunk_index=torch.full((batch_size,), step, dtype=torch.float32, device=self.device)
            )
            
            # Get policy output
            policy_output = self.policy(state, sample=True)
            
            # Extract components
            actions = policy_output['actions']
            log_probs = policy_output['log_probs']
            values = policy_output['state_values']
            
            # Generate dummy rewards (would come from task environment)
            rewards = torch.randn(batch_size, device=self.device) * 0.1
            
            states_list.append(state)
            actions_list.append(actions)
            log_probs_list.append(log_probs)
            rewards_list.append(rewards)
            values_list.append(values)
        
        # Stack into tensors
        actions_tensor = torch.stack(actions_list, dim=1)  # [batch, seq_len, action_dim]
        log_probs_tensor = torch.stack(log_probs_list, dim=1)  # [batch, seq_len]
        rewards_tensor = torch.stack(rewards_list, dim=1)  # [batch, seq_len]
        values_tensor = torch.stack(values_list, dim=1)  # [batch, seq_len]
        
        # Compute advantages using GAE
        advantages = self._compute_advantages(rewards_tensor, values_tensor)
        
        return PGExperience(
            states=states_list,  # Keep as list for now
            actions=actions_tensor,
            log_probs=log_probs_tensor,
            rewards=rewards_tensor,
            values=values_tensor,
            advantages=advantages
        )
    
    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        lambda_gae: float = 0.95
    ) -> torch.Tensor:
        """Compute advantages using Generalized Advantage Estimation."""
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        
        # Add bootstrap value (assume 0 for terminal states)
        next_values = torch.cat([values[:, 1:], torch.zeros(batch_size, 1, device=values.device)], dim=1)
        
        # Compute TD errors
        deltas = rewards + gamma * next_values - values
        
        # Compute GAE advantages
        advantage = 0
        for t in reversed(range(seq_len)):
            advantage = deltas[:, t] + gamma * lambda_gae * advantage
            advantages[:, t] = advantage
        
        return advantages
    
    def _update_policy(self, experience: PGExperience) -> Dict:
        """Update policy using collected experience."""
        # Flatten experience for batch processing
        batch_size, seq_len = experience.actions.shape[:2]
        
        # Flatten tensors
        actions_flat = experience.actions.view(-1, experience.actions.shape[-1])
        log_probs_old = experience.log_probs.view(-1)
        rewards_flat = experience.rewards.view(-1)
        advantages_flat = experience.advantages.view(-1)
        
        # Normalize advantages
        advantages_normalized = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        # Multiple update epochs
        policy_losses = []
        value_losses = []
        kl_divs = []
        
        for epoch in range(4):  # PPO-style multiple epochs
            # Re-evaluate states to get current policy outputs
            policy_losses_epoch = []
            
            for step in range(seq_len):
                states_step = experience.states[step]
                
                # Get current policy output
                policy_output = self.policy(states_step, sample=False)
                log_probs_new = policy_output['log_probs']
                values_new = policy_output['state_values']
                
                # Compute policy loss
                step_start = step * batch_size
                step_end = (step + 1) * batch_size
                
                ratio = torch.exp(log_probs_new - log_probs_old[step_start:step_end])
                advantages_step = advantages_normalized[step_start:step_end]
                
                # PPO clipped objective
                surr1 = ratio * advantages_step
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_step
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_targets = rewards_flat[step_start:step_end] + advantages_step
                value_loss = F.mse_loss(values_new, value_targets)
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss
                
                # Update
                self.policy_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()
                
                policy_losses_epoch.append(policy_loss.item())
                
                # Compute KL divergence
                kl_div = (log_probs_old[step_start:step_end] - log_probs_new).mean()
                kl_divs.append(kl_div.item())
            
            policy_losses.extend(policy_losses_epoch)
            
            # Early stopping if KL divergence too high
            if np.mean(kl_divs[-batch_size:]) > 1.5 * self.kl_target:
                logger.warning(f"Early stopping due to high KL divergence: {np.mean(kl_divs[-batch_size:]):.4f}")
                break
        
        # Update tracking
        self.policy_losses.extend(policy_losses)
        self.kl_divergences.extend(kl_divs)
        self.step += len(policy_losses)
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses) if value_losses else 0,
            'kl_divergence': np.mean(kl_divs)
        }
    
    def _save_checkpoint(self, output_dir: str, filename: str):
        """Save training checkpoint."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'router_state_dict': self.router.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'step': self.step,
            'episode': self.episode,
            'config': self.config.__dict__,
            'kl_target': self.kl_target,
            'clip_ratio': self.clip_ratio
        }
        
        torch.save(checkpoint, output_path / filename)
        logger.info(f"Saved PG checkpoint: {output_path / filename}")


def create_bc_trainer(
    policy: MacroPolicy,
    config: Dict,
    device: Optional[torch.device] = None
) -> BCTrainer:
    """Factory function for BC trainer."""
    training_config = TrainingConfig(
        batch_size=config.get('batch_size', 32),
        learning_rate=config.get('learning_rate', 3e-4),
        num_epochs=config.get('num_epochs', 10),
        warmup_steps=config.get('warmup_steps', 1000),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    return BCTrainer(policy, training_config, device)


def create_pg_trainer(
    router: AgenticRouter,
    config: Dict,
    device: Optional[torch.device] = None
) -> PGTrainer:
    """Factory function for PG trainer."""
    training_config = TrainingConfig(
        batch_size=config.get('batch_size', 32),
        learning_rate=config.get('learning_rate', 1e-4),
        num_epochs=config.get('num_epochs', 1000),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    return PGTrainer(
        router,
        training_config,
        device,
        kl_target=config.get('kl_target', 0.01),
        clip_ratio=config.get('clip_ratio', 0.2)
    )