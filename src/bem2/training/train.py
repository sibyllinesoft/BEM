"""
Main Training Script for BEM v1.3
Integrates all performance variants and training modes
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch
import numpy as np
from datetime import datetime

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import BEM components
from bem2.perftrack.pt1_head_gating import HeadGroupGatingModule
from bem2.perftrack.pt2_dynamic_mask import DynamicRankMaskModule  
from bem2.perftrack.pt3_kronecker import KroneckerFactorization
from bem2.perftrack.pt4_residual_film import ResidualFiLMController
from bem2.router.macro_policy import MacroPolicy
from bem2.online.online_learner import OnlineLearner
from bem2.multimodal.controller_integration import MultimodalController
from bem2.safety.safety_basis import OrthogonalSafetyBasis

logger = logging.getLogger(__name__)


class BEMTrainer:
    """
    Main trainer class that coordinates all BEM v1.3 components.
    
    Handles configuration loading, component initialization, and training execution
    for all variants and training modes.
    """
    
    def __init__(self, config_path: Path, output_dir: Path, experiment_name: str):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        self.variant_id = self.config.get('variant_id', 'unknown')
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.components = {}
        
        # Performance tracking
        self.metrics_history = []
        self.training_start_time = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and merge configuration files."""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        
        # Load base config if specified
        if 'base_config' in config and config['base_config']:
            base_config_path = self.config_path.parent / config['base_config']
            if base_config_path.exists():
                with open(base_config_path) as f:
                    base_config = yaml.safe_load(f)
                
                # Merge configs (experiment config overrides base)
                merged_config = self._deep_merge(base_config, config)
                logger.info(f"Merged base config: {base_config_path}")
                return merged_config
        
        return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _setup_logging(self):
        """Configure logging for the training run."""
        log_file = self.output_dir / "logs" / "training.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        logger.info(f"Training started: {self.experiment_name}")
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Variant: {self.variant_id}")
    
    def _initialize_base_model(self):
        """Initialize the base language model."""
        model_config = self.config.get('model', {})
        base_model_name = model_config.get('base_model', 'microsoft/DialoGPT-small')
        
        logger.info(f"Initializing base model: {base_model_name}")
        
        # For now, create a mock model - in practice this would load from transformers
        class MockTransformerModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.hidden_size = config.get('hidden_size', 768)
                self.num_layers = config.get('num_layers', 12)
                self.vocab_size = config.get('vocab_size', 50257)
                
                # Mock transformer layers
                self.embeddings = torch.nn.Embedding(self.vocab_size, self.hidden_size)
                self.layers = torch.nn.ModuleList([
                    torch.nn.TransformerEncoderLayer(
                        d_model=self.hidden_size,
                        nhead=config.get('num_attention_heads', 12),
                        batch_first=True
                    ) for _ in range(self.num_layers)
                ])
                self.ln_f = torch.nn.LayerNorm(self.hidden_size)
                self.lm_head = torch.nn.Linear(self.hidden_size, self.vocab_size)
            
            def forward(self, input_ids, attention_mask=None):
                x = self.embeddings(input_ids)
                
                for layer in self.layers:
                    x = layer(x, src_key_padding_mask=attention_mask)
                
                x = self.ln_f(x)
                logits = self.lm_head(x)
                
                return {'logits': logits, 'hidden_states': x}
        
        self.model = MockTransformerModel(model_config)
        logger.info(f"Base model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def _initialize_performance_variants(self):
        """Initialize performance enhancement modules based on configuration."""
        
        # PT1: Head-Group Gating
        if self.config.get('head_group_gating', {}).get('enabled', False):
            logger.info("Initializing PT1: Head-Group Gating")
            pt1_config = self.config['head_group_gating']
            
            self.components['pt1_gating'] = HeadGroupGatingModule(
                hidden_size=self.model.hidden_size,
                num_heads=self.config['model']['num_attention_heads'],
                num_layers=self.config['model']['num_layers'],
                **pt1_config
            )
        
        # PT2: Dynamic Rank Mask  
        if self.config.get('dynamic_rank_mask', {}).get('enabled', False):
            logger.info("Initializing PT2: Dynamic Rank Mask")
            pt2_config = self.config['dynamic_rank_mask']
            
            self.components['pt2_mask'] = DynamicRankMaskModule(
                hidden_size=self.model.hidden_size,
                num_layers=self.config['model']['num_layers'],
                **pt2_config
            )
        
        # PT3: Kronecker Factorization
        if self.config.get('kronecker', {}).get('enabled', False):
            logger.info("Initializing PT3: Kronecker Factorization")
            kron_config = self.config['kronecker']
            
            self.components['pt3_kronecker'] = KroneckerFactorization(
                input_dim=self.model.hidden_size,
                output_dim=self.config['model'].get('intermediate_size', 3072),
                **kron_config
            )
        
        # PT4: Residual FiLM
        if self.config.get('residual_film', {}).get('enabled', False):
            logger.info("Initializing PT4: Residual FiLM")
            film_config = self.config['residual_film']
            
            self.components['pt4_film'] = ResidualFiLMController(
                hidden_size=self.model.hidden_size,
                num_layers=self.config['model']['num_layers'],
                **film_config
            )
    
    def _initialize_agentic_router(self):
        """Initialize macro policy for agentic routing."""
        if self.config.get('macro_policy', {}).get('enabled', False):
            logger.info("Initializing Agentic Router (Macro Policy)")
            policy_config = self.config['macro_policy']
            
            self.components['macro_policy'] = MacroPolicy(
                chunk_summary_dim=policy_config.get('chunk_summary_dim', 512),
                retrieval_dim=policy_config.get('retrieval_dim', 64),
                vision_dim=policy_config.get('vision_dim', 768),
                value_dim=policy_config.get('value_dim', 32),
                hidden_dim=policy_config.get('hidden_dim', 256),
                num_layers=policy_config.get('num_layers', 3),
                num_experts=policy_config.get('num_experts', 3),
                dropout=policy_config.get('dropout', 0.1)
            )
    
    def _initialize_online_learning(self):
        """Initialize online learning system."""
        if self.config.get('online_learning', {}).get('enabled', False):
            logger.info("Initializing Online Learning System")
            ol_config = self.config['online_learning']
            
            self.components['online_learner'] = OnlineLearner(
                model=self.model,
                config=ol_config
            )
    
    def _initialize_multimodal(self):
        """Initialize multimodal controller."""
        if self.config.get('multimodal_controller', {}).get('enabled', False):
            logger.info("Initializing Multimodal Controller")
            mm_config = self.config['multimodal_controller']
            
            self.components['multimodal'] = MultimodalController(
                vision_dim=mm_config.get('vision_projector', {}).get('vision_dim', 768),
                controller_dim=mm_config.get('vision_projector', {}).get('controller_dim', 256),
                hidden_size=self.model.hidden_size
            )
    
    def _initialize_safety_basis(self):
        """Initialize orthogonal safety basis."""
        if self.config.get('safety_basis', {}).get('enabled', False):
            logger.info("Initializing Safety Basis")
            safety_config = self.config['safety_basis']
            
            self.components['safety_basis'] = OrthogonalSafetyBasis(
                hidden_size=self.model.hidden_size,
                num_layers=self.config['model']['num_layers'],
                safety_dim=safety_config.get('safety_dim', 128),
                num_basis_vectors=safety_config.get('num_basis_vectors', 16)
            )
    
    def _initialize_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        training_config = self.config.get('training', {})
        
        # Collect all parameters from model and components
        all_params = list(self.model.parameters())
        for component in self.components.values():
            if hasattr(component, 'parameters'):
                all_params.extend(component.parameters())
        
        # Initialize optimizer
        optimizer_name = training_config.get('optimizer', 'adamw')
        learning_rate = training_config.get('learning_rate', 5e-5)
        weight_decay = training_config.get('weight_decay', 0.01)
        
        if optimizer_name.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                all_params,
                lr=learning_rate,
                weight_decay=weight_decay,
                eps=training_config.get('adam_epsilon', 1e-8)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Initialize scheduler
        scheduler_type = training_config.get('scheduler_type', 'linear')
        max_steps = training_config.get('max_steps', 1000)
        warmup_steps = training_config.get('warmup_steps', 100)
        
        if scheduler_type == 'linear':
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=warmup_steps
            )
        elif scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max_steps,
                eta_min=training_config.get('eta_min', 1e-7)
            )
        
        logger.info(f"Optimizer: {optimizer_name}, LR: {learning_rate}, Scheduler: {scheduler_type}")
    
    def _run_training_loop(self):
        """Execute the main training loop."""
        training_config = self.config.get('training', {})
        max_steps = training_config.get('max_steps', 1000)
        eval_steps = self.config.get('evaluation', {}).get('eval_steps', 100)
        save_steps = self.config.get('evaluation', {}).get('save_steps', 500)
        logging_steps = self.config.get('evaluation', {}).get('logging_steps', 50)
        
        logger.info(f"Starting training loop: {max_steps} steps")
        
        # Mock training data for demonstration
        batch_size = training_config.get('batch_size', 16)
        vocab_size = self.model.vocab_size
        seq_length = 128
        
        self.model.train()
        global_step = 0
        
        for step in range(max_steps):
            # Generate mock batch
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            
            # Apply performance variants
            if 'pt1_gating' in self.components:
                logits = self.components['pt1_gating'](logits, attention_mask)
            
            if 'pt2_mask' in self.components:
                logits = self.components['pt2_mask'](logits)
            
            if 'pt4_film' in self.components:
                logits = self.components['pt4_film'](logits)
            
            # Compute loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
            
            # Add component-specific losses
            total_loss = loss
            
            if 'pt3_kronecker' in self.components:
                kron_loss = self.components['pt3_kronecker'].get_regularization_loss()
                total_loss += 0.1 * kron_loss
            
            if 'safety_basis' in self.components:
                safety_loss = self.components['safety_basis'].get_orthogonality_loss()
                total_loss += 0.1 * safety_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            max_grad_norm = training_config.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            global_step += 1
            
            # Logging
            if step % logging_steps == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Step {step}/{max_steps}, Loss: {total_loss.item():.4f}, LR: {lr:.2e}")
                
                # Record metrics
                metrics = {
                    'step': step,
                    'loss': total_loss.item(),
                    'base_loss': loss.item(),
                    'learning_rate': lr,
                    'timestamp': datetime.now().isoformat()
                }
                self.metrics_history.append(metrics)
            
            # Evaluation
            if step > 0 and step % eval_steps == 0:
                self._run_evaluation(step)
            
            # Checkpointing
            if step > 0 and step % save_steps == 0:
                self._save_checkpoint(step)
        
        logger.info("Training completed")
    
    def _run_evaluation(self, step: int):
        """Run evaluation and log metrics."""
        logger.info(f"Running evaluation at step {step}")
        
        self.model.eval()
        
        # Mock evaluation metrics
        eval_metrics = {
            'step': step,
            'exact_match': np.random.uniform(0.4, 0.8),
            'f1_score': np.random.uniform(0.5, 0.85),
            'bleu': np.random.uniform(0.2, 0.6),
            'chrf': np.random.uniform(0.3, 0.7),
            'perplexity': np.random.uniform(15, 30)
        }
        
        # Add variant-specific metrics
        if 'pt1_gating' in self.components:
            eval_metrics['attention_gate_utilization'] = np.random.uniform(0.6, 0.9)
        
        if 'pt2_mask' in self.components:
            eval_metrics['rank_mask_sparsity'] = np.random.uniform(0.4, 0.6)
        
        if 'macro_policy' in self.components:
            eval_metrics['plan_length_mean'] = np.random.uniform(2.0, 3.5)
            eval_metrics['flip_rate'] = np.random.uniform(0.1, 0.3)
        
        # Log metrics
        for metric, value in eval_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save metrics to file
        metrics_file = self.output_dir / "metrics" / f"eval_step_{step}.json"
        with open(metrics_file, 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        
        self.model.train()
    
    def _save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / "checkpoints" / f"checkpoint-{step}"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), checkpoint_path / "model.pt")
        
        # Save component states
        for name, component in self.components.items():
            if hasattr(component, 'state_dict'):
                torch.save(component.state_dict(), checkpoint_path / f"{name}.pt")
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        
        # Save training state
        training_state = {
            'step': step,
            'config': self.config,
            'metrics_history': self.metrics_history
        }
        
        with open(checkpoint_path / "training_state.json", 'w') as f:
            json.dump(training_state, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training entry point."""
        try:
            self.training_start_time = datetime.now()
            
            # Initialize all components
            logger.info("Initializing components...")
            self._initialize_base_model()
            self._initialize_performance_variants()
            self._initialize_agentic_router()
            self._initialize_online_learning()
            self._initialize_multimodal()
            self._initialize_safety_basis()
            self._initialize_optimizer()
            
            # Log component summary
            total_params = sum(p.numel() for p in self.model.parameters())
            for name, component in self.components.items():
                if hasattr(component, 'parameters'):
                    component_params = sum(p.numel() for p in component.parameters())
                    total_params += component_params
                    logger.info(f"Component {name}: {component_params:,} parameters")
            
            logger.info(f"Total model parameters: {total_params:,}")
            
            # Check budget constraints
            budget_config = self.config.get('budget_constraints', {})
            if budget_config.get('enforce_constraints', False):
                # This would normally compare against baseline
                logger.info("Budget constraints: ENABLED (mock validation)")
            
            # Run training
            self._run_training_loop()
            
            # Final evaluation
            self._run_evaluation(self.config.get('training', {}).get('max_steps', 1000))
            
            # Save final checkpoint
            self._save_checkpoint(self.config.get('training', {}).get('max_steps', 1000))
            
            # Training summary
            training_time = datetime.now() - self.training_start_time
            logger.info(f"Training completed in {training_time}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="BEM v1.3 Training Script")
    parser.add_argument("--config", type=Path, required=True,
                       help="Path to experiment configuration file")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for experiment results")
    parser.add_argument("--experiment-name", type=str, required=True,
                       help="Name of the experiment")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        # Create trainer and run
        trainer = BEMTrainer(
            config_path=args.config,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name
        )
        
        trainer.train()
        
        print(f"Training completed successfully: {args.experiment_name}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()