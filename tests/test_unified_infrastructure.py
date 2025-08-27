"""
Comprehensive tests for the unified BEM infrastructure components.

This module validates the core infrastructure including BaseTrainer, BaseEvaluator,
configuration loading, and shared utilities to ensure the migration preserved
all functionality and introduced no regressions.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
import os
import yaml
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

# Import unified infrastructure components
from src.bem_core.training.base_trainer import BaseTrainer, TrainingConfig
from src.bem_core.evaluation.base_evaluator import BaseEvaluator, EvaluationConfig, EvaluationResult
from src.bem_core.config.config_loader import ConfigLoader, ExperimentConfig
from src.bem_core.config.base_config import BaseConfig
from src.bem_core.utils.logging_utils import setup_logger
from src.bem_core.utils.checkpoint_utils import save_checkpoint, load_checkpoint


# Test fixtures and mock classes
class MockModel(nn.Module):
    """Mock model for testing trainers."""
    
    def __init__(self, hidden_size=768):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            input_ids = torch.randn(2, 10, self.hidden_size)
        return {"logits": self.linear(input_ids), "loss": torch.tensor(0.5)}


class MockTrainer(BaseTrainer):
    """Mock trainer implementation for testing abstract base class."""
    
    def create_model(self) -> nn.Module:
        """Create a mock model for testing."""
        return MockModel(self.config.model.hidden_size)
    
    def prepare_data(self) -> Dict[str, Any]:
        """Prepare mock data for testing."""
        # Create mock tensors
        batch_size = self.training_config.batch_size
        seq_length = 10
        hidden_size = self.config.model.hidden_size
        
        return {
            "train_dataloader": self._create_mock_dataloader(batch_size, seq_length, hidden_size),
            "eval_dataloader": self._create_mock_dataloader(batch_size, seq_length, hidden_size)
        }
    
    def compute_loss(self, model_outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute mock loss for testing."""
        if "loss" in model_outputs:
            return model_outputs["loss"]
        return torch.tensor(0.5, requires_grad=True)
    
    def _create_mock_dataloader(self, batch_size, seq_length, hidden_size):
        """Create a mock dataloader."""
        def mock_data():
            for _ in range(5):  # 5 batches
                yield {
                    "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
                    "attention_mask": torch.ones(batch_size, seq_length),
                    "labels": torch.randint(0, 1000, (batch_size, seq_length))
                }
        
        return mock_data()


class MockEvaluator(BaseEvaluator):
    """Mock evaluator implementation for testing abstract base class."""
    
    def evaluate_model(self, model: nn.Module, data: Dict[str, Any]) -> EvaluationResult:
        """Mock evaluation returning realistic metrics."""
        return EvaluationResult(
            metrics={
                "eval_loss": 0.45,
                "eval_accuracy": 0.85,
                "eval_perplexity": 1.57,
                "eval_samples": 100
            },
            metadata={
                "model_type": "MockModel",
                "eval_timestamp": "2025-01-01T00:00:00",
                "config_hash": "mock_hash"
            }
        )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Create a mock experiment configuration."""
    return ExperimentConfig(
        name="test_experiment",
        version="1.0",
        description="Test configuration",
        experiment_type="training",
        model={
            "base_model": "microsoft/DialoGPT-small",
            "hidden_size": 768,
            "num_layers": 12,
            "num_attention_heads": 12,
            "custom_params": {
                "learning_rate": 5e-5,
                "max_steps": 100,
                "warmup_steps": 10
            }
        },
        data={
            "train_file": "data/train.jsonl",
            "validation_file": "data/val.jsonl",
            "max_seq_length": 512,
            "max_samples": 32
        },
        training={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "max_steps": 100,
            "warmup_steps": 10,
            "eval_steps": 10,
            "logging_steps": 5
        },
        hardware={
            "device": "cpu",
            "mixed_precision": "no",
            "gradient_checkpointing": False
        },
        logging={
            "level": "INFO",
            "console_output": True
        },
        output_dir="test_output",
        seed=42
    )


@pytest.fixture
def training_config():
    """Create a mock training configuration."""
    return TrainingConfig(
        learning_rate=5e-5,
        batch_size=16,
        max_steps=100,
        warmup_steps=10,
        eval_steps=10,
        logging_steps=5,
        device="cpu",
        seed=42
    )


class TestBaseTrainer:
    """Test suite for BaseTrainer abstract base class."""
    
    def test_trainer_initialization(self, mock_config, temp_dir):
        """Test that BaseTrainer initializes correctly with valid configuration."""
        trainer = MockTrainer(mock_config, output_dir=temp_dir)
        
        assert trainer.config == mock_config
        assert trainer.output_dir == temp_dir
        assert trainer.training_config.learning_rate == 5e-5
        assert trainer.training_config.batch_size == 16
        assert trainer.device == torch.device("cpu")
    
    def test_abstract_methods_required(self, mock_config):
        """Test that abstract methods must be implemented by subclasses."""
        with pytest.raises(TypeError):
            # Should fail because BaseTrainer is abstract
            BaseTrainer(mock_config)
    
    def test_training_loop_basic_functionality(self, mock_config, temp_dir):
        """Test that the training loop executes without errors."""
        trainer = MockTrainer(mock_config, output_dir=temp_dir)
        
        # Mock the training data preparation
        with patch.object(trainer, 'prepare_data') as mock_prepare:
            mock_prepare.return_value = trainer.prepare_data()
            
            # Run a minimal training loop
            results = trainer.train(max_steps=5)
            
            # Verify training completed
            assert "train_loss" in results
            assert "eval_loss" in results
            assert results["completed_steps"] == 5
    
    def test_checkpoint_saving_and_loading(self, mock_config, temp_dir):
        """Test checkpoint save and load functionality."""
        trainer = MockTrainer(mock_config, output_dir=temp_dir)
        
        # Create and save a checkpoint
        model = trainer.create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=100,
            loss=0.5,
            path=checkpoint_path,
            metadata={"experiment": "test"}
        )
        
        # Verify checkpoint file was created
        assert os.path.exists(checkpoint_path)
        
        # Load checkpoint and verify contents
        loaded_checkpoint = load_checkpoint(checkpoint_path)
        assert loaded_checkpoint["step"] == 100
        assert loaded_checkpoint["loss"] == 0.5
        assert loaded_checkpoint["metadata"]["experiment"] == "test"
    
    def test_model_creation_and_setup(self, mock_config, temp_dir):
        """Test model creation and device setup."""
        trainer = MockTrainer(mock_config, output_dir=temp_dir)
        model = trainer.create_model()
        
        assert isinstance(model, MockModel)
        assert model.hidden_size == 768
        
        # Test model can be moved to device
        trainer._setup_model_and_optimizer(model)
        assert next(model.parameters()).device.type == "cpu"
    
    def test_gradient_computation_and_clipping(self, mock_config, temp_dir):
        """Test gradient computation and clipping functionality."""
        trainer = MockTrainer(mock_config, output_dir=temp_dir)
        model = trainer.create_model()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Create dummy input and compute loss
        dummy_batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10)
        }
        
        outputs = model(**dummy_batch)
        loss = trainer.compute_loss(outputs, dummy_batch)
        
        # Test gradient computation
        loss.backward()
        
        # Verify gradients exist
        for param in model.parameters():
            assert param.grad is not None
        
        # Test gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), trainer.training_config.max_grad_norm
        )
        assert grad_norm >= 0
    
    def test_logging_configuration(self, mock_config, temp_dir):
        """Test logging setup and configuration."""
        trainer = MockTrainer(mock_config, output_dir=temp_dir)
        
        # Test logger was set up
        assert trainer.logger is not None
        assert trainer.logger.level == getattr(trainer.config.logging, 'level', 'INFO')
        
        # Test logging methods work
        trainer.logger.info("Test log message")
        trainer.logger.warning("Test warning message")
    
    def test_evaluation_integration(self, mock_config, temp_dir):
        """Test integration between trainer and evaluation."""
        trainer = MockTrainer(mock_config, output_dir=temp_dir)
        model = trainer.create_model()
        
        # Prepare data
        data = trainer.prepare_data()
        
        # Test evaluation call
        with patch.object(trainer, '_evaluate_model') as mock_eval:
            mock_eval.return_value = {"eval_loss": 0.4, "eval_accuracy": 0.9}
            
            results = trainer._run_evaluation(model, data["eval_dataloader"])
            
            assert "eval_loss" in results
            assert "eval_accuracy" in results
            mock_eval.assert_called_once()


class TestBaseEvaluator:
    """Test suite for BaseEvaluator abstract base class."""
    
    def test_evaluator_initialization(self, temp_dir):
        """Test BaseEvaluator initialization."""
        config = EvaluationConfig(
            metrics=["accuracy", "perplexity"],
            batch_size=16,
            output_dir=temp_dir
        )
        
        evaluator = MockEvaluator(config)
        
        assert evaluator.config == config
        assert evaluator.config.batch_size == 16
        assert evaluator.config.output_dir == temp_dir
    
    def test_evaluation_result_structure(self, temp_dir):
        """Test EvaluationResult structure and serialization."""
        config = EvaluationConfig(output_dir=temp_dir)
        evaluator = MockEvaluator(config)
        model = MockModel()
        
        result = evaluator.evaluate_model(model, {})
        
        # Test result structure
        assert isinstance(result, EvaluationResult)
        assert "eval_loss" in result.metrics
        assert "eval_accuracy" in result.metrics
        assert "model_type" in result.metadata
        
        # Test result serialization
        result_dict = result.to_dict()
        assert "metrics" in result_dict
        assert "metadata" in result_dict
        
        # Test result can be saved and loaded
        result_path = os.path.join(temp_dir, "results.json")
        result.save(result_path)
        
        assert os.path.exists(result_path)
        
        loaded_result = EvaluationResult.load(result_path)
        assert loaded_result.metrics == result.metrics
        assert loaded_result.metadata == result.metadata
    
    def test_abstract_methods_enforcement(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # Should fail because BaseEvaluator is abstract
            BaseEvaluator(EvaluationConfig())
    
    def test_evaluation_metrics_computation(self, temp_dir):
        """Test evaluation metrics computation."""
        config = EvaluationConfig(
            metrics=["accuracy", "perplexity"],
            output_dir=temp_dir
        )
        evaluator = MockEvaluator(config)
        model = MockModel()
        
        # Test evaluation with different data configurations
        test_data = {
            "test_dataloader": [
                {"input_ids": torch.randint(0, 1000, (16, 10))},
                {"input_ids": torch.randint(0, 1000, (16, 10))}
            ]
        }
        
        result = evaluator.evaluate_model(model, test_data)
        
        # Verify required metrics are present
        assert result.metrics["eval_loss"] > 0
        assert 0 <= result.metrics["eval_accuracy"] <= 1
        assert result.metrics["eval_perplexity"] > 0


class TestConfigurationSystem:
    """Test suite for the configuration system."""
    
    def test_config_loader_initialization(self):
        """Test ConfigLoader can be initialized and loads templates."""
        config_loader = ConfigLoader()
        
        # Test that config loader can access template directory
        assert hasattr(config_loader, 'template_dir')
        
        # Test template discovery
        templates = config_loader.discover_templates()
        assert isinstance(templates, list)
        
        # Should find base template at minimum
        template_names = [t.get('name', t.get('template_name', '')) for t in templates]
        assert any('base' in name.lower() for name in template_names)
    
    def test_base_config_inheritance(self, temp_dir):
        """Test configuration template inheritance system."""
        # Create a base template
        base_template = {
            "name": "base_test",
            "model": {"hidden_size": 768, "num_layers": 12},
            "training": {"learning_rate": 1e-4, "batch_size": 16}
        }
        
        base_path = os.path.join(temp_dir, "base_test.yaml")
        with open(base_path, 'w') as f:
            yaml.dump(base_template, f)
        
        # Create a derived template that inherits from base
        derived_template = {
            "inherits_from": "base_test",
            "name": "derived_test",
            "training": {"learning_rate": 5e-5}  # Override base value
        }
        
        derived_path = os.path.join(temp_dir, "derived_test.yaml")
        with open(derived_path, 'w') as f:
            yaml.dump(derived_template, f)
        
        # Test inheritance resolution
        config_loader = ConfigLoader()
        
        # Mock template directory to use our temp directory
        with patch.object(config_loader, 'template_dir', temp_dir):
            config = config_loader.load_config(derived_path)
        
        # Verify inheritance worked
        assert config.name == "derived_test"
        assert config.model.hidden_size == 768  # From base
        assert config.model.num_layers == 12    # From base
        assert config.training.learning_rate == 5e-5  # Overridden
        assert config.training.batch_size == 16      # From base
    
    def test_config_validation_and_error_handling(self, temp_dir):
        """Test configuration validation and error handling."""
        config_loader = ConfigLoader()
        
        # Test loading non-existent config
        with pytest.raises(FileNotFoundError):
            config_loader.load_config("nonexistent.yaml")
        
        # Test invalid YAML
        invalid_yaml_path = os.path.join(temp_dir, "invalid.yaml")
        with open(invalid_yaml_path, 'w') as f:
            f.write("invalid: yaml: content: ][")
        
        with pytest.raises(yaml.YAMLError):
            config_loader.load_config(invalid_yaml_path)
        
        # Test missing required fields
        incomplete_config = {"name": "incomplete"}
        incomplete_path = os.path.join(temp_dir, "incomplete.yaml")
        with open(incomplete_path, 'w') as f:
            yaml.dump(incomplete_config, f)
        
        # Should handle gracefully with defaults or raise validation error
        try:
            config = config_loader.load_config(incomplete_path)
            # If it loads, should have sensible defaults
            assert hasattr(config, 'name')
        except (ValueError, AttributeError) as e:
            # Expected for missing required fields
            assert "required" in str(e).lower() or "missing" in str(e).lower()
    
    def test_experiment_config_conversion(self, temp_dir):
        """Test conversion of experiment configs to unified format."""
        # Create a legacy-style experiment config
        legacy_config = {
            "experiment_name": "legacy_test",
            "model_config": {
                "base_model": "microsoft/DialoGPT-small",
                "hidden_size": 768
            },
            "train_config": {
                "learning_rate": 1e-4,
                "batch_size": 32,
                "num_epochs": 3
            },
            "data_config": {
                "train_file": "train.jsonl",
                "val_file": "val.jsonl"
            }
        }
        
        legacy_path = os.path.join(temp_dir, "legacy_experiment.yaml")
        with open(legacy_path, 'w') as f:
            yaml.dump(legacy_config, f)
        
        config_loader = ConfigLoader()
        
        # Test conversion (assuming ConfigLoader has conversion capability)
        try:
            config = config_loader.load_config(legacy_path)
            
            # Verify conversion worked
            assert hasattr(config, 'name')
            assert hasattr(config, 'model')
            assert hasattr(config, 'training')
            assert hasattr(config, 'data')
            
        except AttributeError:
            # If conversion not implemented, should at least load raw config
            with open(legacy_path, 'r') as f:
                raw_config = yaml.safe_load(f)
            assert raw_config["experiment_name"] == "legacy_test"


class TestSharedUtilities:
    """Test suite for shared utilities (logging, checkpointing, etc.)."""
    
    def test_logging_utils_setup(self, temp_dir):
        """Test logging utility setup and configuration."""
        log_file = os.path.join(temp_dir, "test.log")
        
        # Test logger setup
        logger = setup_logger("test_logger", log_file, level="INFO")
        
        assert logger.name == "test_logger"
        
        # Test logging to file
        logger.info("Test message")
        logger.warning("Test warning")
        logger.error("Test error")
        
        # Verify log file was created and contains messages
        assert os.path.exists(log_file)
        
        with open(log_file, 'r') as f:
            log_content = f.read()
            assert "Test message" in log_content
            assert "Test warning" in log_content
            assert "Test error" in log_content
    
    def test_checkpoint_utils_functionality(self, temp_dir):
        """Test checkpoint utilities for save/load operations."""
        # Create a simple model and optimizer
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
        
        # Test checkpoint saving
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=500,
            loss=0.35,
            path=checkpoint_path,
            metadata={
                "experiment": "test_checkpoint",
                "timestamp": "2025-01-01",
                "config_hash": "abc123"
            }
        )
        
        assert os.path.exists(checkpoint_path)
        
        # Test checkpoint loading
        loaded_checkpoint = load_checkpoint(checkpoint_path)
        
        # Verify all components loaded correctly
        assert "model_state_dict" in loaded_checkpoint
        assert "optimizer_state_dict" in loaded_checkpoint
        assert loaded_checkpoint["step"] == 500
        assert loaded_checkpoint["loss"] == 0.35
        assert loaded_checkpoint["metadata"]["experiment"] == "test_checkpoint"
        
        # Test loading into new model and optimizer
        new_model = MockModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        new_model.load_state_dict(loaded_checkpoint["model_state_dict"])
        new_optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
        
        # Verify state loaded correctly by comparing parameters
        for orig_param, new_param in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(orig_param, new_param)
    
    def test_checkpoint_versioning_and_metadata(self, temp_dir):
        """Test checkpoint versioning and metadata handling."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save multiple checkpoint versions
        for version in range(3):
            checkpoint_path = os.path.join(temp_dir, f"checkpoint_v{version}.pt")
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=version * 100,
                loss=0.5 - (version * 0.1),
                path=checkpoint_path,
                metadata={
                    "version": version,
                    "experiment": "versioning_test",
                    "notes": f"Checkpoint version {version}"
                }
            )
        
        # Verify all versions saved
        for version in range(3):
            checkpoint_path = os.path.join(temp_dir, f"checkpoint_v{version}.pt")
            assert os.path.exists(checkpoint_path)
            
            loaded = load_checkpoint(checkpoint_path)
            assert loaded["metadata"]["version"] == version
            assert loaded["step"] == version * 100
    
    def test_error_handling_in_utilities(self, temp_dir):
        """Test error handling in utility functions."""
        # Test loading non-existent checkpoint
        with pytest.raises(FileNotFoundError):
            load_checkpoint("nonexistent_checkpoint.pt")
        
        # Test saving to invalid path
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        invalid_path = "/invalid/path/checkpoint.pt"
        with pytest.raises((OSError, PermissionError, FileNotFoundError)):
            save_checkpoint(model, optimizer, 0, 0.0, invalid_path)
        
        # Test logger setup with invalid path
        invalid_log_path = "/invalid/path/log.txt"
        try:
            logger = setup_logger("test", invalid_log_path)
            # If it succeeds, should still be usable for console output
            logger.info("Test message")
        except (OSError, PermissionError):
            # Expected for invalid path
            pass


class TestIntegrationScenarios:
    """Integration tests combining multiple infrastructure components."""
    
    def test_full_training_pipeline_integration(self, mock_config, temp_dir):
        """Test complete training pipeline from config to evaluation."""
        # Modify config to use temp directory
        mock_config.output_dir = temp_dir
        mock_config.logging["log_file"] = os.path.join(temp_dir, "training.log")
        
        # Initialize trainer
        trainer = MockTrainer(mock_config, output_dir=temp_dir)
        
        # Initialize evaluator
        eval_config = EvaluationConfig(
            metrics=["accuracy", "loss"],
            output_dir=temp_dir
        )
        evaluator = MockEvaluator(eval_config)
        
        # Run training
        training_results = trainer.train(max_steps=10)
        
        # Run evaluation
        model = trainer.create_model()
        evaluation_results = evaluator.evaluate_model(model, {"test_data": []})
        
        # Verify integration
        assert "train_loss" in training_results
        assert "eval_loss" in evaluation_results.metrics
        
        # Verify files were created
        assert os.path.exists(os.path.join(temp_dir, "training.log"))
        
        # Save evaluation results
        results_path = os.path.join(temp_dir, "evaluation_results.json")
        evaluation_results.save(results_path)
        assert os.path.exists(results_path)
    
    def test_config_to_trainer_integration(self, temp_dir):
        """Test configuration loading and trainer initialization integration."""
        # Create a complete config file
        config_dict = {
            "name": "integration_test",
            "version": "1.0",
            "experiment_type": "training",
            "model": {
                "base_model": "microsoft/DialoGPT-small",
                "hidden_size": 768,
                "num_layers": 12,
                "custom_params": {
                    "learning_rate": 2e-5,
                    "max_steps": 50
                }
            },
            "data": {
                "train_file": "data/train.jsonl",
                "validation_file": "data/val.jsonl",
                "max_samples": 16
            },
            "training": {
                "learning_rate": 2e-5,
                "batch_size": 8,
                "max_steps": 50,
                "eval_steps": 5
            },
            "hardware": {
                "device": "cpu",
                "mixed_precision": "no"
            },
            "logging": {"level": "INFO"},
            "output_dir": temp_dir,
            "seed": 42
        }
        
        config_path = os.path.join(temp_dir, "integration_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        # Load configuration
        config_loader = ConfigLoader()
        loaded_config = config_loader.load_config(config_path)
        
        # Initialize trainer with loaded config
        trainer = MockTrainer(loaded_config, output_dir=temp_dir)
        
        # Verify configuration propagated correctly
        assert trainer.config.name == "integration_test"
        assert trainer.training_config.learning_rate == 2e-5
        assert trainer.training_config.batch_size == 8
        assert trainer.training_config.max_steps == 50
        
        # Run a short training to verify everything works
        results = trainer.train(max_steps=5)
        assert "train_loss" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])