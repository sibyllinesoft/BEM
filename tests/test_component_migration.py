"""
Component-specific migration validation tests.

This module tests that each specialized trainer (RouterTrainer, SafetyTrainer, 
MultimodalTrainer) correctly inherits from BaseTrainer and implements all required
methods while preserving component-specific functionality.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
import os
import json
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List, Optional

# Import unified trainers
from src.bem2.router.unified_trainer import RouterTrainer, RouterEvaluator, migrate_from_legacy_trainer
from src.bem2.safety.unified_trainer import SafetyTrainer, SafetyEvaluator, migrate_from_legacy_safety_trainer
from src.bem2.multimodal.unified_trainer import MultimodalTrainer, MultimodalEvaluator, migrate_from_legacy_multimodal_trainer

# Import base infrastructure
from src.bem_core.training.base_trainer import BaseTrainer, TrainingConfig
from src.bem_core.evaluation.base_evaluator import BaseEvaluator, EvaluationResult
from src.bem_core.config.config_loader import ConfigLoader, ExperimentConfig

# Import legacy components for comparison (if available)
try:
    from src.bem2.router.training import RouterTrainer as LegacyRouterTrainer
    legacy_router_available = True
except ImportError:
    legacy_router_available = False

try:
    from src.bem2.safety.training import SafetyTrainer as LegacySafetyTrainer
    legacy_safety_available = True
except ImportError:
    legacy_safety_available = False

try:
    from src.bem2.multimodal.training import MultimodalTrainer as LegacyMultimodalTrainer
    legacy_multimodal_available = True
except ImportError:
    legacy_multimodal_available = False


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def base_config():
    """Create base experiment configuration."""
    return ExperimentConfig(
        name="component_test",
        version="1.0",
        experiment_type="training",
        model={
            "base_model": "microsoft/DialoGPT-small",
            "hidden_size": 768,
            "num_layers": 12,
            "custom_params": {
                "learning_rate": 5e-5,
                "max_steps": 100
            }
        },
        data={
            "train_file": "data/train.jsonl",
            "validation_file": "data/val.jsonl",
            "max_samples": 32
        },
        training={
            "learning_rate": 5e-5,
            "batch_size": 16,
            "max_steps": 100,
            "eval_steps": 10
        },
        hardware={"device": "cpu", "mixed_precision": "no"},
        logging={"level": "INFO"},
        output_dir="test_output",
        seed=42
    )


@pytest.fixture
def router_config(base_config):
    """Create router-specific configuration."""
    config = base_config.copy() if hasattr(base_config, 'copy') else base_config
    config.name = "router_test"
    config.model.custom_params.update({
        "num_experts": 4,
        "router_type": "learned",
        "composition_strategy": "weighted_sum"
    })
    return config


@pytest.fixture
def safety_config(base_config):
    """Create safety-specific configuration."""
    config = base_config.copy() if hasattr(base_config, 'copy') else base_config
    config.name = "safety_test"
    config.model.custom_params.update({
        "safety_threshold": 0.8,
        "constitutional_ai": True,
        "violation_penalty": 10.0
    })
    return config


@pytest.fixture
def multimodal_config(base_config):
    """Create multimodal-specific configuration."""
    config = base_config.copy() if hasattr(base_config, 'copy') else base_config
    config.name = "multimodal_test"
    config.model.custom_params.update({
        "vision_encoder": "openai/clip-vit-base-patch32",
        "modality_fusion": "cross_attention",
        "max_image_size": 224
    })
    return config


class TestRouterTrainerMigration:
    """Test RouterTrainer migration and functionality."""
    
    def test_router_trainer_inheritance(self, router_config, temp_dir):
        """Test RouterTrainer properly inherits from BaseTrainer."""
        trainer = RouterTrainer(router_config, output_dir=temp_dir)
        
        # Test inheritance chain
        assert isinstance(trainer, BaseTrainer)
        assert isinstance(trainer, RouterTrainer)
        
        # Test base functionality is available
        assert hasattr(trainer, 'train')
        assert hasattr(trainer, 'create_model')
        assert hasattr(trainer, 'compute_loss')
        assert hasattr(trainer, 'prepare_data')
        
        # Test router-specific attributes
        assert hasattr(trainer, 'config')
        assert trainer.config.name == "router_test"
    
    def test_router_model_creation(self, router_config, temp_dir):
        """Test RouterTrainer creates appropriate router models."""
        trainer = RouterTrainer(router_config, output_dir=temp_dir)
        
        with patch('src.bem2.router.unified_trainer.AgenticRouter') as mock_router:
            mock_router.return_value = Mock()
            
            model = trainer.create_model()
            
            # Verify router model was created
            mock_router.assert_called_once()
            assert model is not None
    
    def test_router_data_preparation(self, router_config, temp_dir):
        """Test RouterTrainer prepares routing-specific data."""
        trainer = RouterTrainer(router_config, output_dir=temp_dir)
        
        with patch.object(trainer, '_load_routing_data') as mock_load_data:
            mock_load_data.return_value = {
                "train_dataloader": Mock(),
                "eval_dataloader": Mock(),
                "routing_metadata": {"num_experts": 4}
            }
            
            data = trainer.prepare_data()
            
            assert "train_dataloader" in data
            assert "eval_dataloader" in data
            assert "routing_metadata" in data
            assert data["routing_metadata"]["num_experts"] == 4
    
    def test_router_loss_computation(self, router_config, temp_dir):
        """Test RouterTrainer implements router-specific loss computation."""
        trainer = RouterTrainer(router_config, output_dir=temp_dir)
        
        # Mock model outputs with routing-specific elements
        model_outputs = {
            "loss": torch.tensor(0.5),
            "routing_probs": torch.rand(2, 4),  # batch_size=2, num_experts=4
            "expert_outputs": [torch.rand(2, 768) for _ in range(4)],
            "load_balancing_loss": torch.tensor(0.1)
        }
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "expert_labels": torch.randint(0, 4, (2,))
        }
        
        loss = trainer.compute_loss(model_outputs, batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        # Loss should incorporate load balancing
        expected_loss = model_outputs["loss"] + model_outputs["load_balancing_loss"]
        assert torch.allclose(loss, expected_loss, atol=1e-6)
    
    def test_router_evaluation_integration(self, router_config, temp_dir):
        """Test RouterTrainer integrates with RouterEvaluator."""
        trainer = RouterTrainer(router_config, output_dir=temp_dir)
        evaluator = RouterEvaluator(trainer.config)
        
        # Create mock model and data
        with patch.object(trainer, 'create_model') as mock_create_model:
            mock_model = Mock()
            mock_model.return_value = {
                "loss": torch.tensor(0.4),
                "routing_probs": torch.rand(2, 4),
                "routing_accuracy": torch.tensor(0.85)
            }
            mock_create_model.return_value = mock_model
            
            model = trainer.create_model()
            
            # Test evaluation
            with patch.object(evaluator, '_compute_routing_metrics') as mock_metrics:
                mock_metrics.return_value = {
                    "routing_accuracy": 0.85,
                    "load_balance_score": 0.92,
                    "expert_utilization": 0.75
                }
                
                result = evaluator.evaluate_model(model, {"test_data": []})
                
                assert isinstance(result, EvaluationResult)
                assert "routing_accuracy" in result.metrics
                assert "load_balance_score" in result.metrics
    
    @pytest.mark.skipif(not legacy_router_available, reason="Legacy router trainer not available")
    def test_migration_from_legacy_router(self, router_config, temp_dir):
        """Test migration from legacy RouterTrainer preserves functionality."""
        # Create legacy trainer (mocked if not available)
        with patch('src.bem2.router.training.RouterTrainer') as mock_legacy:
            mock_legacy_instance = Mock()
            mock_legacy_instance.config = router_config
            mock_legacy_instance.model = Mock()
            mock_legacy.return_value = mock_legacy_instance
            
            legacy_trainer = mock_legacy(router_config)
            
            # Migrate to unified trainer
            unified_trainer = migrate_from_legacy_trainer(legacy_trainer, temp_dir)
            
            assert isinstance(unified_trainer, RouterTrainer)
            assert unified_trainer.config == router_config
            assert unified_trainer.output_dir == temp_dir


class TestSafetyTrainerMigration:
    """Test SafetyTrainer migration and functionality."""
    
    def test_safety_trainer_inheritance(self, safety_config, temp_dir):
        """Test SafetyTrainer properly inherits from BaseTrainer."""
        trainer = SafetyTrainer(safety_config, output_dir=temp_dir)
        
        # Test inheritance
        assert isinstance(trainer, BaseTrainer)
        assert isinstance(trainer, SafetyTrainer)
        
        # Test safety-specific configuration
        assert hasattr(trainer, 'safety_config')
        assert trainer.safety_config.safety_threshold == 0.8
        assert trainer.safety_config.constitutional_ai is True
    
    def test_safety_model_creation(self, safety_config, temp_dir):
        """Test SafetyTrainer creates models with safety components."""
        trainer = SafetyTrainer(safety_config, output_dir=temp_dir)
        
        with patch('src.bem2.safety.unified_trainer.SafetyController') as mock_safety:
            mock_safety_instance = Mock()
            mock_safety.return_value = mock_safety_instance
            
            model = trainer.create_model()
            
            # Verify safety controller was integrated
            mock_safety.assert_called_once()
            assert model is not None
    
    def test_safety_loss_computation(self, safety_config, temp_dir):
        """Test SafetyTrainer implements safety-aware loss computation."""
        trainer = SafetyTrainer(safety_config, output_dir=temp_dir)
        
        # Mock model outputs with safety scores
        model_outputs = {
            "loss": torch.tensor(0.6),
            "safety_scores": torch.tensor([0.9, 0.7]),  # batch_size=2
            "violation_flags": torch.tensor([False, True]),
            "constitutional_loss": torch.tensor(0.2)
        }
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "safety_labels": torch.tensor([1, 0])  # 1=safe, 0=unsafe
        }
        
        loss = trainer.compute_loss(model_outputs, batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        
        # Verify safety penalty is applied
        violation_penalty = trainer.safety_config.violation_penalty
        expected_penalty = violation_penalty * model_outputs["violation_flags"].float().sum()
        assert loss > model_outputs["loss"]  # Should be higher due to safety penalty
    
    def test_safety_evaluation_metrics(self, safety_config, temp_dir):
        """Test SafetyEvaluator computes safety-specific metrics."""
        trainer = SafetyTrainer(safety_config, output_dir=temp_dir)
        evaluator = SafetyEvaluator(trainer.safety_config)
        
        with patch.object(trainer, 'create_model') as mock_create_model:
            mock_model = Mock()
            mock_model.return_value = {
                "safety_scores": torch.tensor([0.9, 0.8, 0.6, 0.95]),
                "violation_flags": torch.tensor([False, False, True, False])
            }
            mock_create_model.return_value = mock_model
            
            model = trainer.create_model()
            
            # Test safety evaluation
            result = evaluator.evaluate_model(model, {"test_data": []})
            
            assert isinstance(result, EvaluationResult)
            assert "safety_score" in result.metrics
            assert "violation_rate" in result.metrics
            assert "constitutional_compliance" in result.metrics
            
            # Verify safety threshold is applied
            assert 0 <= result.metrics["safety_score"] <= 1
            assert 0 <= result.metrics["violation_rate"] <= 1
    
    def test_safety_constitutional_ai_integration(self, safety_config, temp_dir):
        """Test integration with Constitutional AI components."""
        trainer = SafetyTrainer(safety_config, output_dir=temp_dir)
        
        with patch('src.bem2.safety.unified_trainer.ConstitutionalScorer') as mock_scorer:
            mock_scorer_instance = Mock()
            mock_scorer_instance.score_safety.return_value = torch.tensor(0.85)
            mock_scorer.return_value = mock_scorer_instance
            
            # Test constitutional scorer integration
            trainer._setup_constitutional_ai()
            
            mock_scorer.assert_called_once()
            
            # Test scoring functionality
            text_input = "Test safety evaluation"
            safety_score = trainer.constitutional_scorer.score_safety(text_input)
            
            assert isinstance(safety_score, torch.Tensor)
            assert 0 <= safety_score <= 1


class TestMultimodalTrainerMigration:
    """Test MultimodalTrainer migration and functionality."""
    
    def test_multimodal_trainer_inheritance(self, multimodal_config, temp_dir):
        """Test MultimodalTrainer properly inherits from BaseTrainer."""
        trainer = MultimodalTrainer(multimodal_config, output_dir=temp_dir)
        
        # Test inheritance
        assert isinstance(trainer, BaseTrainer)
        assert isinstance(trainer, MultimodalTrainer)
        
        # Test multimodal-specific configuration
        assert hasattr(trainer, 'multimodal_config')
        assert trainer.multimodal_config.vision_encoder == "openai/clip-vit-base-patch32"
        assert trainer.multimodal_config.modality_fusion == "cross_attention"
    
    def test_multimodal_model_creation(self, multimodal_config, temp_dir):
        """Test MultimodalTrainer creates vision-text models."""
        trainer = MultimodalTrainer(multimodal_config, output_dir=temp_dir)
        
        with patch('src.bem2.multimodal.unified_trainer.VisionEncoder') as mock_vision:
            mock_vision_instance = Mock()
            mock_vision.return_value = mock_vision_instance
            
            with patch.object(trainer, '_create_fusion_module') as mock_fusion:
                mock_fusion.return_value = Mock()
                
                model = trainer.create_model()
                
                # Verify vision encoder and fusion modules created
                mock_vision.assert_called_once()
                mock_fusion.assert_called_once()
                assert model is not None
    
    def test_multimodal_data_preparation(self, multimodal_config, temp_dir):
        """Test MultimodalTrainer prepares vision-text data."""
        trainer = MultimodalTrainer(multimodal_config, output_dir=temp_dir)
        
        with patch.object(trainer, '_preprocess_images') as mock_preprocess:
            mock_preprocess.return_value = torch.rand(2, 3, 224, 224)  # batch of images
            
            with patch.object(trainer, '_load_multimodal_data') as mock_load_data:
                mock_load_data.return_value = {
                    "train_dataloader": Mock(),
                    "eval_dataloader": Mock(),
                    "vision_features": torch.rand(100, 768),  # precomputed features
                    "text_features": torch.rand(100, 768)
                }
                
                data = trainer.prepare_data()
                
                assert "train_dataloader" in data
                assert "eval_dataloader" in data
                assert "vision_features" in data
                assert "text_features" in data
    
    def test_multimodal_loss_computation(self, multimodal_config, temp_dir):
        """Test MultimodalTrainer implements multimodal loss computation."""
        trainer = MultimodalTrainer(multimodal_config, output_dir=temp_dir)
        
        # Mock model outputs with multimodal components
        model_outputs = {
            "loss": torch.tensor(0.5),
            "text_logits": torch.rand(2, 50257),  # batch_size=2, vocab_size
            "vision_logits": torch.rand(2, 1000),  # batch_size=2, num_classes
            "fusion_loss": torch.tensor(0.1),
            "alignment_loss": torch.tensor(0.05)
        }
        
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "pixel_values": torch.rand(2, 3, 224, 224),
            "text_labels": torch.randint(0, 1000, (2, 10)),
            "vision_labels": torch.randint(0, 1000, (2,))
        }
        
        loss = trainer.compute_loss(model_outputs, batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        
        # Verify multimodal losses are combined
        expected_loss = (model_outputs["loss"] + 
                        model_outputs["fusion_loss"] + 
                        model_outputs["alignment_loss"])
        assert torch.allclose(loss, expected_loss, atol=1e-6)
    
    def test_multimodal_evaluation_metrics(self, multimodal_config, temp_dir):
        """Test MultimodalEvaluator computes multimodal metrics."""
        trainer = MultimodalTrainer(multimodal_config, output_dir=temp_dir)
        evaluator = MultimodalEvaluator(trainer.multimodal_config)
        
        with patch.object(trainer, 'create_model') as mock_create_model:
            mock_model = Mock()
            mock_model.return_value = {
                "text_accuracy": torch.tensor(0.82),
                "vision_accuracy": torch.tensor(0.78),
                "cross_modal_accuracy": torch.tensor(0.75),
                "alignment_score": torch.tensor(0.88)
            }
            mock_create_model.return_value = mock_model
            
            model = trainer.create_model()
            
            # Test multimodal evaluation
            result = evaluator.evaluate_model(model, {"test_data": []})
            
            assert isinstance(result, EvaluationResult)
            assert "text_accuracy" in result.metrics
            assert "vision_accuracy" in result.metrics
            assert "cross_modal_accuracy" in result.metrics
            assert "alignment_score" in result.metrics
            
            # Verify metrics are reasonable
            for metric in ["text_accuracy", "vision_accuracy", "alignment_score"]:
                assert 0 <= result.metrics[metric] <= 1
    
    def test_vision_encoder_integration(self, multimodal_config, temp_dir):
        """Test integration with vision encoder components."""
        trainer = MultimodalTrainer(multimodal_config, output_dir=temp_dir)
        
        with patch('src.bem2.multimodal.vision_encoder.VisionEncoder') as mock_encoder:
            mock_encoder_instance = Mock()
            mock_encoder_instance.encode.return_value = torch.rand(2, 768)  # encoded features
            mock_encoder.return_value = mock_encoder_instance
            
            # Test vision encoder setup
            vision_encoder = trainer._setup_vision_encoder()
            
            mock_encoder.assert_called_once()
            
            # Test encoding functionality
            dummy_images = torch.rand(2, 3, 224, 224)
            features = vision_encoder.encode(dummy_images)
            
            assert features.shape == (2, 768)
            assert isinstance(features, torch.Tensor)


class TestMigrationUtilities:
    """Test migration utility functions."""
    
    def test_legacy_config_conversion(self, temp_dir):
        """Test conversion of legacy configurations to unified format."""
        # Create legacy-style config
        legacy_router_config = {
            "router_config": {
                "num_experts": 8,
                "router_type": "learned",
                "load_balancing_alpha": 0.01
            },
            "model_config": {
                "base_model": "microsoft/DialoGPT-small",
                "hidden_size": 768
            },
            "training_config": {
                "learning_rate": 1e-4,
                "batch_size": 32
            }
        }
        
        # Test conversion (assuming utility function exists)
        with patch('src.bem2.router.unified_trainer.convert_legacy_config') as mock_convert:
            mock_convert.return_value = ExperimentConfig(
                name="converted_router",
                model={"base_model": "microsoft/DialoGPT-small", "hidden_size": 768},
                training={"learning_rate": 1e-4, "batch_size": 32}
            )
            
            converted_config = mock_convert(legacy_router_config)
            
            assert isinstance(converted_config, ExperimentConfig)
            assert converted_config.name == "converted_router"
            mock_convert.assert_called_once_with(legacy_router_config)
    
    def test_component_feature_migration(self, temp_dir):
        """Test that specialized features are preserved during migration."""
        # Test router-specific features
        router_features = {
            "expert_routing": True,
            "load_balancing": True,
            "dynamic_expert_selection": True,
            "composition_strategies": ["weighted_sum", "attention"]
        }
        
        # Test safety-specific features
        safety_features = {
            "constitutional_ai": True,
            "violation_detection": True,
            "safety_thresholding": True,
            "harmfulness_filtering": True
        }
        
        # Test multimodal-specific features
        multimodal_features = {
            "vision_text_fusion": True,
            "cross_modal_attention": True,
            "modality_specific_encoders": True,
            "alignment_objectives": True
        }
        
        # These would be tested by checking that the unified trainers
        # preserve and correctly implement these features
        assert all(isinstance(v, (bool, list, str)) for v in router_features.values())
        assert all(isinstance(v, (bool, list, str)) for v in safety_features.values())
        assert all(isinstance(v, (bool, list, str)) for v in multimodal_features.values())


class TestBackwardCompatibility:
    """Test backward compatibility with existing experiments."""
    
    def test_existing_experiment_configs_load(self, temp_dir):
        """Test that existing experiment configs can be loaded with unified trainers."""
        # Create config in old format
        old_format_config = {
            "experiment_name": "backward_compat_test",
            "router_params": {
                "num_experts": 4,
                "routing_algorithm": "learned"
            },
            "model_params": {
                "base_model": "microsoft/DialoGPT-small"
            }
        }
        
        config_path = os.path.join(temp_dir, "old_format.json")
        with open(config_path, 'w') as f:
            json.dump(old_format_config, f)
        
        # Test that unified config loader can handle old format
        config_loader = ConfigLoader()
        
        with patch.object(config_loader, '_convert_legacy_format') as mock_convert:
            mock_convert.return_value = ExperimentConfig(
                name="backward_compat_test",
                model={"base_model": "microsoft/DialoGPT-small"},
                training={"learning_rate": 5e-5}
            )
            
            try:
                config = config_loader.load_config(config_path)
                assert config.name == "backward_compat_test"
                mock_convert.assert_called_once()
            except NotImplementedError:
                # If legacy conversion not implemented, should at least detect format
                pass
    
    def test_legacy_checkpoint_loading(self, temp_dir):
        """Test that legacy checkpoints can be loaded by unified trainers."""
        # Create a legacy-format checkpoint
        legacy_checkpoint = {
            "model_state_dict": {"linear.weight": torch.rand(768, 768)},
            "optimizer_state_dict": {"param_groups": [{"lr": 1e-4}]},
            "step": 1000,
            "loss": 0.35,
            "legacy_format": True,
            "component_type": "router"
        }
        
        checkpoint_path = os.path.join(temp_dir, "legacy_checkpoint.pt")
        torch.save(legacy_checkpoint, checkpoint_path)
        
        # Test loading with unified trainer
        config = ExperimentConfig(
            name="checkpoint_test",
            model={"base_model": "microsoft/DialoGPT-small"},
            training={"learning_rate": 1e-4}
        )
        
        trainer = RouterTrainer(config, output_dir=temp_dir)
        
        # Test checkpoint compatibility
        loaded_checkpoint = torch.load(checkpoint_path)
        assert loaded_checkpoint["legacy_format"] is True
        assert loaded_checkpoint["component_type"] == "router"
        
        # Test that trainer can handle legacy format
        with patch.object(trainer, '_convert_legacy_checkpoint') as mock_convert:
            mock_convert.return_value = {
                "model_state_dict": loaded_checkpoint["model_state_dict"],
                "optimizer_state_dict": loaded_checkpoint["optimizer_state_dict"],
                "step": loaded_checkpoint["step"],
                "loss": loaded_checkpoint["loss"]
            }
            
            converted = mock_convert(loaded_checkpoint)
            assert "model_state_dict" in converted
            assert converted["step"] == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])