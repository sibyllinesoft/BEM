"""
Tests for shift_generator.py - Distribution shift generation and validation.

This module tests the distribution shift generators used to evaluate model
robustness across domain, temporal, and adversarial shifts.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np
from pathlib import Path
import tempfile
import json
from typing import Dict, List, Any

# Import the modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "pipeline"))

from shift_generator import (
    DomainShiftGenerator,
    TemporalShiftGenerator, 
    AdversarialShiftGenerator,
    ShiftValidator,
    ShiftMetrics,
    ShiftGenerationPipeline
)


class TestDomainShiftGenerator(unittest.TestCase):
    """Test domain shift generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = DomainShiftGenerator()
        self.sample_data = [
            {"text": "The weather is nice today", "label": "positive", "domain": "weather"},
            {"text": "This movie is terrible", "label": "negative", "domain": "entertainment"},
            {"text": "The stock market crashed", "label": "negative", "domain": "finance"}
        ]
    
    def test_domain_identification(self):
        """Test automatic domain identification from data."""
        domains = self.generator.identify_domains(self.sample_data)
        
        self.assertIsInstance(domains, dict)
        self.assertIn("weather", domains)
        self.assertIn("entertainment", domains)
        self.assertIn("finance", domains)
    
    def test_vocabulary_shift_generation(self):
        """Test vocabulary-based domain shift."""
        source_domain = "weather"
        target_domain = "finance"
        
        shifted_data = self.generator.generate_vocabulary_shift(
            self.sample_data, source_domain, target_domain
        )
        
        self.assertIsInstance(shifted_data, list)
        self.assertGreater(len(shifted_data), 0)
        
        # Check that domain-specific vocabulary has been modified
        for item in shifted_data:
            self.assertIn("text", item)
            self.assertIn("label", item)
            self.assertIn("shift_type", item)
            self.assertEqual(item["shift_type"], "vocabulary")
    
    def test_style_shift_generation(self):
        """Test stylistic domain shift."""
        source_style = "formal"
        target_style = "informal"
        
        shifted_data = self.generator.generate_style_shift(
            self.sample_data, source_style, target_style
        )
        
        self.assertIsInstance(shifted_data, list)
        for item in shifted_data:
            self.assertIn("shift_type", item)
            self.assertEqual(item["shift_type"], "style")
    
    def test_topic_shift_generation(self):
        """Test topic-based domain shift."""
        source_topics = ["weather", "climate"]
        target_topics = ["finance", "economics"]
        
        shifted_data = self.generator.generate_topic_shift(
            self.sample_data, source_topics, target_topics
        )
        
        self.assertIsInstance(shifted_data, list)
        for item in shifted_data:
            self.assertIn("shift_type", item)
            self.assertEqual(item["shift_type"], "topic")


class TestTemporalShiftGenerator(unittest.TestCase):
    """Test temporal shift generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = TemporalShiftGenerator()
        self.sample_data = [
            {"text": "Obama was president", "timestamp": "2015-01-01", "entities": ["Obama"]},
            {"text": "Trump won the election", "timestamp": "2016-11-01", "entities": ["Trump"]},
            {"text": "Biden is the current president", "timestamp": "2021-01-01", "entities": ["Biden"]}
        ]
    
    def test_temporal_entity_shift(self):
        """Test temporal entity replacement."""
        cutoff_date = "2017-01-01"
        
        shifted_data = self.generator.generate_temporal_entity_shift(
            self.sample_data, cutoff_date
        )
        
        self.assertIsInstance(shifted_data, list)
        for item in shifted_data:
            self.assertIn("shift_type", item)
            self.assertEqual(item["shift_type"], "temporal_entity")
    
    def test_event_sequence_shift(self):
        """Test temporal event sequence modification."""
        shifted_data = self.generator.generate_event_sequence_shift(self.sample_data)
        
        self.assertIsInstance(shifted_data, list)
        for item in shifted_data:
            self.assertIn("shift_type", item)
            self.assertEqual(item["shift_type"], "event_sequence")
    
    def test_temporal_reference_shift(self):
        """Test temporal reference modification."""
        reference_date = "2020-01-01"
        
        shifted_data = self.generator.generate_temporal_reference_shift(
            self.sample_data, reference_date
        )
        
        self.assertIsInstance(shifted_data, list)
        for item in shifted_data:
            self.assertIn("shift_type", item)
            self.assertEqual(item["shift_type"], "temporal_reference")


class TestAdversarialShiftGenerator(unittest.TestCase):
    """Test adversarial shift generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = AdversarialShiftGenerator()
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        
        # Mock tokenizer behavior
        self.mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.mock_tokenizer.decode.return_value = "mocked text"
        
        self.sample_data = [
            {"text": "This is a positive example", "label": "positive"},
            {"text": "This is a negative example", "label": "negative"}
        ]
    
    def test_perturbation_based_attacks(self):
        """Test perturbation-based adversarial examples."""
        attacked_data = self.generator.generate_perturbation_attacks(
            self.sample_data, self.mock_model, self.mock_tokenizer
        )
        
        self.assertIsInstance(attacked_data, list)
        for item in attacked_data:
            self.assertIn("shift_type", item)
            self.assertEqual(item["shift_type"], "adversarial_perturbation")
    
    def test_paraphrase_attacks(self):
        """Test paraphrase-based adversarial examples."""
        attacked_data = self.generator.generate_paraphrase_attacks(
            self.sample_data, self.mock_model
        )
        
        self.assertIsInstance(attacked_data, list)
        for item in attacked_data:
            self.assertIn("shift_type", item)
            self.assertEqual(item["shift_type"], "adversarial_paraphrase")
    
    def test_semantic_attacks(self):
        """Test semantic adversarial examples."""
        attacked_data = self.generator.generate_semantic_attacks(
            self.sample_data, self.mock_model, self.mock_tokenizer
        )
        
        self.assertIsInstance(attacked_data, list)
        for item in attacked_data:
            self.assertIn("shift_type", item)
            self.assertEqual(item["shift_type"], "adversarial_semantic")


class TestShiftValidator(unittest.TestCase):
    """Test shift validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ShiftValidator()
        self.original_data = [
            {"text": "Original text 1", "label": "positive"},
            {"text": "Original text 2", "label": "negative"}
        ]
        self.shifted_data = [
            {"text": "Shifted text 1", "label": "positive", "shift_type": "domain"},
            {"text": "Shifted text 2", "label": "negative", "shift_type": "domain"}
        ]
    
    def test_statistical_validation(self):
        """Test statistical validation of shifts."""
        validation_results = self.validator.validate_shift_statistical(
            self.original_data, self.shifted_data
        )
        
        self.assertIsInstance(validation_results, dict)
        self.assertIn("distribution_distance", validation_results)
        self.assertIn("vocabulary_overlap", validation_results)
        self.assertIn("semantic_similarity", validation_results)
    
    def test_semantic_validation(self):
        """Test semantic validation of shifts."""
        mock_model = Mock()
        mock_model.encode.return_value = torch.randn(2, 768)  # Mock embeddings
        
        validation_results = self.validator.validate_shift_semantic(
            self.original_data, self.shifted_data, mock_model
        )
        
        self.assertIsInstance(validation_results, dict)
        self.assertIn("semantic_distance", validation_results)
        self.assertIn("embedding_variance", validation_results)
    
    def test_quality_validation(self):
        """Test quality validation of shifts."""
        validation_results = self.validator.validate_shift_quality(
            self.original_data, self.shifted_data
        )
        
        self.assertIsInstance(validation_results, dict)
        self.assertIn("fluency_score", validation_results)
        self.assertIn("coherence_score", validation_results)
        self.assertIn("label_preservation", validation_results)


class TestShiftMetrics(unittest.TestCase):
    """Test shift metrics calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = ShiftMetrics()
    
    def test_distribution_distance(self):
        """Test distribution distance calculation."""
        original_embeddings = torch.randn(100, 768)
        shifted_embeddings = torch.randn(100, 768)
        
        distance = self.metrics.calculate_distribution_distance(
            original_embeddings, shifted_embeddings
        )
        
        self.assertIsInstance(distance, float)
        self.assertGreaterEqual(distance, 0.0)
    
    def test_vocabulary_overlap(self):
        """Test vocabulary overlap calculation."""
        original_texts = ["hello world", "goodbye moon"]
        shifted_texts = ["hello universe", "goodbye sun"]
        
        overlap = self.metrics.calculate_vocabulary_overlap(
            original_texts, shifted_texts
        )
        
        self.assertIsInstance(overlap, float)
        self.assertGreaterEqual(overlap, 0.0)
        self.assertLessEqual(overlap, 1.0)
    
    def test_semantic_similarity(self):
        """Test semantic similarity calculation."""
        original_embeddings = torch.randn(10, 768)
        shifted_embeddings = torch.randn(10, 768)
        
        similarity = self.metrics.calculate_semantic_similarity(
            original_embeddings, shifted_embeddings
        )
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, -1.0)
        self.assertLessEqual(similarity, 1.0)


class TestShiftGenerationPipeline(unittest.TestCase):
    """Test the complete shift generation pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = ShiftGenerationPipeline()
        self.sample_data = [
            {"text": "Sample text 1", "label": "positive", "domain": "test"},
            {"text": "Sample text 2", "label": "negative", "domain": "test"}
        ]
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.domain_generator)
        self.assertIsNotNone(self.pipeline.temporal_generator)
        self.assertIsNotNone(self.pipeline.adversarial_generator)
        self.assertIsNotNone(self.pipeline.validator)
    
    @patch('shift_generator.torch.cuda.is_available')
    def test_generate_all_shifts(self, mock_cuda):
        """Test generation of all shift types."""
        mock_cuda.return_value = False  # Force CPU mode for testing
        
        shift_config = {
            "domain_shifts": ["vocabulary", "style"],
            "temporal_shifts": ["entity", "reference"],
            "adversarial_shifts": ["perturbation"],
            "validation_enabled": True
        }
        
        with patch.object(self.pipeline.domain_generator, 'generate_vocabulary_shift') as mock_vocab:
            with patch.object(self.pipeline.domain_generator, 'generate_style_shift') as mock_style:
                with patch.object(self.pipeline.temporal_generator, 'generate_temporal_entity_shift') as mock_temporal:
                    mock_vocab.return_value = self.sample_data
                    mock_style.return_value = self.sample_data
                    mock_temporal.return_value = self.sample_data
                    
                    results = self.pipeline.generate_all_shifts(
                        self.sample_data, shift_config
                    )
                    
                    self.assertIsInstance(results, dict)
                    self.assertIn("shifts", results)
                    self.assertIn("validation", results)
                    self.assertIn("metadata", results)
    
    def test_save_and_load_shifts(self):
        """Test saving and loading shift data."""
        shifts_data = {
            "domain": self.sample_data,
            "temporal": self.sample_data,
            "adversarial": self.sample_data
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_shifts.json"
            
            # Test saving
            self.pipeline.save_shifts(shifts_data, save_path)
            self.assertTrue(save_path.exists())
            
            # Test loading
            loaded_data = self.pipeline.load_shifts(save_path)
            self.assertEqual(len(loaded_data), len(shifts_data))
            self.assertIn("domain", loaded_data)
            self.assertIn("temporal", loaded_data)
            self.assertIn("adversarial", loaded_data)
    
    def test_pipeline_metrics_tracking(self):
        """Test pipeline metrics tracking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            
            # Initialize pipeline with metrics tracking
            pipeline = ShiftGenerationPipeline(metrics_path=metrics_path)
            
            # Mock some metrics
            test_metrics = {
                "total_shifts_generated": 100,
                "validation_success_rate": 0.95,
                "processing_time": 120.5
            }
            
            pipeline.save_metrics(test_metrics)
            loaded_metrics = pipeline.load_metrics()
            
            self.assertEqual(loaded_metrics["total_shifts_generated"], 100)
            self.assertEqual(loaded_metrics["validation_success_rate"], 0.95)
            self.assertEqual(loaded_metrics["processing_time"], 120.5)


class TestShiftGenerationIntegration(unittest.TestCase):
    """Integration tests for shift generation components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.sample_dataset = [
            {"text": "The weather is sunny today", "label": "positive", "domain": "weather", "timestamp": "2023-01-01"},
            {"text": "This movie is boring", "label": "negative", "domain": "entertainment", "timestamp": "2023-02-01"},
            {"text": "The stock price increased", "label": "positive", "domain": "finance", "timestamp": "2023-03-01"}
        ]
    
    def test_end_to_end_shift_generation(self):
        """Test complete end-to-end shift generation workflow."""
        pipeline = ShiftGenerationPipeline()
        
        # Configure comprehensive shift generation
        config = {
            "domain_shifts": ["vocabulary", "style", "topic"],
            "temporal_shifts": ["entity", "reference", "sequence"],
            "adversarial_shifts": ["perturbation", "paraphrase"],
            "validation_enabled": True,
            "quality_threshold": 0.7,
            "diversity_threshold": 0.3
        }
        
        # Mock the expensive operations for testing
        with patch.object(pipeline.adversarial_generator, 'generate_perturbation_attacks') as mock_perturbation:
            with patch.object(pipeline.adversarial_generator, 'generate_paraphrase_attacks') as mock_paraphrase:
                mock_perturbation.return_value = self.sample_dataset
                mock_paraphrase.return_value = self.sample_dataset
                
                results = pipeline.generate_all_shifts(self.sample_dataset, config)
                
                # Verify structure
                self.assertIn("shifts", results)
                self.assertIn("validation", results)
                self.assertIn("metadata", results)
                
                # Verify shift types are present
                shifts = results["shifts"]
                self.assertIn("domain", shifts)
                self.assertIn("temporal", shifts)
                self.assertIn("adversarial", shifts)
    
    def test_shift_validation_pipeline(self):
        """Test shift validation pipeline integration."""
        pipeline = ShiftGenerationPipeline()
        validator = pipeline.validator
        
        # Create realistic shift data
        original_data = self.sample_dataset
        shifted_data = [
            {**item, "text": item["text"].replace("sunny", "rainy"), "shift_type": "domain"}
            for item in original_data
        ]
        
        # Test comprehensive validation
        validation_results = validator.validate_comprehensive(original_data, shifted_data)
        
        self.assertIn("statistical", validation_results)
        self.assertIn("quality", validation_results)
        self.assertIn("overall_score", validation_results)
        
        # Verify score is reasonable
        overall_score = validation_results["overall_score"]
        self.assertIsInstance(overall_score, float)
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 1.0)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)