"""
Tests for retrieval_ablator.py - Retrieval-aware behavior validation and ablation studies.

This module tests the retrieval ablation system that validates retrieval-aware
behavior patterns and performs systematic ablation studies on retrieval components.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import json
from typing import Dict, List, Any, Tuple, Optional

# Import the modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "pipeline"))

from retrieval_ablator import (
    RetrievalContextAnalyzer,
    AblationStudyDesigner,
    BehaviorValidator,
    RetrievalAwareEvaluator,
    AblationResultsAnalyzer,
    RetrievalAblationPipeline,
    ContextualPerformanceTracker
)


class TestRetrievalContextAnalyzer(unittest.TestCase):
    """Test retrieval context analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = RetrievalContextAnalyzer()
        
        # Mock retrieval contexts
        self.sample_contexts = [
            {
                "query": "What is machine learning?",
                "retrieved_docs": [
                    {"text": "Machine learning is a subset of AI...", "score": 0.95, "source": "wiki"},
                    {"text": "ML algorithms learn from data...", "score": 0.87, "source": "textbook"},
                    {"text": "Neural networks are a type of ML...", "score": 0.82, "source": "paper"}
                ],
                "context_length": 512,
                "retrieval_time": 0.05
            },
            {
                "query": "Explain neural networks",
                "retrieved_docs": [
                    {"text": "Neural networks mimic brain structure...", "score": 0.91, "source": "paper"},
                    {"text": "Deep learning uses multiple layers...", "score": 0.85, "source": "book"},
                ],
                "context_length": 340,
                "retrieval_time": 0.03
            }
        ]
    
    def test_analyze_retrieval_quality(self):
        """Test retrieval quality analysis."""
        quality_metrics = self.analyzer.analyze_retrieval_quality(self.sample_contexts)
        
        self.assertIsInstance(quality_metrics, dict)
        self.assertIn("average_relevance_score", quality_metrics)
        self.assertIn("retrieval_diversity", quality_metrics)
        self.assertIn("context_coverage", quality_metrics)
        self.assertIn("source_distribution", quality_metrics)
        
        # Check metric bounds
        avg_relevance = quality_metrics["average_relevance_score"]
        self.assertGreaterEqual(avg_relevance, 0.0)
        self.assertLessEqual(avg_relevance, 1.0)
    
    def test_analyze_context_utilization(self):
        """Test context utilization analysis."""
        # Mock model attention weights over retrieved context
        attention_weights = [
            torch.softmax(torch.randn(1, 8, 512), dim=-1),  # Context 1
            torch.softmax(torch.randn(1, 8, 340), dim=-1),  # Context 2
        ]
        
        utilization_metrics = self.analyzer.analyze_context_utilization(
            self.sample_contexts, attention_weights
        )
        
        self.assertIsInstance(utilization_metrics, dict)
        self.assertIn("attention_entropy", utilization_metrics)
        self.assertIn("context_focus_score", utilization_metrics)
        self.assertIn("unused_context_ratio", utilization_metrics)
        self.assertIn("doc_level_attention", utilization_metrics)
    
    def test_detect_retrieval_biases(self):
        """Test retrieval bias detection."""
        bias_analysis = self.analyzer.detect_retrieval_biases(self.sample_contexts)
        
        self.assertIsInstance(bias_analysis, dict)
        self.assertIn("source_bias", bias_analysis)
        self.assertIn("length_bias", bias_analysis)
        self.assertIn("recency_bias", bias_analysis)
        self.assertIn("position_bias", bias_analysis)
        
        # Source bias should detect distribution across sources
        source_bias = bias_analysis["source_bias"]
        self.assertIn("source_distribution", source_bias)
        self.assertIn("bias_score", source_bias)
    
    def test_compute_retrieval_diversity(self):
        """Test retrieval diversity computation."""
        diversity_score = self.analyzer.compute_retrieval_diversity(self.sample_contexts[0])
        
        self.assertIsInstance(diversity_score, float)
        self.assertGreaterEqual(diversity_score, 0.0)
        self.assertLessEqual(diversity_score, 1.0)
    
    def test_analyze_context_redundancy(self):
        """Test context redundancy analysis."""
        redundancy_analysis = self.analyzer.analyze_context_redundancy(
            self.sample_contexts[0]["retrieved_docs"]
        )
        
        self.assertIsInstance(redundancy_analysis, dict)
        self.assertIn("similarity_matrix", redundancy_analysis)
        self.assertIn("redundancy_score", redundancy_analysis)
        self.assertIn("unique_information_ratio", redundancy_analysis)


class TestAblationStudyDesigner(unittest.TestCase):
    """Test ablation study design functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.designer = AblationStudyDesigner()
        
        # Define retrieval components for ablation
        self.retrieval_components = {
            "retriever_type": ["dense", "sparse", "hybrid"],
            "num_retrieved_docs": [1, 3, 5, 10],
            "context_length": [256, 512, 1024],
            "reranking": [True, False],
            "query_expansion": [True, False]
        }
    
    def test_design_systematic_ablation(self):
        """Test systematic ablation study design."""
        ablation_design = self.designer.design_systematic_ablation(
            self.retrieval_components
        )
        
        self.assertIsInstance(ablation_design, dict)
        self.assertIn("ablation_matrix", ablation_design)
        self.assertIn("baseline_config", ablation_design)
        self.assertIn("ablation_groups", ablation_design)
        self.assertIn("total_experiments", ablation_design)
        
        # Check that baseline config is reasonable
        baseline = ablation_design["baseline_config"]
        for component in self.retrieval_components.keys():
            self.assertIn(component, baseline)
    
    def test_generate_ablation_variants(self):
        """Test ablation variant generation."""
        baseline_config = {
            "retriever_type": "hybrid",
            "num_retrieved_docs": 5,
            "context_length": 512,
            "reranking": True,
            "query_expansion": True
        }
        
        variants = self.designer.generate_ablation_variants(
            baseline_config,
            components_to_ablate=["retriever_type", "num_retrieved_docs"]
        )
        
        self.assertIsInstance(variants, list)
        self.assertGreater(len(variants), 0)
        
        # Each variant should differ from baseline in exactly one component
        for variant in variants:
            differences = sum(
                1 for key in baseline_config 
                if baseline_config[key] != variant["config"][key]
            )
            self.assertEqual(differences, 1)
    
    def test_design_hierarchical_ablation(self):
        """Test hierarchical ablation design."""
        hierarchical_design = self.designer.design_hierarchical_ablation(
            self.retrieval_components,
            hierarchy=["retriever_type", "num_retrieved_docs", "context_length"]
        )
        
        self.assertIsInstance(hierarchical_design, dict)
        self.assertIn("levels", hierarchical_design)
        self.assertIn("dependency_graph", hierarchical_design)
        self.assertIn("execution_order", hierarchical_design)
    
    def test_optimize_ablation_coverage(self):
        """Test ablation coverage optimization."""
        # Create large space that needs optimization
        large_space = {
            "param_a": list(range(10)),
            "param_b": list(range(8)),
            "param_c": list(range(6)),
            "param_d": [True, False]
        }
        
        optimized_design = self.designer.optimize_ablation_coverage(
            large_space,
            max_experiments=50,
            coverage_strategy="latin_hypercube"
        )
        
        self.assertIsInstance(optimized_design, dict)
        self.assertIn("selected_configs", optimized_design)
        self.assertIn("coverage_score", optimized_design)
        self.assertLessEqual(len(optimized_design["selected_configs"]), 50)


class TestBehaviorValidator(unittest.TestCase):
    """Test behavior validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = BehaviorValidator()
        
        # Mock model outputs and contexts
        self.mock_outputs = {
            "with_retrieval": [
                {"answer": "Machine learning is a subset of AI that learns from data.", "confidence": 0.92},
                {"answer": "Neural networks are inspired by the human brain.", "confidence": 0.88}
            ],
            "without_retrieval": [
                {"answer": "Machine learning involves algorithms and data.", "confidence": 0.75},
                {"answer": "Neural networks use interconnected nodes.", "confidence": 0.71}
            ]
        }
        
        self.mock_contexts = [
            {"query": "What is machine learning?", "has_relevant_context": True},
            {"query": "Explain neural networks", "has_relevant_context": True}
        ]
    
    def test_validate_retrieval_dependency(self):
        """Test retrieval dependency validation."""
        dependency_analysis = self.validator.validate_retrieval_dependency(
            self.mock_outputs["with_retrieval"],
            self.mock_outputs["without_retrieval"],
            self.mock_contexts
        )
        
        self.assertIsInstance(dependency_analysis, dict)
        self.assertIn("dependency_score", dependency_analysis)
        self.assertIn("improvement_metrics", dependency_analysis)
        self.assertIn("retrieval_necessity", dependency_analysis)
        
        # Dependency score should be reasonable
        dependency_score = dependency_analysis["dependency_score"]
        self.assertIsInstance(dependency_score, float)
        self.assertGreaterEqual(dependency_score, 0.0)
        self.assertLessEqual(dependency_score, 1.0)
    
    def test_analyze_context_sensitivity(self):
        """Test context sensitivity analysis."""
        # Create different context conditions
        context_variations = [
            {"query": "What is ML?", "context_quality": "high", "context_length": 500},
            {"query": "What is ML?", "context_quality": "medium", "context_length": 300},
            {"query": "What is ML?", "context_quality": "low", "context_length": 100},
        ]
        
        outputs_by_context = [
            {"answer": "Detailed ML explanation...", "confidence": 0.95},
            {"answer": "Basic ML explanation...", "confidence": 0.80},
            {"answer": "Vague ML explanation...", "confidence": 0.65},
        ]
        
        sensitivity_analysis = self.validator.analyze_context_sensitivity(
            context_variations, outputs_by_context
        )
        
        self.assertIsInstance(sensitivity_analysis, dict)
        self.assertIn("quality_sensitivity", sensitivity_analysis)
        self.assertIn("length_sensitivity", sensitivity_analysis)
        self.assertIn("optimal_context_properties", sensitivity_analysis)
    
    def test_detect_hallucination_patterns(self):
        """Test hallucination pattern detection."""
        # Mock outputs with potential hallucinations
        outputs_with_hallucinations = [
            {
                "answer": "The capital of France is Paris, established in 1850.",  # Date is wrong
                "context": "Paris is the capital of France.",
                "factual_claims": ["Paris is capital", "established in 1850"]
            },
            {
                "answer": "Einstein discovered gravity in 1905.",  # Wrong attribution
                "context": "Einstein published theory of relativity.",
                "factual_claims": ["Einstein discovered gravity", "in 1905"]
            }
        ]
        
        hallucination_analysis = self.validator.detect_hallucination_patterns(
            outputs_with_hallucinations
        )
        
        self.assertIsInstance(hallucination_analysis, dict)
        self.assertIn("hallucination_rate", hallucination_analysis)
        self.assertIn("hallucination_types", hallucination_analysis)
        self.assertIn("context_adherence_score", hallucination_analysis)
    
    def test_validate_answer_grounding(self):
        """Test answer grounding validation."""
        answer = "Machine learning algorithms learn patterns from training data to make predictions."
        context_docs = [
            {"text": "Machine learning uses algorithms to find patterns in data."},
            {"text": "Training data helps models make accurate predictions."},
            {"text": "Deep learning is a subset of machine learning."}
        ]
        
        grounding_analysis = self.validator.validate_answer_grounding(answer, context_docs)
        
        self.assertIsInstance(grounding_analysis, dict)
        self.assertIn("grounding_score", grounding_analysis)
        self.assertIn("supporting_evidence", grounding_analysis)
        self.assertIn("unsupported_claims", grounding_analysis)
        self.assertIn("citation_coverage", grounding_analysis)


class TestRetrievalAwareEvaluator(unittest.TestCase):
    """Test retrieval-aware evaluation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = RetrievalAwareEvaluator()
        
        # Mock evaluation data
        self.evaluation_data = [
            {
                "query": "What causes climate change?",
                "ground_truth": "Climate change is primarily caused by greenhouse gas emissions.",
                "prediction_with_retrieval": "Climate change is caused by increased CO2 and other greenhouse gases.",
                "prediction_without_retrieval": "Climate change has various causes.",
                "retrieved_context": [
                    {"text": "CO2 emissions are the main driver of climate change.", "relevance": 0.95},
                    {"text": "Greenhouse gases trap heat in the atmosphere.", "relevance": 0.89}
                ]
            },
            {
                "query": "How do vaccines work?",
                "ground_truth": "Vaccines train the immune system to recognize and fight pathogens.",
                "prediction_with_retrieval": "Vaccines stimulate immune response to create antibodies against diseases.",
                "prediction_without_retrieval": "Vaccines help prevent diseases.",
                "retrieved_context": [
                    {"text": "Vaccines contain antigens that trigger immune response.", "relevance": 0.92},
                    {"text": "Antibodies are produced to fight specific pathogens.", "relevance": 0.88}
                ]
            }
        ]
    
    def test_compute_retrieval_aware_metrics(self):
        """Test retrieval-aware metrics computation."""
        metrics = self.evaluator.compute_retrieval_aware_metrics(self.evaluation_data)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("retrieval_gain", metrics)
        self.assertIn("context_utilization_score", metrics)
        self.assertIn("grounding_accuracy", metrics)
        self.assertIn("retrieval_precision", metrics)
        self.assertIn("answer_completeness", metrics)
        
        # Check metric ranges
        retrieval_gain = metrics["retrieval_gain"]
        self.assertIsInstance(retrieval_gain, float)
    
    def test_evaluate_context_attribution(self):
        """Test context attribution evaluation."""
        sample_item = self.evaluation_data[0]
        
        attribution_analysis = self.evaluator.evaluate_context_attribution(
            sample_item["prediction_with_retrieval"],
            sample_item["retrieved_context"]
        )
        
        self.assertIsInstance(attribution_analysis, dict)
        self.assertIn("attribution_score", attribution_analysis)
        self.assertIn("context_usage", attribution_analysis)
        self.assertIn("information_synthesis", attribution_analysis)
    
    def test_analyze_retrieval_failure_modes(self):
        """Test retrieval failure mode analysis."""
        # Create data with retrieval failures
        failure_data = [
            {
                "query": "Recent AI breakthroughs",
                "prediction_with_retrieval": "Old information about AI from 2020.",
                "retrieved_context": [{"text": "AI made progress in 2020...", "relevance": 0.6}],
                "failure_type": "outdated_context"
            },
            {
                "query": "How to cure cancer?",
                "prediction_with_retrieval": "Cancer can be cured with these methods...",
                "retrieved_context": [{"text": "Some cancer treatments exist...", "relevance": 0.4}],
                "failure_type": "insufficient_context"
            }
        ]
        
        failure_analysis = self.evaluator.analyze_retrieval_failure_modes(failure_data)
        
        self.assertIsInstance(failure_analysis, dict)
        self.assertIn("failure_categories", failure_analysis)
        self.assertIn("failure_frequency", failure_analysis)
        self.assertIn("impact_analysis", failure_analysis)
    
    def test_compute_contextual_rouge(self):
        """Test contextual ROUGE computation."""
        prediction = "Machine learning algorithms learn from data to make predictions."
        reference = "ML algorithms use training data to predict outcomes."
        context = [{"text": "Machine learning uses data for training algorithms."}]
        
        contextual_rouge = self.evaluator.compute_contextual_rouge(
            prediction, reference, context
        )
        
        self.assertIsInstance(contextual_rouge, dict)
        self.assertIn("rouge_1", contextual_rouge)
        self.assertIn("rouge_2", contextual_rouge)
        self.assertIn("rouge_l", contextual_rouge)
        self.assertIn("context_bonus", contextual_rouge)


class TestAblationResultsAnalyzer(unittest.TestCase):
    """Test ablation results analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = AblationResultsAnalyzer()
        
        # Mock ablation results
        self.ablation_results = {
            "baseline": {"accuracy": 0.85, "f1": 0.82, "retrieval_time": 0.05},
            "no_reranking": {"accuracy": 0.81, "f1": 0.79, "retrieval_time": 0.03},
            "fewer_docs": {"accuracy": 0.78, "f1": 0.76, "retrieval_time": 0.02},
            "sparse_retrieval": {"accuracy": 0.80, "f1": 0.78, "retrieval_time": 0.04},
            "no_query_expansion": {"accuracy": 0.83, "f1": 0.81, "retrieval_time": 0.04},
        }
    
    def test_compute_component_importance(self):
        """Test component importance computation."""
        importance_analysis = self.analyzer.compute_component_importance(
            self.ablation_results,
            baseline_key="baseline",
            primary_metric="accuracy"
        )
        
        self.assertIsInstance(importance_analysis, dict)
        self.assertIn("importance_scores", importance_analysis)
        self.assertIn("importance_ranking", importance_analysis)
        self.assertIn("performance_drops", importance_analysis)
        
        # Check that importance scores are reasonable
        importance_scores = importance_analysis["importance_scores"]
        for component, score in importance_scores.items():
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
    
    def test_analyze_interaction_effects(self):
        """Test interaction effects analysis."""
        # Add interaction results
        interaction_results = self.ablation_results.copy()
        interaction_results["no_reranking_fewer_docs"] = {
            "accuracy": 0.75, "f1": 0.73, "retrieval_time": 0.01
        }
        interaction_results["sparse_no_expansion"] = {
            "accuracy": 0.77, "f1": 0.75, "retrieval_time": 0.03
        }
        
        interaction_analysis = self.analyzer.analyze_interaction_effects(
            interaction_results,
            component_pairs=[
                ("no_reranking", "fewer_docs"),
                ("sparse_retrieval", "no_query_expansion")
            ]
        )
        
        self.assertIsInstance(interaction_analysis, dict)
        self.assertIn("interaction_scores", interaction_analysis)
        self.assertIn("synergistic_effects", interaction_analysis)
        self.assertIn("antagonistic_effects", interaction_analysis)
    
    def test_identify_performance_bottlenecks(self):
        """Test performance bottleneck identification."""
        bottleneck_analysis = self.analyzer.identify_performance_bottlenecks(
            self.ablation_results,
            metrics=["accuracy", "f1", "retrieval_time"]
        )
        
        self.assertIsInstance(bottleneck_analysis, dict)
        self.assertIn("bottleneck_components", bottleneck_analysis)
        self.assertIn("efficiency_analysis", bottleneck_analysis)
        self.assertIn("optimization_suggestions", bottleneck_analysis)
    
    def test_generate_ablation_insights(self):
        """Test ablation insights generation."""
        insights = self.analyzer.generate_ablation_insights(
            self.ablation_results,
            significance_threshold=0.02
        )
        
        self.assertIsInstance(insights, dict)
        self.assertIn("key_findings", insights)
        self.assertIn("component_rankings", insights)
        self.assertIn("trade_off_analysis", insights)
        self.assertIn("recommendations", insights)


class TestContextualPerformanceTracker(unittest.TestCase):
    """Test contextual performance tracking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = ContextualPerformanceTracker()
    
    def test_track_performance_by_context(self):
        """Test performance tracking by context type."""
        performance_data = [
            {"context_type": "factual", "accuracy": 0.90, "confidence": 0.85},
            {"context_type": "factual", "accuracy": 0.88, "confidence": 0.82},
            {"context_type": "opinion", "accuracy": 0.75, "confidence": 0.70},
            {"context_type": "opinion", "accuracy": 0.78, "confidence": 0.73},
            {"context_type": "procedural", "accuracy": 0.85, "confidence": 0.80},
        ]
        
        context_performance = self.tracker.track_performance_by_context(performance_data)
        
        self.assertIsInstance(context_performance, dict)
        self.assertIn("factual", context_performance)
        self.assertIn("opinion", context_performance)
        self.assertIn("procedural", context_performance)
        
        # Check statistics for each context type
        for context_type, stats in context_performance.items():
            self.assertIn("mean_accuracy", stats)
            self.assertIn("std_accuracy", stats)
            self.assertIn("count", stats)
    
    def test_analyze_context_difficulty(self):
        """Test context difficulty analysis."""
        context_examples = [
            {"context": "Simple factual statement", "performance": 0.95, "complexity": "low"},
            {"context": "Complex technical explanation with multiple concepts", "performance": 0.70, "complexity": "high"},
            {"context": "Ambiguous statement with multiple interpretations", "performance": 0.65, "complexity": "high"},
        ]
        
        difficulty_analysis = self.tracker.analyze_context_difficulty(context_examples)
        
        self.assertIsInstance(difficulty_analysis, dict)
        self.assertIn("difficulty_correlation", difficulty_analysis)
        self.assertIn("complexity_factors", difficulty_analysis)
        self.assertIn("performance_predictors", difficulty_analysis)


class TestRetrievalAblationPipeline(unittest.TestCase):
    """Test the complete retrieval ablation pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = RetrievalAblationPipeline()
        
        # Mock model and evaluator
        self.mock_model = Mock()
        self.mock_model.predict.return_value = {
            "answer": "Test answer",
            "confidence": 0.85
        }
        
        # Mock evaluation dataset
        self.mock_dataset = [
            {
                "query": "Test query 1",
                "ground_truth": "Test answer 1",
                "context": [{"text": "Test context 1", "score": 0.9}]
            },
            {
                "query": "Test query 2", 
                "ground_truth": "Test answer 2",
                "context": [{"text": "Test context 2", "score": 0.8}]
            }
        ]
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.context_analyzer)
        self.assertIsNotNone(self.pipeline.ablation_designer)
        self.assertIsNotNone(self.pipeline.behavior_validator)
        self.assertIsNotNone(self.pipeline.retrieval_evaluator)
        self.assertIsNotNone(self.pipeline.results_analyzer)
        self.assertIsNotNone(self.pipeline.performance_tracker)
    
    def test_run_ablation_study(self):
        """Test running a complete ablation study."""
        ablation_config = {
            "retriever_type": ["dense", "sparse"],
            "num_docs": [3, 5],
            "reranking": [True, False]
        }
        
        # Mock the ablation execution
        with patch.object(self.pipeline, '_execute_single_ablation') as mock_execute:
            mock_execute.return_value = {
                "accuracy": 0.85,
                "f1": 0.82,
                "retrieval_time": 0.05
            }
            
            results = self.pipeline.run_ablation_study(
                model=self.mock_model,
                dataset=self.mock_dataset,
                ablation_config=ablation_config,
                max_experiments=4
            )
            
            self.assertIsInstance(results, dict)
            self.assertIn("ablation_results", results)
            self.assertIn("component_importance", results)
            self.assertIn("insights", results)
    
    def test_comprehensive_retrieval_analysis(self):
        """Test comprehensive retrieval analysis."""
        with patch.object(self.mock_model, 'predict') as mock_predict:
            mock_predict.return_value = {
                "answer": "Test answer",
                "confidence": 0.85,
                "attention_weights": torch.randn(1, 8, 100)
            }
            
            analysis_results = self.pipeline.comprehensive_retrieval_analysis(
                model=self.mock_model,
                dataset=self.mock_dataset
            )
            
            self.assertIsInstance(analysis_results, dict)
            self.assertIn("context_analysis", analysis_results)
            self.assertIn("behavior_validation", analysis_results)
            self.assertIn("performance_metrics", analysis_results)
    
    def test_save_and_load_results(self):
        """Test saving and loading ablation results."""
        test_results = {
            "baseline": {"accuracy": 0.85, "f1": 0.82},
            "ablation_1": {"accuracy": 0.80, "f1": 0.78},
            "metadata": {"experiment_id": "test_001"}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results_path = Path(temp_dir) / "ablation_results.json"
            
            # Save results
            self.pipeline.save_results(test_results, results_path)
            self.assertTrue(results_path.exists())
            
            # Load results
            loaded_results = self.pipeline.load_results(results_path)
            self.assertEqual(loaded_results["baseline"]["accuracy"], 0.85)
            self.assertEqual(loaded_results["metadata"]["experiment_id"], "test_001")


class TestRetrievalAblationIntegration(unittest.TestCase):
    """Integration tests for retrieval ablation components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.pipeline = RetrievalAblationPipeline()
        
        # Create more realistic dataset
        self.realistic_dataset = [
            {
                "query": "What are the benefits of renewable energy?",
                "ground_truth": "Renewable energy reduces emissions, creates jobs, and provides energy security.",
                "context": [
                    {"text": "Renewable energy sources like solar and wind reduce greenhouse gas emissions.", "score": 0.92},
                    {"text": "The renewable energy sector creates millions of jobs worldwide.", "score": 0.88},
                    {"text": "Energy independence reduces reliance on fossil fuel imports.", "score": 0.85}
                ]
            },
            {
                "query": "How does photosynthesis work?",
                "ground_truth": "Photosynthesis converts light energy into chemical energy using chlorophyll.",
                "context": [
                    {"text": "Chlorophyll absorbs light energy in plant leaves.", "score": 0.94},
                    {"text": "Light energy is converted to chemical energy in the form of glucose.", "score": 0.91},
                    {"text": "Carbon dioxide and water are the raw materials for photosynthesis.", "score": 0.89}
                ]
            }
        ]
    
    def test_end_to_end_ablation_workflow(self):
        """Test complete end-to-end ablation workflow."""
        # Define comprehensive ablation study
        ablation_config = {
            "retriever_type": ["dense", "sparse", "hybrid"],
            "num_retrieved_docs": [1, 3, 5],
            "context_length": [256, 512],
            "reranking": [True, False],
            "query_expansion": [True, False]
        }
        
        # Mock model with realistic behavior
        mock_model = Mock()
        
        def mock_predict_behavior(query, context=None, **kwargs):
            """Mock model that performs better with more context."""
            base_accuracy = 0.70
            if context:
                # Better performance with more context
                context_bonus = min(0.15, len(context) * 0.03)
                # Reranking bonus
                if kwargs.get('reranking', False):
                    context_bonus += 0.05
                accuracy = base_accuracy + context_bonus
            else:
                accuracy = base_accuracy
            
            return {
                "answer": f"Answer for: {query[:30]}...",
                "confidence": accuracy,
                "accuracy": accuracy  # For testing purposes
            }
        
        mock_model.predict.side_effect = mock_predict_behavior
        
        # Mock the expensive operations for testing
        with patch.object(self.pipeline.ablation_designer, 'optimize_ablation_coverage') as mock_optimize:
            # Return a manageable subset of configurations
            mock_optimize.return_value = {
                "selected_configs": [
                    {"retriever_type": "dense", "num_retrieved_docs": 5, "context_length": 512, "reranking": True, "query_expansion": True},
                    {"retriever_type": "sparse", "num_retrieved_docs": 3, "context_length": 256, "reranking": False, "query_expansion": False},
                    {"retriever_type": "hybrid", "num_retrieved_docs": 5, "context_length": 512, "reranking": True, "query_expansion": False},
                ],
                "coverage_score": 0.85
            }
            
            # Run ablation study
            results = self.pipeline.run_ablation_study(
                model=mock_model,
                dataset=self.realistic_dataset,
                ablation_config=ablation_config,
                max_experiments=10
            )
            
            # Verify results structure
            self.assertIsInstance(results, dict)
            self.assertIn("ablation_results", results)
            self.assertIn("component_importance", results)
            self.assertIn("insights", results)
            
            # Verify that different configurations were tested
            ablation_results = results["ablation_results"]
            self.assertGreaterEqual(len(ablation_results), 2)
            
            # Verify component importance analysis
            importance = results["component_importance"]
            self.assertIn("importance_scores", importance)
            self.assertIn("importance_ranking", importance)
    
    def test_retrieval_behavior_validation_workflow(self):
        """Test retrieval behavior validation workflow."""
        mock_model = Mock()
        
        # Mock different behaviors with and without retrieval
        def mock_context_sensitive_predict(query, context=None, **kwargs):
            if context and len(context) > 0:
                return {
                    "answer": f"Context-aware answer for {query[:20]}...",
                    "confidence": 0.88,
                    "uses_context": True
                }
            else:
                return {
                    "answer": f"Generic answer for {query[:20]}...",
                    "confidence": 0.65,
                    "uses_context": False
                }
        
        mock_model.predict.side_effect = mock_context_sensitive_predict
        
        # Run comprehensive analysis
        analysis_results = self.pipeline.comprehensive_retrieval_analysis(
            model=mock_model,
            dataset=self.realistic_dataset
        )
        
        # Verify analysis results
        self.assertIn("behavior_validation", analysis_results)
        self.assertIn("context_analysis", analysis_results)
        self.assertIn("performance_metrics", analysis_results)
        
        # Verify behavior validation detected context dependency
        behavior_validation = analysis_results["behavior_validation"]
        self.assertIn("dependency_score", behavior_validation)
        
        dependency_score = behavior_validation["dependency_score"]
        self.assertGreater(dependency_score, 0.1)  # Should detect some dependency
    
    def test_ablation_results_analysis_integration(self):
        """Test integration of ablation results analysis."""
        # Create realistic ablation results
        mock_results = {
            "baseline_full": {"accuracy": 0.87, "f1": 0.84, "retrieval_time": 0.12},
            "no_reranking": {"accuracy": 0.82, "f1": 0.79, "retrieval_time": 0.08},
            "fewer_docs": {"accuracy": 0.79, "f1": 0.76, "retrieval_time": 0.05},
            "sparse_only": {"accuracy": 0.83, "f1": 0.80, "retrieval_time": 0.06},
            "no_query_expansion": {"accuracy": 0.85, "f1": 0.82, "retrieval_time": 0.10},
            "minimal_setup": {"accuracy": 0.74, "f1": 0.71, "retrieval_time": 0.03}
        }
        
        # Analyze results
        analysis = self.pipeline.results_analyzer.compute_component_importance(
            mock_results,
            baseline_key="baseline_full",
            primary_metric="accuracy"
        )
        
        # Verify analysis identifies most important components
        importance_ranking = analysis["importance_ranking"]
        self.assertIsInstance(importance_ranking, list)
        self.assertGreater(len(importance_ranking), 0)
        
        # Generate insights
        insights = self.pipeline.results_analyzer.generate_ablation_insights(
            mock_results,
            significance_threshold=0.02
        )
        
        self.assertIn("key_findings", insights)
        self.assertIn("recommendations", insights)
        
        # Verify recommendations are actionable
        recommendations = insights["recommendations"]
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)