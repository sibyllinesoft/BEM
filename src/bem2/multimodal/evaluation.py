"""
Evaluation Framework for BEM 2.0 Multimodal Conditioning

Implements specialized evaluation for VQA tasks, hallucination detection,
and multimodal performance metrics. Focuses on the key objectives:
- +≥2% EM/F1 on VQA slice
- Hallucination reduction
- ≤+15% p50 latency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from dataclasses import dataclass, asdict
import numpy as np
import json
import time
from pathlib import Path
import re
from collections import defaultdict, Counter

try:
    from sklearn.metrics import accuracy_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class VQAMetrics(NamedTuple):
    """VQA evaluation metrics."""
    exact_match: float
    f1_score: float
    accuracy: float
    bleu_score: Optional[float]
    rouge_l: Optional[float]
    answer_relevance: float  # Custom metric for answer relevance


class HallucinationMetrics(NamedTuple):
    """Hallucination detection metrics."""
    object_hallucination_rate: float      # Objects mentioned but not present
    attribute_hallucination_rate: float   # Wrong attributes
    spatial_hallucination_rate: float     # Wrong spatial relationships
    factual_consistency_score: float      # Consistency with visual facts
    overall_hallucination_rate: float     # Combined hallucination rate


class LatencyMetrics(NamedTuple):
    """Latency performance metrics."""
    preprocessing_latency_ms: float       # Vision preprocessing time
    inference_latency_ms: float          # Model inference time
    postprocessing_latency_ms: float     # Output processing time
    total_latency_ms: float              # End-to-end latency
    p50_latency_ms: float                # 50th percentile
    p95_latency_ms: float                # 95th percentile
    p99_latency_ms: float                # 99th percentile


@dataclass
class MultimodalEvaluationResult:
    """Comprehensive evaluation result."""
    vqa_metrics: VQAMetrics
    hallucination_metrics: HallucinationMetrics
    latency_metrics: LatencyMetrics
    coverage_stats: Dict[str, float]
    consistency_stats: Dict[str, float]
    gate_stats: Dict[str, float]
    num_samples: int
    evaluation_time: float


class VQAEvaluator:
    """
    Specialized evaluator for Visual Question Answering tasks.
    Implements exact match, F1, and specialized VQA metrics.
    """
    
    def __init__(
        self,
        answer_vocab: Optional[List[str]] = None,
        use_stemming: bool = True,
        case_sensitive: bool = False
    ):
        self.answer_vocab = answer_vocab or []
        self.use_stemming = use_stemming
        self.case_sensitive = case_sensitive
        
        # Common VQA answer patterns
        self.number_pattern = re.compile(r'^\d+(\.\d+)?$')
        self.yes_no_pattern = re.compile(r'^(yes|no)$', re.IGNORECASE)
        self.color_words = {'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 
                           'pink', 'purple', 'orange', 'gray', 'grey'}
        
        # Initialize stemmer if available
        self.stemmer = None
        if use_stemming:
            try:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
            except ImportError:
                pass
    
    def _preprocess_answer(self, answer: str) -> str:
        """Preprocess answer for evaluation."""
        if not self.case_sensitive:
            answer = answer.lower()
        
        # Remove articles and common stop words
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were'}
        tokens = answer.split()
        tokens = [t for t in tokens if t not in stop_words]
        
        # Apply stemming if available
        if self.stemmer is not None:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        return ' '.join(tokens).strip()
    
    def _compute_exact_match(self, predicted: str, ground_truth: List[str]) -> float:
        """Compute exact match score."""
        pred_processed = self._preprocess_answer(predicted)
        
        for gt in ground_truth:
            gt_processed = self._preprocess_answer(gt)
            if pred_processed == gt_processed:
                return 1.0
        
        return 0.0
    
    def _compute_f1_score(self, predicted: str, ground_truth: List[str]) -> float:
        """Compute F1 score between prediction and ground truth."""
        pred_tokens = set(self._preprocess_answer(predicted).split())
        
        max_f1 = 0.0
        for gt in ground_truth:
            gt_tokens = set(self._preprocess_answer(gt).split())
            
            if len(pred_tokens) == 0 and len(gt_tokens) == 0:
                f1 = 1.0
            elif len(pred_tokens) == 0 or len(gt_tokens) == 0:
                f1 = 0.0
            else:
                intersection = pred_tokens & gt_tokens
                precision = len(intersection) / len(pred_tokens)
                recall = len(intersection) / len(gt_tokens)
                
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
            
            max_f1 = max(max_f1, f1)
        
        return max_f1
    
    def _compute_answer_relevance(self, predicted: str, question: str, ground_truth: List[str]) -> float:
        """Compute relevance of answer to question (simplified heuristic)."""
        pred_lower = predicted.lower()
        question_lower = question.lower()
        
        relevance_score = 0.0
        
        # Check for question type alignment
        if 'how many' in question_lower or 'count' in question_lower:
            if self.number_pattern.match(predicted.strip()):
                relevance_score += 0.5
        
        if any(word in question_lower for word in ['is', 'are', 'does', 'do', 'can', 'will']):
            if self.yes_no_pattern.match(predicted.strip()):
                relevance_score += 0.5
        
        if 'color' in question_lower or 'what color' in question_lower:
            pred_tokens = set(pred_lower.split())
            if pred_tokens & self.color_words:
                relevance_score += 0.5
        
        # Check overlap with ground truth concepts
        gt_tokens = set()
        for gt in ground_truth:
            gt_tokens.update(self._preprocess_answer(gt).split())
        
        pred_tokens = set(self._preprocess_answer(predicted).split())
        if len(gt_tokens) > 0:
            overlap = len(pred_tokens & gt_tokens) / len(gt_tokens)
            relevance_score += overlap * 0.5
        
        return min(relevance_score, 1.0)
    
    def evaluate_batch(
        self, 
        predictions: List[str],
        ground_truths: List[List[str]],  # Multiple ground truths per sample
        questions: List[str]
    ) -> VQAMetrics:
        """
        Evaluate a batch of VQA predictions.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answer lists  
            questions: List of questions
            
        Returns:
            VQA metrics
        """
        if len(predictions) != len(ground_truths) or len(predictions) != len(questions):
            raise ValueError("Predictions, ground truths, and questions must have same length")
        
        exact_matches = []
        f1_scores = []
        accuracies = []
        relevance_scores = []
        
        for pred, gt_list, question in zip(predictions, ground_truths, questions):
            # Exact match
            em = self._compute_exact_match(pred, gt_list)
            exact_matches.append(em)
            
            # F1 score
            f1 = self._compute_f1_score(pred, gt_list)
            f1_scores.append(f1)
            
            # Accuracy (same as EM for VQA)
            accuracies.append(em)
            
            # Answer relevance
            relevance = self._compute_answer_relevance(pred, question, gt_list)
            relevance_scores.append(relevance)
        
        return VQAMetrics(
            exact_match=np.mean(exact_matches),
            f1_score=np.mean(f1_scores),
            accuracy=np.mean(accuracies),
            bleu_score=None,  # Could be computed if needed
            rouge_l=None,     # Could be computed if needed
            answer_relevance=np.mean(relevance_scores)
        )


class HallucinationDetector:
    """
    Detects various types of hallucinations in multimodal outputs.
    """
    
    def __init__(
        self,
        object_vocab: Optional[List[str]] = None,
        attribute_vocab: Optional[Dict[str, List[str]]] = None,
        spatial_relations: Optional[List[str]] = None
    ):
        # Common object vocabulary (simplified COCO classes)
        self.object_vocab = object_vocab or [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Attribute vocabulary
        self.attribute_vocab = attribute_vocab or {
            'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'pink', 
                     'purple', 'orange', 'gray', 'grey'],
            'size': ['small', 'large', 'big', 'tiny', 'huge', 'medium'],
            'material': ['wooden', 'metal', 'plastic', 'glass', 'fabric', 'leather'],
            'shape': ['round', 'square', 'rectangular', 'circular', 'oval', 'triangular']
        }
        
        # Spatial relations
        self.spatial_relations = spatial_relations or [
            'left of', 'right of', 'above', 'below', 'in front of', 'behind', 'next to',
            'near', 'far from', 'inside', 'outside', 'on top of', 'under', 'between'
        ]
        
        # Compile regex patterns
        self.object_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(obj) for obj in self.object_vocab) + r')\b',
            re.IGNORECASE
        )
        
        all_attributes = []
        for attr_list in self.attribute_vocab.values():
            all_attributes.extend(attr_list)
        self.attribute_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(attr) for attr in all_attributes) + r')\b',
            re.IGNORECASE
        )
        
        self.spatial_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(rel) for rel in self.spatial_relations) + r')\b',
            re.IGNORECASE
        )
    
    def _extract_mentioned_objects(self, text: str) -> List[str]:
        """Extract objects mentioned in text."""
        matches = self.object_pattern.findall(text)
        return [match.lower() for match in matches]
    
    def _extract_mentioned_attributes(self, text: str) -> List[str]:
        """Extract attributes mentioned in text."""
        matches = self.attribute_pattern.findall(text)
        return [match.lower() for match in matches]
    
    def _extract_spatial_relations(self, text: str) -> List[str]:
        """Extract spatial relations mentioned in text."""
        matches = self.spatial_pattern.findall(text)
        return [match.lower() for match in matches]
    
    def detect_object_hallucinations(
        self,
        text: str,
        detected_objects: List[str]
    ) -> float:
        """
        Detect object hallucinations.
        
        Args:
            text: Generated text
            detected_objects: Objects actually detected in image
            
        Returns:
            Hallucination rate (0.0 = no hallucinations, 1.0 = all hallucinated)
        """
        mentioned_objects = self._extract_mentioned_objects(text)
        detected_objects_lower = [obj.lower() for obj in detected_objects]
        
        if not mentioned_objects:
            return 0.0  # No objects mentioned, no hallucination
        
        hallucinated_count = 0
        for obj in mentioned_objects:
            if obj not in detected_objects_lower:
                hallucinated_count += 1
        
        return hallucinated_count / len(mentioned_objects)
    
    def detect_attribute_hallucinations(
        self,
        text: str,
        object_attributes: Dict[str, List[str]]  # object -> list of attributes
    ) -> float:
        """
        Detect attribute hallucinations.
        
        Args:
            text: Generated text
            object_attributes: Dict mapping objects to their detected attributes
            
        Returns:
            Hallucination rate
        """
        mentioned_attributes = self._extract_mentioned_attributes(text)
        mentioned_objects = self._extract_mentioned_objects(text)
        
        if not mentioned_attributes:
            return 0.0
        
        # Get all valid attributes from detected objects
        valid_attributes = set()
        for obj in mentioned_objects:
            if obj in object_attributes:
                valid_attributes.update(attr.lower() for attr in object_attributes[obj])
        
        if not valid_attributes:
            # If no valid attributes available, can't determine hallucination
            return 0.0
        
        hallucinated_count = 0
        for attr in mentioned_attributes:
            if attr not in valid_attributes:
                hallucinated_count += 1
        
        return hallucinated_count / len(mentioned_attributes)
    
    def detect_spatial_hallucinations(
        self,
        text: str,
        spatial_facts: List[str]  # List of true spatial relationships
    ) -> float:
        """
        Detect spatial relationship hallucinations.
        
        Args:
            text: Generated text
            spatial_facts: List of true spatial relationships
            
        Returns:
            Hallucination rate
        """
        mentioned_relations = self._extract_spatial_relations(text)
        
        if not mentioned_relations:
            return 0.0
        
        # Simplified: check if mentioned relations are in facts
        # In practice, would need more sophisticated parsing
        spatial_facts_lower = [fact.lower() for fact in spatial_facts]
        
        hallucinated_count = 0
        for relation in mentioned_relations:
            # Check if this relation appears in any fact
            relation_found = any(relation in fact for fact in spatial_facts_lower)
            if not relation_found:
                hallucinated_count += 1
        
        return hallucinated_count / len(mentioned_relations)
    
    def compute_factual_consistency(
        self,
        text: str,
        visual_facts: List[str]  # List of facts extracted from image
    ) -> float:
        """
        Compute overall factual consistency with visual content.
        
        Args:
            text: Generated text
            visual_facts: List of visual facts
            
        Returns:
            Consistency score (0.0 = inconsistent, 1.0 = fully consistent)
        """
        if not visual_facts:
            return 1.0  # No facts to contradict
        
        text_lower = text.lower()
        
        consistent_facts = 0
        for fact in visual_facts:
            fact_lower = fact.lower()
            # Simple keyword matching - could be improved with semantic similarity
            if any(word in text_lower for word in fact_lower.split()):
                consistent_facts += 1
        
        return consistent_facts / len(visual_facts) if visual_facts else 1.0
    
    def evaluate_hallucinations(
        self,
        texts: List[str],
        detected_objects_list: List[List[str]],
        object_attributes_list: List[Dict[str, List[str]]],
        spatial_facts_list: List[List[str]],
        visual_facts_list: List[List[str]]
    ) -> HallucinationMetrics:
        """
        Evaluate hallucinations across a batch of texts.
        
        Args:
            texts: Generated texts
            detected_objects_list: Detected objects for each image
            object_attributes_list: Object attributes for each image
            spatial_facts_list: Spatial facts for each image
            visual_facts_list: Visual facts for each image
            
        Returns:
            Hallucination metrics
        """
        object_hallucination_rates = []
        attribute_hallucination_rates = []
        spatial_hallucination_rates = []
        factual_consistency_scores = []
        
        for i, text in enumerate(texts):
            # Object hallucinations
            obj_rate = self.detect_object_hallucinations(
                text, detected_objects_list[i]
            )
            object_hallucination_rates.append(obj_rate)
            
            # Attribute hallucinations
            attr_rate = self.detect_attribute_hallucinations(
                text, object_attributes_list[i]
            )
            attribute_hallucination_rates.append(attr_rate)
            
            # Spatial hallucinations
            spatial_rate = self.detect_spatial_hallucinations(
                text, spatial_facts_list[i]
            )
            spatial_hallucination_rates.append(spatial_rate)
            
            # Factual consistency
            factual_score = self.compute_factual_consistency(
                text, visual_facts_list[i]
            )
            factual_consistency_scores.append(factual_score)
        
        # Overall hallucination rate
        overall_rates = []
        for obj, attr, spatial in zip(object_hallucination_rates, 
                                     attribute_hallucination_rates,
                                     spatial_hallucination_rates):
            overall = (obj + attr + spatial) / 3
            overall_rates.append(overall)
        
        return HallucinationMetrics(
            object_hallucination_rate=np.mean(object_hallucination_rates),
            attribute_hallucination_rate=np.mean(attribute_hallucination_rates),
            spatial_hallucination_rate=np.mean(spatial_hallucination_rates),
            factual_consistency_score=np.mean(factual_consistency_scores),
            overall_hallucination_rate=np.mean(overall_rates)
        )


class LatencyProfiler:
    """
    Profiles latency of multimodal inference pipeline.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all timing measurements."""
        self.preprocessing_times = []
        self.inference_times = []
        self.postprocessing_times = []
        self.total_times = []
    
    def profile_batch(
        self,
        preprocessing_fn,
        inference_fn,
        postprocessing_fn,
        inputs,
        batch_size: int = 1
    ) -> LatencyMetrics:
        """
        Profile a batch of multimodal inference.
        
        Args:
            preprocessing_fn: Function for vision preprocessing
            inference_fn: Function for model inference
            postprocessing_fn: Function for output postprocessing
            inputs: Input batch
            batch_size: Batch size
            
        Returns:
            Latency metrics
        """
        batch_preprocessing_times = []
        batch_inference_times = []
        batch_postprocessing_times = []
        batch_total_times = []
        
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size]
            
            total_start = time.time()
            
            # Preprocessing
            preprocess_start = time.time()
            preprocessed = preprocessing_fn(batch_inputs)
            preprocess_time = (time.time() - preprocess_start) * 1000  # ms
            
            # Inference
            inference_start = time.time()
            outputs = inference_fn(preprocessed)
            inference_time = (time.time() - inference_start) * 1000  # ms
            
            # Postprocessing
            postprocess_start = time.time()
            final_outputs = postprocessing_fn(outputs)
            postprocess_time = (time.time() - postprocess_start) * 1000  # ms
            
            total_time = (time.time() - total_start) * 1000  # ms
            
            # Store per-sample times (divide by batch size)
            samples_in_batch = len(batch_inputs)
            batch_preprocessing_times.extend([preprocess_time / samples_in_batch] * samples_in_batch)
            batch_inference_times.extend([inference_time / samples_in_batch] * samples_in_batch)
            batch_postprocessing_times.extend([postprocess_time / samples_in_batch] * samples_in_batch)
            batch_total_times.extend([total_time / samples_in_batch] * samples_in_batch)
        
        # Update cumulative times
        self.preprocessing_times.extend(batch_preprocessing_times)
        self.inference_times.extend(batch_inference_times)
        self.postprocessing_times.extend(batch_postprocessing_times)
        self.total_times.extend(batch_total_times)
        
        # Compute percentiles
        total_times_array = np.array(batch_total_times)
        p50 = np.percentile(total_times_array, 50)
        p95 = np.percentile(total_times_array, 95)
        p99 = np.percentile(total_times_array, 99)
        
        return LatencyMetrics(
            preprocessing_latency_ms=np.mean(batch_preprocessing_times),
            inference_latency_ms=np.mean(batch_inference_times),
            postprocessing_latency_ms=np.mean(batch_postprocessing_times),
            total_latency_ms=np.mean(batch_total_times),
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99
        )
    
    def get_cumulative_metrics(self) -> LatencyMetrics:
        """Get cumulative latency metrics across all profiled batches."""
        if not self.total_times:
            return LatencyMetrics(0, 0, 0, 0, 0, 0, 0)
        
        total_times_array = np.array(self.total_times)
        p50 = np.percentile(total_times_array, 50)
        p95 = np.percentile(total_times_array, 95)
        p99 = np.percentile(total_times_array, 99)
        
        return LatencyMetrics(
            preprocessing_latency_ms=np.mean(self.preprocessing_times),
            inference_latency_ms=np.mean(self.inference_times),
            postprocessing_latency_ms=np.mean(self.postprocessing_times),
            total_latency_ms=np.mean(self.total_times),
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99
        )


class MultimodalEvaluator:
    """
    Comprehensive evaluator for multimodal BEM systems.
    Combines VQA evaluation, hallucination detection, and latency profiling.
    """
    
    def __init__(
        self,
        vqa_config: Optional[Dict[str, Any]] = None,
        hallucination_config: Optional[Dict[str, Any]] = None
    ):
        vqa_config = vqa_config or {}
        hallucination_config = hallucination_config or {}
        
        self.vqa_evaluator = VQAEvaluator(**vqa_config)
        self.hallucination_detector = HallucinationDetector(**hallucination_config)
        self.latency_profiler = LatencyProfiler()
    
    def evaluate_multimodal_system(
        self,
        model,
        vision_encoder,
        test_data: List[Dict[str, Any]],  # List of {image, question, answers, visual_facts}
        batch_size: int = 8
    ) -> MultimodalEvaluationResult:
        """
        Comprehensive evaluation of multimodal system.
        
        Args:
            model: Multimodal BEM model
            vision_encoder: Vision encoder
            test_data: Test dataset
            batch_size: Evaluation batch size
            
        Returns:
            Comprehensive evaluation result
        """
        start_time = time.time()
        
        all_predictions = []
        all_ground_truths = []
        all_questions = []
        all_visual_facts = []
        all_detected_objects = []
        all_object_attributes = []
        all_spatial_facts = []
        
        coverage_scores = []
        consistency_scores = []
        gate_activations = []
        
        # Process in batches
        for i in range(0, len(test_data), batch_size):
            batch_data = test_data[i:i + batch_size]
            
            # Extract batch components
            batch_images = [item['image'] for item in batch_data]
            batch_questions = [item['question'] for item in batch_data]
            batch_answers = [item['answers'] for item in batch_data]
            batch_visual_facts = [item.get('visual_facts', []) for item in batch_data]
            batch_detected_objects = [item.get('detected_objects', []) for item in batch_data]
            batch_object_attributes = [item.get('object_attributes', {}) for item in batch_data]
            batch_spatial_facts = [item.get('spatial_facts', []) for item in batch_data]
            
            # Define processing functions for latency profiling
            def preprocess_batch(images):
                return vision_encoder.preprocess_images(images)
            
            def inference_batch(pixel_values):
                with torch.no_grad():
                    # Extract vision features
                    vision_features = vision_encoder(pixel_values)
                    
                    # Generate text (simplified - would need actual text generation)
                    # For now, return mock predictions
                    batch_preds = ["mock prediction"] * len(pixel_values)
                    
                    return {
                        'predictions': batch_preds,
                        'vision_features': vision_features
                    }
            
            def postprocess_batch(outputs):
                return outputs['predictions']
            
            # Profile latency
            latency_metrics = self.latency_profiler.profile_batch(
                preprocess_batch,
                inference_batch,
                postprocess_batch,
                batch_images,
                batch_size
            )
            
            # Get actual predictions (mock for now)
            pixel_values = preprocess_batch(batch_images)
            outputs = inference_batch(pixel_values)
            predictions = postprocess_batch(outputs)
            
            # Collect results
            all_predictions.extend(predictions)
            all_ground_truths.extend(batch_answers)
            all_questions.extend(batch_questions)
            all_visual_facts.extend(batch_visual_facts)
            all_detected_objects.extend(batch_detected_objects)
            all_object_attributes.extend(batch_object_attributes)
            all_spatial_facts.extend(batch_spatial_facts)
            
            # Collect multimodal metrics
            vision_features = outputs['vision_features']
            if hasattr(vision_features, 'coverage_score') and vision_features.coverage_score is not None:
                coverage_scores.extend(vision_features.coverage_score.cpu().numpy())
            if hasattr(vision_features, 'consistency_score') and vision_features.consistency_score is not None:
                consistency_scores.extend(vision_features.consistency_score.cpu().numpy())
        
        # Evaluate VQA metrics
        vqa_metrics = self.vqa_evaluator.evaluate_batch(
            all_predictions, all_ground_truths, all_questions
        )
        
        # Evaluate hallucination metrics
        hallucination_metrics = self.hallucination_detector.evaluate_hallucinations(
            all_predictions,
            all_detected_objects,
            all_object_attributes,
            all_spatial_facts,
            all_visual_facts
        )
        
        # Get cumulative latency metrics
        cumulative_latency = self.latency_profiler.get_cumulative_metrics()
        
        # Compute statistics
        coverage_stats = {
            'mean': np.mean(coverage_scores) if coverage_scores else 0.0,
            'std': np.std(coverage_scores) if coverage_scores else 0.0,
            'min': np.min(coverage_scores) if coverage_scores else 0.0,
            'max': np.max(coverage_scores) if coverage_scores else 0.0
        }
        
        consistency_stats = {
            'mean': np.mean(consistency_scores) if consistency_scores else 0.0,
            'std': np.std(consistency_scores) if consistency_scores else 0.0,
            'min': np.min(consistency_scores) if consistency_scores else 0.0,
            'max': np.max(consistency_scores) if consistency_scores else 0.0
        }
        
        gate_stats = {
            'activation_rate': np.mean(gate_activations) if gate_activations else 0.0,
            'total_processed': len(all_predictions)
        }
        
        evaluation_time = time.time() - start_time
        
        return MultimodalEvaluationResult(
            vqa_metrics=vqa_metrics,
            hallucination_metrics=hallucination_metrics,
            latency_metrics=cumulative_latency,
            coverage_stats=coverage_stats,
            consistency_stats=consistency_stats,
            gate_stats=gate_stats,
            num_samples=len(test_data),
            evaluation_time=evaluation_time
        )
    
    def save_results(self, results: MultimodalEvaluationResult, output_path: Path):
        """Save evaluation results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict for JSON serialization
        results_dict = {}
        for field_name in results.__dataclass_fields__:
            value = getattr(results, field_name)
            if hasattr(value, '_asdict'):  # NamedTuple
                results_dict[field_name] = value._asdict()
            else:
                results_dict[field_name] = value
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def print_summary(self, results: MultimodalEvaluationResult):
        """Print evaluation summary."""
        print("🎯 BEM 2.0 Multimodal Evaluation Results")
        print("=" * 50)
        
        print("\n📊 VQA Metrics:")
        print(f"  Exact Match: {results.vqa_metrics.exact_match:.3f}")
        print(f"  F1 Score: {results.vqa_metrics.f1_score:.3f}")
        print(f"  Accuracy: {results.vqa_metrics.accuracy:.3f}")
        print(f"  Answer Relevance: {results.vqa_metrics.answer_relevance:.3f}")
        
        print("\n👻 Hallucination Metrics:")
        print(f"  Object Hallucination: {results.hallucination_metrics.object_hallucination_rate:.3f}")
        print(f"  Attribute Hallucination: {results.hallucination_metrics.attribute_hallucination_rate:.3f}")
        print(f"  Spatial Hallucination: {results.hallucination_metrics.spatial_hallucination_rate:.3f}")
        print(f"  Factual Consistency: {results.hallucination_metrics.factual_consistency_score:.3f}")
        print(f"  Overall Hallucination: {results.hallucination_metrics.overall_hallucination_rate:.3f}")
        
        print("\n⚡ Latency Metrics:")
        print(f"  Preprocessing: {results.latency_metrics.preprocessing_latency_ms:.1f}ms")
        print(f"  Inference: {results.latency_metrics.inference_latency_ms:.1f}ms")
        print(f"  Postprocessing: {results.latency_metrics.postprocessing_latency_ms:.1f}ms")
        print(f"  Total (mean): {results.latency_metrics.total_latency_ms:.1f}ms")
        print(f"  P50: {results.latency_metrics.p50_latency_ms:.1f}ms")
        print(f"  P95: {results.latency_metrics.p95_latency_ms:.1f}ms")
        
        print(f"\n📈 Coverage & Consistency:")
        print(f"  Coverage (mean): {results.coverage_stats['mean']:.3f}")
        print(f"  Consistency (mean): {results.consistency_stats['mean']:.3f}")
        print(f"  Gate Activation Rate: {results.gate_stats['activation_rate']:.3f}")
        
        print(f"\n⏱️  Evaluation Time: {results.evaluation_time:.1f}s")
        print(f"   Samples Evaluated: {results.num_samples}")


def create_vqa_test_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Create mock VQA test data for evaluation."""
    test_data = []
    
    for i in range(num_samples):
        # Mock data - in practice would load real VQA dataset
        item = {
            'image': f"mock_image_{i}",  # Would be actual image
            'question': f"What is in the image {i}?",
            'answers': [f"object_{i}", f"item_{i}"],  # Multiple valid answers
            'visual_facts': [f"there is an object_{i} in the image"],
            'detected_objects': [f"object_{i}"],
            'object_attributes': {f"object_{i}": ["red", "large"]},
            'spatial_facts': [f"object_{i} is in the center"]
        }
        test_data.append(item)
    
    return test_data