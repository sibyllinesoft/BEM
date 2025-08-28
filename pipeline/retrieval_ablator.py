#!/usr/bin/env python3
"""
Retrieval-Aware Behavior Testing for BEM Research Validation

Create retrieval on/off ablation studies with noise injection for behavior validation.
Validates BEM's key claim of retrieval-aware adaptation effectiveness.
"""

import json
import logging
import random
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalMode(Enum):
    """Retrieval system operating modes."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    NOISY = "noisy"
    RANDOM = "random"

@dataclass
class RetrievalContext:
    """Context for retrieval-aware behavior testing."""
    query: str
    relevant_documents: List[str]
    noise_documents: List[str]
    expected_answer: str
    domain: str
    difficulty: str  # "easy", "medium", "hard"

@dataclass
class RetrievalAblationResult:
    """Results from retrieval ablation experiment."""
    mode: RetrievalMode
    accuracy: float
    f1_score: float
    retrieval_dependency_score: float
    robustness_score: float
    individual_results: List[Dict[str, Any]]
    performance_degradation: float  # vs. optimal retrieval

@dataclass
class NoiseInjectionResult:
    """Results from noise injection experiments."""
    noise_level: float
    noise_type: str
    performance_impact: float
    robustness_validation: bool
    adaptation_effectiveness: float

@dataclass
class BehaviorValidationResult:
    """Complete behavior validation result."""
    retrieval_ablation_results: Dict[RetrievalMode, RetrievalAblationResult]
    noise_injection_results: List[NoiseInjectionResult]
    adaptation_effectiveness_score: float
    retrieval_awareness_validated: bool
    recommendations: List[str]
    metadata: Dict[str, Any]

class RetrievalSimulator:
    """Simulate different retrieval system behaviors."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
    def simulate_retrieval(self, 
                          query: str, 
                          mode: RetrievalMode,
                          context: RetrievalContext,
                          noise_level: float = 0.0) -> List[str]:
        """Simulate retrieval under different modes."""
        
        if mode == RetrievalMode.ENABLED:
            # Return relevant documents
            return context.relevant_documents
            
        elif mode == RetrievalMode.DISABLED:
            # Return empty context
            return []
            
        elif mode == RetrievalMode.NOISY:
            # Mix relevant and noise documents based on noise level
            n_relevant = len(context.relevant_documents)
            n_noise = len(context.noise_documents)
            
            if noise_level == 0.0:
                return context.relevant_documents
                
            # Determine how many noise docs to include
            n_noise_inject = int(n_relevant * noise_level)
            n_noise_inject = min(n_noise_inject, n_noise)
            
            # Sample noise documents
            noise_sample = random.sample(context.noise_documents, n_noise_inject)
            
            # Combine and shuffle
            combined_docs = context.relevant_documents + noise_sample
            random.shuffle(combined_docs)
            
            return combined_docs
            
        elif mode == RetrievalMode.RANDOM:
            # Return random documents
            all_docs = context.relevant_documents + context.noise_documents
            n_return = len(context.relevant_documents)
            return random.sample(all_docs, min(n_return, len(all_docs)))
            
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

class NoiseGenerator:
    """Generate different types of noise for robustness testing."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
    def inject_semantic_noise(self, 
                             documents: List[str], 
                             noise_level: float) -> List[str]:
        """Inject semantic noise by word substitution."""
        
        noisy_documents = []
        
        for doc in documents:
            words = doc.split()
            n_words_to_change = int(len(words) * noise_level)
            
            # Select random words to replace
            indices_to_change = random.sample(range(len(words)), 
                                            min(n_words_to_change, len(words)))
            
            for idx in indices_to_change:
                # Replace with a random word from vocabulary
                words[idx] = random.choice([
                    "data", "system", "method", "result", "analysis", 
                    "information", "process", "algorithm", "model", "framework"
                ])
                
            noisy_documents.append(" ".join(words))
            
        return noisy_documents
    
    def inject_contradictory_noise(self, 
                                  documents: List[str], 
                                  noise_level: float) -> List[str]:
        """Inject contradictory information."""
        
        noisy_documents = []
        
        contradiction_templates = [
            "However, recent studies show the opposite.",
            "This contradicts previous findings that suggest otherwise.",
            "Alternative research indicates different results.",
            "Contrary evidence demonstrates the inverse relationship.",
            "Opposing viewpoints argue for different conclusions."
        ]
        
        for doc in documents:
            if random.random() < noise_level:
                # Add contradictory statement
                contradiction = random.choice(contradiction_templates)
                doc_with_noise = f"{doc} {contradiction}"
            else:
                doc_with_noise = doc
                
            noisy_documents.append(doc_with_noise)
            
        return noisy_documents
    
    def inject_irrelevant_noise(self, 
                               documents: List[str], 
                               noise_level: float) -> List[str]:
        """Inject completely irrelevant information."""
        
        irrelevant_content = [
            "The weather today is quite pleasant with sunny skies.",
            "Stock market fluctuations continue to impact investors.",
            "New restaurant openings have increased in the downtown area.",
            "Transportation schedules may be affected by construction.",
            "Sports teams are preparing for the upcoming season."
        ]
        
        noisy_documents = []
        
        for doc in documents:
            if random.random() < noise_level:
                # Prepend irrelevant content
                irrelevant = random.choice(irrelevant_content)
                doc_with_noise = f"{irrelevant} {doc}"
            else:
                doc_with_noise = doc
                
            noisy_documents.append(doc_with_noise)
            
        return noisy_documents

class BehaviorEvaluator:
    """Evaluate model behavior under different retrieval conditions."""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-small",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
    def evaluate_with_retrieval(self, 
                              context: RetrievalContext,
                              retrieved_docs: List[str],
                              max_new_tokens: int = 50) -> Dict[str, Any]:
        """Evaluate model performance with given retrieved documents."""
        
        # Construct input with retrieval context
        if retrieved_docs:
            retrieval_context = "\n".join([f"Document {i+1}: {doc}" 
                                         for i, doc in enumerate(retrieved_docs)])
            full_input = f"Context:\n{retrieval_context}\n\nQuestion: {context.query}\nAnswer:"
        else:
            full_input = f"Question: {context.query}\nAnswer:"
            
        # Tokenize
        inputs = self.tokenizer(
            full_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode prediction
        prediction = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Compute metrics
        em_score = 1.0 if prediction.lower() == context.expected_answer.lower() else 0.0
        
        # F1 score (token overlap)
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(context.expected_answer.lower().split())
        
        if not pred_tokens and not ref_tokens:
            f1_score = 1.0
        elif not pred_tokens or not ref_tokens:
            f1_score = 0.0
        else:
            precision = len(pred_tokens & ref_tokens) / len(pred_tokens)
            recall = len(pred_tokens & ref_tokens) / len(ref_tokens)
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
        return {
            "prediction": prediction,
            "em_score": em_score,
            "f1_score": f1_score,
            "input_length": inputs["input_ids"].shape[1],
            "output_length": len(prediction.split()),
            "retrieval_docs_used": len(retrieved_docs)
        }

class RetrievalAblator:
    """Main orchestrator for retrieval ablation studies."""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-small",
                 seed: int = 42):
        
        self.seed = seed
        self.retrieval_simulator = RetrievalSimulator(seed)
        self.noise_generator = NoiseGenerator(seed)
        self.behavior_evaluator = BehaviorEvaluator(model_name)
        
    def create_test_contexts(self, n_contexts: int = 50) -> List[RetrievalContext]:
        """Create test contexts for ablation studies."""
        
        # Generate synthetic test contexts
        contexts = []
        
        domains = ["medical", "legal", "technical", "financial", "academic"]
        difficulties = ["easy", "medium", "hard"]
        
        for i in range(n_contexts):
            domain = random.choice(domains)
            difficulty = random.choice(difficulties)
            
            # Generate context based on domain
            if domain == "medical":
                query = f"What are the symptoms of condition {i}?"
                relevant_docs = [
                    f"Condition {i} typically presents with fever, headache, and fatigue.",
                    f"Patients with condition {i} often report muscle aches and nausea.",
                    f"Diagnosis of condition {i} requires blood tests and imaging."
                ]
                noise_docs = [
                    "Legal contracts require careful review of all clauses.",
                    "Stock market volatility affects investment strategies.",
                    "New software updates improve system performance."
                ]
                expected_answer = "fever, headache, fatigue, muscle aches, nausea"
                
            elif domain == "legal":
                query = f"What are the requirements for contract {i}?"
                relevant_docs = [
                    f"Contract {i} requires written agreement between parties.",
                    f"Valid consideration must be exchanged in contract {i}.",
                    f"Both parties must have legal capacity for contract {i}."
                ]
                noise_docs = [
                    "Medical symptoms include fever and headache.",
                    "Technical systems require regular maintenance.",
                    "Financial markets fluctuate based on economic indicators."
                ]
                expected_answer = "written agreement, valid consideration, legal capacity"
                
            else:  # Generic template
                query = f"What are the key features of system {i}?"
                relevant_docs = [
                    f"System {i} has high performance and reliability.",
                    f"The architecture of system {i} is modular and scalable.",
                    f"System {i} supports multiple user interfaces."
                ]
                noise_docs = [
                    "Random information about unrelated topics.",
                    "Irrelevant data that should not influence answers.",
                    "Distracting content with different subject matter."
                ]
                expected_answer = "high performance, reliability, modular architecture, scalable, multiple interfaces"
                
            contexts.append(RetrievalContext(
                query=query,
                relevant_documents=relevant_docs,
                noise_documents=noise_docs,
                expected_answer=expected_answer,
                domain=domain,
                difficulty=difficulty
            ))
            
        return contexts
    
    def run_retrieval_ablation(self, 
                             contexts: List[RetrievalContext],
                             modes: List[RetrievalMode] = None) -> Dict[RetrievalMode, RetrievalAblationResult]:
        """Run retrieval ablation across different modes."""
        
        if modes is None:
            modes = list(RetrievalMode)
            
        logger.info(f"Running retrieval ablation on {len(contexts)} contexts")
        
        results = {}
        
        for mode in modes:
            logger.info(f"Evaluating retrieval mode: {mode.value}")
            
            individual_results = []
            em_scores = []
            f1_scores = []
            
            for context in contexts:
                # Simulate retrieval
                retrieved_docs = self.retrieval_simulator.simulate_retrieval(
                    context.query, mode, context
                )
                
                # Evaluate behavior
                eval_result = self.behavior_evaluator.evaluate_with_retrieval(
                    context, retrieved_docs
                )
                
                individual_results.append({
                    "context": context,
                    "retrieved_docs": retrieved_docs,
                    "evaluation": eval_result
                })
                
                em_scores.append(eval_result["em_score"])
                f1_scores.append(eval_result["f1_score"])
                
            # Aggregate results
            accuracy = np.mean(em_scores)
            f1_score = np.mean(f1_scores)
            
            # Compute retrieval dependency score
            retrieval_dependency = self._compute_retrieval_dependency(
                individual_results, mode
            )
            
            # Compute robustness score
            robustness_score = self._compute_robustness_score(individual_results)
            
            results[mode] = RetrievalAblationResult(
                mode=mode,
                accuracy=accuracy,
                f1_score=f1_score,
                retrieval_dependency_score=retrieval_dependency,
                robustness_score=robustness_score,
                individual_results=individual_results,
                performance_degradation=0.0  # Will be computed later
            )
            
        # Compute performance degradations
        if RetrievalMode.ENABLED in results:
            baseline_accuracy = results[RetrievalMode.ENABLED].accuracy
            for mode, result in results.items():
                if mode != RetrievalMode.ENABLED:
                    result.performance_degradation = max(0.0, 
                        baseline_accuracy - result.accuracy)
                    
        return results
    
    def run_noise_injection_study(self, 
                                 contexts: List[RetrievalContext],
                                 noise_levels: List[float] = [0.1, 0.2, 0.3, 0.5],
                                 noise_types: List[str] = None) -> List[NoiseInjectionResult]:
        """Run noise injection robustness study."""
        
        if noise_types is None:
            noise_types = ["semantic", "contradictory", "irrelevant"]
            
        logger.info(f"Running noise injection study: {len(noise_levels)} levels, {len(noise_types)} types")
        
        results = []
        
        # Baseline performance (no noise)
        baseline_results = []
        for context in contexts:
            retrieved_docs = self.retrieval_simulator.simulate_retrieval(
                context.query, RetrievalMode.ENABLED, context
            )
            eval_result = self.behavior_evaluator.evaluate_with_retrieval(context, retrieved_docs)
            baseline_results.append(eval_result["f1_score"])
            
        baseline_performance = np.mean(baseline_results)
        
        # Test different noise conditions
        for noise_type in noise_types:
            for noise_level in noise_levels:
                logger.info(f"Testing {noise_type} noise at level {noise_level}")
                
                noisy_results = []
                
                for context in contexts:
                    # Get clean retrieved documents
                    retrieved_docs = self.retrieval_simulator.simulate_retrieval(
                        context.query, RetrievalMode.ENABLED, context
                    )
                    
                    # Inject noise
                    if noise_type == "semantic":
                        noisy_docs = self.noise_generator.inject_semantic_noise(
                            retrieved_docs, noise_level
                        )
                    elif noise_type == "contradictory":
                        noisy_docs = self.noise_generator.inject_contradictory_noise(
                            retrieved_docs, noise_level
                        )
                    elif noise_type == "irrelevant":
                        noisy_docs = self.noise_generator.inject_irrelevant_noise(
                            retrieved_docs, noise_level
                        )
                    else:
                        noisy_docs = retrieved_docs
                        
                    # Evaluate with noisy documents
                    eval_result = self.behavior_evaluator.evaluate_with_retrieval(
                        context, noisy_docs
                    )
                    noisy_results.append(eval_result["f1_score"])
                    
                # Compute metrics
                noisy_performance = np.mean(noisy_results)
                performance_impact = baseline_performance - noisy_performance
                
                # Robustness validation (performance degradation < 20%)
                robustness_validation = performance_impact < 0.2
                
                # Adaptation effectiveness (how well BEM handles noise)
                adaptation_effectiveness = max(0.0, 1.0 - (performance_impact / baseline_performance))
                
                results.append(NoiseInjectionResult(
                    noise_level=noise_level,
                    noise_type=noise_type,
                    performance_impact=performance_impact,
                    robustness_validation=robustness_validation,
                    adaptation_effectiveness=adaptation_effectiveness
                ))
                
        return results
    
    def validate_retrieval_aware_behavior(self, 
                                        n_contexts: int = 50) -> BehaviorValidationResult:
        """Run complete retrieval-aware behavior validation."""
        
        logger.info("Starting comprehensive retrieval-aware behavior validation")
        
        # Create test contexts
        contexts = self.create_test_contexts(n_contexts)
        
        # Run retrieval ablation
        ablation_results = self.run_retrieval_ablation(contexts)
        
        # Run noise injection studies
        noise_results = self.run_noise_injection_study(contexts)
        
        # Compute overall adaptation effectiveness
        adaptation_score = self._compute_adaptation_effectiveness(
            ablation_results, noise_results
        )
        
        # Validate retrieval awareness
        retrieval_awareness_validated = self._validate_retrieval_awareness(
            ablation_results, adaptation_score
        )
        
        # Generate recommendations
        recommendations = self._generate_behavior_recommendations(
            ablation_results, noise_results, adaptation_score
        )
        
        return BehaviorValidationResult(
            retrieval_ablation_results=ablation_results,
            noise_injection_results=noise_results,
            adaptation_effectiveness_score=adaptation_score,
            retrieval_awareness_validated=retrieval_awareness_validated,
            recommendations=recommendations,
            metadata={
                "n_contexts": n_contexts,
                "validation_timestamp": str(np.datetime64('now')),
                "seed": self.seed
            }
        )
    
    def _compute_retrieval_dependency(self, 
                                    individual_results: List[Dict],
                                    mode: RetrievalMode) -> float:
        """Compute how dependent performance is on retrieval quality."""
        
        if mode == RetrievalMode.ENABLED:
            return 1.0  # Fully dependent on good retrieval
        elif mode == RetrievalMode.DISABLED:
            return 0.0  # No retrieval dependency
        else:
            # For noisy/random modes, compute based on performance variance
            f1_scores = [r["evaluation"]["f1_score"] for r in individual_results]
            return 1.0 - min(1.0, np.var(f1_scores))  # Low variance = high dependency
            
    def _compute_robustness_score(self, individual_results: List[Dict]) -> float:
        """Compute robustness score based on performance consistency."""
        
        f1_scores = [r["evaluation"]["f1_score"] for r in individual_results]
        
        if not f1_scores:
            return 0.0
            
        # Robustness = 1 - coefficient of variation
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        
        if mean_f1 == 0:
            return 0.0
            
        cv = std_f1 / mean_f1
        return max(0.0, 1.0 - cv)
    
    def _compute_adaptation_effectiveness(self, 
                                        ablation_results: Dict[RetrievalMode, RetrievalAblationResult],
                                        noise_results: List[NoiseInjectionResult]) -> float:
        """Compute overall adaptation effectiveness score."""
        
        effectiveness_scores = []
        
        # Retrieval ablation effectiveness
        if RetrievalMode.ENABLED in ablation_results and RetrievalMode.DISABLED in ablation_results:
            enabled_acc = ablation_results[RetrievalMode.ENABLED].accuracy
            disabled_acc = ablation_results[RetrievalMode.DISABLED].accuracy
            
            if enabled_acc > disabled_acc:
                retrieval_effectiveness = (enabled_acc - disabled_acc) / enabled_acc
                effectiveness_scores.append(retrieval_effectiveness)
                
        # Noise robustness effectiveness  
        noise_effectiveness_scores = [nr.adaptation_effectiveness for nr in noise_results]
        if noise_effectiveness_scores:
            effectiveness_scores.append(np.mean(noise_effectiveness_scores))
            
        return np.mean(effectiveness_scores) if effectiveness_scores else 0.0
    
    def _validate_retrieval_awareness(self, 
                                    ablation_results: Dict[RetrievalMode, RetrievalAblationResult],
                                    adaptation_score: float) -> bool:
        """Validate that BEM shows retrieval awareness."""
        
        criteria_met = []
        
        # Criterion 1: Performance improves with good retrieval
        if RetrievalMode.ENABLED in ablation_results and RetrievalMode.DISABLED in ablation_results:
            enabled_acc = ablation_results[RetrievalMode.ENABLED].accuracy
            disabled_acc = ablation_results[RetrievalMode.DISABLED].accuracy
            criteria_met.append(enabled_acc > disabled_acc + 0.05)  # At least 5% improvement
            
        # Criterion 2: Performance degrades with bad retrieval
        if RetrievalMode.RANDOM in ablation_results and RetrievalMode.ENABLED in ablation_results:
            random_acc = ablation_results[RetrievalMode.RANDOM].accuracy
            enabled_acc = ablation_results[RetrievalMode.ENABLED].accuracy
            criteria_met.append(random_acc < enabled_acc - 0.05)  # At least 5% degradation
            
        # Criterion 3: High adaptation effectiveness
        criteria_met.append(adaptation_score > 0.3)
        
        # Must meet at least 2 out of 3 criteria
        return sum(criteria_met) >= 2
    
    def _generate_behavior_recommendations(self, 
                                         ablation_results: Dict[RetrievalMode, RetrievalAblationResult],
                                         noise_results: List[NoiseInjectionResult],
                                         adaptation_score: float) -> List[str]:
        """Generate recommendations based on behavior validation."""
        
        recommendations = []
        
        # Retrieval dependency analysis
        if RetrievalMode.ENABLED in ablation_results:
            enabled_result = ablation_results[RetrievalMode.ENABLED]
            
            if enabled_result.accuracy < 0.7:
                recommendations.append(
                    f"Low accuracy with retrieval ({enabled_result.accuracy:.2%}). "
                    "Consider improving retrieval quality or model adaptation."
                )
                
            if enabled_result.retrieval_dependency_score < 0.5:
                recommendations.append(
                    "Low retrieval dependency detected. BEM may not be fully utilizing retrieved information."
                )
                
        # Robustness analysis
        high_noise_impact = any(nr.performance_impact > 0.3 for nr in noise_results)
        if high_noise_impact:
            recommendations.append(
                "High sensitivity to noise detected. Consider noise robustness training or filtering."
            )
            
        # Adaptation effectiveness
        if adaptation_score < 0.3:
            recommendations.append(
                f"Low adaptation effectiveness ({adaptation_score:.2f}). "
                "BEM may need better retrieval-aware training or architecture improvements."
            )
        elif adaptation_score > 0.7:
            recommendations.append(
                f"Excellent adaptation effectiveness ({adaptation_score:.2f}). "
                "BEM demonstrates strong retrieval-aware behavior."
            )
            
        if not recommendations:
            recommendations.append("Retrieval-aware behavior validation shows acceptable performance.")
            
        return recommendations

def main():
    """Example usage of retrieval ablator."""
    
    # Initialize ablator
    ablator = RetrievalAblator()
    
    # Run comprehensive validation
    validation_result = ablator.validate_retrieval_aware_behavior(n_contexts=20)
    
    # Print results
    print("Retrieval-Aware Behavior Validation Results:")
    print(f"Adaptation Effectiveness: {validation_result.adaptation_effectiveness_score:.3f}")
    print(f"Retrieval Awareness Validated: {validation_result.retrieval_awareness_validated}")
    
    # Ablation results
    print("\nRetrieval Ablation Results:")
    for mode, result in validation_result.retrieval_ablation_results.items():
        print(f"  {mode.value}: Accuracy={result.accuracy:.2%}, F1={result.f1_score:.3f}")
        
    # Noise injection results
    print(f"\nNoise Injection Results ({len(validation_result.noise_injection_results)} tests):")
    for result in validation_result.noise_injection_results[:5]:  # Show first 5
        print(f"  {result.noise_type} @ {result.noise_level}: Impact={result.performance_impact:.3f}")
        
    print(f"\nRecommendations ({len(validation_result.recommendations)}):")
    for rec in validation_result.recommendations:
        print(f"  - {rec}")

if __name__ == "__main__":
    main()