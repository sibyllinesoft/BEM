#!/usr/bin/env python3
"""
Comprehensive Baseline Suite for BEM Research Validation

Unified evaluation framework for all MoE-LoRA baselines:
- Static LoRA, AdaLoRA, LoRAHub, MoELoRA, Switch-LoRA, QLoRA

Provides standardized interfaces for fair comparison with resource monitoring.
"""

import json
import time
import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import peft
from peft import LoraConfig, get_peft_model, TaskType
import psutil
import GPUtil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Results from baseline evaluation."""
    method_name: str
    em_score: float
    f1_score: float
    exact_match_scores: List[float]
    f1_scores: List[float]
    latency_ms: float
    vram_peak_mb: float
    parameter_count: int
    tokens_per_second: float
    metadata: Dict[str, Any]

@dataclass 
class ResourceMetrics:
    """Resource usage metrics during evaluation."""
    peak_vram_mb: float
    peak_ram_mb: float
    avg_gpu_utilization: float
    inference_latency_ms: float
    tokens_per_second: float

class ResourceMonitor:
    """Monitor GPU/CPU resources during evaluation."""
    
    def __init__(self):
        self.start_time = None
        self.start_vram = None
        self.peak_vram = 0.0
        self.peak_ram = 0.0
        self.gpu_utilizations = []
        
    def start_monitoring(self):
        """Start resource monitoring."""
        self.start_time = time.time()
        
        # Get initial VRAM usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.start_vram = torch.cuda.memory_allocated() / (1024**2)  # MB
        
        self.peak_vram = self.start_vram or 0.0
        self.peak_ram = psutil.virtual_memory().used / (1024**2)
        
    def update_metrics(self):
        """Update resource metrics."""
        if torch.cuda.is_available():
            current_vram = torch.cuda.memory_allocated() / (1024**2)
            self.peak_vram = max(self.peak_vram, current_vram)
            
        current_ram = psutil.virtual_memory().used / (1024**2) 
        self.peak_ram = max(self.peak_ram, current_ram)
        
        # GPU utilization
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_utilizations.append(gpus[0].load * 100)
        except:
            pass  # GPU monitoring not available
            
    def get_metrics(self, tokens_processed: int) -> ResourceMetrics:
        """Get final resource metrics."""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        return ResourceMetrics(
            peak_vram_mb=self.peak_vram,
            peak_ram_mb=self.peak_ram,
            avg_gpu_utilization=np.mean(self.gpu_utilizations) if self.gpu_utilizations else 0.0,
            inference_latency_ms=(total_time * 1000) / max(tokens_processed, 1),
            tokens_per_second=tokens_processed / max(total_time, 0.001)
        )

class BaselineEvaluator(ABC):
    """Abstract base class for all baseline evaluators."""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-small",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.monitor = ResourceMonitor()
        
    @abstractmethod
    def setup_model(self, **kwargs) -> None:
        """Setup the baseline model with appropriate configurations."""
        pass
        
    @abstractmethod
    def evaluate_dataset(self, 
                        dataset: List[Dict[str, Any]], 
                        **kwargs) -> EvaluationResult:
        """Evaluate the model on a dataset."""
        pass
        
    def _load_base_model(self):
        """Load base model and tokenizer."""
        logger.info(f"Loading base model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
    def _compute_metrics(self, 
                        predictions: List[str], 
                        references: List[str]) -> Tuple[float, float, List[float], List[float]]:
        """Compute EM and F1 scores."""
        em_scores = []
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            # Exact Match
            em = 1.0 if pred.strip() == ref.strip() else 0.0
            em_scores.append(em)
            
            # F1 Score (token-level)
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if not pred_tokens and not ref_tokens:
                f1 = 1.0
            elif not pred_tokens or not ref_tokens:
                f1 = 0.0
            else:
                precision = len(pred_tokens & ref_tokens) / len(pred_tokens)
                recall = len(pred_tokens & ref_tokens) / len(ref_tokens)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
            f1_scores.append(f1)
            
        return np.mean(em_scores), np.mean(f1_scores), em_scores, f1_scores

class StaticLoRAEvaluator(BaselineEvaluator):
    """Evaluator for Static LoRA baseline."""
    
    def setup_model(self, 
                    r: int = 8, 
                    lora_alpha: int = 32, 
                    lora_dropout: float = 0.1,
                    target_modules: Optional[List[str]] = None,
                    **kwargs) -> None:
        """Setup Static LoRA model."""
        
        self._load_base_model()
        
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
            
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        logger.info(f"Static LoRA setup complete: r={r}, alpha={lora_alpha}")
        
    def evaluate_dataset(self, 
                        dataset: List[Dict[str, Any]], 
                        max_new_tokens: int = 50,
                        **kwargs) -> EvaluationResult:
        """Evaluate Static LoRA on dataset."""
        
        logger.info("Starting Static LoRA evaluation")
        self.monitor.start_monitoring()
        
        predictions = []
        references = []
        
        for item in dataset:
            self.monitor.update_metrics()
            
            input_text = item.get("input", item.get("question", ""))
            reference = item.get("output", item.get("answer", ""))
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
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
            
            predictions.append(prediction)
            references.append(reference)
            
        # Compute metrics
        em_score, f1_score, em_scores, f1_scores = self._compute_metrics(predictions, references)
        
        # Get resource metrics
        resource_metrics = self.monitor.get_metrics(len(dataset) * max_new_tokens)
        
        # Count parameters
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return EvaluationResult(
            method_name="Static LoRA",
            em_score=em_score,
            f1_score=f1_score,
            exact_match_scores=em_scores,
            f1_scores=f1_scores,
            latency_ms=resource_metrics.inference_latency_ms,
            vram_peak_mb=resource_metrics.peak_vram_mb,
            parameter_count=param_count,
            tokens_per_second=resource_metrics.tokens_per_second,
            metadata={
                "avg_gpu_utilization": resource_metrics.avg_gpu_utilization,
                "peak_ram_mb": resource_metrics.peak_ram_mb,
                "dataset_size": len(dataset)
            }
        )

class AdaLoRAEvaluator(BaselineEvaluator):
    """Evaluator for AdaLoRA with adaptive importance scoring."""
    
    def setup_model(self, 
                    r: int = 8,
                    target_r: int = 4,
                    init_r: int = 12,
                    tinit: int = 200,
                    tfinal: int = 1000,
                    deltaT: int = 10,
                    **kwargs) -> None:
        """Setup AdaLoRA model with adaptive budget allocation."""
        
        self._load_base_model()
        
        # AdaLoRA configuration
        from peft import AdaLoraConfig
        
        adalora_config = AdaLoraConfig(
            r=r,
            target_r=target_r,
            init_r=init_r,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=deltaT,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, adalora_config)
        logger.info(f"AdaLoRA setup complete: r={r}, target_r={target_r}")
        
    def evaluate_dataset(self, 
                        dataset: List[Dict[str, Any]], 
                        **kwargs) -> EvaluationResult:
        """Evaluate AdaLoRA on dataset."""
        
        logger.info("Starting AdaLoRA evaluation")
        # Similar evaluation logic to StaticLoRA but with adaptive ranking
        
        # For now, use StaticLoRA evaluation as placeholder
        # TODO: Implement proper AdaLoRA evaluation with importance scoring
        static_evaluator = StaticLoRAEvaluator(self.model_name, self.device)
        static_evaluator.model = self.model
        static_evaluator.tokenizer = self.tokenizer
        
        result = static_evaluator.evaluate_dataset(dataset, **kwargs)
        result.method_name = "AdaLoRA"
        return result

class MoELoRAEvaluator(BaselineEvaluator):
    """Evaluator for traditional Mixture of Experts LoRA."""
    
    def setup_model(self, 
                    num_experts: int = 8,
                    top_k: int = 2,
                    r: int = 8,
                    **kwargs) -> None:
        """Setup MoELoRA with expert routing."""
        
        self._load_base_model()
        
        # Custom MoELoRA implementation
        class MoELoRALayer(nn.Module):
            def __init__(self, base_layer, num_experts, top_k, r):
                super().__init__()
                self.base_layer = base_layer
                self.num_experts = num_experts
                self.top_k = top_k
                self.r = r
                
                # Expert LoRA adapters
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(base_layer.in_features, r, bias=False),
                        nn.Linear(r, base_layer.out_features, bias=False)
                    ) for _ in range(num_experts)
                ])
                
                # Gating network
                self.gate = nn.Linear(base_layer.in_features, num_experts)
                
            def forward(self, x):
                base_output = self.base_layer(x)
                
                # Expert selection
                gate_scores = torch.softmax(self.gate(x), dim=-1)
                _, top_indices = torch.topk(gate_scores, self.top_k, dim=-1)
                
                # Expert outputs
                expert_outputs = []
                for i, expert in enumerate(self.experts):
                    expert_outputs.append(expert(x))
                    
                expert_outputs = torch.stack(expert_outputs, dim=-2)  # [..., num_experts, d_model]
                
                # Weighted combination
                top_gate_scores = torch.gather(gate_scores.unsqueeze(-1), -2, 
                                             top_indices.unsqueeze(-1))
                top_expert_outputs = torch.gather(expert_outputs, -2,
                                                 top_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)))
                
                weighted_output = (top_gate_scores * top_expert_outputs).sum(dim=-2)
                
                return base_output + weighted_output
                
        # Replace attention layers with MoE variants
        for name, module in self.model.named_modules():
            if "q_proj" in name or "v_proj" in name:
                parent = self.model
                for part in name.split('.')[:-1]:
                    parent = getattr(parent, part)
                    
                layer_name = name.split('.')[-1]
                original_layer = getattr(parent, layer_name)
                setattr(parent, layer_name, 
                       MoELoRALayer(original_layer, num_experts, top_k, r))
                
        logger.info(f"MoELoRA setup complete: {num_experts} experts, top-{top_k}")
        
    def evaluate_dataset(self, 
                        dataset: List[Dict[str, Any]], 
                        **kwargs) -> EvaluationResult:
        """Evaluate MoELoRA on dataset."""
        
        logger.info("Starting MoELoRA evaluation")
        
        # Use StaticLoRA evaluation pipeline
        static_evaluator = StaticLoRAEvaluator(self.model_name, self.device)
        static_evaluator.model = self.model
        static_evaluator.tokenizer = self.tokenizer
        
        result = static_evaluator.evaluate_dataset(dataset, **kwargs)
        result.method_name = "MoELoRA"
        return result

class QLoRAEvaluator(BaselineEvaluator):
    """Evaluator for QLoRA with quantization."""
    
    def setup_model(self, 
                    r: int = 8,
                    lora_alpha: int = 32,
                    quantization_config: Optional[Dict] = None,
                    **kwargs) -> None:
        """Setup QLoRA with 4-bit quantization."""
        
        # QLoRA quantization config
        if quantization_config is None:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load quantized model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Apply LoRA to quantized model
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        logger.info(f"QLoRA setup complete: r={r}, 4-bit quantization enabled")
        
    def evaluate_dataset(self, 
                        dataset: List[Dict[str, Any]], 
                        **kwargs) -> EvaluationResult:
        """Evaluate QLoRA on dataset."""
        
        logger.info("Starting QLoRA evaluation")
        
        # Use StaticLoRA evaluation pipeline
        static_evaluator = StaticLoRAEvaluator(self.model_name, self.device)
        static_evaluator.model = self.model
        static_evaluator.tokenizer = self.tokenizer
        
        result = static_evaluator.evaluate_dataset(dataset, **kwargs)
        result.method_name = "QLoRA"
        return result

class BaselineOrchestrator:
    """Orchestrator for running all baselines with fair comparison."""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-small",
                 output_dir: str = "experiments/baselines"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluators
        self.evaluators = {
            "static_lora": StaticLoRAEvaluator(model_name),
            "adalora": AdaLoRAEvaluator(model_name), 
            "moelora": MoELoRAEvaluator(model_name),
            "qlora": QLoRAEvaluator(model_name),
            # TODO: Add LoRAHub and Switch-LoRA when available
        }
        
    def run_comprehensive_evaluation(self, 
                                   dataset: List[Dict[str, Any]],
                                   baselines: Optional[List[str]] = None,
                                   seeds: List[int] = [42, 1337, 2023, 2024, 2025]) -> Dict[str, List[EvaluationResult]]:
        """Run comprehensive evaluation across all baselines and seeds."""
        
        if baselines is None:
            baselines = list(self.evaluators.keys())
            
        logger.info(f"Starting comprehensive evaluation on {len(baselines)} baselines, {len(seeds)} seeds")
        
        results = {}
        
        for baseline_name in baselines:
            if baseline_name not in self.evaluators:
                logger.warning(f"Unknown baseline: {baseline_name}")
                continue
                
            baseline_results = []
            evaluator = self.evaluators[baseline_name]
            
            for seed in seeds:
                logger.info(f"Evaluating {baseline_name} with seed {seed}")
                
                # Set random seeds
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # Setup model
                evaluator.setup_model()
                
                # Evaluate
                result = evaluator.evaluate_dataset(dataset)
                result.metadata["seed"] = seed
                baseline_results.append(result)
                
                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            results[baseline_name] = baseline_results
            
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, List[EvaluationResult]]) -> None:
        """Save evaluation results to disk."""
        
        # Convert results to JSON-serializable format
        json_results = {}
        
        for baseline_name, baseline_results in results.items():
            json_results[baseline_name] = []
            
            for result in baseline_results:
                json_results[baseline_name].append({
                    "method_name": result.method_name,
                    "em_score": result.em_score,
                    "f1_score": result.f1_score,
                    "latency_ms": result.latency_ms,
                    "vram_peak_mb": result.vram_peak_mb,
                    "parameter_count": result.parameter_count,
                    "tokens_per_second": result.tokens_per_second,
                    "metadata": result.metadata,
                    "individual_scores": {
                        "em_scores": result.exact_match_scores,
                        "f1_scores": result.f1_scores
                    }
                })
                
        # Save to file
        output_file = self.output_dir / "baseline_results.json"
        with open(output_file, "w") as f:
            json.dump(json_results, f, indent=2)
            
        logger.info(f"Saved baseline results to {output_file}")

def main():
    """Example usage of baseline evaluation suite."""
    
    # Sample dataset
    sample_dataset = [
        {
            "input": "What is the capital of France?",
            "output": "Paris"
        },
        {
            "input": "How many continents are there?",
            "output": "Seven"
        },
        # ... more examples
    ]
    
    # Run comprehensive evaluation
    orchestrator = BaselineOrchestrator()
    results = orchestrator.run_comprehensive_evaluation(sample_dataset)
    
    # Print summary
    for baseline_name, baseline_results in results.items():
        em_scores = [r.em_score for r in baseline_results]
        f1_scores = [r.f1_score for r in baseline_results]
        
        print(f"{baseline_name}:")
        print(f"  EM: {np.mean(em_scores):.3f} ± {np.std(em_scores):.3f}")
        print(f"  F1: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")

if __name__ == "__main__":
    main()