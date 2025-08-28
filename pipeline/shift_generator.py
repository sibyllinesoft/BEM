#!/usr/bin/env python3
"""
Distribution Shift Generation System for BEM Research Validation

Systematic generation of domain shifts, temporal shifts, and adversarial examples
for comprehensive robustness evaluation against all MoE-LoRA competitors.
"""

import json
import random
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from datasets import Dataset, load_dataset
import nltk
from nltk.corpus import wordnet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShiftType(Enum):
    """Types of distribution shifts to generate."""
    DOMAIN = "domain"
    TEMPORAL = "temporal" 
    ADVERSARIAL = "adversarial"

@dataclass
class ShiftConfig:
    """Configuration for shift generation."""
    shift_type: ShiftType
    source: str
    target: str
    intensity: float = 1.0
    seed: int = 42
    
@dataclass
class ShiftResult:
    """Results from shift generation."""
    original_data: List[Dict[str, Any]]
    shifted_data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    shift_config: ShiftConfig

class DomainShiftGenerator:
    """Generate domain shifts (medical↔legal, tech↔finance, etc.)"""
    
    DOMAIN_MAPPINGS = {
        "medical": {
            "target_domains": ["legal", "financial", "technical"],
            "keywords": ["patient", "diagnosis", "treatment", "symptoms", "medication"],
            "context_markers": ["clinical", "medical", "healthcare", "therapeutic"],
        },
        "legal": {
            "target_domains": ["medical", "financial", "technical"],
            "keywords": ["contract", "agreement", "plaintiff", "defendant", "litigation"],
            "context_markers": ["legal", "judicial", "statutory", "contractual"],
        },
        "technical": {
            "target_domains": ["medical", "legal", "financial"],
            "keywords": ["algorithm", "implementation", "optimization", "parameters", "function"],
            "context_markers": ["technical", "engineering", "computational", "systematic"],
        },
        "financial": {
            "target_domains": ["medical", "legal", "technical"],
            "keywords": ["portfolio", "investment", "revenue", "profit", "assets"],
            "context_markers": ["financial", "economic", "monetary", "fiscal"],
        }
    }
    
    def __init__(self, tokenizer: Optional[AutoTokenizer] = None):
        self.tokenizer = tokenizer
        
    def generate_domain_shift(self, 
                            dataset: List[Dict[str, Any]], 
                            config: ShiftConfig) -> ShiftResult:
        """Generate domain shift by modifying context and terminology."""
        
        logger.info(f"Generating {config.source} → {config.target} domain shift")
        
        if config.source not in self.DOMAIN_MAPPINGS:
            raise ValueError(f"Unsupported source domain: {config.source}")
            
        source_mapping = self.DOMAIN_MAPPINGS[config.source]
        if config.target not in source_mapping["target_domains"]:
            raise ValueError(f"Unsupported target domain: {config.target}")
            
        target_mapping = self.DOMAIN_MAPPINGS[config.target]
        
        shifted_data = []
        for item in dataset:
            shifted_item = self._apply_domain_shift(
                item, source_mapping, target_mapping, config.intensity
            )
            shifted_data.append(shifted_item)
            
        metadata = {
            "shift_type": "domain",
            "source_domain": config.source,
            "target_domain": config.target,
            "intensity": config.intensity,
            "items_shifted": len(shifted_data),
            "terminology_changes": len(source_mapping["keywords"]),
        }
        
        return ShiftResult(
            original_data=dataset,
            shifted_data=shifted_data,
            metadata=metadata,
            shift_config=config
        )
    
    def _apply_domain_shift(self, 
                          item: Dict[str, Any], 
                          source_mapping: Dict, 
                          target_mapping: Dict, 
                          intensity: float) -> Dict[str, Any]:
        """Apply domain-specific transformations to a single item."""
        
        shifted_item = item.copy()
        
        # Replace domain-specific keywords
        for field in ["input", "output", "text", "question", "answer"]:
            if field in item:
                text = item[field]
                
                # Replace keywords with probability based on intensity
                for src_keyword, tgt_keyword in zip(
                    source_mapping["keywords"], target_mapping["keywords"]
                ):
                    if random.random() < intensity:
                        text = text.replace(src_keyword, tgt_keyword)
                
                # Replace context markers
                for src_marker, tgt_marker in zip(
                    source_mapping["context_markers"], target_mapping["context_markers"]
                ):
                    if random.random() < intensity * 0.7:  # Lower probability for context
                        text = text.replace(src_marker, tgt_marker)
                        
                shifted_item[field] = text
                
        return shifted_item

class TemporalShiftGenerator:
    """Generate temporal shifts (≤2020 vs ≥2024 data)"""
    
    def __init__(self, cutoff_date: date = date(2022, 1, 1)):
        self.cutoff_date = cutoff_date
        
    def generate_temporal_shift(self, 
                              dataset: List[Dict[str, Any]], 
                              config: ShiftConfig) -> ShiftResult:
        """Generate temporal shift by filtering data by date."""
        
        logger.info(f"Generating temporal shift with cutoff {self.cutoff_date}")
        
        # Separate data into pre/post cutoff
        pre_cutoff = []
        post_cutoff = []
        
        for item in dataset:
            item_date = self._extract_date(item)
            if item_date:
                if item_date <= self.cutoff_date:
                    pre_cutoff.append(item)
                else:
                    post_cutoff.append(item)
            else:
                # If no date, randomly assign based on typical distribution
                if random.random() < 0.7:  # 70% pre-cutoff
                    pre_cutoff.append(item)
                else:
                    post_cutoff.append(item)
        
        # Select source and target based on config
        if config.source == "pre_2022":
            source_data = pre_cutoff
            target_data = post_cutoff
        else:
            source_data = post_cutoff
            target_data = pre_cutoff
            
        # Apply temporal linguistic shifts
        shifted_data = [
            self._apply_temporal_linguistic_shift(item, config.intensity) 
            for item in target_data
        ]
        
        metadata = {
            "shift_type": "temporal",
            "cutoff_date": str(self.cutoff_date),
            "pre_cutoff_count": len(pre_cutoff),
            "post_cutoff_count": len(post_cutoff),
            "source_count": len(source_data),
            "target_count": len(target_data),
        }
        
        return ShiftResult(
            original_data=source_data,
            shifted_data=shifted_data,
            metadata=metadata,
            shift_config=config
        )
    
    def _extract_date(self, item: Dict[str, Any]) -> Optional[date]:
        """Extract date from item metadata."""
        
        # Check common date fields
        for date_field in ["date", "timestamp", "created_at", "publication_date"]:
            if date_field in item:
                try:
                    if isinstance(item[date_field], str):
                        # Try parsing different date formats
                        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"]:
                            try:
                                return datetime.strptime(item[date_field], fmt).date()
                            except ValueError:
                                continue
                    elif isinstance(item[date_field], (int, float)):
                        return datetime.fromtimestamp(item[date_field]).date()
                except:
                    continue
                    
        return None
    
    def _apply_temporal_linguistic_shift(self, 
                                       item: Dict[str, Any], 
                                       intensity: float) -> Dict[str, Any]:
        """Apply temporal linguistic changes (slang evolution, tech terms, etc.)"""
        
        shifted_item = item.copy()
        
        # Temporal replacement patterns (2020 → 2024 evolution)
        temporal_replacements = {
            "COVID-19": "post-pandemic",
            "remote work": "hybrid work",
            "Zoom meeting": "virtual collaboration",
            "social distancing": "flexible spacing",
            "AI assistant": "AI agent",
            "machine learning": "AI/ML",
            "neural network": "deep learning model",
            "cryptocurrency": "digital assets",
            "NFT": "digital collectible",
        }
        
        for field in ["input", "output", "text", "question", "answer"]:
            if field in item:
                text = item[field]
                
                for old_term, new_term in temporal_replacements.items():
                    if random.random() < intensity:
                        text = text.replace(old_term, new_term)
                        
                shifted_item[field] = text
                
        return shifted_item

class AdversarialShiftGenerator:
    """Generate adversarial examples (paraphrases, synonyms, noise)"""
    
    def __init__(self, tokenizer: Optional[AutoTokenizer] = None):
        self.tokenizer = tokenizer
        
        # Ensure NLTK data is available
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading WordNet corpus...")
            nltk.download('wordnet')
            nltk.download('omw-1.4')
    
    def generate_adversarial_shift(self, 
                                 dataset: List[Dict[str, Any]], 
                                 config: ShiftConfig) -> ShiftResult:
        """Generate adversarial examples using various perturbation methods."""
        
        logger.info(f"Generating {config.target} adversarial examples")
        
        adversarial_methods = {
            "synonym_substitution": self._synonym_substitution,
            "paraphrase_generation": self._paraphrase_generation,
            "character_noise": self._character_noise,
            "token_shuffle": self._token_shuffle,
        }
        
        if config.target not in adversarial_methods:
            raise ValueError(f"Unsupported adversarial method: {config.target}")
            
        method = adversarial_methods[config.target]
        
        adversarial_data = []
        for item in dataset:
            adv_item = method(item, config.intensity)
            adversarial_data.append(adv_item)
            
        metadata = {
            "shift_type": "adversarial",
            "method": config.target,
            "intensity": config.intensity,
            "items_perturbed": len(adversarial_data),
        }
        
        return ShiftResult(
            original_data=dataset,
            shifted_data=adversarial_data,
            metadata=metadata,
            shift_config=config
        )
    
    def _synonym_substitution(self, 
                            item: Dict[str, Any], 
                            intensity: float) -> Dict[str, Any]:
        """Replace words with synonyms using WordNet."""
        
        adv_item = item.copy()
        
        for field in ["input", "output", "text", "question", "answer"]:
            if field in item:
                text = item[field]
                words = text.split()
                
                for i, word in enumerate(words):
                    if random.random() < intensity:
                        synonyms = self._get_synonyms(word.lower())
                        if synonyms:
                            words[i] = random.choice(synonyms)
                            
                adv_item[field] = " ".join(words)
                
        return adv_item
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and len(synonym) > 2:
                    synonyms.add(synonym)
                    
        return list(synonyms)
    
    def _paraphrase_generation(self, 
                             item: Dict[str, Any], 
                             intensity: float) -> Dict[str, Any]:
        """Generate paraphrases by restructuring sentences."""
        
        # Simple paraphrase patterns
        paraphrase_patterns = [
            (r"It is (.+) that (.+)", r"The fact that \2 is \1"),
            (r"(.+) because (.+)", r"Due to \2, \1"),
            (r"If (.+), then (.+)", r"When \1, \2 occurs"),
            (r"(.+) is important", r"\1 plays a crucial role"),
        ]
        
        adv_item = item.copy()
        
        for field in ["input", "output", "text", "question", "answer"]:
            if field in item:
                text = item[field]
                
                if random.random() < intensity:
                    # Apply simple paraphrase transformations
                    import re
                    for pattern, replacement in paraphrase_patterns:
                        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                        
                adv_item[field] = text
                
        return adv_item
    
    def _character_noise(self, 
                       item: Dict[str, Any], 
                       intensity: float) -> Dict[str, Any]:
        """Add character-level noise (typos, insertions, deletions)."""
        
        adv_item = item.copy()
        
        for field in ["input", "output", "text", "question", "answer"]:
            if field in item:
                text = list(item[field])
                
                # Character operations based on intensity
                n_operations = int(len(text) * intensity * 0.1)  # Max 10% of characters
                
                for _ in range(n_operations):
                    if not text:
                        break
                        
                    operation = random.choice(["substitute", "insert", "delete"])
                    pos = random.randint(0, len(text) - 1)
                    
                    if operation == "substitute" and text[pos].isalpha():
                        text[pos] = random.choice("abcdefghijklmnopqrstuvwxyz")
                    elif operation == "insert":
                        text.insert(pos, random.choice("abcdefghijklmnopqrstuvwxyz"))
                    elif operation == "delete" and len(text) > 1:
                        text.pop(pos)
                        
                adv_item[field] = "".join(text)
                
        return adv_item
    
    def _token_shuffle(self, 
                     item: Dict[str, Any], 
                     intensity: float) -> Dict[str, Any]:
        """Shuffle token order within sentences."""
        
        adv_item = item.copy()
        
        for field in ["input", "output", "text", "question", "answer"]:
            if field in item:
                text = item[field]
                sentences = text.split('. ')
                
                shuffled_sentences = []
                for sentence in sentences:
                    words = sentence.split()
                    
                    if random.random() < intensity and len(words) > 3:
                        # Shuffle middle portion of sentence, keep first/last
                        middle = words[1:-1]
                        random.shuffle(middle)
                        words = [words[0]] + middle + [words[-1]]
                        
                    shuffled_sentences.append(" ".join(words))
                    
                adv_item[field] = ". ".join(shuffled_sentences)
                
        return adv_item

class ShiftGeneratorOrchestrator:
    """Main orchestrator for systematic distribution shift generation."""
    
    def __init__(self, output_dir: str = "experiments/shifts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.domain_generator = DomainShiftGenerator()
        self.temporal_generator = TemporalShiftGenerator()
        self.adversarial_generator = AdversarialShiftGenerator()
        
    def generate_comprehensive_shifts(self, 
                                    dataset: List[Dict[str, Any]], 
                                    seed: int = 42) -> Dict[str, ShiftResult]:
        """Generate all types of distribution shifts for comprehensive evaluation."""
        
        random.seed(seed)
        np.random.seed(seed)
        
        logger.info("Starting comprehensive distribution shift generation")
        
        shift_results = {}
        
        # Domain shifts
        domain_pairs = [
            ("medical", "legal"),
            ("technical", "financial"), 
            ("academic", "casual"),
        ]
        
        for source, target in domain_pairs:
            config = ShiftConfig(
                shift_type=ShiftType.DOMAIN,
                source=source,
                target=target,
                intensity=0.8,
                seed=seed
            )
            
            result = self.domain_generator.generate_domain_shift(dataset, config)
            shift_results[f"domain_{source}_to_{target}"] = result
            
        # Temporal shifts
        temporal_configs = [
            ShiftConfig(ShiftType.TEMPORAL, "pre_2022", "post_2022", 1.0, seed),
        ]
        
        for config in temporal_configs:
            result = self.temporal_generator.generate_temporal_shift(dataset, config)
            shift_results[f"temporal_{config.source}_to_{config.target}"] = result
            
        # Adversarial shifts  
        adversarial_types = [
            "synonym_substitution",
            "paraphrase_generation", 
            "character_noise",
            "token_shuffle",
        ]
        
        for adv_type in adversarial_types:
            config = ShiftConfig(
                shift_type=ShiftType.ADVERSARIAL,
                source="original",
                target=adv_type,
                intensity=0.3,  # Lower intensity for adversarial
                seed=seed
            )
            
            result = self.adversarial_generator.generate_adversarial_shift(dataset, config)
            shift_results[f"adversarial_{adv_type}"] = result
            
        logger.info(f"Generated {len(shift_results)} distribution shifts")
        return shift_results
    
    def save_shift_results(self, shift_results: Dict[str, ShiftResult]) -> None:
        """Save generated shifts to disk with metadata."""
        
        for shift_name, result in shift_results.items():
            shift_dir = self.output_dir / shift_name
            shift_dir.mkdir(exist_ok=True)
            
            # Save original and shifted data
            self._save_jsonl(result.original_data, shift_dir / "original.jsonl")
            self._save_jsonl(result.shifted_data, shift_dir / "shifted.jsonl")
            
            # Save metadata
            with open(shift_dir / "metadata.json", "w") as f:
                metadata = result.metadata.copy()
                metadata["shift_config"] = {
                    "shift_type": result.shift_config.shift_type.value,
                    "source": result.shift_config.source,
                    "target": result.shift_config.target,
                    "intensity": result.shift_config.intensity,
                    "seed": result.shift_config.seed,
                }
                json.dump(metadata, f, indent=2)
                
        logger.info(f"Saved shift results to {self.output_dir}")
    
    def _save_jsonl(self, data: List[Dict[str, Any]], filepath: Path) -> None:
        """Save data in JSONL format."""
        with open(filepath, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

def main():
    """Example usage of the shift generation system."""
    
    # Sample dataset (in practice, load from your datasets)
    sample_dataset = [
        {
            "input": "What are the symptoms of pneumonia?",
            "output": "Common symptoms include fever, cough, and chest pain.",
            "domain": "medical",
            "date": "2023-01-15"
        },
        {
            "input": "How does contract law work?", 
            "output": "Contract law governs legally binding agreements between parties.",
            "domain": "legal",
            "date": "2023-06-20"
        },
        # ... more examples
    ]
    
    # Generate comprehensive shifts
    orchestrator = ShiftGeneratorOrchestrator("experiments/shifts")
    shift_results = orchestrator.generate_comprehensive_shifts(sample_dataset, seed=42)
    
    # Save results
    orchestrator.save_shift_results(shift_results)
    
    # Print summary
    for shift_name, result in shift_results.items():
        print(f"{shift_name}: {len(result.shifted_data)} items")
        print(f"  Metadata: {result.metadata}")

if __name__ == "__main__":
    main()