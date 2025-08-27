"""
F5.5 - Counterfactual Hard-Negatives mining and training.

Mechanism: Mine near-duplicate but wrong spans; add training loss to reduce ΔW magnitude
when contradiction detectors fire.

Why: Robustness to distractors; sharpens policy-over-memory.
Gate: Index-swap slope ↑; retrieval-off penalty ↓; Slice-B no regress.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import numpy as np
from collections import defaultdict
import re
from difflib import SequenceMatcher
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class HardNegativeConfig:
    """Configuration for hard negative mining and training."""
    # Mining parameters
    min_lexical_overlap: float = 0.7        # Minimum lexical overlap for candidates
    max_lexical_overlap: float = 0.95       # Maximum overlap (avoid exact duplicates)
    min_length_ratio: float = 0.5           # Minimum length ratio (shorter/longer)
    max_edit_distance: int = 10             # Maximum edit distance for candidates
    contradiction_threshold: float = 0.8    # Threshold for contradiction detection
    
    # Training parameters
    contradiction_loss_weight: float = 0.1  # Weight for contradiction loss
    magnitude_reduction_factor: float = 0.5  # Factor to reduce ΔW magnitude
    consistency_loss_weight: float = 0.05    # Weight for consistency loss
    hard_neg_sampling_ratio: float = 0.3     # Ratio of hard negatives in training
    
    # Detection parameters
    use_embedding_similarity: bool = True    # Use embeddings for similarity
    embedding_threshold: float = 0.85       # Embedding similarity threshold
    use_nli_model: bool = True              # Use NLI for contradiction detection
    cache_embeddings: bool = True           # Cache embeddings for efficiency
    
    # Quality control
    max_hard_negs_per_positive: int = 3     # Maximum hard negatives per positive
    contamination_check: bool = True        # Check for label contamination
    diversity_threshold: float = 0.3        # Minimum diversity among hard negatives


class TextSimilarityCalculator:
    """Calculates various text similarity metrics."""
    
    @staticmethod
    def lexical_overlap(text1: str, text2: str, normalize: bool = True) -> float:
        """Calculate lexical overlap between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        
        if normalize:
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0
        else:
            return intersection / min(len(words1), len(words2))
            
    @staticmethod
    def edit_distance_normalized(text1: str, text2: str) -> float:
        """Calculate normalized edit distance."""
        # Simple implementation - for production, use python-Levenshtein
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio()
        
    @staticmethod
    def length_ratio(text1: str, text2: str) -> float:
        """Calculate length ratio (shorter/longer)."""
        len1, len2 = len(text1), len(text2)
        if len1 == 0 and len2 == 0:
            return 1.0
        return min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0
        
    @staticmethod
    def semantic_hash(text: str, hash_size: int = 64) -> str:
        """Create semantic hash for deduplication."""
        # Simple semantic hashing - normalize and hash
        normalized = re.sub(r'\W+', ' ', text.lower()).strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:hash_size//4]


class EmbeddingCache:
    """Caches text embeddings for efficiency."""
    
    def __init__(self, cache_size: int = 10000):
        self.cache = {}
        self.cache_size = cache_size
        self.hits = 0
        self.misses = 0
        
    def get(self, text: str) -> Optional[torch.Tensor]:
        """Get embedding from cache."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            self.hits += 1
            return self.cache[text_hash]
        self.misses += 1
        return None
        
    def put(self, text: str, embedding: torch.Tensor):
        """Store embedding in cache."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        text_hash = hashlib.md5(text.encode()).hexdigest()
        self.cache[text_hash] = embedding.detach().cpu()
        
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            'hit_rate': self.hits / total if total > 0 else 0.0,
            'cache_size': len(self.cache),
            'total_queries': total
        }


class ContradictionDetector(nn.Module):
    """Detects contradictions between text pairs."""
    
    def __init__(self, config: HardNegativeConfig):
        super().__init__()
        self.config = config
        
        # Simple contradiction detection using features
        # In practice, this could use a pre-trained NLI model
        self.feature_dim = 512  # Configurable
        
        # Feature extractors
        self.text_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        # Contradiction classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 64),  # [text1, text2, interaction]
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # entailment, neutral, contradiction
        )
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(
        self, 
        text1_features: torch.Tensor,
        text2_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect contradictions between text pairs.
        
        Args:
            text1_features: Features for first text [batch, feature_dim]
            text2_features: Features for second text [batch, feature_dim]
            
        Returns:
            contradiction_scores: Contradiction scores [batch, 3] (entail, neutral, contradict)
            contradiction_probs: Contradiction probabilities [batch]
        """
        # Encode texts
        enc1 = self.text_encoder(text1_features)  # [batch, 128]
        enc2 = self.text_encoder(text2_features)  # [batch, 128]
        
        # Compute interaction features
        interaction = enc1 * enc2  # Element-wise product
        
        # Concatenate all features
        combined = torch.cat([enc1, enc2, interaction], dim=-1)  # [batch, 384]
        
        # Classify
        scores = self.classifier(combined)  # [batch, 3]
        probs = F.softmax(scores, dim=-1)
        
        # Return contradiction probability (index 2)
        contradiction_probs = probs[:, 2]  # [batch]
        
        return scores, contradiction_probs


class HardNegativeMiner:
    """Mines hard negative examples from datasets."""
    
    def __init__(self, config: HardNegativeConfig):
        self.config = config
        self.similarity_calc = TextSimilarityCalculator()
        self.embedding_cache = EmbeddingCache() if config.cache_embeddings else None
        self.mined_pairs = {}  # Cache mined pairs
        
    def mine_hard_negatives(
        self,
        positive_samples: List[Dict[str, Any]],
        candidate_pool: List[Dict[str, Any]],
        embedding_model: Optional[nn.Module] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Mine hard negative examples from candidate pool.
        
        Args:
            positive_samples: List of positive examples with 'text' and 'label' fields
            candidate_pool: Pool of candidate negatives
            embedding_model: Optional embedding model for semantic similarity
            
        Returns:
            Dictionary mapping positive sample ids to hard negatives
        """
        hard_negatives = {}
        stats = defaultdict(int)
        
        for i, pos_sample in enumerate(positive_samples):
            pos_text = pos_sample['text']
            pos_label = pos_sample.get('label', '')
            
            # Find candidates with high lexical overlap but different labels
            candidates = self._find_candidates(
                pos_text, pos_label, candidate_pool, stats
            )
            
            # Rank candidates by similarity
            if embedding_model and self.config.use_embedding_similarity:
                candidates = self._rank_by_embedding_similarity(
                    pos_text, candidates, embedding_model
                )
            
            # Select diverse hard negatives
            selected = self._select_diverse_negatives(candidates)
            
            if selected:
                hard_negatives[f"pos_{i}"] = selected
                stats['total_hard_negatives'] += len(selected)
            else:
                stats['no_hard_negatives'] += 1
                
        # Log mining statistics
        logger.info(f"Hard negative mining stats: {dict(stats)}")
        
        return hard_negatives
        
    def _find_candidates(
        self,
        pos_text: str,
        pos_label: str,
        candidate_pool: List[Dict[str, Any]],
        stats: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Find candidate hard negatives."""
        candidates = []
        
        for candidate in candidate_pool:
            cand_text = candidate['text']
            cand_label = candidate.get('label', '')
            
            # Skip if same label (not a negative)
            if cand_label == pos_label:
                stats['same_label_skipped'] += 1
                continue
                
            # Check lexical overlap
            overlap = self.similarity_calc.lexical_overlap(pos_text, cand_text)
            if not (self.config.min_lexical_overlap <= overlap <= self.config.max_lexical_overlap):
                stats['lexical_overlap_filtered'] += 1
                continue
                
            # Check length ratio
            length_ratio = self.similarity_calc.length_ratio(pos_text, cand_text)
            if length_ratio < self.config.min_length_ratio:
                stats['length_ratio_filtered'] += 1
                continue
                
            # Check edit distance
            edit_similarity = self.similarity_calc.edit_distance_normalized(pos_text, cand_text)
            if edit_similarity < 0.5:  # Too dissimilar
                stats['edit_distance_filtered'] += 1
                continue
                
            # Add candidate with metadata
            candidate_info = {
                'text': cand_text,
                'label': cand_label,
                'lexical_overlap': overlap,
                'length_ratio': length_ratio,
                'edit_similarity': edit_similarity,
                'original_data': candidate
            }
            
            candidates.append(candidate_info)
            
        stats['candidates_found'] += len(candidates)
        return candidates
        
    def _rank_by_embedding_similarity(
        self,
        pos_text: str,
        candidates: List[Dict[str, Any]],
        embedding_model: nn.Module
    ) -> List[Dict[str, Any]]:
        """Rank candidates by embedding similarity."""
        if not candidates:
            return candidates
            
        # Get embedding for positive text
        pos_embedding = self._get_embedding(pos_text, embedding_model)
        
        # Compute similarities
        for candidate in candidates:
            cand_embedding = self._get_embedding(candidate['text'], embedding_model)
            similarity = F.cosine_similarity(
                pos_embedding.unsqueeze(0), 
                cand_embedding.unsqueeze(0)
            ).item()
            candidate['embedding_similarity'] = similarity
            
        # Sort by similarity (higher is more similar, hence harder negative)
        candidates.sort(key=lambda x: x['embedding_similarity'], reverse=True)
        
        # Filter by embedding threshold
        filtered = [
            c for c in candidates 
            if c['embedding_similarity'] >= self.config.embedding_threshold
        ]
        
        return filtered
        
    def _get_embedding(self, text: str, embedding_model: nn.Module) -> torch.Tensor:
        """Get text embedding with caching."""
        if self.embedding_cache:
            cached = self.embedding_cache.get(text)
            if cached is not None:
                return cached
                
        # Compute embedding (simplified - assumes model takes text directly)
        with torch.no_grad():
            embedding = embedding_model.encode(text)
            
        if self.embedding_cache:
            self.embedding_cache.put(text, embedding)
            
        return embedding
        
    def _select_diverse_negatives(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select diverse hard negatives."""
        if not candidates:
            return []
            
        selected = []
        max_negatives = self.config.max_hard_negs_per_positive
        
        # Select first candidate (highest similarity)
        selected.append(candidates[0])
        
        # Select additional candidates ensuring diversity
        for candidate in candidates[1:]:
            if len(selected) >= max_negatives:
                break
                
            # Check diversity with already selected
            is_diverse = True
            for sel in selected:
                diversity = 1.0 - self.similarity_calc.lexical_overlap(
                    candidate['text'], sel['text']
                )
                if diversity < self.config.diversity_threshold:
                    is_diverse = False
                    break
                    
            if is_diverse:
                selected.append(candidate)
                
        return selected
        
    def get_mining_statistics(self) -> Dict[str, Any]:
        """Get mining statistics."""
        stats = {}
        
        if self.embedding_cache:
            stats['embedding_cache'] = self.embedding_cache.get_stats()
            
        return stats


class ConsistencyScorer(nn.Module):
    """Scores consistency between retrieved passages and queries."""
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Consistency scoring network
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(
        self,
        query_features: torch.Tensor,
        passage_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Score consistency between query and passage.
        
        Args:
            query_features: Query features [batch, feature_dim]
            passage_features: Passage features [batch, feature_dim]
            
        Returns:
            consistency_scores: Consistency scores [batch]
        """
        combined = torch.cat([query_features, passage_features], dim=-1)
        scores = self.scorer(combined).squeeze(-1)
        return scores


class HardNegativeTrainingLoss(nn.Module):
    """Training loss that incorporates hard negative signals."""
    
    def __init__(self, config: HardNegativeConfig):
        super().__init__()
        self.config = config
        
        # Initialize detectors
        self.contradiction_detector = ContradictionDetector(config)
        self.consistency_scorer = ConsistencyScorer()
        
    def forward(
        self,
        bem_output: Dict[str, torch.Tensor],
        query_features: torch.Tensor,
        passage_features: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        target_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute training loss with hard negative regularization.
        
        Args:
            bem_output: Output from BEM module
            query_features: Query features [batch, feature_dim]
            passage_features: Passage features [batch, feature_dim]  
            hard_negative_mask: Boolean mask for hard negatives [batch]
            target_labels: Target labels [batch]
            
        Returns:
            total_loss: Combined loss
            loss_components: Individual loss components
        """
        batch_size = query_features.size(0)
        device = query_features.device
        
        # Standard task loss (placeholder - replace with actual task loss)
        task_loss = F.cross_entropy(bem_output['output'], target_labels)
        
        loss_components = {'task_loss': task_loss}
        
        # Contradiction detection loss
        if hard_negative_mask.any():
            contradiction_scores, contradiction_probs = self.contradiction_detector(
                query_features, passage_features
            )
            
            # Higher contradiction loss for hard negatives
            contradiction_loss = F.cross_entropy(
                contradiction_scores[hard_negative_mask],
                torch.full((hard_negative_mask.sum(),), 2, device=device, dtype=torch.long)  # contradiction class
            )
            
            loss_components['contradiction_loss'] = contradiction_loss * self.config.contradiction_loss_weight
            
            # Magnitude reduction for ΔW when contradictions detected
            if 'codes' in bem_output:
                codes = bem_output['codes']  # [batch, seq_len, code_dim]
                
                # Reduce code magnitude for contradictory samples
                contradiction_mask = contradiction_probs > self.config.contradiction_threshold
                if contradiction_mask.any():
                    code_magnitude = torch.norm(codes[contradiction_mask], p=2, dim=-1).mean()
                    magnitude_loss = code_magnitude * self.config.magnitude_reduction_factor
                    loss_components['magnitude_reduction'] = magnitude_loss
                    
        # Consistency loss
        consistency_scores = self.consistency_scorer(query_features, passage_features)
        
        # Encourage high consistency for positive samples, low for negatives
        target_consistency = (~hard_negative_mask).float()  # 1 for positives, 0 for negatives
        consistency_loss = F.mse_loss(consistency_scores, target_consistency)
        loss_components['consistency_loss'] = consistency_loss * self.config.consistency_loss_weight
        
        # Combine all losses
        total_loss = sum(loss_components.values())
        
        return total_loss, loss_components
        
    def get_contradiction_features(
        self,
        query_features: torch.Tensor,
        passage_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get features for controller from contradiction detector."""
        with torch.no_grad():
            contradiction_scores, contradiction_probs = self.contradiction_detector(
                query_features, passage_features
            )
            consistency_scores = self.consistency_scorer(query_features, passage_features)
            
        return {
            'contradiction_probs': contradiction_probs,
            'consistency_scores': consistency_scores,
            'contradiction_logits': contradiction_scores
        }


class HardNegativeDataset:
    """Dataset that incorporates hard negative examples."""
    
    def __init__(
        self,
        positive_samples: List[Dict[str, Any]],
        hard_negatives: Dict[str, List[Dict[str, Any]]],
        config: HardNegativeConfig
    ):
        self.positive_samples = positive_samples
        self.hard_negatives = hard_negatives
        self.config = config
        
        # Build training pairs
        self.training_pairs = self._build_training_pairs()
        
    def _build_training_pairs(self) -> List[Dict[str, Any]]:
        """Build training pairs with hard negatives."""
        pairs = []
        
        for i, pos_sample in enumerate(self.positive_samples):
            pos_key = f"pos_{i}"
            
            # Add positive sample
            pairs.append({
                'query': pos_sample.get('query', ''),
                'passage': pos_sample['text'],
                'label': 1,  # Positive
                'is_hard_negative': False,
                'sample_weight': 1.0
            })
            
            # Add hard negatives if available
            if pos_key in self.hard_negatives:
                neg_samples = self.hard_negatives[pos_key]
                
                for neg_sample in neg_samples:
                    pairs.append({
                        'query': pos_sample.get('query', ''),
                        'passage': neg_sample['text'],
                        'label': 0,  # Negative
                        'is_hard_negative': True,
                        'sample_weight': 1.0 / len(neg_samples),  # Balance hard negatives
                        'similarity_info': {
                            'lexical_overlap': neg_sample['lexical_overlap'],
                            'embedding_similarity': neg_sample.get('embedding_similarity', 0.0)
                        }
                    })
                    
        return pairs
        
    def __len__(self) -> int:
        return len(self.training_pairs)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.training_pairs[idx]
        
    def get_hard_negative_ratio(self) -> float:
        """Get ratio of hard negatives in dataset."""
        hard_negs = sum(1 for pair in self.training_pairs if pair['is_hard_negative'])
        return hard_negs / len(self.training_pairs) if self.training_pairs else 0.0
        
    def get_contamination_report(self) -> Dict[str, Any]:
        """Generate contamination report."""
        # Check for potential label contamination
        text_to_labels = defaultdict(set)
        
        for pair in self.training_pairs:
            text_hash = TextSimilarityCalculator.semantic_hash(pair['passage'])
            text_to_labels[text_hash].add(pair['label'])
            
        contaminated = {h: labels for h, labels in text_to_labels.items() if len(labels) > 1}
        
        return {
            'total_unique_texts': len(text_to_labels),
            'contaminated_texts': len(contaminated),
            'contamination_rate': len(contaminated) / len(text_to_labels) if text_to_labels else 0.0,
            'hard_negative_ratio': self.get_hard_negative_ratio()
        }


def create_hard_negative_config(**kwargs) -> HardNegativeConfig:
    """Factory function to create HardNegativeConfig with validation."""
    return HardNegativeConfig(**kwargs)


def mine_hard_negatives_from_files(
    positive_file: Path,
    candidate_file: Path,
    output_file: Path,
    config: Optional[HardNegativeConfig] = None
) -> Dict[str, Any]:
    """Mine hard negatives from JSON files."""
    if config is None:
        config = HardNegativeConfig()
        
    # Load data
    with open(positive_file) as f:
        positive_samples = json.load(f)
    with open(candidate_file) as f:
        candidate_pool = json.load(f)
        
    # Mine hard negatives
    miner = HardNegativeMiner(config)
    hard_negatives = miner.mine_hard_negatives(positive_samples, candidate_pool)
    
    # Create dataset
    dataset = HardNegativeDataset(positive_samples, hard_negatives, config)
    
    # Generate reports
    contamination_report = dataset.get_contamination_report()
    mining_stats = miner.get_mining_statistics()
    
    # Save results
    results = {
        'hard_negatives': hard_negatives,
        'contamination_report': contamination_report,
        'mining_stats': mining_stats,
        'config': config.__dict__
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    logger.info(f"Hard negative mining completed. Results saved to {output_file}")
    logger.info(f"Contamination rate: {contamination_report['contamination_rate']:.3f}")
    logger.info(f"Hard negative ratio: {contamination_report['hard_negative_ratio']:.3f}")
    
    return results


def main():
    """CLI interface for hard negative mining."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mine hard negative examples")
    parser.add_argument("--positives", required=True, help="Positive samples JSON file")
    parser.add_argument("--candidates", required=True, help="Candidate pool JSON file")
    parser.add_argument("--output", required=True, help="Output file for results")
    parser.add_argument("--min-overlap", type=float, default=0.7, help="Minimum lexical overlap")
    parser.add_argument("--max-overlap", type=float, default=0.95, help="Maximum lexical overlap")
    parser.add_argument("--max-hard-negs", type=int, default=3, help="Max hard negatives per positive")
    
    args = parser.parse_args()
    
    config = HardNegativeConfig(
        min_lexical_overlap=args.min_overlap,
        max_lexical_overlap=args.max_overlap,
        max_hard_negs_per_positive=args.max_hard_negs
    )
    
    mine_hard_negatives_from_files(
        Path(args.positives),
        Path(args.candidates), 
        Path(args.output),
        config
    )


if __name__ == "__main__":
    main()