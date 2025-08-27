"""
Coverage and consistency features for retrieval-aware BEM routing.

This module computes semantic features that quantify the quality and relevance
of retrieved evidence to guide the hierarchical controller's routing decisions.

Key features:
1. Coverage: Semantic similarity between retrieved docs and current context
2. Consistency: Inter-document agreement using pairwise similarities
3. Relevance scoring: Weight retrieved evidence by trustworthiness
4. Feature normalization: Stable inputs to controller

Based on TODO.md Phase 3 requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from dataclasses import dataclass
import math

from .retrieval import MicroRetriever


@dataclass
class RetrievalFeaturesConfig:
    """Configuration for retrieval features computation."""
    # Coverage features
    coverage_aggregation: str = "weighted_mean"  # "mean", "max", "weighted_mean"
    coverage_threshold: float = 0.3  # Minimum similarity for relevant docs
    coverage_temperature: float = 1.0  # Temperature for softmax weighting
    
    # Consistency features
    consistency_metric: str = "pairwise_cosine"  # "pairwise_cosine", "centroid_distance"
    consistency_threshold: float = 0.5  # Minimum inter-doc similarity
    min_docs_for_consistency: int = 2  # Minimum docs needed for consistency calc
    
    # Relevance scoring
    enable_relevance_weighting: bool = True
    relevance_decay: float = 0.1  # Decay factor for lower-ranked docs
    
    # Feature normalization
    normalize_features: bool = True
    feature_clamp_range: Tuple[float, float] = (-3.0, 3.0)  # Clamp range for stability
    
    # Performance settings
    batch_process: bool = True
    max_docs_per_batch: int = 32  # Limit docs processed per batch


class CoverageCalculator:
    """
    Computes coverage metrics that measure how well retrieved documents
    cover the current context/query.
    """
    
    def __init__(self, config: RetrievalFeaturesConfig):
        self.config = config
    
    def compute_coverage(
        self,
        query_embeddings: torch.Tensor,  # [batch, embed_dim] or [batch, seq_len, embed_dim]
        doc_embeddings: List[torch.Tensor],  # List of [num_docs_i, embed_dim] per query
        doc_similarities: List[List[float]]  # Retrieval similarities per query
    ) -> torch.Tensor:
        """
        Compute coverage scores for each query based on retrieved documents.
        
        Coverage measures how well the retrieved documents match the query's
        semantic content, considering both similarity and diversity.
        
        Args:
            query_embeddings: Query embeddings
            doc_embeddings: List of document embedding tensors per query
            doc_similarities: List of similarity scores per query
            
        Returns:
            coverage_scores: Coverage scores [batch] or [batch, seq_len]
        """
        if query_embeddings.dim() == 3:
            # Handle sequence-level queries by averaging over sequence
            query_embeddings_pooled = query_embeddings.mean(dim=1)  # [batch, embed_dim]
        else:
            query_embeddings_pooled = query_embeddings
        
        batch_size = query_embeddings_pooled.shape[0]
        coverage_scores = []
        
        for i in range(batch_size):
            if i >= len(doc_embeddings) or doc_embeddings[i].numel() == 0:
                # No documents retrieved for this query
                coverage_scores.append(torch.tensor(0.0))
                continue
            
            query_emb = query_embeddings_pooled[i]  # [embed_dim]
            docs_emb = doc_embeddings[i]  # [num_docs, embed_dim]
            similarities = torch.tensor(doc_similarities[i], dtype=torch.float32)  # [num_docs]
            
            # Compute query-document similarities
            query_doc_sims = F.cosine_similarity(
                query_emb.unsqueeze(0), docs_emb, dim=1
            )  # [num_docs]
            
            # Apply relevance weighting based on retrieval rank
            if self.config.enable_relevance_weighting:
                rank_weights = torch.exp(
                    -torch.arange(len(similarities), dtype=torch.float32) * self.config.relevance_decay
                )
                weighted_sims = query_doc_sims * rank_weights
            else:
                weighted_sims = query_doc_sims
            
            # Aggregate coverage score
            if self.config.coverage_aggregation == "mean":
                coverage_score = weighted_sims.mean()
            elif self.config.coverage_aggregation == "max":
                coverage_score = weighted_sims.max()
            elif self.config.coverage_aggregation == "weighted_mean":
                # Use retrieval similarities as weights
                weights = F.softmax(similarities / self.config.coverage_temperature, dim=0)
                coverage_score = (weighted_sims * weights).sum()
            else:
                raise ValueError(f"Unknown coverage aggregation: {self.config.coverage_aggregation}")
            
            coverage_scores.append(coverage_score)
        
        # Stack into tensor
        coverage_tensor = torch.stack(coverage_scores)  # [batch]
        
        # Expand back to sequence dimension if needed
        if query_embeddings.dim() == 3:
            coverage_tensor = coverage_tensor.unsqueeze(1).expand(-1, query_embeddings.shape[1])
        
        return coverage_tensor
    
    def compute_diversity_coverage(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute diversity-aware coverage that considers how well the document set
        covers different aspects of the query.
        
        Args:
            query_embeddings: Query embeddings [batch, embed_dim]
            doc_embeddings: List of document embeddings per query
            
        Returns:
            diversity_scores: Diversity coverage scores [batch]
        """
        if query_embeddings.dim() == 3:
            query_embeddings = query_embeddings.mean(dim=1)
        
        batch_size = query_embeddings.shape[0]
        diversity_scores = []
        
        for i in range(batch_size):
            if i >= len(doc_embeddings) or doc_embeddings[i].numel() == 0:
                diversity_scores.append(torch.tensor(0.0))
                continue
            
            query_emb = query_embeddings[i]  # [embed_dim]
            docs_emb = doc_embeddings[i]  # [num_docs, embed_dim]
            
            # Compute pairwise document similarities
            doc_doc_sims = F.cosine_similarity(
                docs_emb.unsqueeze(1), docs_emb.unsqueeze(0), dim=2
            )  # [num_docs, num_docs]
            
            # Remove self-similarities (diagonal)
            mask = ~torch.eye(doc_doc_sims.shape[0], dtype=torch.bool)
            pairwise_sims = doc_doc_sims[mask]
            
            # Diversity is inversely related to average pairwise similarity
            if len(pairwise_sims) > 0:
                avg_pairwise_sim = pairwise_sims.mean()
                diversity = 1.0 - avg_pairwise_sim  # Higher diversity = lower similarity
            else:
                diversity = torch.tensor(0.0)
            
            # Combine with query-document coverage
            query_doc_sims = F.cosine_similarity(query_emb.unsqueeze(0), docs_emb, dim=1)
            query_coverage = query_doc_sims.mean()
            
            # Balanced diversity-coverage score
            diversity_coverage = 0.7 * query_coverage + 0.3 * diversity
            diversity_scores.append(diversity_coverage)
        
        return torch.stack(diversity_scores)


class ConsistencyCalculator:
    """
    Computes consistency metrics that measure inter-document agreement
    and coherence in the retrieved evidence.
    """
    
    def __init__(self, config: RetrievalFeaturesConfig):
        self.config = config
    
    def compute_consistency(
        self,
        doc_embeddings: List[torch.Tensor],  # List of [num_docs_i, embed_dim] per query
        doc_similarities: List[List[float]]  # Retrieval similarities per query
    ) -> torch.Tensor:
        """
        Compute consistency scores measuring inter-document agreement.
        
        Args:
            doc_embeddings: List of document embeddings per query
            doc_similarities: List of similarity scores per query
            
        Returns:
            consistency_scores: Consistency scores [batch]
        """
        batch_size = len(doc_embeddings)
        consistency_scores = []
        
        for i in range(batch_size):
            if i >= len(doc_embeddings) or doc_embeddings[i].numel() == 0:
                consistency_scores.append(torch.tensor(0.0))
                continue
            
            docs_emb = doc_embeddings[i]  # [num_docs, embed_dim]
            similarities = torch.tensor(doc_similarities[i], dtype=torch.float32)
            
            # Need at least 2 documents for consistency
            if docs_emb.shape[0] < self.config.min_docs_for_consistency:
                consistency_scores.append(torch.tensor(0.0))
                continue
            
            if self.config.consistency_metric == "pairwise_cosine":
                consistency = self._pairwise_cosine_consistency(docs_emb, similarities)
            elif self.config.consistency_metric == "centroid_distance":
                consistency = self._centroid_consistency(docs_emb, similarities)
            else:
                raise ValueError(f"Unknown consistency metric: {self.config.consistency_metric}")
            
            consistency_scores.append(consistency)
        
        return torch.stack(consistency_scores)
    
    def _pairwise_cosine_consistency(
        self,
        doc_embeddings: torch.Tensor,  # [num_docs, embed_dim]
        similarities: torch.Tensor  # [num_docs]
    ) -> torch.Tensor:
        """Compute consistency based on pairwise cosine similarities."""
        num_docs = doc_embeddings.shape[0]
        
        # Compute all pairwise cosine similarities
        pairwise_sims = F.cosine_similarity(
            doc_embeddings.unsqueeze(1), doc_embeddings.unsqueeze(0), dim=2
        )  # [num_docs, num_docs]
        
        # Remove diagonal (self-similarities)
        mask = ~torch.eye(num_docs, dtype=torch.bool)
        pairwise_sims_flat = pairwise_sims[mask]  # [num_docs * (num_docs - 1)]
        
        # Weight by retrieval quality
        if self.config.enable_relevance_weighting:
            # Create weights for pairwise combinations
            weight_matrix = similarities.unsqueeze(1) * similarities.unsqueeze(0)
            pairwise_weights = weight_matrix[mask]
            
            # Weighted average of pairwise similarities
            consistency = (pairwise_sims_flat * pairwise_weights).sum() / pairwise_weights.sum()
        else:
            # Simple average
            consistency = pairwise_sims_flat.mean()
        
        return consistency
    
    def _centroid_consistency(
        self,
        doc_embeddings: torch.Tensor,  # [num_docs, embed_dim]
        similarities: torch.Tensor  # [num_docs]
    ) -> torch.Tensor:
        """Compute consistency based on distance from centroid."""
        # Compute weighted centroid
        if self.config.enable_relevance_weighting:
            weights = F.softmax(similarities, dim=0)
            centroid = (doc_embeddings * weights.unsqueeze(1)).sum(dim=0)
        else:
            centroid = doc_embeddings.mean(dim=0)
        
        # Compute similarities to centroid
        centroid_sims = F.cosine_similarity(doc_embeddings, centroid.unsqueeze(0), dim=1)
        
        # Consistency is average similarity to centroid
        consistency = centroid_sims.mean()
        
        return consistency
    
    def compute_topical_coherence(
        self,
        doc_embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute topical coherence measuring how well documents form coherent topics.
        
        Args:
            doc_embeddings: List of document embeddings per query
            
        Returns:
            coherence_scores: Topical coherence scores [batch]
        """
        batch_size = len(doc_embeddings)
        coherence_scores = []
        
        for i in range(batch_size):
            if i >= len(doc_embeddings) or doc_embeddings[i].numel() == 0:
                coherence_scores.append(torch.tensor(0.0))
                continue
            
            docs_emb = doc_embeddings[i]  # [num_docs, embed_dim]
            
            if docs_emb.shape[0] < 2:
                coherence_scores.append(torch.tensor(0.0))
                continue
            
            # Simple coherence: variance of document embeddings
            # Lower variance = higher coherence
            doc_variance = docs_emb.var(dim=0).mean()  # Average variance across dimensions
            coherence = torch.exp(-doc_variance)  # Convert to similarity-like score
            
            coherence_scores.append(coherence)
        
        return torch.stack(coherence_scores)


class RetrievalFeatureExtractor:
    """
    Main class that extracts comprehensive retrieval features for controller input.
    Combines coverage, consistency, and other semantic features.
    """
    
    def __init__(self, config: RetrievalFeaturesConfig):
        self.config = config
        self.coverage_calculator = CoverageCalculator(config)
        self.consistency_calculator = ConsistencyCalculator(config)
    
    def extract_features(
        self,
        query_embeddings: torch.Tensor,  # [batch, embed_dim] or [batch, seq_len, embed_dim]
        retrieval_results: Dict[str, Any],  # Results from MicroRetriever
        micro_retriever: Optional[MicroRetriever] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract comprehensive retrieval features.
        
        Args:
            query_embeddings: Query/context embeddings
            retrieval_results: Results from micro-retriever
            micro_retriever: Optional retriever for additional computations
            
        Returns:
            features: Dictionary of retrieval features
        """
        # Parse retrieval results
        documents = retrieval_results['documents']  # List[List[str]]
        similarities = retrieval_results['similarities']  # List[List[float]]
        doc_embeddings_list = []
        
        # Get document embeddings from micro-retriever if available
        if micro_retriever is not None:
            for doc_list in documents:
                if doc_list:
                    doc_embs = micro_retriever.index.encode_queries(doc_list)
                    doc_embeddings_list.append(doc_embs)
                else:
                    # Empty embedding tensor for no documents
                    doc_embeddings_list.append(torch.empty(0, micro_retriever.config.embedding_dim))
        else:
            # Create dummy embeddings if retriever not available
            embed_dim = query_embeddings.shape[-1]
            for doc_list in documents:
                if doc_list:
                    # Create random embeddings as placeholder
                    doc_embs = torch.randn(len(doc_list), embed_dim)
                    doc_embeddings_list.append(F.normalize(doc_embs, p=2, dim=1))
                else:
                    doc_embeddings_list.append(torch.empty(0, embed_dim))
        
        # Extract features
        features = {}
        
        # Coverage features
        coverage = self.coverage_calculator.compute_coverage(
            query_embeddings, doc_embeddings_list, similarities
        )
        features['coverage'] = coverage
        
        # Diversity coverage
        diversity_coverage = self.coverage_calculator.compute_diversity_coverage(
            query_embeddings, doc_embeddings_list
        )
        features['diversity_coverage'] = diversity_coverage
        
        # Consistency features
        consistency = self.consistency_calculator.compute_consistency(
            doc_embeddings_list, similarities
        )
        features['consistency'] = consistency
        
        # Topical coherence
        coherence = self.consistency_calculator.compute_topical_coherence(
            doc_embeddings_list
        )
        features['coherence'] = coherence
        
        # Additional derived features
        features['retrieval_confidence'] = self._compute_retrieval_confidence(similarities)
        features['result_density'] = self._compute_result_density(similarities)
        
        # Normalize features if requested
        if self.config.normalize_features:
            features = self._normalize_features(features)
        
        return features
    
    def _compute_retrieval_confidence(
        self,
        similarities: List[List[float]]
    ) -> torch.Tensor:
        """
        Compute confidence in retrieval quality based on similarity scores.
        
        Args:
            similarities: List of similarity lists per query
            
        Returns:
            confidence_scores: Confidence scores [batch]
        """
        confidence_scores = []
        
        for sim_list in similarities:
            if not sim_list:
                confidence_scores.append(torch.tensor(0.0))
                continue
            
            sims = torch.tensor(sim_list, dtype=torch.float32)
            
            # Confidence based on top similarity and score distribution
            top_sim = sims.max()
            sim_std = sims.std() if len(sims) > 1 else torch.tensor(0.0)
            
            # High confidence = high top similarity + low variance (clear winner)
            confidence = top_sim * torch.exp(-sim_std)
            confidence_scores.append(confidence)
        
        return torch.stack(confidence_scores)
    
    def _compute_result_density(
        self,
        similarities: List[List[float]]
    ) -> torch.Tensor:
        """
        Compute density of good retrieval results.
        
        Args:
            similarities: List of similarity lists per query
            
        Returns:
            density_scores: Result density scores [batch]
        """
        density_scores = []
        
        for sim_list in similarities:
            if not sim_list:
                density_scores.append(torch.tensor(0.0))
                continue
            
            sims = torch.tensor(sim_list, dtype=torch.float32)
            
            # Density = fraction of results above threshold
            above_threshold = (sims > self.config.coverage_threshold).float()
            density = above_threshold.mean()
            
            density_scores.append(density)
        
        return torch.stack(density_scores)
    
    def _normalize_features(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Normalize features for stable controller input.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            normalized_features: Normalized feature dictionary
        """
        normalized = {}
        
        for key, feature_tensor in features.items():
            # Z-score normalization
            normalized_feature = (feature_tensor - feature_tensor.mean()) / (feature_tensor.std() + 1e-8)
            
            # Clamp to prevent extreme values
            if self.config.feature_clamp_range:
                min_val, max_val = self.config.feature_clamp_range
                normalized_feature = torch.clamp(normalized_feature, min_val, max_val)
            
            normalized[key] = normalized_feature
        
        return normalized
    
    def combine_features(
        self,
        features: Dict[str, torch.Tensor],
        feature_weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Combine multiple features into a single side signal tensor.
        
        Args:
            features: Dictionary of feature tensors
            feature_weights: Optional weights for combining features
            
        Returns:
            combined_features: Combined feature tensor [batch, feature_dim]
        """
        if feature_weights is None:
            # Default equal weighting
            feature_weights = {key: 1.0 for key in features.keys()}
        
        # Stack features with weights
        weighted_features = []
        for key, feature_tensor in features.items():
            weight = feature_weights.get(key, 1.0)
            
            # Ensure feature tensor is 2D [batch, 1]
            if feature_tensor.dim() == 1:
                feature_tensor = feature_tensor.unsqueeze(1)
            elif feature_tensor.dim() > 2:
                feature_tensor = feature_tensor.view(feature_tensor.shape[0], -1)
            
            weighted_features.append(feature_tensor * weight)
        
        # Concatenate along feature dimension
        combined = torch.cat(weighted_features, dim=1)  # [batch, total_features]
        
        return combined


def create_retrieval_feature_extractor(
    config: Optional[RetrievalFeaturesConfig] = None,
    **config_kwargs
) -> RetrievalFeatureExtractor:
    """
    Factory function to create a retrieval feature extractor.
    
    Args:
        config: RetrievalFeaturesConfig instance
        **config_kwargs: Config overrides
        
    Returns:
        RetrievalFeatureExtractor instance
    """
    if config is None:
        config = RetrievalFeaturesConfig(**config_kwargs)
    else:
        # Update config with any provided kwargs
        for key, value in config_kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return RetrievalFeatureExtractor(config)