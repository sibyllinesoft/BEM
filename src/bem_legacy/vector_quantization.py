"""
Vector Quantization System for BEM - Phase 5 Implementation.

Implements discrete code representation with VQ codebooks, residuals, and 
episodic memory as specified in TODO.md Phase 5.

Key Features:
- VQ Codebook: Learnable discrete code vectors
- Residual Coding: Continuous residuals on top of discrete codes
- Episodic Memory: Store and deduplicate codewords
- LSH Deduplication: Locality-sensitive hashing for efficiency
- Code Evolution: Update codebook based on usage patterns
"""

from typing import Dict, List, Optional, Tuple, Union, NamedTuple, Set
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from collections import defaultdict, Counter, deque
import hashlib
import pickle
import time

from .telemetry import TelemetryCollector


class VQMetrics(NamedTuple):
    """Metrics for vector quantization performance."""
    codebook_utilization: float
    average_quantization_error: float
    commitment_loss: float
    codebook_loss: float
    perplexity: float
    dead_codes: int
    memory_efficiency: float


@dataclass
class VQConfig:
    """Configuration for vector quantization system."""
    
    # Codebook parameters
    codebook_size: int = 512
    code_dim: int = 64
    commitment_cost: float = 0.25
    
    # EMA updates
    use_ema_update: bool = True
    ema_decay: float = 0.99
    ema_epsilon: float = 1e-5
    
    # Residual coding
    enable_residual: bool = True
    residual_layers: int = 2
    residual_dim: int = 32
    
    # Dead code handling
    dead_code_threshold: int = 100  # Steps without usage
    dead_code_replacement: str = "random"  # "random", "split", "reinit"
    
    # Episodic memory
    enable_episodic_memory: bool = True
    memory_size: int = 10000
    memory_update_frequency: int = 100
    
    # LSH for deduplication
    enable_lsh: bool = True
    lsh_num_hashes: int = 8
    lsh_hash_size: int = 16
    similarity_threshold: float = 0.9
    
    # Adaptive parameters
    adaptive_codebook: bool = True
    usage_tracking_window: int = 1000
    rebalancing_frequency: int = 5000
    
    # Quantization method
    quantization_method: str = "straight_through"  # "straight_through", "gumbel", "vq_vae"
    temperature: float = 1.0
    hard_quantization: bool = True


class LSHIndex:
    """Locality-Sensitive Hashing for efficient similarity search."""
    
    def __init__(
        self,
        dim: int,
        num_hashes: int = 8,
        hash_size: int = 16,
        similarity_threshold: float = 0.9
    ):
        self.dim = dim
        self.num_hashes = num_hashes
        self.hash_size = hash_size
        self.similarity_threshold = similarity_threshold
        
        # Random projection matrices for LSH
        self.projections = []
        for _ in range(num_hashes):
            proj = torch.randn(dim, hash_size)
            proj = F.normalize(proj, dim=0)
            self.projections.append(proj)
        
        # Hash tables
        self.hash_tables = [defaultdict(list) for _ in range(num_hashes)]
        self.stored_vectors = {}  # hash -> vector
        
    def _compute_hash(self, vector: torch.Tensor, hash_idx: int) -> str:
        """Compute LSH hash for a vector."""
        projected = torch.matmul(vector, self.projections[hash_idx])
        binary_hash = (projected > 0).int()
        # Convert to string hash
        hash_str = ''.join(str(b.item()) for b in binary_hash)
        return hash_str
    
    def add_vector(self, vector: torch.Tensor, vector_id: str):
        """Add vector to LSH index."""
        vector = vector.detach().cpu()
        
        # Compute hashes for all hash functions
        for i in range(self.num_hashes):
            hash_val = self._compute_hash(vector, i)
            self.hash_tables[i][hash_val].append(vector_id)
        
        # Store the actual vector
        self.stored_vectors[vector_id] = vector
    
    def find_similar(self, query_vector: torch.Tensor) -> List[Tuple[str, float]]:
        """Find similar vectors using LSH."""
        query_vector = query_vector.detach().cpu()
        candidate_ids = set()
        
        # Get candidates from all hash tables
        for i in range(self.num_hashes):
            hash_val = self._compute_hash(query_vector, i)
            if hash_val in self.hash_tables[i]:
                candidate_ids.update(self.hash_tables[i][hash_val])
        
        # Compute actual similarities
        similarities = []
        for candidate_id in candidate_ids:
            if candidate_id in self.stored_vectors:
                stored_vec = self.stored_vectors[candidate_id]
                similarity = F.cosine_similarity(
                    query_vector.unsqueeze(0), 
                    stored_vec.unsqueeze(0)
                ).item()
                
                if similarity >= self.similarity_threshold:
                    similarities.append((candidate_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def remove_vector(self, vector_id: str):
        """Remove vector from LSH index."""
        if vector_id not in self.stored_vectors:
            return
        
        vector = self.stored_vectors[vector_id]
        
        # Remove from hash tables
        for i in range(self.num_hashes):
            hash_val = self._compute_hash(vector, i)
            if hash_val in self.hash_tables[i] and vector_id in self.hash_tables[i][hash_val]:
                self.hash_tables[i][hash_val].remove(vector_id)
                if not self.hash_tables[i][hash_val]:  # Clean empty buckets
                    del self.hash_tables[i][hash_val]
        
        # Remove stored vector
        del self.stored_vectors[vector_id]
    
    def get_statistics(self) -> Dict[str, any]:
        """Get LSH index statistics."""
        total_buckets = sum(len(table) for table in self.hash_tables)
        total_entries = len(self.stored_vectors)
        avg_bucket_size = total_entries / max(total_buckets, 1)
        
        return {
            'total_vectors': total_entries,
            'total_buckets': total_buckets,
            'average_bucket_size': avg_bucket_size,
            'hash_tables_size': [len(table) for table in self.hash_tables]
        }


class EpisodicCodeMemory:
    """Episodic memory for storing and managing code patterns."""
    
    def __init__(
        self,
        max_size: int = 10000,
        code_dim: int = 64,
        enable_lsh: bool = True,
        lsh_config: Optional[Dict] = None
    ):
        self.max_size = max_size
        self.code_dim = code_dim
        
        # Memory storage
        self.memory = {}  # code_id -> code_info
        self.access_times = {}  # code_id -> last_access_time
        self.usage_counts = Counter()
        
        # LSH index for similarity search
        if enable_lsh:
            lsh_config = lsh_config or {}
            self.lsh_index = LSHIndex(
                dim=code_dim,
                num_hashes=lsh_config.get('num_hashes', 8),
                hash_size=lsh_config.get('hash_size', 16),
                similarity_threshold=lsh_config.get('similarity_threshold', 0.9)
            )
        else:
            self.lsh_index = None
        
        self.current_time = 0
        self.logger = logging.getLogger(__name__)
    
    def add_code(
        self, 
        code: torch.Tensor, 
        metadata: Optional[Dict] = None
    ) -> str:
        """Add code to episodic memory."""
        self.current_time += 1
        
        # Generate unique ID for the code
        code_hash = hashlib.sha256(code.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        code_id = f"code_{code_hash}_{self.current_time}"
        
        # Check for similar codes using LSH
        if self.lsh_index:
            similar_codes = self.lsh_index.find_similar(code)
            if similar_codes:
                # Use existing similar code instead of storing duplicate
                most_similar_id, similarity = similar_codes[0]
                self.usage_counts[most_similar_id] += 1
                self.access_times[most_similar_id] = self.current_time
                return most_similar_id
        
        # Store new code
        code_info = {
            'code': code.detach().clone(),
            'metadata': metadata or {},
            'creation_time': self.current_time,
            'last_used': self.current_time
        }
        
        self.memory[code_id] = code_info
        self.access_times[code_id] = self.current_time
        self.usage_counts[code_id] = 1
        
        # Add to LSH index
        if self.lsh_index:
            self.lsh_index.add_vector(code, code_id)
        
        # Evict old codes if memory is full
        if len(self.memory) > self.max_size:
            self._evict_old_codes()
        
        return code_id
    
    def get_code(self, code_id: str) -> Optional[torch.Tensor]:
        """Retrieve code from memory."""
        if code_id not in self.memory:
            return None
        
        # Update access statistics
        self.current_time += 1
        self.access_times[code_id] = self.current_time
        self.usage_counts[code_id] += 1
        self.memory[code_id]['last_used'] = self.current_time
        
        return self.memory[code_id]['code']
    
    def find_similar_codes(
        self, 
        query_code: torch.Tensor, 
        top_k: int = 5
    ) -> List[Tuple[str, torch.Tensor, float]]:
        """Find similar codes in memory."""
        if self.lsh_index:
            # Use LSH for efficient search
            lsh_results = self.lsh_index.find_similar(query_code)
            results = []
            for code_id, similarity in lsh_results[:top_k]:
                if code_id in self.memory:
                    code = self.memory[code_id]['code']
                    results.append((code_id, code, similarity))
            return results
        else:
            # Brute force search
            similarities = []
            for code_id, code_info in self.memory.items():
                code = code_info['code']
                similarity = F.cosine_similarity(
                    query_code.unsqueeze(0), 
                    code.unsqueeze(0)
                ).item()
                similarities.append((code_id, code, similarity))
            
            # Sort and return top-k
            similarities.sort(key=lambda x: x[2], reverse=True)
            return similarities[:top_k]
    
    def _evict_old_codes(self):
        """Evict least recently used codes to make space."""
        # Sort by access time (oldest first)
        sorted_codes = sorted(
            self.access_times.items(),
            key=lambda x: x[1]
        )
        
        # Remove oldest codes until within size limit
        num_to_remove = len(self.memory) - int(self.max_size * 0.9)  # Remove 10% buffer
        
        for i in range(num_to_remove):
            code_id, _ = sorted_codes[i]
            
            # Remove from all data structures
            if code_id in self.memory:
                del self.memory[code_id]
            if code_id in self.access_times:
                del self.access_times[code_id]
            if code_id in self.usage_counts:
                del self.usage_counts[code_id]
            
            # Remove from LSH index
            if self.lsh_index:
                self.lsh_index.remove_vector(code_id)
        
        self.logger.info(f"Evicted {num_to_remove} old codes from episodic memory")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get memory usage statistics."""
        if not self.memory:
            return {
                'total_codes': 0,
                'memory_utilization': 0.0,
                'average_usage': 0.0,
                'lsh_stats': {}
            }
        
        usage_counts_list = list(self.usage_counts.values())
        
        stats = {
            'total_codes': len(self.memory),
            'memory_utilization': len(self.memory) / self.max_size,
            'average_usage': np.mean(usage_counts_list) if usage_counts_list else 0,
            'max_usage': max(usage_counts_list) if usage_counts_list else 0,
            'total_accesses': sum(usage_counts_list),
        }
        
        if self.lsh_index:
            stats['lsh_stats'] = self.lsh_index.get_statistics()
        
        return stats


class VectorQuantizer(nn.Module):
    """Vector quantizer with codebook learning and residuals."""
    
    def __init__(
        self,
        config: VQConfig,
        telemetry_collector: Optional[TelemetryCollector] = None
    ):
        super().__init__()
        self.config = config
        self.telemetry = telemetry_collector
        
        # Codebook embeddings
        self.codebook = nn.Embedding(config.codebook_size, config.code_dim)
        nn.init.uniform_(self.codebook.weight, -1/config.codebook_size, 1/config.codebook_size)
        
        # EMA parameters for codebook updates
        if config.use_ema_update:
            self.register_buffer('ema_cluster_size', torch.zeros(config.codebook_size))
            self.register_buffer('ema_weight', self.codebook.weight.clone())
        
        # Residual layers
        if config.enable_residual:
            self.residual_encoder = nn.Sequential(
                nn.Linear(config.code_dim, config.residual_dim),
                nn.ReLU(),
                *[nn.Sequential(
                    nn.Linear(config.residual_dim, config.residual_dim),
                    nn.ReLU()
                ) for _ in range(config.residual_layers - 1)]
            )
            self.residual_decoder = nn.Linear(config.residual_dim, config.code_dim)
        
        # Usage tracking
        self.register_buffer('code_usage', torch.zeros(config.codebook_size, dtype=torch.long))
        self.register_buffer('last_used', torch.zeros(config.codebook_size, dtype=torch.long))
        self.step_count = 0
        
        # Episodic memory
        if config.enable_episodic_memory:
            self.episodic_memory = EpisodicCodeMemory(
                max_size=config.memory_size,
                code_dim=config.code_dim,
                enable_lsh=config.enable_lsh,
                lsh_config={
                    'num_hashes': config.lsh_num_hashes,
                    'hash_size': config.lsh_hash_size,
                    'similarity_threshold': config.similarity_threshold
                }
            )
        
        self.logger = logging.getLogger(__name__)
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through vector quantizer.
        
        Args:
            inputs: Input tensor [batch_size, seq_len, code_dim]
            training: Whether in training mode
            
        Returns:
            Dictionary with quantized outputs and losses
        """
        batch_size, seq_len, _ = inputs.shape
        flat_inputs = inputs.view(-1, self.config.code_dim)
        
        # Compute distances to codebook entries
        distances = torch.cdist(flat_inputs, self.codebook.weight)
        
        # Get closest codes
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Update usage statistics
        if training:
            self._update_usage_statistics(encoding_indices)
        
        # Get quantized values
        quantized = self.codebook(encoding_indices)
        
        # Straight-through estimator
        if self.config.quantization_method == "straight_through":
            quantized = inputs + (quantized.view_as(inputs) - inputs).detach()
        elif self.config.quantization_method == "gumbel":
            quantized = self._gumbel_quantize(inputs, distances)
        
        # Compute losses
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        
        # EMA updates for codebook
        if training and self.config.use_ema_update:
            self._ema_update(flat_inputs, encoding_indices)
        
        # Residual coding
        residual = torch.zeros_like(inputs)
        if self.config.enable_residual:
            residual_input = inputs - quantized
            encoded_residual = self.residual_encoder(residual_input.view(-1, self.config.code_dim))
            decoded_residual = self.residual_decoder(encoded_residual)
            residual = decoded_residual.view_as(inputs)
            
            # Add residual to quantized output
            quantized = quantized + residual
        
        # Handle dead codes
        if training and self.step_count % 100 == 0:
            self._handle_dead_codes()
        
        # Store codes in episodic memory
        if hasattr(self, 'episodic_memory') and training:
            if self.step_count % self.config.memory_update_frequency == 0:
                self._update_episodic_memory(flat_inputs, encoding_indices)
        
        self.step_count += 1
        
        # Compute metrics
        metrics = self._compute_metrics(
            inputs, quantized, commitment_loss, codebook_loss
        )
        
        if self.telemetry and training:
            self.telemetry.log_vq_metrics(metrics)
        
        return {
            'quantized': quantized,
            'encoding_indices': encoding_indices.view(batch_size, seq_len),
            'distances': distances.view(batch_size, seq_len, -1),
            'residual': residual,
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'metrics': metrics
        }
    
    def _gumbel_quantize(
        self, 
        inputs: torch.Tensor, 
        distances: torch.Tensor
    ) -> torch.Tensor:
        """Apply Gumbel softmax quantization."""
        # Convert distances to logits (negative distances)
        logits = -distances / self.config.temperature
        
        # Add Gumbel noise
        gumbel_noise = torch.empty_like(logits).exponential_(1).log().neg()
        logits_with_noise = logits + gumbel_noise
        
        # Soft assignment
        soft_assignment = F.softmax(logits_with_noise, dim=-1)
        
        # Weighted sum of codebook entries
        quantized = torch.matmul(soft_assignment, self.codebook.weight)
        
        # Hard assignment for discrete codes (straight-through)
        if self.config.hard_quantization:
            hard_assignment = F.one_hot(
                torch.argmax(soft_assignment, dim=-1), 
                num_classes=self.config.codebook_size
            ).float()
            quantized = quantized + (torch.matmul(hard_assignment, self.codebook.weight) - quantized).detach()
        
        return quantized.view_as(inputs)
    
    def _update_usage_statistics(self, encoding_indices: torch.Tensor):
        """Update code usage statistics."""
        unique_indices, counts = torch.unique(encoding_indices, return_counts=True)
        
        for idx, count in zip(unique_indices, counts):
            self.code_usage[idx] += count
            self.last_used[idx] = self.step_count
    
    def _ema_update(self, inputs: torch.Tensor, encoding_indices: torch.Tensor):
        """Update codebook using exponential moving average."""
        # One-hot encoding of indices
        one_hot = F.one_hot(encoding_indices, num_classes=self.config.codebook_size).float()
        
        # Update cluster sizes
        cluster_sizes = one_hot.sum(dim=0)
        self.ema_cluster_size.mul_(self.config.ema_decay).add_(
            cluster_sizes, alpha=1 - self.config.ema_decay
        )
        
        # Update embeddings
        embedding_sum = torch.matmul(one_hot.t(), inputs)
        self.ema_weight.mul_(self.config.ema_decay).add_(
            embedding_sum, alpha=1 - self.config.ema_decay
        )
        
        # Normalize embeddings
        normalized_embeddings = self.ema_weight / (
            self.ema_cluster_size.unsqueeze(1) + self.config.ema_epsilon
        )
        
        # Update codebook
        self.codebook.weight.data.copy_(normalized_embeddings)
    
    def _handle_dead_codes(self):
        """Handle dead (unused) codes in the codebook."""
        dead_mask = (self.step_count - self.last_used) > self.config.dead_code_threshold
        dead_codes = torch.where(dead_mask)[0]
        
        if len(dead_codes) > 0:
            self.logger.info(f"Handling {len(dead_codes)} dead codes")
            
            if self.config.dead_code_replacement == "random":
                # Replace with random vectors
                with torch.no_grad():
                    self.codebook.weight[dead_codes] = torch.randn_like(
                        self.codebook.weight[dead_codes]
                    ) * 0.1
            
            elif self.config.dead_code_replacement == "split":
                # Split most used codes
                most_used_codes = torch.topk(self.code_usage, len(dead_codes)).indices
                with torch.no_grad():
                    for dead_idx, used_idx in zip(dead_codes, most_used_codes):
                        noise = torch.randn_like(self.codebook.weight[used_idx]) * 0.01
                        self.codebook.weight[dead_idx] = self.codebook.weight[used_idx] + noise
            
            # Reset statistics for replaced codes
            self.code_usage[dead_codes] = 0
            self.last_used[dead_codes] = self.step_count
    
    def _update_episodic_memory(self, inputs: torch.Tensor, encoding_indices: torch.Tensor):
        """Update episodic memory with current codes."""
        unique_indices = torch.unique(encoding_indices)
        
        for idx in unique_indices:
            code = self.codebook.weight[idx]
            code_id = self.episodic_memory.add_code(
                code,
                metadata={
                    'codebook_index': idx.item(),
                    'usage_count': self.code_usage[idx].item(),
                    'step': self.step_count
                }
            )
    
    def _compute_metrics(
        self,
        inputs: torch.Tensor,
        quantized: torch.Tensor,
        commitment_loss: torch.Tensor,
        codebook_loss: torch.Tensor
    ) -> VQMetrics:
        """Compute VQ metrics."""
        # Codebook utilization
        active_codes = (self.code_usage > 0).sum().item()
        utilization = active_codes / self.config.codebook_size
        
        # Quantization error
        quantization_error = F.mse_loss(inputs, quantized).item()
        
        # Perplexity (measure of code diversity)
        usage_probs = self.code_usage.float() / (self.code_usage.sum().float() + 1e-8)
        entropy = -torch.sum(usage_probs * torch.log(usage_probs + 1e-8))
        perplexity = torch.exp(entropy).item()
        
        # Dead codes
        dead_codes = ((self.step_count - self.last_used) > self.config.dead_code_threshold).sum().item()
        
        # Memory efficiency (if episodic memory enabled)
        memory_efficiency = 1.0
        if hasattr(self, 'episodic_memory'):
            memory_stats = self.episodic_memory.get_statistics()
            memory_efficiency = memory_stats.get('memory_utilization', 1.0)
        
        return VQMetrics(
            codebook_utilization=utilization,
            average_quantization_error=quantization_error,
            commitment_loss=commitment_loss.item(),
            codebook_loss=codebook_loss.item(),
            perplexity=perplexity,
            dead_codes=dead_codes,
            memory_efficiency=memory_efficiency
        )
    
    def get_codebook_analysis(self) -> Dict[str, any]:
        """Get detailed codebook analysis."""
        analysis = {
            'codebook_size': self.config.codebook_size,
            'active_codes': (self.code_usage > 0).sum().item(),
            'total_usage': self.code_usage.sum().item(),
            'usage_distribution': self.code_usage.cpu().numpy().tolist(),
            'most_used_codes': torch.topk(self.code_usage, 10).indices.cpu().numpy().tolist(),
            'least_used_codes': torch.topk(self.code_usage, 10, largest=False).indices.cpu().numpy().tolist(),
            'dead_codes': ((self.step_count - self.last_used) > self.config.dead_code_threshold).sum().item()
        }
        
        if hasattr(self, 'episodic_memory'):
            analysis['episodic_memory'] = self.episodic_memory.get_statistics()
        
        return analysis
    
    def save_codebook(self, path: str):
        """Save codebook and statistics to file."""
        state = {
            'codebook_weights': self.codebook.weight.cpu(),
            'code_usage': self.code_usage.cpu(),
            'last_used': self.last_used.cpu(),
            'step_count': self.step_count,
            'config': self.config
        }
        
        if self.config.use_ema_update:
            state['ema_cluster_size'] = self.ema_cluster_size.cpu()
            state['ema_weight'] = self.ema_weight.cpu()
        
        torch.save(state, path)
    
    def load_codebook(self, path: str):
        """Load codebook and statistics from file."""
        state = torch.load(path)
        
        self.codebook.weight.data.copy_(state['codebook_weights'])
        self.code_usage.copy_(state['code_usage'])
        self.last_used.copy_(state['last_used'])
        self.step_count = state['step_count']
        
        if self.config.use_ema_update and 'ema_cluster_size' in state:
            self.ema_cluster_size.copy_(state['ema_cluster_size'])
            self.ema_weight.copy_(state['ema_weight'])


def create_vector_quantizer(
    config: Optional[VQConfig] = None,
    telemetry_collector: Optional[TelemetryCollector] = None
) -> VectorQuantizer:
    """Create vector quantizer with default configuration."""
    if config is None:
        config = VQConfig()
    
    return VectorQuantizer(config=config, telemetry_collector=telemetry_collector)


def create_default_vq_config(
    codebook_size: int = 512,
    code_dim: int = 64,
    enable_residual: bool = True,
    enable_episodic_memory: bool = True
) -> VQConfig:
    """Create default VQ configuration."""
    return VQConfig(
        codebook_size=codebook_size,
        code_dim=code_dim,
        enable_residual=enable_residual,
        enable_episodic_memory=enable_episodic_memory,
        use_ema_update=True,
        enable_lsh=True,
        adaptive_codebook=True
    )


# Example usage and testing
if __name__ == "__main__":
    # Test vector quantizer
    config = create_default_vq_config(codebook_size=256, code_dim=32)
    vq = create_vector_quantizer(config)
    
    # Test forward pass
    batch_size, seq_len, code_dim = 4, 16, 32
    inputs = torch.randn(batch_size, seq_len, code_dim)
    
    # Forward pass
    vq.train()
    result = vq(inputs, training=True)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Quantized shape: {result['quantized'].shape}")
    print(f"Encoding indices shape: {result['encoding_indices'].shape}")
    print(f"Commitment loss: {result['commitment_loss']:.4f}")
    print(f"Codebook loss: {result['codebook_loss']:.4f}")
    
    metrics = result['metrics']
    print(f"\nVQ Metrics:")
    print(f"  Codebook utilization: {metrics.codebook_utilization:.3f}")
    print(f"  Quantization error: {metrics.average_quantization_error:.4f}")
    print(f"  Perplexity: {metrics.perplexity:.2f}")
    print(f"  Dead codes: {metrics.dead_codes}")
    
    # Test episodic memory
    if hasattr(vq, 'episodic_memory'):
        memory_stats = vq.episodic_memory.get_statistics()
        print(f"\nEpisodic Memory:")
        print(f"  Total codes stored: {memory_stats['total_codes']}")
        print(f"  Memory utilization: {memory_stats['memory_utilization']:.3f}")
    
    # Test codebook analysis
    analysis = vq.get_codebook_analysis()
    print(f"\nCodebook Analysis:")
    print(f"  Active codes: {analysis['active_codes']}/{analysis['codebook_size']}")
    print(f"  Total usage: {analysis['total_usage']}")
    print(f"  Most used codes: {analysis['most_used_codes'][:5]}")
    
    # Test multiple forward passes to see adaptation
    print(f"\nTesting adaptation over multiple steps...")
    for step in range(10):
        inputs = torch.randn(batch_size, seq_len, code_dim)
        result = vq(inputs, training=True)
        
        if step % 3 == 0:
            metrics = result['metrics']
            print(f"Step {step}: Utilization={metrics.codebook_utilization:.3f}, "
                  f"Perplexity={metrics.perplexity:.2f}, Dead codes={metrics.dead_codes}")
    
    print("Vector quantization test completed!")