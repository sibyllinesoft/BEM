"""
Retrieval-aware BEM module that integrates hierarchical routing with retrieval coupling.

This module combines:
1. Hierarchical BEM from Phase 2
2. Micro-retriever with FAISS index
3. Coverage and consistency features
4. Non-blocking retrieval pipeline
5. Cache-aware retrieval features

This is the complete Phase 3 implementation that enables evidence-based routing
while maintaining the performance and functionality of hierarchical routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
import asyncio
import threading
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

from .hierarchical_bem import HierarchicalBEMModule, FullHierarchicalBEM, HierarchicalBEMConfig
from .controller import HierarchicalController, RoutingLevel, RoutingState
from .retrieval import MicroRetriever, RetrievalConfig, create_micro_retriever
from .retrieval_features import RetrievalFeatureExtractor, RetrievalFeaturesConfig, create_retrieval_feature_extractor
from .telemetry import TelemetryCollector as BEMTelemetry

logger = logging.getLogger(__name__)


@dataclass
class RetrievalBEMConfig:
    """Configuration for retrieval-aware BEM system."""
    # Inherit hierarchical BEM configuration
    hierarchical_config: HierarchicalBEMConfig = None
    
    # Retrieval configuration
    retrieval_config: RetrievalConfig = None
    features_config: RetrievalFeaturesConfig = None
    
    # Integration settings
    retrieval_enabled: bool = True
    precompute_retrieval: bool = True  # Precompute at chunk boundaries
    cache_retrieval_results: bool = True
    max_cache_size: int = 1000
    
    # Performance settings
    non_blocking_retrieval: bool = True  # Don't block generation on retrieval
    retrieval_timeout_ms: int = 100  # Max time to wait for retrieval
    background_retrieval: bool = True  # Use background threads
    max_retrieval_workers: int = 2
    
    # Feature integration
    combine_retrieval_features: bool = True
    feature_weights: Optional[Dict[str, float]] = None
    adaptive_feature_weighting: bool = False
    
    # Cache alignment
    align_with_chunk_boundaries: bool = True  # Align retrieval with N=32 chunks
    cache_chunk_embeddings: bool = True
    
    # Telemetry
    track_retrieval_effectiveness: bool = True
    log_retrieval_stats: bool = True


class RetrievalCache:
    """
    Cache for retrieval results aligned with chunk boundaries.
    Provides fast access to retrieval features without blocking generation.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}  # query_hash -> retrieval_results
        self.access_times = {}  # query_hash -> last_access_time
        self.lock = threading.RLock()
    
    def _compute_query_hash(self, query_text: str, chunk_position: int) -> str:
        """Compute hash for query + chunk position."""
        return f"{hash(query_text)}_{chunk_position}"
    
    def get(self, query_text: str, chunk_position: int) -> Optional[Dict[str, Any]]:
        """Get cached retrieval results."""
        with self.lock:
            query_hash = self._compute_query_hash(query_text, chunk_position)
            if query_hash in self.cache:
                self.access_times[query_hash] = time.time()
                return self.cache[query_hash]
            return None
    
    def put(self, query_text: str, chunk_position: int, results: Dict[str, Any]):
        """Cache retrieval results."""
        with self.lock:
            query_hash = self._compute_query_hash(query_text, chunk_position)
            
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                # Remove the least recently used entry
                lru_hash = min(self.access_times.keys(), key=self.access_times.get)
                del self.cache[lru_hash]
                del self.access_times[lru_hash]
            
            self.cache[query_hash] = results
            self.access_times[query_hash] = time.time()
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': 0.0  # Would need to track hits/misses for real hit rate
            }


class BackgroundRetriever:
    """
    Background retrieval system that performs retrieval without blocking generation.
    Uses thread pool for concurrent retrieval operations.
    """
    
    def __init__(
        self,
        micro_retriever: MicroRetriever,
        feature_extractor: RetrievalFeatureExtractor,
        max_workers: int = 2,
        timeout_ms: int = 100
    ):
        self.micro_retriever = micro_retriever
        self.feature_extractor = feature_extractor
        self.timeout_ms = timeout_ms
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="retrieval")
        self.pending_futures = {}  # query_hash -> future
        self.lock = threading.RLock()
    
    def submit_retrieval(
        self,
        query_text: str,
        query_embedding: torch.Tensor,
        chunk_position: int
    ) -> str:
        """
        Submit retrieval task and return task ID.
        
        Args:
            query_text: Query text for retrieval
            query_embedding: Query embedding for features
            chunk_position: Chunk position for caching
            
        Returns:
            task_id: Unique task identifier
        """
        task_id = f"{hash(query_text)}_{chunk_position}_{time.time()}"
        
        future = self.executor.submit(
            self._retrieval_task,
            query_text, query_embedding, chunk_position
        )
        
        with self.lock:
            self.pending_futures[task_id] = future
        
        return task_id
    
    def get_results(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get retrieval results if ready, otherwise return None.
        Non-blocking - respects timeout.
        
        Args:
            task_id: Task identifier from submit_retrieval
            
        Returns:
            results: Retrieval results or None if not ready
        """
        with self.lock:
            if task_id not in self.pending_futures:
                return None
            
            future = self.pending_futures[task_id]
        
        try:
            # Non-blocking check with timeout
            results = future.result(timeout=self.timeout_ms / 1000.0)
            
            # Clean up completed future
            with self.lock:
                del self.pending_futures[task_id]
            
            return results
        
        except Exception as e:
            # Timeout or other error - return None
            logger.debug(f"Retrieval task {task_id} not ready: {e}")
            return None
    
    def _retrieval_task(
        self,
        query_text: str,
        query_embedding: torch.Tensor,
        chunk_position: int
    ) -> Dict[str, Any]:
        """Execute retrieval and feature extraction."""
        try:
            # Perform retrieval
            retrieval_results = self.micro_retriever.retrieve_for_queries([query_text])
            
            # Extract features
            features = self.feature_extractor.extract_features(
                query_embedding.unsqueeze(0),  # Add batch dimension
                retrieval_results,
                self.micro_retriever
            )
            
            return {
                'retrieval_results': retrieval_results,
                'features': features,
                'chunk_position': chunk_position,
                'timestamp': time.time()
            }
        
        except Exception as e:
            logger.error(f"Retrieval task failed: {e}")
            return {
                'retrieval_results': None,
                'features': None,
                'error': str(e),
                'chunk_position': chunk_position,
                'timestamp': time.time()
            }
    
    def shutdown(self):
        """Shutdown background retriever."""
        self.executor.shutdown(wait=True)


class RetrievalAwareBEMModule(HierarchicalBEMModule):
    """
    BEM module enhanced with retrieval-aware routing capabilities.
    Extends hierarchical BEM with retrieval features integration.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        config: RetrievalBEMConfig,
        micro_retriever: Optional[MicroRetriever] = None,
        layer_name: str = "unknown",
        attach_point: str = "mlp"
    ):
        # Initialize hierarchical BEM
        hierarchical_config = config.hierarchical_config or HierarchicalBEMConfig()
        super().__init__(base_layer, hierarchical_config, layer_name, attach_point)
        
        self.retrieval_config = config
        self.micro_retriever = micro_retriever
        
        # Initialize retrieval components if enabled
        if config.retrieval_enabled and micro_retriever is not None:
            # Feature extractor
            features_config = config.features_config or RetrievalFeaturesConfig()
            self.feature_extractor = create_retrieval_feature_extractor(features_config)
            
            # Retrieval cache
            if config.cache_retrieval_results:
                self.retrieval_cache = RetrievalCache(config.max_cache_size)
            else:
                self.retrieval_cache = None
            
            # Background retriever for non-blocking retrieval
            if config.non_blocking_retrieval and config.background_retrieval:
                self.background_retriever = BackgroundRetriever(
                    micro_retriever,
                    self.feature_extractor,
                    config.max_retrieval_workers,
                    config.retrieval_timeout_ms
                )
            else:
                self.background_retriever = None
        else:
            self.feature_extractor = None
            self.retrieval_cache = None
            self.background_retriever = None
        
        # Retrieval statistics
        self.register_buffer('total_retrievals', torch.tensor(0))
        self.register_buffer('cache_hits', torch.tensor(0))
        self.register_buffer('retrieval_timeouts', torch.tensor(0))
    
    def _get_retrieval_query(
        self,
        hidden_states: torch.Tensor,
        chunk_position: int
    ) -> Tuple[str, torch.Tensor]:
        """
        Extract query text and embedding for retrieval.
        
        Args:
            hidden_states: Hidden states [batch, seq_len, dim]
            chunk_position: Current chunk position
            
        Returns:
            query_text: Query text for retrieval
            query_embedding: Query embedding for features
        """
        # Simple approach: use mean of current chunk as query
        # In practice, you might want more sophisticated query extraction
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        chunk_size = getattr(self.config, 'chunk_size', 32)
        
        # Extract current chunk
        chunk_start = chunk_position
        chunk_end = min(chunk_start + chunk_size, seq_len)
        chunk_states = hidden_states[:, chunk_start:chunk_end, :]
        
        # Pool chunk states to get query embedding
        query_embedding = chunk_states.mean(dim=1).mean(dim=0)  # [hidden_dim]
        
        # Generate query text (placeholder - in practice you'd decode from hidden states)
        query_text = f"chunk_{chunk_position}_query"
        
        return query_text, query_embedding
    
    def _get_side_signals(
        self,
        retrieval_results: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Convert retrieval results to side signals for the controller.
        
        Args:
            retrieval_results: Results from background retrieval
            query_embedding: Query embedding for feature extraction
            
        Returns:
            side_signals: Combined side signals tensor
        """
        if not self.retrieval_config.retrieval_enabled or self.feature_extractor is None:
            return None
        
        if retrieval_results is None or retrieval_results.get('features') is None:
            return None
        
        try:
            features = retrieval_results['features']
            
            # Combine features into side signals
            if self.retrieval_config.combine_retrieval_features:
                side_signals = self.feature_extractor.combine_features(
                    features, 
                    self.retrieval_config.feature_weights
                )
                
                # Ensure correct batch dimension
                if side_signals.dim() == 1:
                    side_signals = side_signals.unsqueeze(0)
                
                return side_signals
            else:
                # Just use coverage as primary signal
                coverage = features.get('coverage', torch.tensor(0.0))
                if coverage.dim() == 0:
                    coverage = coverage.unsqueeze(0).unsqueeze(0)
                elif coverage.dim() == 1:
                    coverage = coverage.unsqueeze(1)
                
                return coverage
        
        except Exception as e:
            logger.warning(f"Error converting retrieval results to side signals: {e}")
            return None
    
    def forward(
        self,
        x: torch.Tensor,
        hidden_states: torch.Tensor,
        controller: HierarchicalController,
        attention_mask: Optional[torch.Tensor] = None,
        side_signals: Optional[torch.Tensor] = None,
        chunk_position: int = 0,
        return_routing_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass with retrieval-aware routing.
        
        Args:
            x: Input tensor
            hidden_states: Hidden states for controller  
            controller: HierarchicalController instance
            attention_mask: Optional attention mask
            side_signals: Optional explicit side signals (overrides retrieval)
            chunk_position: Current chunk position
            return_routing_info: Whether to return routing information
            
        Returns:
            output: Modified output tensor
            routing_info: Routing information (if requested)
        """
        retrieval_info = {}
        enhanced_side_signals = side_signals
        
        # Perform retrieval if enabled and no explicit side signals provided
        if (self.retrieval_config.retrieval_enabled and 
            side_signals is None and 
            self.micro_retriever is not None):
            
            try:
                # Get query for retrieval
                query_text, query_embedding = self._get_retrieval_query(
                    hidden_states, chunk_position
                )
                
                # Check cache first
                retrieval_results = None
                if self.retrieval_cache is not None:
                    retrieval_results = self.retrieval_cache.get(query_text, chunk_position)
                    if retrieval_results is not None:
                        self.cache_hits += 1
                        retrieval_info['cache_hit'] = True
                
                # If not in cache, try background retrieval
                if retrieval_results is None and self.background_retriever is not None:
                    # Submit background retrieval task
                    task_id = self.background_retriever.submit_retrieval(
                        query_text, query_embedding, chunk_position
                    )
                    
                    # Try to get immediate results (non-blocking)
                    retrieval_results = self.background_retriever.get_results(task_id)
                    
                    if retrieval_results is None:
                        self.retrieval_timeouts += 1
                        retrieval_info['retrieval_timeout'] = True
                    else:
                        retrieval_info['background_retrieval'] = True
                        
                        # Cache results for future use
                        if self.retrieval_cache is not None:
                            self.retrieval_cache.put(query_text, chunk_position, retrieval_results)
                
                # Convert to side signals
                if retrieval_results is not None:
                    enhanced_side_signals = self._get_side_signals(retrieval_results, query_embedding)
                    self.total_retrievals += 1
                    
                    retrieval_info.update({
                        'retrieval_success': True,
                        'num_retrieved_docs': len(retrieval_results.get('retrieval_results', {}).get('documents', [[]])[0]),
                        'coverage': retrieval_results.get('features', {}).get('coverage', torch.tensor(0.0)).item(),
                        'consistency': retrieval_results.get('features', {}).get('consistency', torch.tensor(0.0)).item()
                    })
            
            except Exception as e:
                logger.warning(f"Retrieval failed for {self.layer_name}: {e}")
                retrieval_info['retrieval_error'] = str(e)
        
        # Call parent forward with enhanced side signals
        if return_routing_info:
            output, routing_info = super().forward(
                x, hidden_states, controller, attention_mask,
                enhanced_side_signals, chunk_position, return_routing_info
            )
            
            # Add retrieval info to routing info
            routing_info['retrieval_info'] = retrieval_info
            return output, routing_info
        else:
            output = super().forward(
                x, hidden_states, controller, attention_mask,
                enhanced_side_signals, chunk_position, return_routing_info
            )
            return output
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get retrieval-specific statistics."""
        stats = {
            'total_retrievals': self.total_retrievals.item(),
            'cache_hits': self.cache_hits.item(),
            'retrieval_timeouts': self.retrieval_timeouts.item(),
            'cache_hit_rate': (self.cache_hits.float() / max(self.total_retrievals, 1)).item()
        }
        
        if self.retrieval_cache is not None:
            stats['cache_stats'] = self.retrieval_cache.get_stats()
        
        if self.micro_retriever is not None:
            stats['retriever_stats'] = self.micro_retriever.get_retrieval_statistics()
        
        return stats
    
    def reset_retrieval_statistics(self):
        """Reset retrieval statistics."""
        self.total_retrievals.zero_()
        self.cache_hits.zero_()
        self.retrieval_timeouts.zero_()
        
        if self.retrieval_cache is not None:
            self.retrieval_cache.clear()


class FullRetrievalAwareBEM(FullHierarchicalBEM):
    """
    Complete retrieval-aware BEM system that integrates retrieval coupling
    with hierarchical routing across the entire model.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: RetrievalBEMConfig,
        micro_retriever: Optional[MicroRetriever] = None,
        attach_layers: Optional[List[str]] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        # Initialize hierarchical BEM base
        hierarchical_config = config.hierarchical_config or HierarchicalBEMConfig()
        super().__init__(base_model, hierarchical_config, attach_layers, model_config)
        
        self.retrieval_config = config
        self.micro_retriever = micro_retriever
        
        # Replace hierarchical BEM modules with retrieval-aware ones
        if config.retrieval_enabled and micro_retriever is not None:
            self._upgrade_to_retrieval_aware()
        
        # Initialize telemetry
        if config.track_retrieval_effectiveness:
            self.telemetry = BEMTelemetry()
        else:
            self.telemetry = None
    
    def _upgrade_to_retrieval_aware(self):
        """Upgrade existing BEM modules to retrieval-aware versions."""
        for layer_name, bem_module in list(self.bem_modules.items()):
            # Create retrieval-aware version
            retrieval_bem = RetrievalAwareBEMModule(
                base_layer=bem_module.base_layer,
                config=self.retrieval_config,
                micro_retriever=self.micro_retriever,
                layer_name=layer_name,
                attach_point=bem_module.attach_point
            )
            
            # Copy over hierarchical BEM state
            retrieval_bem.lora_U.data.copy_(bem_module.lora_U.data)
            retrieval_bem.lora_V.data.copy_(bem_module.lora_V.data)
            retrieval_bem.total_forward_calls.copy_(bem_module.total_forward_calls)
            retrieval_bem.routing_stats.copy_(bem_module.routing_stats)
            
            # Replace in ModuleDict and base model
            self.bem_modules[layer_name] = retrieval_bem
            self._replace_module(layer_name, retrieval_bem)
    
    def build_retrieval_index(
        self,
        corpus_path: str,
        **index_kwargs
    ):
        """Build retrieval index from corpus."""
        if self.micro_retriever is None:
            raise RuntimeError("No micro-retriever available")
        
        logger.info(f"Building retrieval index from {corpus_path}")
        self.micro_retriever.build_index_from_corpus(corpus_path, **index_kwargs)
        logger.info("Retrieval index built successfully")
    
    def save_retrieval_index(self, path: Optional[str] = None):
        """Save retrieval index."""
        if self.micro_retriever is not None:
            self.micro_retriever.save_index(path)
    
    def load_retrieval_index(self, path: Optional[str] = None):
        """Load retrieval index."""
        if self.micro_retriever is not None:
            self.micro_retriever.load_index(path)
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics including retrieval metrics."""
        stats = super().get_routing_statistics()
        
        # Add retrieval statistics
        retrieval_stats = {
            'total_retrieval_modules': sum(
                1 for module in self.bem_modules.values()
                if isinstance(module, RetrievalAwareBEMModule)
            ),
            'retrieval_enabled': self.retrieval_config.retrieval_enabled,
            'layer_retrieval_stats': {}
        }
        
        for layer_name, bem_module in self.bem_modules.items():
            if isinstance(bem_module, RetrievalAwareBEMModule):
                retrieval_stats['layer_retrieval_stats'][layer_name] = \
                    bem_module.get_retrieval_statistics()
        
        # Global retrieval statistics
        if self.micro_retriever is not None:
            retrieval_stats['global_retriever_stats'] = \
                self.micro_retriever.get_retrieval_statistics()
        
        stats['retrieval'] = retrieval_stats
        
        return stats
    
    def run_index_swap_evaluation(
        self,
        test_queries: List[str],
        alternative_index_path: str
    ) -> Dict[str, Any]:
        """
        Run index-swap evaluation to test policy over memory.
        
        This test validates that BEM behavior follows evidence quality
        rather than memorized patterns.
        
        Args:
            test_queries: List of test queries
            alternative_index_path: Path to alternative index
            
        Returns:
            evaluation_results: Policy vs memory test results
        """
        if self.micro_retriever is None:
            raise RuntimeError("No micro-retriever available for index swap test")
        
        logger.info("Running index-swap evaluation")
        
        # Save original index
        original_index_path = "temp_original_index.faiss"
        self.micro_retriever.save_index(original_index_path)
        
        try:
            # Test with original index
            original_results = []
            for query in test_queries:
                result = self.micro_retriever.retrieve_for_queries([query])
                original_results.append(result)
            
            # Swap to alternative index
            self.micro_retriever.load_index(alternative_index_path)
            
            # Test with alternative index
            alternative_results = []
            for query in test_queries:
                result = self.micro_retriever.retrieve_for_queries([query])
                alternative_results.append(result)
            
            # Analyze behavior differences
            behavior_changes = []
            for orig, alt in zip(original_results, alternative_results):
                # Compare retrieval similarities and document content
                orig_sims = orig['similarities'][0] if orig['similarities'] else []
                alt_sims = alt['similarities'][0] if alt['similarities'] else []
                
                sim_change = abs(np.mean(orig_sims) - np.mean(alt_sims)) if orig_sims and alt_sims else 0
                behavior_changes.append(sim_change)
            
            # Restore original index
            self.micro_retriever.load_index(original_index_path)
            
            return {
                'mean_behavior_change': np.mean(behavior_changes),
                'std_behavior_change': np.std(behavior_changes),
                'policy_over_memory': np.mean(behavior_changes) > 0.1,  # Threshold for meaningful change
                'num_queries_tested': len(test_queries)
            }
        
        finally:
            # Cleanup
            import os
            if os.path.exists(original_index_path):
                os.remove(original_index_path)
            if os.path.exists(original_index_path.replace('.faiss', '.pkl')):
                os.remove(original_index_path.replace('.faiss', '.pkl'))
    
    def shutdown(self):
        """Shutdown retrieval-aware BEM system."""
        # Shutdown background retrievers
        for bem_module in self.bem_modules.values():
            if isinstance(bem_module, RetrievalAwareBEMModule) and bem_module.background_retriever:
                bem_module.background_retriever.shutdown()


def create_retrieval_aware_bem(
    base_model: nn.Module,
    config: Optional[RetrievalBEMConfig] = None,
    corpus_path: Optional[str] = None,
    attach_layers: Optional[List[str]] = None,
    **config_kwargs
) -> FullRetrievalAwareBEM:
    """
    Factory function to create a complete retrieval-aware BEM system.
    
    Args:
        base_model: Base model to wrap
        config: RetrievalBEMConfig instance  
        corpus_path: Optional path to corpus for building index
        attach_layers: Specific layers to attach BEMs to
        **config_kwargs: Config overrides
        
    Returns:
        FullRetrievalAwareBEM instance
    """
    if config is None:
        config = RetrievalBEMConfig()
        
        # Set default sub-configs
        if config.hierarchical_config is None:
            config.hierarchical_config = HierarchicalBEMConfig(**config_kwargs)
        if config.retrieval_config is None:
            config.retrieval_config = RetrievalConfig()
        if config.features_config is None:
            config.features_config = RetrievalFeaturesConfig()
    
    # Create micro-retriever if retrieval is enabled
    micro_retriever = None
    if config.retrieval_enabled:
        micro_retriever = create_micro_retriever(config.retrieval_config)
        
        # Build index if corpus provided
        if corpus_path is not None:
            micro_retriever.build_index_from_corpus(corpus_path)
    
    # Extract model config
    model_config = None
    if hasattr(base_model, 'config'):
        model_config = base_model.config
        if hasattr(model_config, '__dict__'):
            model_config = model_config.__dict__
    
    # Create retrieval-aware BEM
    retrieval_bem = FullRetrievalAwareBEM(
        base_model=base_model,
        config=config,
        micro_retriever=micro_retriever,
        attach_layers=attach_layers,
        model_config=model_config
    )
    
    return retrieval_bem