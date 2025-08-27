"""
Micro-retriever infrastructure for BEM Phase 3.

This module implements the retrieval coupling architecture:
1. FAISS index with frozen sentence-transformer encoder
2. Efficient batched retrieval for multiple queries
3. Index building and management from domain corpus
4. HyDE (Hypothetical Document Embeddings) support

Based on TODO.md Phase 3 requirements for retrieval-aware routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import logging

# Optional FAISS import with fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

# Optional sentence-transformers import with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval system."""
    # Index configuration
    index_path: str = "indices/domain.faiss"
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384  # all-MiniLM-L6-v2 dimension
    max_corpus_size: int = 2_000_000  # 2M chunks max
    
    # Retrieval parameters
    num_retrieved: int = 8  # Number of documents to retrieve
    retrieval_batch_size: int = 32  # Batch size for retrieval
    
    # Coverage and consistency parameters
    coverage_threshold: float = 0.3  # Minimum similarity for coverage
    consistency_threshold: float = 0.5  # Minimum inter-doc similarity
    
    # HyDE parameters
    enable_hyde: bool = False
    hyde_num_hypotheses: int = 3
    
    # Performance settings
    use_gpu: bool = True  # Use GPU for FAISS if available
    precompute_norms: bool = True  # Precompute document norms for efficiency
    
    # Cache settings
    cache_embeddings: bool = True
    max_cache_size: int = 10000  # Maximum embeddings to cache


class DocumentIndex:
    """
    Document index wrapper around FAISS for efficient similarity search.
    Supports both CPU and GPU operation with frozen encoder.
    """
    
    def __init__(
        self,
        config: RetrievalConfig,
        encoder: Optional[SentenceTransformer] = None
    ):
        self.config = config
        self.embedding_dim = config.embedding_dim
        
        # Initialize encoder
        if encoder is not None:
            self.encoder = encoder
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            self.encoder = SentenceTransformer(config.encoder_name)
            # Freeze encoder parameters
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            raise RuntimeError(
                "sentence-transformers not available. Install with: "
                "pip install sentence-transformers"
            )
        
        # Initialize FAISS index
        if not FAISS_AVAILABLE:
            raise RuntimeError(
                "FAISS not available. Install with: "
                "pip install faiss-cpu faiss-gpu"
            )
        
        # Create FAISS index
        self.index = None
        self.documents = []
        self.document_ids = []
        self.metadata = {}
        
        # Embedding cache
        self.embedding_cache = {} if config.cache_embeddings else None
        
        # Load existing index if available
        if Path(config.index_path).exists():
            self.load_index()
        else:
            logger.info(f"No existing index found at {config.index_path}")
    
    def _create_faiss_index(self, use_gpu: bool = None) -> faiss.Index:
        """Create FAISS index with appropriate configuration."""
        if use_gpu is None:
            use_gpu = self.config.use_gpu and torch.cuda.is_available()
        
        # Create base index (Inner Product for normalized embeddings)
        index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add GPU support if requested and available
        if use_gpu and hasattr(faiss, 'StandardGpuResources'):
            try:
                gpu_res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
                logger.info("Using GPU FAISS index")
            except Exception as e:
                logger.warning(f"GPU FAISS not available, falling back to CPU: {e}")
                index = faiss.IndexFlatIP(self.embedding_dim)
        
        return index
    
    def build_index(
        self,
        documents: List[str],
        document_ids: Optional[List[str]] = None,
        batch_size: int = 32,
        show_progress: bool = True
    ):
        """
        Build FAISS index from document corpus.
        
        Args:
            documents: List of document texts
            document_ids: Optional list of document IDs
            batch_size: Batch size for encoding
            show_progress: Whether to show progress
        """
        if len(documents) > self.config.max_corpus_size:
            logger.warning(
                f"Corpus size {len(documents)} exceeds maximum {self.config.max_corpus_size}. "
                f"Using first {self.config.max_corpus_size} documents."
            )
            documents = documents[:self.config.max_corpus_size]
        
        logger.info(f"Building index for {len(documents)} documents")
        
        # Generate document IDs if not provided
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.documents = documents
        self.document_ids = document_ids
        
        # Create FAISS index
        self.index = self._create_faiss_index()
        
        # Encode documents in batches
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            if show_progress and i % (batch_size * 10) == 0:
                logger.info(f"Encoding batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}")
            
            # Encode batch
            with torch.no_grad():
                embeddings = self.encoder.encode(
                    batch_docs,
                    convert_to_tensor=True,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                # Normalize embeddings for cosine similarity via inner product
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
        
        # Concatenate all embeddings
        embeddings_matrix = np.vstack(all_embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings_matrix)
        
        # Store metadata
        self.metadata = {
            'num_documents': len(documents),
            'embedding_dim': self.embedding_dim,
            'encoder_name': self.config.encoder_name,
            'build_timestamp': torch.tensor(0)  # Placeholder
        }
        
        logger.info(f"Built index with {len(documents)} documents")
    
    def save_index(self, path: Optional[str] = None):
        """Save FAISS index and metadata to disk."""
        if path is None:
            path = self.config.index_path
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if hasattr(self.index, 'index'):
            # GPU index - move to CPU first
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(path))
        else:
            faiss.write_index(self.index, str(path))
        
        # Save metadata and documents
        metadata_path = path.with_suffix('.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'document_ids': self.document_ids,
                'metadata': self.metadata,
                'config': self.config
            }, f)
        
        logger.info(f"Saved index to {path}")
    
    def load_index(self, path: Optional[str] = None):
        """Load FAISS index and metadata from disk."""
        if path is None:
            path = self.config.index_path
        
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Index not found at {path}")
        
        # Load FAISS index
        cpu_index = faiss.read_index(str(path))
        
        # Move to GPU if requested
        if self.config.use_gpu and torch.cuda.is_available():
            try:
                gpu_res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
                logger.info("Loaded index on GPU")
            except Exception as e:
                logger.warning(f"GPU FAISS not available, using CPU: {e}")
                self.index = cpu_index
        else:
            self.index = cpu_index
        
        # Load metadata and documents
        metadata_path = path.with_suffix('.pkl')
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data.get('documents', [])
                self.document_ids = data.get('document_ids', [])
                self.metadata = data.get('metadata', {})
                
                # Update config if saved config exists
                if 'config' in data:
                    saved_config = data['config']
                    # Only update non-path related config
                    self.config.embedding_dim = saved_config.embedding_dim
                    self.config.encoder_name = saved_config.encoder_name
        
        logger.info(f"Loaded index with {len(self.documents)} documents")
    
    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        """
        Encode queries using the frozen encoder with caching.
        
        Args:
            queries: List of query strings
            
        Returns:
            embeddings: Query embeddings [num_queries, embedding_dim]
        """
        if self.embedding_cache is not None:
            # Check cache first
            cached_embeddings = []
            uncached_queries = []
            uncached_indices = []
            
            for i, query in enumerate(queries):
                if query in self.embedding_cache:
                    cached_embeddings.append((i, self.embedding_cache[query]))
                else:
                    uncached_queries.append(query)
                    uncached_indices.append(i)
            
            # Encode uncached queries
            if uncached_queries:
                with torch.no_grad():
                    new_embeddings = self.encoder.encode(
                        uncached_queries,
                        convert_to_tensor=True,
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )
                    new_embeddings = F.normalize(new_embeddings, p=2, dim=1)
                
                # Add to cache (with size limit)
                if len(self.embedding_cache) < self.config.max_cache_size:
                    for query, embedding in zip(uncached_queries, new_embeddings):
                        self.embedding_cache[query] = embedding
            
            # Combine cached and new embeddings
            all_embeddings = torch.zeros(len(queries), self.embedding_dim)
            
            # Fill in cached embeddings
            for i, embedding in cached_embeddings:
                all_embeddings[i] = embedding
            
            # Fill in new embeddings
            if uncached_queries:
                for i, embedding in zip(uncached_indices, new_embeddings):
                    all_embeddings[i] = embedding
            
            return all_embeddings
        
        else:
            # No caching - encode directly
            with torch.no_grad():
                embeddings = self.encoder.encode(
                    queries,
                    convert_to_tensor=True,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                embeddings = F.normalize(embeddings, p=2, dim=1)
                return embeddings
    
    def retrieve(
        self,
        queries: List[str],
        k: Optional[int] = None
    ) -> Tuple[List[List[str]], List[List[float]], List[List[int]]]:
        """
        Retrieve top-k documents for queries.
        
        Args:
            queries: List of query strings
            k: Number of documents to retrieve (default: config.num_retrieved)
            
        Returns:
            documents: List of retrieved document lists
            similarities: List of similarity score lists
            indices: List of document index lists
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        if k is None:
            k = self.config.num_retrieved
        
        # Encode queries
        query_embeddings = self.encode_queries(queries)
        
        # Convert to numpy for FAISS
        if query_embeddings.is_cuda:
            query_embeddings_np = query_embeddings.cpu().numpy().astype(np.float32)
        else:
            query_embeddings_np = query_embeddings.numpy().astype(np.float32)
        
        # Search FAISS index
        similarities, indices = self.index.search(query_embeddings_np, k)
        
        # Convert results to lists and retrieve documents
        retrieved_documents = []
        retrieved_similarities = []
        retrieved_indices = []
        
        for i, (sim_scores, doc_indices) in enumerate(zip(similarities, indices)):
            query_docs = []
            query_sims = []
            query_indices = []
            
            for sim, idx in zip(sim_scores, doc_indices):
                if idx >= 0 and idx < len(self.documents):  # Valid index
                    query_docs.append(self.documents[idx])
                    query_sims.append(float(sim))
                    query_indices.append(int(idx))
            
            retrieved_documents.append(query_docs)
            retrieved_similarities.append(query_sims)
            retrieved_indices.append(query_indices)
        
        return retrieved_documents, retrieved_similarities, retrieved_indices


class HyDEGenerator:
    """
    Hypothetical Document Embeddings (HyDE) generator.
    Creates hypothetical documents from queries to improve retrieval.
    """
    
    def __init__(
        self,
        config: RetrievalConfig,
        generator_model: Optional[nn.Module] = None
    ):
        self.config = config
        self.generator_model = generator_model
        
        if not config.enable_hyde:
            return
        
        if generator_model is None:
            logger.warning(
                "HyDE enabled but no generator model provided. "
                "HyDE will use simple query expansion instead."
            )
    
    def generate_hypotheses(self, queries: List[str]) -> List[List[str]]:
        """
        Generate hypothetical documents for queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            hypotheses: List of hypothesis lists for each query
        """
        if not self.config.enable_hyde:
            return [[query] for query in queries]
        
        hypotheses = []
        
        for query in queries:
            if self.generator_model is not None:
                # Use model to generate hypotheses
                query_hypotheses = self._generate_with_model(query)
            else:
                # Simple query expansion as fallback
                query_hypotheses = self._expand_query(query)
            
            hypotheses.append(query_hypotheses)
        
        return hypotheses
    
    def _generate_with_model(self, query: str) -> List[str]:
        """Generate hypotheses using a language model."""
        # Placeholder for actual model generation
        # In practice, you'd use a small LM to generate hypothetical answers
        return [query] * self.config.hyde_num_hypotheses
    
    def _expand_query(self, query: str) -> List[str]:
        """Simple query expansion as fallback."""
        # Simple expansions - in practice you'd use more sophisticated methods
        expansions = [
            query,
            f"What is {query}?",
            f"How to {query}?"
        ]
        return expansions[:self.config.hyde_num_hypotheses]


class MicroRetriever(nn.Module):
    """
    Main micro-retriever class that combines index, HyDE, and provides
    a clean interface for retrieval-aware routing.
    """
    
    def __init__(
        self,
        config: RetrievalConfig,
        encoder: Optional[SentenceTransformer] = None
    ):
        super().__init__()
        
        self.config = config
        
        # Initialize document index
        self.index = DocumentIndex(config, encoder)
        
        # Initialize HyDE generator
        self.hyde_generator = HyDEGenerator(config)
        
        # Performance tracking
        self.register_buffer('total_retrievals', torch.tensor(0))
        self.register_buffer('cache_hits', torch.tensor(0))
    
    def build_index_from_corpus(
        self,
        corpus_path: str,
        text_field: str = 'text',
        id_field: str = 'id',
        max_docs: Optional[int] = None
    ):
        """
        Build index from a corpus file.
        
        Args:
            corpus_path: Path to corpus file (jsonl format)
            text_field: Field name for document text
            id_field: Field name for document ID
            max_docs: Maximum number of documents to index
        """
        documents = []
        document_ids = []
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_docs and i >= max_docs:
                    break
                
                try:
                    doc = json.loads(line.strip())
                    documents.append(doc.get(text_field, ''))
                    document_ids.append(doc.get(id_field, f'doc_{i}'))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON on line {i+1}")
                    continue
        
        if not documents:
            raise ValueError(f"No documents found in {corpus_path}")
        
        logger.info(f"Loaded {len(documents)} documents from {corpus_path}")
        self.index.build_index(documents, document_ids)
    
    def retrieve_for_queries(
        self,
        queries: List[str],
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main retrieval interface that returns comprehensive retrieval results.
        
        Args:
            queries: List of query strings
            k: Number of documents to retrieve
            
        Returns:
            retrieval_results: Dictionary with documents, scores, and metadata
        """
        if k is None:
            k = self.config.num_retrieved
        
        # Generate hypotheses if HyDE is enabled
        if self.config.enable_hyde:
            hypotheses = self.hyde_generator.generate_hypotheses(queries)
            # Flatten hypotheses for retrieval
            all_queries = []
            query_mapping = []  # Maps flattened index back to original query
            for i, hyp_list in enumerate(hypotheses):
                for hyp in hyp_list:
                    all_queries.append(hyp)
                    query_mapping.append(i)
        else:
            all_queries = queries
            query_mapping = list(range(len(queries)))
        
        # Retrieve documents
        retrieved_docs, similarities, indices = self.index.retrieve(all_queries, k)
        
        # Aggregate results back to original queries if using HyDE
        if self.config.enable_hyde:
            aggregated_docs = [[] for _ in range(len(queries))]
            aggregated_sims = [[] for _ in range(len(queries))]
            aggregated_indices = [[] for _ in range(len(queries))]
            
            for i, (docs, sims, idxs) in enumerate(zip(retrieved_docs, similarities, indices)):
                orig_query_idx = query_mapping[i]
                aggregated_docs[orig_query_idx].extend(docs)
                aggregated_sims[orig_query_idx].extend(sims)
                aggregated_indices[orig_query_idx].extend(idxs)
            
            # Deduplicate and re-rank
            final_docs = []
            final_sims = []
            final_indices = []
            
            for i in range(len(queries)):
                # Simple deduplication by document index
                seen_indices = set()
                query_docs = []
                query_sims = []
                query_indices = []
                
                for doc, sim, idx in zip(aggregated_docs[i], aggregated_sims[i], aggregated_indices[i]):
                    if idx not in seen_indices:
                        query_docs.append(doc)
                        query_sims.append(sim)
                        query_indices.append(idx)
                        seen_indices.add(idx)
                        
                        if len(query_docs) >= k:
                            break
                
                final_docs.append(query_docs)
                final_sims.append(query_sims)
                final_indices.append(query_indices)
            
            retrieved_docs, similarities, indices = final_docs, final_sims, final_indices
        
        # Update statistics
        self.total_retrievals += len(queries)
        
        return {
            'documents': retrieved_docs,
            'similarities': similarities,
            'indices': indices,
            'query_embeddings': self.index.encode_queries(queries),
            'num_queries': len(queries),
            'k': k
        }
    
    def save_index(self, path: Optional[str] = None):
        """Save the retrieval index."""
        self.index.save_index(path)
    
    def load_index(self, path: Optional[str] = None):
        """Load the retrieval index."""
        self.index.load_index(path)
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get retrieval performance statistics."""
        return {
            'total_retrievals': self.total_retrievals.item(),
            'cache_hits': self.cache_hits.item(),
            'index_size': len(self.index.documents),
            'embedding_dim': self.config.embedding_dim,
            'cache_size': len(self.index.embedding_cache) if self.index.embedding_cache else 0
        }


def create_micro_retriever(
    config: Optional[RetrievalConfig] = None,
    **config_kwargs
) -> MicroRetriever:
    """
    Factory function to create a micro-retriever.
    
    Args:
        config: RetrievalConfig instance
        **config_kwargs: Config overrides
        
    Returns:
        MicroRetriever instance
    """
    if config is None:
        config = RetrievalConfig(**config_kwargs)
    else:
        # Update config with any provided kwargs
        for key, value in config_kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return MicroRetriever(config)