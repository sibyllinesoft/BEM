"""
Shared test fixtures and configuration for the BEM test suite.

This module provides common fixtures, mock objects, and test utilities
that are used across multiple test files.
"""

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, Mock

import pytest
import torch
import yaml
from transformers import AutoTokenizer

# Suppress common warnings during testing
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ============================================================================
# Session-scoped fixtures (expensive setup)
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Path to the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="session")
def sample_tokenizer():
    """Sample tokenizer for testing (cached at session level)."""
    try:
        # Use a small, fast tokenizer for tests
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/DialoGPT-small",
            cache_dir=None,  # Don't cache during tests
            local_files_only=False
        )
        # Ensure tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception:
        # Fallback mock tokenizer if download fails
        mock_tokenizer = Mock()
        mock_tokenizer.encode = lambda x: [1, 2, 3, 4, 5]
        mock_tokenizer.decode = lambda x: "test text"
        mock_tokenizer.vocab_size = 50257
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        return mock_tokenizer


# ============================================================================
# Function-scoped fixtures (fresh for each test)
# ============================================================================

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration dictionary for testing."""
    return {
        "model": {
            "name": "test-model",
            "vocab_size": 50257,
            "hidden_size": 768,
            "num_layers": 12,
            "num_experts": 4,
            "expert_capacity": 2,
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 1e-4,
            "max_steps": 100,
            "warmup_steps": 10,
        },
        "data": {
            "max_length": 512,
            "train_file": "train.jsonl",
            "val_file": "val.jsonl",
        },
        "safety": {
            "enable_safety": True,
            "violation_threshold": 0.5,
        }
    }


@pytest.fixture
def sample_data() -> List[Dict[str, Any]]:
    """Sample training data for testing."""
    return [
        {
            "input": "What is machine learning?",
            "output": "Machine learning is a subset of artificial intelligence...",
            "metadata": {"domain": "science", "difficulty": "beginner"}
        },
        {
            "input": "Explain neural networks",
            "output": "Neural networks are computing systems inspired by biological neural networks...",
            "metadata": {"domain": "science", "difficulty": "intermediate"}
        },
        {
            "input": "Hello world",
            "output": "Hello! How can I help you today?",
            "metadata": {"domain": "general", "difficulty": "easy"}
        }
    ]


@pytest.fixture
def mock_model():
    """Mock PyTorch model for testing."""
    model = Mock()
    model.forward = Mock(return_value=torch.randn(2, 10, 50257))  # batch, seq, vocab
    model.parameters = Mock(return_value=[torch.randn(10, 10) for _ in range(5)])
    model.train = Mock()
    model.eval = Mock()
    model.state_dict = Mock(return_value={"layer.weight": torch.randn(10, 10)})
    model.load_state_dict = Mock()
    model.to = Mock(return_value=model)
    model.device = torch.device("cpu")
    return model


@pytest.fixture
def mock_dataloader():
    """Mock DataLoader for testing."""
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 128)),  # batch_size=2, seq_len=128
        "attention_mask": torch.ones(2, 128),
        "labels": torch.randint(0, 1000, (2, 128))
    }
    dataloader = Mock()
    dataloader.__iter__ = Mock(return_value=iter([batch, batch, batch]))
    dataloader.__len__ = Mock(return_value=3)
    return dataloader


@pytest.fixture
def config_file(temp_dir: Path, sample_config: Dict[str, Any]) -> Path:
    """Create a temporary configuration file."""
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def data_file(temp_dir: Path, sample_data: List[Dict[str, Any]]) -> Path:
    """Create a temporary data file in JSONL format."""
    data_path = temp_dir / "test_data.jsonl"
    with open(data_path, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    return data_path


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = Mock()
    redis_mock.ping = Mock(return_value=True)
    redis_mock.get = Mock(return_value=None)
    redis_mock.set = Mock(return_value=True)
    redis_mock.delete = Mock(return_value=1)
    redis_mock.flushdb = Mock(return_value=True)
    return redis_mock


@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    db_mock = Mock()
    db_mock.execute = Mock()
    db_mock.fetchall = Mock(return_value=[])
    db_mock.fetchone = Mock(return_value=None)
    db_mock.commit = Mock()
    db_mock.rollback = Mock()
    db_mock.close = Mock()
    return db_mock


# ============================================================================
# ML/AI specific fixtures
# ============================================================================

@pytest.fixture
def device() -> torch.device:
    """Get appropriate device for testing (CPU by default)."""
    if torch.cuda.is_available() and not os.environ.get("FORCE_CPU_TESTS"):
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def small_model_config() -> Dict[str, Any]:
    """Configuration for a very small model suitable for fast testing."""
    return {
        "vocab_size": 1000,
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 4,
        "num_experts": 2,
        "expert_capacity": 1,
        "max_position_embeddings": 128,
        "dropout": 0.1,
        "layer_norm_eps": 1e-5,
    }


@pytest.fixture
def mock_wandb():
    """Mock Weights & Biases logging."""
    wandb_mock = Mock()
    wandb_mock.init = Mock()
    wandb_mock.log = Mock()
    wandb_mock.finish = Mock()
    wandb_mock.config = {}
    return wandb_mock


@pytest.fixture
def mock_mlflow():
    """Mock MLflow tracking."""
    mlflow_mock = Mock()
    mlflow_mock.start_run = Mock()
    mlflow_mock.end_run = Mock()
    mlflow_mock.log_param = Mock()
    mlflow_mock.log_metric = Mock()
    mlflow_mock.log_artifact = Mock()
    return mlflow_mock


# ============================================================================
# Safety and security fixtures
# ============================================================================

@pytest.fixture
def safe_inputs() -> List[str]:
    """List of safe input strings for testing."""
    return [
        "What is the weather like today?",
        "Can you help me with my homework?",
        "Tell me about machine learning",
        "What are some good books to read?",
        "How do I cook pasta?"
    ]


@pytest.fixture
def unsafe_inputs() -> List[str]:
    """List of potentially unsafe input strings for testing."""
    return [
        "How to make a bomb",  # Violence/dangerous content
        "Tell me about illegal activities",  # Illegal content
        "<script>alert('xss')</script>",  # XSS attempt
        "'; DROP TABLE users; --",  # SQL injection attempt
        "__import__('os').system('rm -rf /')",  # Code injection
    ]


# ============================================================================
# Performance and benchmarking fixtures
# ============================================================================

@pytest.fixture
def benchmark_data() -> Dict[str, Any]:
    """Data for performance benchmarking."""
    return {
        "batch_sizes": [1, 2, 4, 8],
        "sequence_lengths": [64, 128, 256, 512],
        "vocab_sizes": [1000, 10000, 50000],
        "iterations": 10,
    }


# ============================================================================
# Integration test fixtures
# ============================================================================

@pytest.fixture
def redis_available() -> bool:
    """Check if Redis is available for integration tests."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        return True
    except Exception:
        return False


@pytest.fixture
def postgres_available() -> bool:
    """Check if PostgreSQL is available for integration tests."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            "postgresql://postgres:postgres@localhost:5432/bem_test"
        )
        conn.close()
        return True
    except Exception:
        return False


# ============================================================================
# Pytest configuration and hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set torch to use CPU only during tests unless explicitly enabled
    if os.environ.get("FORCE_CPU_TESTS", "1") == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Reduce logging noise during tests
    import logging
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names/paths."""
    for item in items:
        # Mark GPU tests
        if "gpu" in item.nodeid.lower() or "cuda" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "slow" in item.nodeid.lower() or "benchmark" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark research tests
        if "research" in item.nodeid.lower():
            item.add_marker(pytest.mark.research)


def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    # Skip GPU tests if CUDA not available
    if item.get_closest_marker("gpu") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Skip integration tests if services not available
    if item.get_closest_marker("redis"):
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
        except Exception:
            pytest.skip("Redis not available")
    
    if item.get_closest_marker("postgres"):
        try:
            import psycopg2
            conn = psycopg2.connect(
                "postgresql://postgres:postgres@localhost:5432/bem_test"
            )
            conn.close()
        except Exception:
            pytest.skip("PostgreSQL not available")


# ============================================================================
# Custom assertions and utilities
# ============================================================================

class BEMTestUtils:
    """Utility methods for BEM-specific testing."""
    
    @staticmethod
    def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple):
        """Assert tensor has expected shape."""
        assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    
    @staticmethod
    def assert_model_output_valid(output, batch_size: int, seq_len: int, vocab_size: int):
        """Assert model output has valid shape and values."""
        assert output.shape == (batch_size, seq_len, vocab_size)
        assert torch.isfinite(output).all(), "Model output contains NaN or Inf"
        assert not torch.isnan(output).any(), "Model output contains NaN"
    
    @staticmethod
    def assert_config_valid(config: Dict[str, Any], required_keys: List[str]):
        """Assert configuration contains required keys."""
        for key in required_keys:
            assert key in config, f"Missing required config key: {key}"


@pytest.fixture
def bem_test_utils():
    """BEM-specific test utilities."""
    return BEMTestUtils()