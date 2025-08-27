"""Base configuration classes for BEM experiments.

Defines the foundational configuration structure that all BEM components
and experiments inherit from, ensuring consistency and validation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class BaseConfig:
    """Base configuration class for all BEM components.
    
    Provides common configuration patterns and validation methods
    that all experiment and component configs inherit from.
    """
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_config()
        self._resolve_paths()
    
    def _validate_config(self) -> None:
        """Validate configuration values. Override in subclasses."""
        pass
    
    def _resolve_paths(self) -> None:
        """Resolve relative paths to absolute paths."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, (str, Path)) and field_name.endswith(('_path', '_dir', '_file')):
                if not os.path.isabs(str(field_value)):
                    setattr(self, field_name, str(Path(field_value).resolve()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Unknown config field: {key}")


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for model architecture."""
    
    # Base model
    base_model: str = "microsoft/DialoGPT-small"
    model_path: Optional[str] = None
    
    # Architecture parameters
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: Optional[int] = None
    
    # Precision and optimization
    torch_dtype: str = "float32"  # "float16", "bfloat16", "float32"
    attn_implementation: str = "eager"  # "eager", "flash_attention_2", "sdpa"
    
    # Model-specific parameters (to be extended by subclasses)
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def _validate_config(self) -> None:
        """Validate model configuration."""
        super()._validate_config()
        
        # Validate dtype
        valid_dtypes = {"float16", "bfloat16", "float32"}
        if self.torch_dtype not in valid_dtypes:
            raise ValueError(f"torch_dtype must be one of {valid_dtypes}")
        
        # Validate attention implementation
        valid_attn = {"eager", "flash_attention_2", "sdpa"}
        if self.attn_implementation not in valid_attn:
            raise ValueError(f"attn_implementation must be one of {valid_attn}")


@dataclass 
class DataConfig(BaseConfig):
    """Configuration for data loading and preprocessing."""
    
    # Data files
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Data parameters
    max_seq_length: int = 512
    max_samples: Optional[int] = None
    
    # Preprocessing
    add_special_tokens: bool = True
    truncation: bool = True
    padding: str = "max_length"  # "max_length", "longest", "do_not_pad"
    
    # Data loading
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    
    def _validate_config(self) -> None:
        """Validate data configuration."""
        super()._validate_config()
        
        # Validate padding strategy
        valid_padding = {"max_length", "longest", "do_not_pad"}
        if self.padding not in valid_padding:
            raise ValueError(f"padding must be one of {valid_padding}")
        
        # Check that at least train file is provided
        if not self.train_file:
            raise ValueError("train_file is required")


@dataclass
class HardwareConfig(BaseConfig):
    """Configuration for hardware and system settings."""
    
    # Device configuration
    device: Optional[str] = None  # Auto-detect if None
    gpu_ids: Optional[List[int]] = None
    
    # Memory and precision
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    gradient_checkpointing: bool = False
    
    # Distributed training
    distributed: bool = False
    local_rank: int = -1
    
    # Resource requirements
    min_gpu_memory_gb: Optional[int] = None
    preferred_gpu: Optional[str] = None
    
    def _validate_config(self) -> None:
        """Validate hardware configuration."""
        super()._validate_config()
        
        # Validate mixed precision
        valid_precision = {"no", "fp16", "bf16"}
        if self.mixed_precision not in valid_precision:
            raise ValueError(f"mixed_precision must be one of {valid_precision}")


@dataclass
class LoggingConfig(BaseConfig):
    """Configuration for logging and monitoring."""
    
    # Basic logging
    level: str = "INFO"
    log_file: Optional[str] = None
    console_output: bool = True
    
    # Experiment tracking
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_config: Dict[str, Any] = field(default_factory=dict)
    
    # MLflow tracking
    mlflow_enabled: bool = False
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    
    # Telemetry
    log_frequency: int = 10
    metrics_to_log: List[str] = field(default_factory=list)
    
    def _validate_config(self) -> None:
        """Validate logging configuration."""
        super()._validate_config()
        
        # Validate log level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")


@dataclass
class ExperimentConfig(BaseConfig):
    """Complete experiment configuration.
    
    Combines all configuration aspects for a complete experiment setup.
    This is the main configuration class that experiments should use.
    """
    
    # Experiment metadata
    name: str = "experiment"
    version: str = "1.0"
    description: str = ""
    experiment_type: str = "training"  # "training", "evaluation", "analysis"
    variant_id: Optional[str] = None
    
    # Base configuration inheritance
    base_config: Optional[str] = None  # Path to base config to inherit from
    
    # Configuration sections
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig) 
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Output configuration
    output_dir: str = "logs/experiment"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Budget constraints
    budget_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Quality gates
    quality_gates: Dict[str, Any] = field(default_factory=dict)
    
    def _validate_config(self) -> None:
        """Validate complete experiment configuration."""
        super()._validate_config()
        
        # Validate experiment type
        valid_types = {"training", "evaluation", "analysis", "benchmark"}
        if self.experiment_type not in valid_types:
            raise ValueError(f"experiment_type must be one of {valid_types}")
        
        # Ensure output directory is set
        if not self.output_dir:
            self.output_dir = f"logs/{self.name}"
        
        # Validate metric direction
        if self.metric_for_best_model and not isinstance(self.greater_is_better, bool):
            raise ValueError("greater_is_better must be a boolean")
    
    def get_output_path(self, filename: str = "") -> Path:
        """Get full path for output file."""
        output_path = Path(self.output_dir)
        if filename:
            output_path = output_path / filename
        return output_path
    
    def inherit_from_base(self, base_config: 'ExperimentConfig') -> None:
        """Inherit configuration from base config.
        
        Args:
            base_config: Base configuration to inherit from
        """
        # Inherit model config
        base_model_dict = base_config.model.to_dict()
        current_model_dict = self.model.to_dict()
        base_model_dict.update(current_model_dict)
        self.model = ModelConfig(**base_model_dict)
        
        # Inherit data config  
        base_data_dict = base_config.data.to_dict()
        current_data_dict = self.data.to_dict()
        base_data_dict.update(current_data_dict)
        self.data = DataConfig(**base_data_dict)
        
        # Inherit hardware config
        base_hardware_dict = base_config.hardware.to_dict()
        current_hardware_dict = self.hardware.to_dict()
        base_hardware_dict.update(current_hardware_dict)
        self.hardware = HardwareConfig(**base_hardware_dict)
        
        # Inherit logging config
        base_logging_dict = base_config.logging.to_dict()
        current_logging_dict = self.logging.to_dict()
        base_logging_dict.update(current_logging_dict)
        self.logging = LoggingConfig(**base_logging_dict)
        
        # Inherit other fields if not set
        for field_name in ['budget_constraints', 'quality_gates']:
            base_value = getattr(base_config, field_name, {})
            current_value = getattr(self, field_name, {})
            if base_value and not current_value:
                setattr(self, field_name, base_value.copy())
