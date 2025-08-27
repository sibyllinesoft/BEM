"""Configuration loading utilities with inheritance and validation.

Provides unified configuration loading across all BEM experiments,
with support for template inheritance and configuration validation.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base_config import ExperimentConfig, ModelConfig, DataConfig, HardwareConfig, LoggingConfig
from ..training.base_trainer import TrainingConfig


class ConfigLoader:
    """Unified configuration loader for BEM experiments."""
    
    def __init__(self, template_dir: Optional[Union[str, Path]] = None):
        """Initialize config loader.
        
        Args:
            template_dir: Directory containing configuration templates
        """
        if template_dir is None:
            # Default to templates directory in the same package
            template_dir = Path(__file__).parent / "templates"
        
        self.template_dir = Path(template_dir)
        self._template_cache = {}
    
    def load_config(self, config_path: Union[str, Path]) -> ExperimentConfig:
        """Load experiment configuration with inheritance.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Complete experiment configuration
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load the configuration file
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle base configuration inheritance
        if 'base_config' in config_dict and config_dict['base_config']:
            base_config_path = config_dict['base_config']
            
            # Resolve relative paths
            if not os.path.isabs(base_config_path):
                base_config_path = config_path.parent / base_config_path
            
            base_config = self.load_config(base_config_path)
            config_dict = self._merge_configs(base_config.to_dict(), config_dict)
        
        # Create configuration object
        return self._dict_to_config(config_dict)
    
    def load_template(self, template_name: str) -> Dict[str, Any]:
        """Load configuration template.
        
        Args:
            template_name: Name of template file (without .yaml extension)
            
        Returns:
            Template configuration as dictionary
        """
        if template_name in self._template_cache:
            return self._template_cache[template_name].copy()
        
        template_path = self.template_dir / f"{template_name}.yaml"
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template_dict = yaml.safe_load(f)
        
        self._template_cache[template_name] = template_dict
        return template_dict.copy()
    
    def create_from_template(
        self, 
        template_name: str, 
        overrides: Optional[Dict[str, Any]] = None
    ) -> ExperimentConfig:
        """Create configuration from template with overrides.
        
        Args:
            template_name: Name of template to use
            overrides: Configuration overrides
            
        Returns:
            Experiment configuration
        """
        template_dict = self.load_template(template_name)
        
        if overrides:
            template_dict = self._merge_configs(template_dict, overrides)
        
        return self._dict_to_config(template_dict)
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries.
        
        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert configuration dictionary to ExperimentConfig object.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ExperimentConfig object
        """
        # Extract nested configurations
        model_config = ModelConfig(**config_dict.pop('model', {}))
        data_config = DataConfig(**config_dict.pop('data', {}))
        hardware_config = HardwareConfig(**config_dict.pop('hardware', {}))
        logging_config = LoggingConfig(**config_dict.pop('logging', {}))
        
        # Remove training config if present (handled separately)
        config_dict.pop('training', None)
        
        # Create main config
        config = ExperimentConfig(
            model=model_config,
            data=data_config, 
            hardware=hardware_config,
            logging=logging_config,
            **config_dict
        )
        
        return config
    
    def save_config(self, config: ExperimentConfig, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


# Global config loader instance
_default_loader = ConfigLoader()


def load_experiment_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """Load experiment configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Complete experiment configuration
    """
    return _default_loader.load_config(config_path)


def load_training_config(config_path: Union[str, Path]) -> TrainingConfig:
    """Load training configuration from experiment config.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Training configuration
    """
    config_path = Path(config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract training configuration
    training_dict = config_dict.get('training', {})
    
    # Add common fields from experiment config
    training_dict.setdefault('seed', config_dict.get('seed', 42))
    training_dict.setdefault('deterministic', config_dict.get('deterministic', True))
    
    # Add device from hardware config
    hardware_config = config_dict.get('hardware', {})
    training_dict.setdefault('device', hardware_config.get('device'))
    
    return TrainingConfig(**training_dict)


def create_config_from_template(
    template_name: str, 
    overrides: Optional[Dict[str, Any]] = None
) -> ExperimentConfig:
    """Create configuration from template with overrides.
    
    Args:
        template_name: Name of template to use
        overrides: Configuration overrides
        
    Returns:
        Experiment configuration
    """
    return _default_loader.create_from_template(template_name, overrides)


def save_experiment_config(config: ExperimentConfig, output_path: Union[str, Path]) -> None:
    """Save experiment configuration to file.
    
    Args:
        config: Configuration to save
        output_path: Output file path
    """
    _default_loader.save_config(config, output_path)