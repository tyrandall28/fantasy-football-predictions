"""
Configuration management utilities.

Provides functions to load and validate configuration settings
from YAML files for the fantasy football prediction pipeline.
"""

import yaml
import os
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default config.yaml
        
    Returns:
        Dictionary containing configuration settings
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If YAML file is malformed
    """
    if config_path is None:
        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")


def get_model_config(model_name: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model (e.g., 'random_forest', 'xgboost')
        config: Configuration dictionary. If None, loads default config
        
    Returns:
        Dictionary containing model-specific configuration
        
    Raises:
        KeyError: If model configuration doesn't exist
    """
    if config is None:
        config = load_config()
    
    try:
        return config['model'][model_name]
    except KeyError:
        available_models = list(config['model'].keys())
        raise KeyError(f"Model '{model_name}' not found in configuration. "
                      f"Available models: {available_models}")


def get_data_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get data configuration settings.
    
    Args:
        config: Configuration dictionary. If None, loads default config
        
    Returns:
        Dictionary containing data configuration
    """
    if config is None:
        config = load_config()
    
    return config['data']


def get_feature_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get feature engineering configuration settings.
    
    Args:
        config: Configuration dictionary. If None, loads default config
        
    Returns:
        Dictionary containing feature configuration
    """
    if config is None:
        config = load_config()
    
    return config['features']


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary has required keys.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If required configuration keys are missing
    """
    required_keys = ['model', 'data', 'features', 'evaluation']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required configuration key '{key}' is missing")
    
    # Validate model configurations exist
    if not config['model']:
        raise ValueError("No model configurations found")
    
    # Validate data paths
    data_config = config['data']
    required_data_keys = ['raw_data_dir', 'processed_data_dir', 'test_size']
    
    for key in required_data_keys:
        if key not in data_config:
            raise ValueError(f"Required data configuration key '{key}' is missing")
    
    return True 