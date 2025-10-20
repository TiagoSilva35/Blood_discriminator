"""
Utility functions for the blood discriminator project
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        if save_path.endswith('.yaml') or save_path.endswith('.yml'):
            yaml.dump(config, f, default_flow_style=False)
        elif save_path.endswith('.json'):
            json.dump(config, f, indent=2)
        else:
            raise ValueError("Save path must be .yaml, .yml, or .json")


def ensure_dir(directory: str):
    """
    Ensure directory exists.
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_results(results: Dict[str, Any], save_path: str):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        save_path: Path to save results
    """
    ensure_dir(os.path.dirname(save_path))
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
