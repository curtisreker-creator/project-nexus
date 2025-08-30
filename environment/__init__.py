# File: environment/__init__.py
import yaml
import os
from .grid_world import GridWorld

def load_config(config_path="configs/default.yaml"):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_environment(config_path="configs/default.yaml", **kwargs):
    """Create GridWorld environment from configuration"""
    config = load_config(config_path)
    env_config = config['environment']
    
    # Override with any provided kwargs
    env_config.update(kwargs)
    
    return GridWorld(
        size=tuple(env_config['grid_size']),
        n_agents=env_config.get('max_agents', 1),
        max_resources=env_config['max_resources'],
        max_steps=env_config['max_steps']
    )