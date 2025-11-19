"""Configuration loader utility"""

import yaml
from pathlib import Path


def load_config(config_path='config/config.yaml'):
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to config file

    Returns:
        dict: Configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_config_value(config, *keys, default=None):
    """
    Safely get nested config value.

    Args:
        config (dict): Configuration dictionary
        *keys: Nested keys to access
        default: Default value if key not found

    Returns:
        Value at nested key or default

    Example:
        >>> config = {'model': {'timesteps': 10}}
        >>> get_config_value(config, 'model', 'timesteps')
        10
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
