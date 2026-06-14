"""Configuration management for the framework."""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration manager for the explainability framework."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration.

        Args:
            config_dict: Optional dictionary with configuration settings
        """
        self.config = config_dict or self._get_default_config()

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "model": {"device": "cuda", "precision": "fp32"},
            "lime": {"num_samples": 1000, "num_features": 10},
            "shap": {"num_samples": 100, "background_samples": 100},
            "gradcam": {"use_cuda": True},
            "visualization": {"dpi": 150, "figsize": (12, 8)},
        }

    def load_from_file(self, filepath: str) -> None:
        """Load configuration from JSON file.

        Args:
            filepath: Path to configuration JSON file
        """
        with open(filepath) as f:
            self.config = json.load(f)

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file.

        Args:
            filepath: Path to save configuration JSON file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=4)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def __repr__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.config, indent=2)
