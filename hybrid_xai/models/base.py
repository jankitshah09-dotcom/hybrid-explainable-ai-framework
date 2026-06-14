"""Base model wrapper class."""

from abc import ABC, abstractmethod
from typing import Any, Union
import torch


class BaseModel(ABC):
    """Abstract base class for model wrappers."""

    def __init__(self, model: Any, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize model wrapper.

        Args:
            model: PyTorch or TensorFlow model
            device: Device to run model on
        """
        self.model = model
        self.device = device
        self.model.to(device) if hasattr(model, "to") else None
        self.model.eval() if hasattr(model, "eval") else None

    @abstractmethod
    def predict(self, x: Any) -> Any:
        """Make prediction on input.

        Args:
            x: Input data

        Returns:
            Model predictions
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Any:
        """Get feature importance from model.

        Returns:
            Feature importance scores
        """
        pass

    def __call__(self, x: Any) -> Any:
        """Call model as function."""
        return self.predict(x)
