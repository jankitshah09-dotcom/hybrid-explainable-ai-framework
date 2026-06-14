"""CNN model wrapper."""

import torch
import torch.nn as nn
from typing import Any
from .base import BaseModel


class CNNWrapper(BaseModel):
    """Wrapper for CNN models."""

    def __init__(self, model: nn.Module, num_classes: int = 10, **kwargs):
        """Initialize CNN wrapper.

        Args:
            model: PyTorch CNN model
            num_classes: Number of output classes
            **kwargs: Additional arguments for BaseModel
        """
        super().__init__(model, **kwargs)
        self.num_classes = num_classes

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions on input images.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Model predictions
        """
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(self.device)
            output = self.model(x)
            return output.cpu().numpy()

    def get_feature_importance(self) -> Any:
        """Get feature importance (for CNN: layer weights).

        Returns:
            Layer weight statistics
        """
        importance = {}
        for name, param in self.model.named_parameters():
            if len(param.shape) > 1:
                importance[name] = param.data.abs().mean().item()
        return importance
