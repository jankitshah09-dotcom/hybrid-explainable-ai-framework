"""Grad-CAM explainer implementation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List
from .base_explainer import BaseExplainer

try:
    from captum.attr import GradCAM
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False


class GradCAMExplainer(BaseExplainer):
    """Grad-CAM explanation method."""

    def __init__(self, model: Any, target_layer: str = None):
        """Initialize Grad-CAM explainer.

        Args:
            model: Model to explain
            target_layer: Name of layer to visualize (auto-detected if None)
        """
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum not installed. Install with: pip install captum")
        super().__init__(model, "Grad-CAM")
        self.target_layer = target_layer
        self.grad_cam = None

    def explain(self, x: torch.Tensor, target_class: int = None, **kwargs) -> Dict[str, Any]:
        """Generate Grad-CAM explanation.

        Args:
            x: Input tensor
            target_class: Target class for attribution
            **kwargs: Additional arguments

        Returns:
            Grad-CAM heatmap
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Auto-detect target layer if not specified
        if self.target_layer is None:
            target_layer = self._get_last_conv_layer()
        else:
            target_layer = dict(self.model.model.named_modules())[self.target_layer]

        self.grad_cam = GradCAM(self.model.model, target_layer)
        attributions = self.grad_cam.attribute(x, target=target_class)
        
        return {
            "method": "Grad-CAM",
            "attributions": attributions.cpu().numpy(),
            "target_class": target_class,
        }

    def _get_last_conv_layer(self) -> nn.Module:
        """Get last convolutional layer in model.

        Returns:
            Last conv layer
        """
        last_conv = None
        for module in self.model.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv
