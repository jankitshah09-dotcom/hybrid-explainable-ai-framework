"""Grad-CAM explainer implementation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional
from .base_explainer import BaseExplainer

# Try Captum import with better error handling
CAPTUM_AVAILABLE = False
try:
    from captum.attr import GradCAM
    CAPTUM_AVAILABLE = True
except ImportError:
    try:
        import captum
        from captum.attr import GradCAM
        CAPTUM_AVAILABLE = True
    except (ImportError, AttributeError):
        CAPTUM_AVAILABLE = False


class GradCAMExplainer(BaseExplainer):
    """Grad-CAM explanation method."""

    def __init__(self, model: Any, target_layer: Optional[str] = None):
        """Initialize Grad-CAM explainer.

        Args:
            model: Model to explain (should have .model attribute with PyTorch module)
            target_layer: Name of layer to visualize (auto-detected if None)
        """
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum not installed. Install with: pip install captum")
        super().__init__(model, "Grad-CAM")
        self.target_layer = target_layer
        self.grad_cam = None

    def explain(
        self, 
        x: torch.Tensor, 
        target_class: Optional[int] = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """Generate Grad-CAM explanation.

        Args:
            x: Input tensor or array
            target_class: Target class for attribution
            **kwargs: Additional arguments

        Returns:
            Grad-CAM heatmap and attribution
        """
        try:
            # Convert to tensor if needed
n            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            
            # Move to model device
            if hasattr(self.model, 'device'):\n                x = x.to(self.model.device)\n            elif torch.cuda.is_available():\n                x = x.cuda()\n            \n            # Handle batch dimension\n            if x.dim() == 3:\n                x = x.unsqueeze(0)\n            \n            # Get underlying PyTorch model\n            pt_model = self.model.model if hasattr(self.model, 'model') else self.model\n            \n            # Auto-detect target layer if not specified\n            if self.target_layer is None:\n                target_layer = self._get_last_conv_layer(pt_model)\n            else:\n                target_layer = dict(pt_model.named_modules())[self.target_layer]\n\n            # Create GradCAM\n            self.grad_cam = GradCAM(pt_model, target_layer)\n            \n            # Generate attributions\n            with torch.no_grad():\n                attributions = self.grad_cam.attribute(x, target=target_class)\n            \n            return {\n                \"method\": \"Grad-CAM\",\n                \"attributions\": attributions.cpu().detach().numpy(),\n                \"target_class\": target_class,\n            }\n        except Exception as e:\n            return {\n                \"method\": \"Grad-CAM\",\n                \"error\": str(e),\n                \"attributions\": None,\n                \"target_class\": target_class,\n            }\n\n    def _get_last_conv_layer(self, model: nn.Module) -> nn.Module:\n        \"\"\"Get last convolutional layer in model.\n\n        Args:\n            model: PyTorch model\n\n        Returns:\n            Last conv layer\n        \"\"\"\n        last_conv = None\n        for module in model.modules():\n            if isinstance(module, nn.Conv2d):\n                last_conv = module\n        \n        if last_conv is None:\n            raise RuntimeError(\"No Conv2d layer found in model\")\n        \n        return last_conv\n