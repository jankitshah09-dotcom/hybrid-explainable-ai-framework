"""LIME explainer implementation."""

import numpy as np
import torch
from typing import Any, Dict
from .base_explainer import BaseExplainer

try:
    import lime
    import lime.lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class LimeExplainer(BaseExplainer):
    """LIME explanation method."""

    def __init__(self, model: Any, num_samples: int = 1000, num_features: int = 10):
        """Initialize LIME explainer.

        Args:
            model: Model to explain
            num_samples: Number of samples for LIME
            num_features: Number of features to explain
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME not installed. Install with: pip install lime")
        super().__init__(model, "LIME")
        self.num_samples = num_samples
        self.num_features = num_features
        self.explainer = None

    def explain(self, x: np.ndarray, top_labels: int = 5, **kwargs) -> Dict[str, Any]:
        """Generate LIME explanation.

        Args:
            x: Input image (can be CHW or HWC format)
            top_labels: Number of top labels to explain
            **kwargs: Additional arguments

        Returns:
            Explanation with mask and scores
        """
        # Convert to HWC format if needed (from CHW)
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        # Handle batch dimension
        if len(x.shape) == 4:
            x = x[0]
        
        # Convert CHW to HWC if needed
        if x.shape[0] in [1, 3, 4]:  # Likely channel-first
            x = np.transpose(x, (1, 2, 0))
        
        # Handle grayscale to RGB
        if len(x.shape) == 2:
            x = np.stack([x, x, x], axis=-1)
        
        # Normalize to [0, 1] range if needed
        if x.max() > 1.1:
            x = x / 255.0
        
        # Ensure float type
        x = x.astype(np.float32)
        
        if self.explainer is None:
            self.explainer = lime.lime_image.LimeImageExplainer()

        # Wrapper to handle model predictions
        def predict_fn(images):
            if isinstance(images, np.ndarray):
                # LIME passes HWC format, model expects CHW
                if images.ndim == 3:
                    images = np.transpose(images[np.newaxis, ...], (0, 3, 1, 2))
                elif images.ndim == 4 and images.shape[-1] in [1, 3, 4]:
                    images = np.transpose(images, (0, 3, 1, 2))
            return self.model.predict(images)

        try:
            explanation = self.explainer.explain_instance(
                x,
                predict_fn,
                top_labels=top_labels,
                num_samples=self.num_samples,
                num_features=self.num_features,
            )
            
            return {
                "method": "LIME",
                "explanation": explanation,
                "segments": explanation.segments,
            }
        except Exception as e:
            return {
                "method": "LIME",
                "error": str(e),
                "segments": None,
            }
