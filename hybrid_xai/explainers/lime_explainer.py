"""LIME explainer implementation."""

import numpy as np
from typing import Any, Dict, Callable
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
            x: Input image (height, width, channels)
            top_labels: Number of top labels to explain
            **kwargs: Additional arguments

        Returns:
            Explanation with mask and scores
        """
        if len(x.shape) == 4:
            x = x[0]

        # Normalize to [0, 1] range if needed
        if x.max() > 1.0:
            x = x / 255.0

        if self.explainer is None:
            self.explainer = lime.lime_image.LimeImageExplainer()

        def predict_fn(images):
            return self.model.predict(images)

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
