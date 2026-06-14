"""SHAP explainer implementation."""

import numpy as np
from typing import Any, Dict, Optional
from .base_explainer import BaseExplainer

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ShapExplainer(BaseExplainer):
    """SHAP explanation method."""

    def __init__(self, model: Any, background_samples: int = 100):
        """Initialize SHAP explainer.

        Args:
            model: Model to explain
            background_samples: Number of background samples for SHAP background
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        super().__init__(model, "SHAP")
        self.background_samples = background_samples
        self.explainer = None

    def explain(
        self, 
        x: np.ndarray, 
        background_data: Optional[np.ndarray] = None,
        num_samples: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate SHAP explanation.

        Args:
            x: Input to explain
            background_data: Background dataset for SHAP (if None, generates random)
            num_samples: Number of evaluation samples for SHAP (alternative parameter)
            **kwargs: Additional arguments

        Returns:
            Explanation with SHAP values
        """
        try:
            # Prepare background data
            if background_data is None:
                if isinstance(x, np.ndarray):
                    shape = x.shape
                    background_data = np.random.randn(self.background_samples, *shape[1:]).astype(np.float32) * 0.1
                else:
                    background_data = np.random.randn(self.background_samples, 224, 224, 3).astype(np.float32) * 0.1
            else:
                background_data = background_data[: self.background_samples]

            # Create explainer if not exists
            if self.explainer is None:
                self.explainer = shap.KernelExplainer(
                    self.model.predict, 
                    background_data
                )

            # Handle input format
            if isinstance(x, np.ndarray) and x.ndim == 3:
                x = np.expand_dims(x, 0)
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(x, nsamples=num_samples)

            return {
                "method": "SHAP",
                "shap_values": shap_values,
                "base_value": self.explainer.expected_value,
            }
        except Exception as e:
            return {
                "method": "SHAP",
                "error": str(e),
                "shap_values": None,
                "base_value": None,
            }
