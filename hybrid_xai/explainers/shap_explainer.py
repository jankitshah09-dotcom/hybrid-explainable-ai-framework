"""SHAP explainer implementation."""

import numpy as np
from typing import Any, Dict
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
            background_samples: Number of background samples
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        super().__init__(model, "SHAP")
        self.background_samples = background_samples
        self.explainer = None

    def explain(self, x: np.ndarray, background_data: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """Generate SHAP explanation.

        Args:
            x: Input to explain
            background_data: Background dataset for SHAP
            **kwargs: Additional arguments

        Returns:
            Explanation with SHAP values
        """
        if background_data is None:
            background_data = np.random.randn(*x.shape).astype(np.float32) * 0.1
        else:
            background_data = background_data[: self.background_samples]

        if self.explainer is None:
            self.explainer = shap.KernelExplainer(
                self.model.predict, background_data
            )

        shap_values = self.explainer.shap_values(x)

        return {
            "method": "SHAP",
            "shap_values": shap_values,
            "base_value": self.explainer.expected_value,
        }
