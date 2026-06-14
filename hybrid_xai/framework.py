"""Main explainability framework."""

from typing import Any, Dict, List, Optional
import numpy as np
from .explainers import LimeExplainer, ShapExplainer, GradCAMExplainer
from .visualization import ExplanationVisualizer, ComparativeVisualizer


class ExplainabilityFramework:
    """Main framework for model explainability."""

    def __init__(self, model: Any, config: Optional[Dict] = None):
        """Initialize framework.

        Args:
            model: Model to explain (wrapped)
            config: Optional configuration dictionary
        """
        self.model = model
        self.config = config or {}
        self.explainers = {}
        self._initialize_explainers()

    def _initialize_explainers(self) -> None:
        """Initialize all explainers."""
        try:
            self.explainers["lime"] = LimeExplainer(
                self.model,
                num_samples=self.config.get("lime_samples", 1000),
                num_features=self.config.get("lime_features", 10),
            )
        except ImportError:
            pass

        try:
            self.explainers["shap"] = ShapExplainer(
                self.model,
                background_samples=self.config.get("shap_background", 100),
            )
        except ImportError:
            pass

        try:
            self.explainers["gradcam"] = GradCAMExplainer(self.model)
        except ImportError:
            pass

    def explain(
        self,
        input_data: Any,
        methods: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate explanations using specified methods.

        Args:
            input_data: Input to explain
            methods: List of methods to use (lime, shap, gradcam)
            **kwargs: Additional arguments for explainers

        Returns:
            Dictionary of explanations
        """
        if methods is None:
            methods = list(self.explainers.keys())

        explanations = {}
        for method in methods:
            if method in self.explainers:
                explainer = self.explainers[method]
                explanations[method] = explainer.explain(input_data, **kwargs)
            else:
                print(f"Warning: {method} not available")

        return explanations

    def visualize(
        self,
        explanations: Dict[str, Any],
        image: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        """Visualize explanations.

        Args:
            explanations: Dictionary of explanations
            image: Original image (optional)
            save_path: Path to save visualization

        Returns:
            Matplotlib figure
        """
        if image is not None:
            fig = ComparativeVisualizer.compare_explanations(image, explanations)
        else:
            fig = None

        if save_path:
            ExplanationVisualizer.save_figure(fig, save_path)

        return fig

    def get_supported_methods(self) -> List[str]:
        """Get list of supported explanation methods.

        Returns:
            List of method names
        """
        return list(self.explainers.keys())
