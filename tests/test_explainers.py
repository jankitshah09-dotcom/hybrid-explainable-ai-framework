"""Test explainers."""

import pytest
import numpy as np
from hybrid_xai.models import load_pretrained_model
from hybrid_xai.explainers import LimeExplainer, ShapExplainer


class TestExplainers:
    """Test explainer implementations."""

    @pytest.fixture
    def model(self):
        """Load test model."""
        return load_pretrained_model("resnet18", num_classes=10)

    @pytest.fixture
    def dummy_image(self):
        """Create dummy image."""
        return np.random.randn(224, 224, 3).astype(np.float32) / 255.0

    def test_lime_explainer_initialization(self, model):
        """Test LIME explainer initialization."""
        try:
            explainer = LimeExplainer(model)
            assert explainer is not None
        except ImportError:
            pytest.skip("LIME not installed")

    def test_shap_explainer_initialization(self, model):
        """Test SHAP explainer initialization."""
        try:
            explainer = ShapExplainer(model)
            assert explainer is not None
        except ImportError:
            pytest.skip("SHAP not installed")

    def test_lime_explain(self, model, dummy_image):
        """Test LIME explanation."""
        try:
            explainer = LimeExplainer(model, num_samples=100)
            explanation = explainer.explain(dummy_image, top_labels=3)
            assert "method" in explanation
            assert explanation["method"] == "LIME"
        except ImportError:
            pytest.skip("LIME not installed")
