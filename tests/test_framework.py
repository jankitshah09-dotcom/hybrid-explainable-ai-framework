"""Test framework main functionality."""

import pytest
import numpy as np
import torch
from hybrid_xai.framework import ExplainabilityFramework
from hybrid_xai.models import load_pretrained_model


class TestFramework:
    """Test main framework."""

    @pytest.fixture
    def model(self):
        """Load test model."""
        return load_pretrained_model("resnet18", num_classes=10)

    @pytest.fixture
    def framework(self, model):
        """Create framework instance."""
        return ExplainabilityFramework(model)

    @pytest.fixture
    def dummy_image(self):
        """Create dummy image."""
        return np.random.randn(1, 3, 224, 224).astype(np.float32)

    def test_framework_initialization(self, framework):
        """Test framework initialization."""
        assert framework is not None
        assert len(framework.get_supported_methods()) > 0

    def test_supported_methods(self, framework):
        """Test supported methods."""
        methods = framework.get_supported_methods()
        assert isinstance(methods, list)
        assert len(methods) > 0

    def test_model_prediction(self, model, dummy_image):
        """Test model prediction."""
        output = model.predict(dummy_image)
        assert output.shape[0] == 1
        assert output.shape[1] == 10

    def test_explain_interface(self, framework, dummy_image):
        """Test explain interface."""
        methods = framework.get_supported_methods()
        if methods:
            explanations = framework.explain(
                dummy_image,
                methods=[methods[0]]
            )
            assert isinstance(explanations, dict)
