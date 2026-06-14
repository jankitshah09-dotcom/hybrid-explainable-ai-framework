"""Test models."""

import pytest
import numpy as np
from hybrid_xai.models import load_pretrained_model


class TestModels:
    """Test model wrappers."""

    def test_load_resnet18(self):
        """Test loading ResNet18."""
        model = load_pretrained_model("resnet18", num_classes=10)
        assert model is not None

    def test_load_resnet50(self):
        """Test loading ResNet50."""
        model = load_pretrained_model("resnet50", num_classes=10)
        assert model is not None

    def test_load_vgg16(self):
        """Test loading VGG16."""
        model = load_pretrained_model("vgg16", num_classes=10)
        assert model is not None

    def test_model_prediction(self):
        """Test model prediction."""
        model = load_pretrained_model("resnet18", num_classes=10)
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output = model.predict(dummy_input)
        
        assert output.shape[0] == 1
        assert output.shape[1] == 10

    def test_feature_importance(self):
        """Test feature importance extraction."""
        model = load_pretrained_model("resnet18", num_classes=10)
        importance = model.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0
