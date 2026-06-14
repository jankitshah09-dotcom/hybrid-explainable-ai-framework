"""Model loading utilities."""

import torch
import torchvision.models as models
from typing import Literal
from .cnn import CNNWrapper


def load_pretrained_model(
    model_name: str = "resnet18",
    task: Literal["image_classification", "object_detection"] = "image_classification",
    num_classes: int = 10,
    pretrained: bool = True,
    device: str = None,
) -> CNNWrapper:
    """Load pretrained model.

    Args:
        model_name: Name of model architecture (resnet18, vgg16, etc.)
        task: Type of task
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        device: Device to load model on

    Returns:
        Wrapped model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if task == "image_classification":
        if model_name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            model.fc = torch.nn.Linear(512, num_classes)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            model.fc = torch.nn.Linear(2048, num_classes)
        elif model_name == "vgg16":
            model = models.vgg16(pretrained=pretrained)
            model.classifier[6] = torch.nn.Linear(4096, num_classes)
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = torch.nn.Linear(1280, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return CNNWrapper(model, num_classes=num_classes, device=device)
    else:
        raise NotImplementedError(f"Task {task} not yet supported")
