"""Base class for explainers."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseExplainer(ABC):
    """Abstract base class for explainers."""

    def __init__(self, model: Any, name: str):
        """Initialize explainer.

        Args:
            model: Model to explain
            name: Name of the explainer
        """
        self.model = model
        self.name = name

    @abstractmethod
    def explain(self, x: Any, **kwargs) -> Dict[str, Any]:
        """Generate explanation for input.

        Args:
            x: Input to explain
            **kwargs: Additional arguments

        Returns:
            Explanation dictionary
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name}Explainer"
