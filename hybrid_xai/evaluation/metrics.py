Evaluation and Metrics
======================

This module provides metrics for evaluating explanation quality.
"""

import numpy as np
from typing import Any, Callable


class ExplanationMetrics:
    """Metrics for evaluating explanations."""

    @staticmethod
    def faithfulness(
        model: Callable,
        input_data: np.ndarray,
        explanation: np.ndarray,
        num_samples: int = 10,
    ) -> float:
        """Calculate faithfulness metric (drop-in accuracy).

        Args:
            model: Model prediction function
            input_data: Input to evaluate
            explanation: Explanation (importance scores)
            num_samples: Number of samples for evaluation

        Returns:
            Faithfulness score
        """
        original_pred = model(input_data[np.newaxis, ...])
        scores = []
        
        for _ in range(num_samples):
            mask = np.random.random(explanation.shape) < 0.5
            masked_input = input_data * mask
            masked_pred = model(masked_input[np.newaxis, ...])
            
            similarity = np.mean(
                np.sign(explanation[mask]) == np.sign(masked_pred[0] - original_pred[0])
            )
            scores.append(similarity)
        
        return np.mean(scores)

    @staticmethod
    def sparsity(explanation: np.ndarray) -> float:
        """Calculate sparsity (percentage of zeros).

        Args:
            explanation: Explanation array

        Returns:
            Sparsity score
        """
        return np.sum(explanation == 0) / explanation.size

    @staticmethod
    def sensitivity(
        model: Callable,
        input_data: np.ndarray,
        perturbation_size: float = 0.1,
    ) -> float:
        """Calculate sensitivity to input perturbations.

        Args:
            model: Model prediction function
            input_data: Input to evaluate
            perturbation_size: Size of perturbation

        Returns:
            Sensitivity score
        """
        original_output = model(input_data[np.newaxis, ...])
        
        perturbation = np.random.randn(*input_data.shape) * perturbation_size
        perturbed_output = model((input_data + perturbation)[np.newaxis, ...])
        
        return np.linalg.norm(original_output - perturbed_output)
