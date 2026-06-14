"""Comparative visualization of multiple explanations."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any


class ComparativeVisualizer:
    """Compare explanations from different methods."""

    @staticmethod
    def compare_explanations(
        image: np.ndarray,
        explanations: Dict[str, Any],
        figsize: tuple = (15, 5),
    ) -> plt.Figure:
        """Create side-by-side comparison of explanations.

        Args:
            image: Original image
            explanations: Dictionary of explanations from different methods
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        num_methods = len(explanations) + 1
        fig, axes = plt.subplots(1, num_methods, figsize=figsize)
        
        if num_methods == 1:
            axes = [axes]
        
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        for idx, (method, exp_data) in enumerate(explanations.items(), 1):
            if "attributions" in exp_data:
                vis_data = np.mean(np.abs(exp_data["attributions"][0]), axis=0)
            elif "heatmap" in exp_data:
                vis_data = exp_data["heatmap"]
            else:
                vis_data = np.zeros_like(image[:, :, 0])
            
            im = axes[idx].imshow(vis_data, cmap="jet")
            axes[idx].set_title(method)
            axes[idx].axis("off")
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig
