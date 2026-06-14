"""Visualization for individual explanations."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class ExplanationVisualizer:
    """Visualize explanations from different methods."""

    @staticmethod
    def plot_image(image: np.ndarray, title: str = "Image", figsize: tuple = (6, 6)):
        """Plot image.

        Args:
            image: Image array
            title: Plot title
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        if len(image.shape) == 3 and image.shape[2] == 3:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
        return fig, ax

    @staticmethod
    def plot_gradcam(image: np.ndarray, heatmap: np.ndarray, title: str = "Grad-CAM"):
        """Plot Grad-CAM heatmap over image.

        Args:
            image: Original image
            heatmap: Grad-CAM heatmap
            title: Plot title
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        im = axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title(title)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1])
        
        return fig, axes

    @staticmethod
    def plot_lime_segments(image: np.ndarray, segments: np.ndarray, title: str = "LIME Segments"):
        """Plot LIME segments.

        Args:
            image: Original image
            segments: Segment mask
            title: Plot title
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        axes[1].imshow(segments, cmap="tab20")
        axes[1].set_title(title)
        axes[1].axis("off")
        
        return fig, axes

    @staticmethod
    def save_figure(fig, filepath: str, dpi: int = 150):
        """Save figure to file.

        Args:
            fig: Matplotlib figure
            filepath: Path to save
            dpi: DPI for saving
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
