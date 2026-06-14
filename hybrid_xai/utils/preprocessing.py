"""Data preprocessing utilities."""

import numpy as np
from typing import Union, Tuple


class DataPreprocessor:
    """Preprocessing utilities for different data types."""

    @staticmethod
    def normalize_image(image: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
        """Normalize image data.

        Args:
            image: Input image array
            mean: Mean for normalization
            std: Standard deviation for normalization

        Returns:
            Normalized image
        """
        return (image - mean) / std

    @staticmethod
    def denormalize_image(image: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
        """Denormalize image data.

        Args:
            image: Normalized image array
            mean: Mean used for normalization
            std: Std used for normalization

        Returns:
            Denormalized image
        """
        return image * std + mean

    @staticmethod
    def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size.

        Args:
            image: Input image array
            size: Target (height, width)

        Returns:
            Resized image
        """
        from scipy import ndimage
        return ndimage.zoom(image, (size[0] / image.shape[0], size[1] / image.shape[1], 1), order=1)

    @staticmethod
    def standardize_array(array: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Standardize array (z-score normalization).

        Args:
            array: Input array

        Returns:
            Standardized array, mean, std
        """
        mean = np.mean(array)
        std = np.std(array)
        return (array - mean) / (std + 1e-8), mean, std
