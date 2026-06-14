"""Benchmarking utilities."""

import time
from typing import Any, Dict, Callable
import numpy as np


class ExplanationBenchmark:
    """Benchmark explanation methods."""

    @staticmethod
    def benchmark_explain_time(
        explainer: Callable,
        input_data: np.ndarray,
        num_runs: int = 5,
    ) -> Dict[str, float]:
        """Benchmark explanation time.

        Args:
            explainer: Explainer function
            input_data: Input to explain
            num_runs: Number of benchmark runs

        Returns:
            Timing statistics
        """
        times = []
        for _ in range(num_runs):
            start = time.time()
            explainer(input_data)
            times.append(time.time() - start)
        
        return {
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
        }

    @staticmethod
    def benchmark_memory(
        explainer: Callable,
        input_data: np.ndarray,
    ) -> Dict[str, float]:
        """Benchmark memory usage.

        Args:
            explainer: Explainer function
            input_data: Input to explain

        Returns:
            Memory statistics
        """
        import tracemalloc
        
        tracemalloc.start()
        explainer(input_data)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {"current_mb": current / 1024**2, "peak_mb": peak / 1024**2}
