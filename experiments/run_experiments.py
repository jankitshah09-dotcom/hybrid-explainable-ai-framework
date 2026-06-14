"""Run comprehensive experiments on multiple domains."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hybrid_xai.models import load_pretrained_model
from hybrid_xai.framework import ExplainabilityFramework
from hybrid_xai.evaluation import ExplanationMetrics, ExplanationBenchmark


def run_vision_experiments(config: dict) -> dict:
    """Run computer vision experiments.

    Args:
        config: Configuration dictionary

    Returns:
        Experiment results
    """
    print("\n" + "="*50)
    print("COMPUTER VISION EXPERIMENTS (CIFAR-10)")
    print("="*50)

    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])

    testset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Load model
    model = load_pretrained_model("resnet18", num_classes=10, pretrained=True)
    framework = ExplainabilityFramework(model)

    results = {"model": "resnet18", "dataset": "CIFAR-10", "experiments": {}}

    # Test on first few samples
    for idx, (images, labels) in enumerate(testloader):
        if idx >= config.get("num_samples", 3):
            break

        print(f"\nSample {idx+1}: Label={labels[0].item()}")

        # Get explanations
        explanations = framework.explain(
            images.numpy(),
            methods=["lime", "shap", "gradcam"],
            target_class=labels[0].item()
        )

        results["experiments"][f"sample_{idx}"] = {
            "label": labels[0].item(),
            "methods_used": list(explanations.keys())
        }

        print(f"  Methods available: {list(explanations.keys())}")

    return results


def benchmark_methods(config: dict) -> dict:
    """Benchmark explanation methods.

    Args:
        config: Configuration dictionary

    Returns:
        Benchmark results
    """
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARKS")
    print("="*50)

    model = load_pretrained_model("resnet18", num_classes=10)
    dummy_input = np.random.randn(224, 224, 3).astype(np.float32) / 255.0

    results = {"benchmark": {}}

    # Benchmark prediction time
    print("\nBenchmarking model prediction...")
    pred_times = []
    for _ in range(10):
        import time
        start = time.time()
        model.predict(np.random.randn(1, 3, 224, 224).astype(np.float32))
        pred_times.append(time.time() - start)

    results["benchmark"]["prediction"] = {
        "mean_ms": np.mean(pred_times) * 1000,
        "std_ms": np.std(pred_times) * 1000,
    }
    print(f"  Prediction: {results['benchmark']['prediction']['mean_ms']:.2f}ms")

    return results


def main():
    """Run all experiments."""
    parser = argparse.ArgumentParser(description="Run XAI Framework Experiments")
    parser.add_argument("--config", default="experiments/config.yaml", help="Config file")
    parser.add_argument("--output", default="experiments/results/", help="Output directory")
    args = parser.parse_args()

    # Load config (simple version)
    config = {"num_samples": 3}

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = {}

    try:
        all_results["vision"] = run_vision_experiments(config)
    except Exception as e:
        print(f"Error in vision experiments: {e}")
        all_results["vision"] = {"error": str(e)}

    try:
        all_results["benchmark"] = benchmark_methods(config)
    except Exception as e:
        print(f"Error in benchmarking: {e}")
        all_results["benchmark"] = {"error": str(e)}

    # Save results
    output_file = Path(args.output) / "results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4, default=str)

    print(f"\n\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
