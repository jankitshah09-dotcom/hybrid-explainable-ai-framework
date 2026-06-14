# Getting Started with Hybrid XAI Framework

## Quick Setup

### 1. Clone and Install
```bash
git clone https://github.com/jankitshah09-dotcom/hybrid-explainable-ai-framework.git
cd hybrid-explainable-ai-framework

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### 2. Test Installation
```bash
python -c "from hybrid_xai import ExplainabilityFramework; print('✓ Installation successful!')"
```

## Basic Usage

### Example 1: Image Classification Explanation
```python
import numpy as np
from hybrid_xai.models import load_pretrained_model
from hybrid_xai.framework import ExplainabilityFramework
from hybrid_xai.visualization import ComparativeVisualizer

# Load a pretrained ResNet model
model = load_pretrained_model('resnet18', num_classes=10, pretrained=True)

# Create framework
framework = ExplainabilityFramework(model)

# Prepare an image (random for demo)
image = np.random.randn(224, 224, 3).astype(np.float32) / 255.0

# Get explanations from all methods
explanations = framework.explain(
    image,
    methods=['lime', 'shap', 'gradcam'],
    target_class=5
)

# Visualize results
fig = ComparativeVisualizer.compare_explanations(image, explanations)
fig.savefig('explanations.png')
```

### Example 2: Using Individual Explainers
```python
from hybrid_xai.explainers import LimeExplainer, ShapExplainer, GradCAMExplainer

model = load_pretrained_model('resnet18')

# LIME Explanation
lime_explainer = LimeExplainer(model, num_samples=1000)
lime_exp = lime_explainer.explain(image)

# SHAP Explanation
shap_explainer = ShapExplainer(model)
shap_exp = shap_explainer.explain(image)

# Grad-CAM Explanation
gradcam_explainer = GradCAMExplainer(model)
gradcam_exp = gradcam_explainer.explain(image, target_class=5)
```

### Example 3: Evaluation and Benchmarking
```python
from hybrid_xai.evaluation import ExplanationMetrics, ExplanationBenchmark

# Calculate metrics
metrics = ExplanationMetrics()
faithfulness = metrics.faithfulness(model.predict, image, explanations['lime'])
sparsity = metrics.sparsity(explanations['lime'])

# Benchmark performance
benchmark = ExplanationBenchmark()
timing = benchmark.benchmark_explain_time(
    lambda x: framework.explain(x, methods=['lime']),
    image,
    num_runs=5
)
print(f"Average explanation time: {timing['mean']*1000:.2f}ms")
```

## Running the Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run app/dashboard.py
```

Then open `http://localhost:8501` in your browser.

## Running Experiments

Execute the full experimental pipeline:

```bash
python experiments/run_experiments.py --config experiments/config.yaml --output experiments/results/
```

## Running Tests

```bash
pytest tests/ -v --cov=hybrid_xai
```

## Building Documentation

```bash
cd docs/
make html
```

Documentation will be in `docs/_build/html/index.html`.

## Supported Models

### Computer Vision (Image Classification)
- ResNet18, ResNet50
- VGG16
- EfficientNetB0

### NLP
- BERT (coming soon)
- GPT models (coming soon)

### Time Series
- LSTM (coming soon)
- Transformer models (coming soon)

## Configuration

Customize behavior via configuration:

```python
from hybrid_xai.utils import Config

config = Config({
    'lime': {'num_samples': 2000, 'num_features': 15},
    'shap': {'num_samples': 200, 'background_samples': 150},
    'gradcam': {'use_cuda': True},
})

framework = ExplainabilityFramework(model, config=config.config)
```

## Common Issues & Troubleshooting

### CUDA/GPU Issues
If you don't have a GPU, install CPU-only PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Memory Issues
Reduce num_samples in explainer configuration:
```python
config = {'lime_samples': 500, 'shap_background': 50}
```

### LIME/SHAP Import Errors
Install optional dependencies:
```bash
pip install lime shap captum
```

## Next Steps

1. **Read the Notebooks**: Check `notebooks/` for detailed examples
2. **API Documentation**: See `docs/` for full API reference
3. **Run Experiments**: Execute `experiments/run_experiments.py` for full pipeline
4. **Explore Dashboard**: Use Streamlit dashboard for interactive exploration
5. **Research Paper**: Read `dissertation/dissertation.tex` for methodology

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Citation

If you use this framework, please cite:

```bibtex
@thesis{shah2026hybrid,
  title={Hybrid Explainable AI Framework for Deep Learning Models},
  author={Shah, Jankit},
  year={2026},
  school={University of East London}
}
```

## License

MIT License - see LICENSE file

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues/discussions
- Review documentation in `docs/`
