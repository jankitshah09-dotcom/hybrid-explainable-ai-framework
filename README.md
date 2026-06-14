# Hybrid Explainable AI Framework for Deep Learning Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

A comprehensive Python framework that combines **LIME**, **SHAP**, and **Grad-CAM** to provide explainability and interpretability for deep learning models across multiple domains (Computer Vision, NLP, Time Series).

This project is designed as a dissertation research on **Hybrid Explainable AI**, demonstrating how multiple explanation methods can be unified into a single framework for better model interpretability.

## Key Features

✨ **Multi-Method Explainability**
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Grad-CAM (Gradient-based Class Activation Mapping)

🏗️ **Model Support**
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs/LSTMs)
- Transformer-based models
- Custom PyTorch/TensorFlow models

📊 **Domain Coverage**
- Computer Vision (Image Classification)
- Natural Language Processing (Text Classification)
- Time Series Analysis

📈 **Comprehensive Toolkit**
- Unified API for all explanation methods
- Comparative visualizations
- Performance benchmarking
- Fidelity and robustness metrics
- Interactive Streamlit dashboard

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/jankitshah09-dotcom/hybrid-explainable-ai-framework.git
cd hybrid-explainable-ai-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
from hybrid_xai.framework import ExplainabilityFramework
from hybrid_xai.models import load_pretrained_model
import torch

# Load a pretrained model
model = load_pretrained_model('resnet18', task='image_classification')

# Initialize framework
framework = ExplainabilityFramework(model)

# Get explanations using all methods
explanations = framework.explain(
    input_data=image_tensor,
    methods=['lime', 'shap', 'gradcam'],
    target_class=5
)

# Visualize explanations
framework.visualize(explanations, save_path='./outputs/')
```

## Project Structure

```
hybrid-explainable-ai-framework/
├── hybrid_xai/
│   ├── __init__.py
│   ├── framework.py              # Main framework class
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py               # Base model wrapper
│   │   ├── cnn.py                # CNN implementations
│   │   ├── rnn.py                # RNN/LSTM implementations
│   │   └── transformer.py        # Transformer implementations
│   ├── explainers/
│   │   ├── __init__.py
│   │   ├── base_explainer.py     # Abstract base class
│   │   ├── lime_explainer.py     # LIME implementation
│   │   ├── shap_explainer.py     # SHAP implementation
│   │   └── gradcam_explainer.py  # Grad-CAM implementation
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── explainer_viz.py      # Explanation visualizations
│   │   └── comparative_viz.py    # Side-by-side comparisons
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Fidelity & robustness metrics
│   │   └── benchmarks.py         # Performance benchmarks
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       └── preprocessing.py      # Data preprocessing utilities
├── notebooks/
│   ├── 01_image_classification_demo.ipynb
│   ├── 02_text_classification_demo.ipynb
│   ├── 03_time_series_demo.ipynb
│   └── 04_comparative_analysis.ipynb
├── experiments/
│   ├── config.yaml               # Experiment configurations
│   ├── run_experiments.py        # Main experiment runner
│   └── results/                  # Experiment results
├── app/
│   └── dashboard.py              # Streamlit interactive dashboard
├── tests/
│   ├── __init__.py
│   ├── test_explainers.py
│   ├── test_models.py
│   └── test_framework.py
├── docs/
│   ├── conf.py                   # Sphinx configuration
│   ├── index.rst
│   └── api.rst
├── dissertation/
│   ├── dissertation.tex          # LaTeX dissertation template
│   └── figures/
├── slides/
│   └── presentation.pptx         # Presentation slides
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## Usage Examples

### 1. Image Classification

See `notebooks/01_image_classification_demo.ipynb` for a complete example.

### 2. Text Classification

See `notebooks/02_text_classification_demo.ipynb` for a complete example.

### 3. Time Series Analysis

See `notebooks/03_time_series_demo.ipynb` for a complete example.

### 4. Interactive Dashboard

```bash
streamlit run app/dashboard.py
```

## Documentation

Full API documentation is available at: https://hybrid-explainable-ai.readthedocs.io/

To build documentation locally:

```bash
cd docs/
make html
```

## Dissertation Paper

The dissertation research paper is located in the `dissertation/` folder. It covers:

- Introduction to Explainable AI
- Literature review of LIME, SHAP, and Grad-CAM
- Hybrid framework design and architecture
- Multi-domain experimental evaluation
- Comparative analysis of explanation methods
- Conclusions and future work

## Running Experiments

```bash
python experiments/run_experiments.py --config experiments/config.yaml
```

## Testing

```bash
pytest tests/ -v --cov=hybrid_xai
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@thesis{shah2026hybrid,
  title={Hybrid Explainable AI Framework for Deep Learning Models},
  author={Shah, Jankit},
  year={2026},
  school={University of East London}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions or suggestions, please open an issue or contact [your_email@example.com].

## Acknowledgments

- LIME: https://github.com/marcotcr/lime
- SHAP: https://github.com/shap/shap
- Captum (Grad-CAM): https://github.com/pytorch/captum
