Hybrid Explainable AI Framework
===============================

Welcome to the documentation for the Hybrid Explainable AI Framework.

Contents:

.. toctree::
   :maxdepth: 2

   api
   models
   explainers
   visualization

Introduction
------------

This framework combines multiple explainability methods to provide comprehensive
explanations for deep learning models:

- **LIME**: Local model-agnostic explanations
- **SHAP**: Game theory-based explanations
- **Grad-CAM**: Gradient-based visual explanations

Quick Start
-----------

Install the framework:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

Basic usage:

.. code-block:: python

   from hybrid_xai.framework import ExplainabilityFramework
   from hybrid_xai.models import load_pretrained_model

   model = load_pretrained_model('resnet18')
   framework = ExplainabilityFramework(model)
   explanations = framework.explain(image, methods=['lime', 'shap', 'gradcam'])

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
