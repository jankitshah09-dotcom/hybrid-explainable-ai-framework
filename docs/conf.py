# Configuration file for the Sphinx documentation builder.

project = "Hybrid Explainable AI Framework"
copyright = "2026, Jankit Shah"
author = "Jankit Shah"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_mock_imports = ["torch", "torchvision", "lime", "shap", "captum"]
