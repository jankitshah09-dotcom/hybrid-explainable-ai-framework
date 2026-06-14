from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hybrid-explainable-ai",
    version="0.1.0",
    author="Jankit Shah",
    author_email="your_email@example.com",
    description="A hybrid framework combining LIME, SHAP, and Grad-CAM for DL model explainability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jankitshah09-dotcom/hybrid-explainable-ai-framework",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "lime>=0.2.0",
        "shap>=0.40.0",
        "captum>=0.6.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
    ],
)
