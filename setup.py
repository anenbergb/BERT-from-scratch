from setuptools import setup, find_packages

setup(
    name="bert",
    version="1.0.0",
    url="https://github.com/anenbergb/BERT-from-scratch",
    author="Bryan Anenberg",
    author_email="anenbergb@gmail.com",
    description="A fully from scratch implementation of BERT language model",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tensorboard",
        "loguru",
        "matplotlib",
        "tqdm",
        "types-tqdm",
        "tabulate",
        "datasets",
        "transformers",
    ],
    extras_require={
        "torch": [
            "torch",
            "torchvision",
        ],
        "notebook": [
            "jupyter",
            "itkwidgets",
            "jupyter_contrib_nbextensions",
            "plotly",
            "seaborn",
        ],
        "dev": ["black", "mypy", "flake8", "isort", "ipdb"],
    },
)
