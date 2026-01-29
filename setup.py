from setuptools import setup, find_packages

setup(
    name="novami",
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "polars",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "rdkit",
        "scipy",
        "torch",
        "torch_geometric"
    ],
)
