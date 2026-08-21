from setuptools import setup, find_packages

setup(
    name="novami",
    version="0.4.2",
    author="Mateusz Iwan",
    author_email="mateusz.iwan@hotmail.com",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "novami": ["files/*.joblib"],
    },
    install_requires=[
        "numpy",
        "pandas",
        "polars",
        "matplotlib",
        "seaborn",
        "scipy",
    ],
    extras_require={
        "full": [
            "scikit-learn",
            "rdkit",
            "torch",
            "torch_geometric",
        ],
    },
)
