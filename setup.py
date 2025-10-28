# setup.py
from setuptools import setup, find_packages

setup(
    name="fluid-sr-thesis",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "h5py",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "Pillow",
        "MegaFlow2D",
    ],
    author="Diya",
    author_email="diyanag@iisc.ac.in",
    description="Super-resolution for fluid flows",
    python_requires=">=3.8",
)