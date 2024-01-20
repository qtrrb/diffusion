from setuptools import setup, find_packages

setup(
    name="diffusion",
    version="0.0.0",
    description="A fork of stable diffusion",
    author="Quentin Torroba",
    packages=find_packages(include=["diffusion", "diffusion.*"]),
    install_requires=[
        "numpy>=1.23.1",
        "safetensors>=0.3.0",
        "omegaconf==2.1.1",
        "einops>=0.3.0",
        "transformers>=4.19.2",
        "open-clip-torch>=2.7.0",
        "k-diffusion>=0.0.14",
        "pydantic>=1.10.7",
        "fastapi>=0.89.1",
        "uvicorn>=0.20.0",
        "torch>=1.12.1",
        "torchvision>=0.13.1",
    ],
)
