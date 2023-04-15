from setuptools import setup, find_packages

setup(
    name="distill_ai",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch==1.10.1",
        "torchmetrics==0.7.0",
        "pytorch-lightning==1.5.9",
    ],
    author="Samir Salman",
)
