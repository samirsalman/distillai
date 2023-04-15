from setuptools import setup, find_packages

setup(
    name="distill_ai",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch==2.0.0",
        "torchmetrics==0.9.3",
        "pytorch-lightning==1.8.6",
    ],
    author="Samir Salman",
)
