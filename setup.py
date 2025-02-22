"""
Casino of Life - Setup configuration
"""
from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Get version from package __init__.py
def get_version():
    with open(os.path.join("casino_of_life", "__init__.py"), "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.2.4"  # fallback version

setup(
    name="casino-of-life",
    version=get_version(),
    packages=find_packages(include=["casino_of_life", "casino_of_life.*"]),
    package_dir={"": "."},
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "g4f>=0.1.0",
        "websockets>=10.0",
        "aiohttp>=3.8.0",
        "pydantic>=1.8.0",
        "python-dotenv>=0.19.0",
        "gymnasium>=0.21.0",
        "stable-retro>=0.9.0",
        "stable-baselines3>=1.5.0",
        "numpy>=1.19.0",
        "torch>=1.9.0",
    ],
    author="Cimai",
    author_email="info@cimai.biz",
    description="A package for training AI agents to play retro games using natural language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cimai-Decentralized-Games/casino-of-life",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        'casino_of_life': [
            'data/stable/*',
            'data/contrib/*',
            'scenarios/*/*.json',
            'game_wrappers/mk2/data/*',
            'game_wrappers/street-fighter/data/*',
        ]
    },
    entry_points={
        'console_scripts': [
            'casino-of-life=casino_of_life.main:main',
        ],
    }
)
