#!/usr/bin/env python3
"""
Alternative setup.py for OECT Transfer Analysis package.

This file provides an alternative to pyproject.toml for compatibility
with older Python environments or specific build requirements.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements from files if they exist
def read_requirements(filename):
    """Read requirements from a file."""
    try:
        with open(filename, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

# Core requirements
install_requires = [
    "oect-transfer>=0.4.2",
    "numpy>=1.20.0", 
    "pandas>=1.3.0",
    "matplotlib>=3.5.0"
]

# Optional requirements for animation
animation_requires = [
    "opencv-python>=4.5.0",
    "Pillow>=8.0.0"
]

# Development requirements
dev_requires = [
    "black>=22.0",
    "flake8>=4.0", 
    "mypy>=0.950",
    "pytest>=6.0",
    "pytest-cov>=2.0",
    "build"
]

# Documentation requirements  
docs_requires = [
    "sphinx>=4.0",
    "sphinx-rtd-theme>=1.0",
    "myst-parser>=0.17"
]

setup(
    name="oect-transfer-analysis",
    version="1.0.0",
    
    # Package metadata
    author="lidonghao",
    author_email="lidonghao100@outlook.com",
    description="Advanced analysis tools for OECT transfer curves - time series analysis, visualization, and animation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # URLs
    url="https://github.com/yourusername/oect-transfer-analysis",
    project_urls={
        "Homepage": "https://github.com/yourusername/oect-transfer-analysis",
        "Documentation": "https://oect-transfer-analysis.readthedocs.io/",
        "Repository": "https://github.com/yourusername/oect-transfer-analysis.git",
        "Bug Tracker": "https://github.com/yourusername/oect-transfer-analysis/issues",
    },
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include additional files
    include_package_data=True,
    package_data={
        "oect_transfer_analysis": ["*.json", "*.yaml"],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=install_requires,
    
    # Optional dependencies
    extras_require={
        "animation": animation_requires,
        "dev": dev_requires,
        "docs": docs_requires,
        "all": animation_requires + dev_requires + docs_requires,
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords
    keywords=[
        "oect", "transfer", "analysis", "visualization", 
        "time-series", "animation", "electrochemical", 
        "transistor", "degradation", "stability"
    ],
    
    # Entry points (if needed for CLI tools)
    entry_points={
        "console_scripts": [
            # "oect-analyze=oect_transfer_analysis.cli:main",  # If CLI is added later
        ],
    },
    
    # Zip safety
    zip_safe=False,
)