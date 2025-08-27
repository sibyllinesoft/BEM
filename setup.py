"""
Setup script for BEM (Block-wise Expert Modules) package.
"""

import os
from pathlib import Path
from setuptools import setup, find_packages

# Get version from version file
version_file = Path(__file__).parent / "src" / "bem_core" / "_version.py"
version_vars = {}
with open(version_file) as f:
    exec(f.read(), version_vars)

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Development requirements
dev_requirements = [
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.7.0", 
    "flake8>=6.0.0",
    "bandit[toml]>=1.7.5",
    "safety>=2.3.0",
    "pre-commit>=3.5.0",
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-xdist>=3.3.0",
    "pytest-timeout>=2.2.0",
    "pytest-mock>=3.12.0"
]

# Documentation requirements
docs_requirements = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.4.0",
    "mkdocs-gen-files>=0.5.0"
]

setup(
    # Basic package information
    name=version_vars["__title__"],
    version=version_vars["__version__"],
    description=version_vars["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=version_vars["__url__"],
    
    # Author information
    author=version_vars["__author__"],
    author_email=version_vars["__author_email__"],
    
    # License and classification
    license=version_vars["__license__"],
    classifiers=[
        # Development Status
        "Development Status :: 5 - Production/Stable",
        
        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Operating System
        "Operating System :: OS Independent",
        
        # Framework
        "Framework :: Jupyter",
    ],
    
    # Keywords and search terms
    keywords=" ".join(version_vars["__keywords__"]),
    
    # Package structure
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Include additional files
    include_package_data=True,
    package_data={
        "bem_core": ["config/templates/*.yaml"],
        "bem2": ["**/*.yaml", "**/*.json"],
    },
    
    # Dependencies
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "docs": docs_requirements,
        "full": dev_requirements + docs_requirements,
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "bem-info=bem_core._version:print_version_info",
            "bem-demo=scripts.demos.demo_simple_bem:main",
        ],
    },
    
    # Project URLs
    project_urls={
        "Homepage": version_vars["__url__"],
        "Bug Reports": f"{version_vars['__url__']}/issues",
        "Source": version_vars["__url__"],
        "Documentation": f"{version_vars['__url__']}/blob/main/docs/",
        "Changelog": f"{version_vars['__url__']}/blob/main/CHANGELOG.md",
        "Research Paper": "https://arxiv.org/abs/XXXX.XXXX",  # Update when available
    },
    
    # Additional metadata
    zip_safe=False,
    platforms=["any"],
    
    # Long description and content type
    provides=["bem"],
)