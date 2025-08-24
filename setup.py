#!/usr/bin/env python
"""
Factory Test Station Setup Script

This script provides installation and build capabilities for the Factory Test Station system.
"""

from setuptools import setup, find_packages
import os

# Read version from main module
__version__ = "1.0.0"

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Factory Test Station System"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="factory-test-station",
    version=__version__,
    description="Factory Test Station System for Hardware Testing",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Factory Test Station Team",
    author_email="factory-test@company.com",
    url="https://github.com/company/factory-test-station",
    
    # Package discovery
    packages=find_packages(exclude=["tests*", "reference*"]),
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "GUI": ["UI_dep/*", "logo/*"],
        "stations": ["config/*"],
        "": ["*.md", "*.txt", "*.cfg"],
    },
    
    # Requirements
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "gui": [
            "tkinter",
            "flask",
            "flask-socketio", 
        ],
        "windows": [
            "pywin32; sys_platform=='win32'",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.7",
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "factory-test-station=stations.project_station_run:main",
            "factory-test-console=stations.console_test_runner:main",
        ],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Manufacturing",
        "Topic :: System :: Hardware :: Testing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Environment :: X11 Applications :: Qt",
        "Environment :: Win32 (MS Windows)",
    ],
    
    # Additional metadata
    keywords="factory testing hardware automation gui console",
    project_urls={
        "Bug Reports": "https://github.com/company/factory-test-station/issues",
        "Source": "https://github.com/company/factory-test-station",
        "Documentation": "https://factory-test-station.readthedocs.io/",
    },
)