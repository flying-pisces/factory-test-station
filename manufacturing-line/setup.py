#!/usr/bin/env python3
"""Setup script for Manufacturing Line Control System."""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='manufacturing-line-control',
    version='1.0.0',
    description='Multi-tier manufacturing line orchestration system',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='Factory Automation Team',
    author_email='automation@factory.com',
    url='https://github.com/[org]/manufacturing-line',
    
    packages=find_packages(exclude=['tests*', 'docs*']),
    
    install_requires=read_requirements(),
    
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-asyncio>=0.21.0',
            'black>=23.7.0',
            'isort>=5.12.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0'
        ],
        'vision': [
            'opencv-python>=4.8.0',
            'pillow>=10.0.0'
        ],
        'ai': [
            'tensorflow>=2.13.0',
            'scikit-learn>=1.3.0',
            'torch>=2.0.0'
        ]
    },
    
    python_requires='>=3.8',
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Manufacturing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Hardware :: Hardware Drivers',
        'Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator'
    ],
    
    keywords='manufacturing, automation, factory, production, testing',
    
    entry_points={
        'console_scripts': [
            'line-controller=line_controller.main:main',
            'station-manager=line_controller.station_manager:main',
            'conveyor-control=conveyors.control:main',
            'operator-manager=operators.manager:main'
        ]
    },
    
    package_data={
        'manufacturing-line-control': [
            'config/*.json',
            'config/*.yaml',
            'web-portal/static/*',
            'web-portal/templates/*'
        ]
    },
    
    include_package_data=True,
    zip_safe=False,
    
    project_urls={
        'Bug Reports': 'https://github.com/[org]/manufacturing-line/issues',
        'Source': 'https://github.com/[org]/manufacturing-line',
        'Documentation': 'https://docs.factory.com/manufacturing-line'
    }
)