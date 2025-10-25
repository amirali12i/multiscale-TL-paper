"""
Setup script for TL Regularization package
"""

from setuptools import setup, find_packages
import os

# Read README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f 
                   if line.strip() and not line.startswith('#')]

setup(
    name='tl-regularization',
    version='1.0.0',
    author='Anonymous',
    author_email='anonymous@institution.edu',
    description='Triebel-Lizorkin Norm Regularization for Deep Neural Networks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your-username/tl-regularization',
    packages=find_packages(exclude=['tests', 'experiments', 'notebooks']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.10',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.3.0',
            'pytest-cov>=4.1.0',
            'black>=23.3.0',
            'flake8>=6.0.0',
            'mypy>=1.3.0',
            'pre-commit>=3.3.0',
        ],
        'docs': [
            'sphinx>=6.2.0',
            'sphinx-rtd-theme>=1.2.0',
        ],
        'notebooks': [
            'jupyter>=1.0.0',
            'ipywidgets>=8.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'tl-train=experiments.glue.train_bert:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
