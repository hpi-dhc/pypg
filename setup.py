"""
====================================
Set Up File
====================================
"""

import setuptools

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="pypg",
    version="0.0.5",
    author="Ariane Sasso",
    author_email="ariane.sasso@gmail.com",
    description="PyPG: A library for PPG (PhotoPlethysmoGram) processing",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/hpi-dhc/pypg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'scipy'
    ]
)
