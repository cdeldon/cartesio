#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read()

setup(
    author="Carlo Del Don",
    author_email="carlo.deldon@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Simple geometry accelerated with Numba",
    entry_points={
        "console_scripts": [
            "cartesio=cartesio.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="cartesio",
    name="cartesio",
    packages=find_packages(include=["cartesio", "cartesio.*"]),
    test_suite="tests",
    url="https://github.com/cdeldon/cartesio",
    version="version='0.2.0'",
    zip_safe=False,
)
