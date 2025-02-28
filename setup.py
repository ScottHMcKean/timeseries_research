"""
setup.py configuration script describing how to build and package this project.

This file is primarily used by the setuptools library and typically should not
be executed directly. See README.md for how to deploy, test, and run
the timeseries_research project.
"""
from setuptools import setup, find_packages

import sys
sys.path.append('./src')

import datetime
import dbtsr

setup(
    name="dbtsr",
    version=dbtsr.__version__ + "+" + datetime.datetime.now(datetime.UTC).strftime("%Y%m%d.%H%M%S"),
    url="https://github.com/ScottHMcKean/timeseries_research",
    author="scott.mckean@databricks.com",
    description="Databricks Time Series Research",
    packages=find_packages(where='./src'),
    package_dir={'': 'src'},
    entry_points={
        "packages": [
            "main=dbtsr.main:main"
        ]
    },
    install_requires=[
        "setuptools"
    ],
)
