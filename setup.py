"""
__author__: satheesh.k
Created: Wednesday, 2nd December 2020 6:08:21 pm
"""

from os import path

from labelvis import __version__
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="labelvis",
    version=__version__,
    description="python library to visualize object detection labels",
    long_description_content_type="text/markdown",
    long_description=long_description,
    author="Satheesh Katipomu",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Utilities",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    url="https://github.com/satheeshkatipomu/labelvis",
    packages=find_packages(),
    platforms=["linux", "unix"],
    python_requires=">=3.7",
)