"""Setup configuration for buoy-data package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="buoy-data",
    version="1.0.0",
    author="C.J. Walsh",
    author_email="cj@odewebdesigns.com",
    description="An aggregator that pulls data from NOAA's National Data Buoy Center",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/odds-get-evened/buoy-data",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.31.0",
        "sqlalchemy>=2.0.0",
    ],
    extras_require={
        "ml": [
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
)
