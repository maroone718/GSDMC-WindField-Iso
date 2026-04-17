from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gsdmc-windfield-iso",
    version="0.1.0",
    author="Zhao ZhongYi",
    description=(
        "Python implementation of GSDMC for wind-speed isosurface "
        "extraction from 3D NetCDF wind-field data"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.6",
        "netCDF4>=1.5",
    ],
    extras_require={
        "dev": ["pytest>=7"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
