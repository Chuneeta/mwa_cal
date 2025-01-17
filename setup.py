from setuptools import setup, find_packages

setup(
    name="autocal",  # Package name
    version="0.1.0",  # Version number
    author="Ridhima Nunhokee",
    author_email="cnunhokee@gmail.com",
    description="Bandpass calibration using autocorrelations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Chuneeta/autocal",  # URL of your project
    packages=find_packages(),  # Automatically find all packages
    install_requires=[
        "numpy",  # List dependencies here
        "pandas",
		"matplotlib",
		"astropy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

