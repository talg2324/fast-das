import setuptools

__version__ = '1.0.0'

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fast-das",                        # This is the name of the package
    version=__version__,                        # The initial release version
    author="talg",                          # Full name of the author
    description="Parallel beamforming wrapper for native code",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    py_modules=["fastDAS"],                 # Name of the python package
    package_dir={'fastDAS' : 'fastDAS'},        # Directory of the source code of the package
    install_requires=['numpy'],              # Install other dependencies if any
    package_data={'bin': ['*.dll', '*.so']},
    include_package_data=True
)