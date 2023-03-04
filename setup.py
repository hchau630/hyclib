from setuptools import setup, find_packages
import pathlib

this_directory = pathlib.Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='hyclib',
    version='0.1.0',
    description='Commonly used tools across my own personal projects',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/hchau630/hyclib",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'statsmodels',
        'torch',
        'tomli',
        'h5py',
        'pandas',
        'tables', # optional dependency for pandas that is needed here
        'tqdm',
        'mat73',
        'platformdirs',
    ],
    include_package_data=True,
)
