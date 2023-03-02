from setuptools import setup, find_packages
import pathlib

this_directory = pathlib.Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='hyc-utils',
    version='0.5.30',
    description='Commonly used tools across my own personal projects',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
    extras_require={
        'test': ['pytest', 'scipy'],
    },
    include_package_data=True,
)
