from setuptools import setup, find_packages

setup(
    name='hyc-utils',
    version='0.5.27',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'statsmodels',
        'torch',
        'tomli',
        'h5py',
        'pandas',
        'tables', # optional dependency for pandas
        'tqdm',
        'mat73',
    ],
    extras_require={
        'test': ['pytest', 'scipy'],
    },
)
