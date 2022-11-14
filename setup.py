from setuptools import setup, find_packages

setup(
    name='hyc-utils',
    version='0.5.4',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'statsmodels',
        'torch',
        'tomli',
    ],
)
