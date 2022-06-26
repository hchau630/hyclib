from setuptools import setup, find_packages

setup(
    name='hyc-utils',
    version='0.4.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'torch',
    ],
    extras_require={
        'dev': ['pytest','twine'],
    }
)
