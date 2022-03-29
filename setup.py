from setuptools import setup, find_packages

setup(
    name='hyc-utils',
    version='0.3.2',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
    ],
    extras_require={
        'dev': ['pytest','torch','numpy','twine'],
    }
)
