from setuptools import setup, find_packages

setup(
    name='hyc-utils',
    version='0.2.0',
    packages=find_packages(),
    extras_require={
        'packaging': ['twine'],
        'test': ['pytest','torch','numpy'],
    }
)
