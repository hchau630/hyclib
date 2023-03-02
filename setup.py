from setuptools import setup, find_packages

with open('README.me') as f:
    README = f.read()

setup(
    name='hyc-utils',
    version='0.5.30',
    description='Commonly used tools across my own personal projects'
    long_description=README,
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
