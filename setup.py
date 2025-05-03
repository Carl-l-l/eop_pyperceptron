from setuptools import setup, find_packages

setup(
    name='py_perceptron',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.4',
        'scikit-learn>=1.6.1',
        'pandas>=2.2.3',
        'matplotlib>=3.10.1',
    ],
)