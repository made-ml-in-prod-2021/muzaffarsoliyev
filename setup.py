from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='homework1',
    author='Muzaffar Soliyev',
    install_requires=[
        "click==7.1.2",
        "python-dotenv>=0.5.1",
        "scikit-learn==0.24.1",
        "dataclasses==0.6",
        "pyyaml==5.3.1",
        "marshmallow-dataclass==8.3.0",
        "pandas==1.1.5",
        "hydra-core==1.0.6"
    ],
    license='MIT',
)
