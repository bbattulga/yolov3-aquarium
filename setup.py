from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "keras==2.1.5",
    "tensorflow==1.6.0",
    "matplotlib",
    "numpy",
    "Pillow",
    "tensorboard",
    "termcolor",
    "google-cloud-storage",
    "setuptools",
    "opencv-python",
    "SciPy"
]

setup(
    name="trainer",
    version="0.6",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='aquarium trainer'
)
