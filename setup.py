from pathlib import Path

from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

# Define our package
setup(
    name="VertebralColumn",
    version=0.1,
    description="Projet de classification de Colonne Vertebrale.",
    author="Pauline CHEVALIER",
    author_email="pauline.chevalier@insa-rouen.fr",
    url="",
    python_requires=">=3.7",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
)
