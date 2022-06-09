import io
import os
from setuptools import (
    setup,
    find_packages,
)


def read(
    fname,
):
    with io.open(
        os.path.join(
            os.path.dirname(__file__),
            fname,
        ),
        encoding="utf-8",
    ) as f:
        return f.read()


setup(
    name="schnetpack-gschnet",
    packages=find_packages("src"),
    package_dir={"": "src"},
    version="0.1",
    author="Niklas Gebauer",
    scripts=[],
    include_package_data=True,
    install_requires=[
        "schnetpack>=1.0.0.dev0",
        "torch>=1.9",
        "pytorch_lightning>=1.3.5",
        "hydra-core==1.1.0",
        "hydra-colorlog>=1.1.0",
        "numpy",
        "ase>=3.21",
        "torchmetrics",
        "h5py",
        "tqdm",
        "pyyaml",
        "pre-commit",
        "black",
    ],
)
