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
    version="1.0.0",
    author="Niklas Gebauer",
    scripts=[],
    include_package_data=True,
    install_requires=[
        "schnetpack>=2.0.3",
        "torch>=1.9",
        "pytorch_lightning>=2.0",
        "hydra-core>=1.1.0",
        "hydra-colorlog>=1.1.0",
        "numpy",
        "ase>=3.21",
        "torchmetrics",
        "h5py",
        "tqdm",
        "pyyaml",
        "tensorboard",
        "pre-commit",
        "black",
    ],
)
