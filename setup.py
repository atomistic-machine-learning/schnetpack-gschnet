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
    version="0.1-alpha",
    author="Niklas Gebauer",
    scripts=[],
    include_package_data=True,
    install_requires=[
        "schnetpack==2.0.1",
        "torch==1.13.1",
        "pytorch_lightning==1.9",
        "hydra-core==1.3.2",
        "hydra-colorlog==1.2.0",
        "numpy",
        "ase==3.22.1",
        "torchmetrics",
        "h5py",
        "tqdm",
        "pyyaml",
        "tensorboard"
        "pre-commit",
        "black",
    ],
)
