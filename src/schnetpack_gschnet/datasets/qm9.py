import io
import logging
import os
import re
import shutil
import tarfile
import tempfile
from typing import List, Optional, Dict
from urllib import request as request

import numpy as np
from ase import Atoms
from ase.io.extxyz import read_xyz
from tqdm import tqdm

import torch
from schnetpack_gschnet.data import GenerativeAtomsDataModule
from schnetpack_gschnet import properties
from schnetpack.data import (
    AtomsDataFormat,
    BaseAtomsData,
    create_dataset,
    load_dataset,
    AtomsDataModuleError,
    SplittingStrategy,
)

__all__ = [
    "QM9Gen",
]


class QM9Gen(GenerativeAtomsDataModule):
    """QM9 benchmark database for organic molecules.

    The QM9 database contains small organic molecules with up to nine non-hydrogen atoms
    from including C, O, N, F. This class adds convenient functions to download QM9 from
    figshare and load the data into pytorch.

    References:

        .. [#qm9_1] https://ndownloader.figshare.com/files/3195404

    """

    # properties
    A = "rotational_constant_A"
    B = "rotational_constant_B"
    C = "rotational_constant_C"
    mu = "dipole_moment"
    alpha = "isotropic_polarizability"
    homo = "homo"
    lumo = "lumo"
    gap = "gap"
    r2 = "electronic_spatial_extent"
    zpve = "zpve"
    U0 = "energy_U0"
    U = "energy_U"
    H = "enthalpy_H"
    G = "free_energy"
    Cv = "heat_capacity"

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        placement_cutoff: float,
        use_covalent_radii: bool = True,
        covalent_radius_factor: float = 1.1,
        num_train: Optional[int] = None,
        num_val: Optional[int] = None,
        num_test: Optional[int] = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = None,
        remove_uncharacterized: bool = False,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 4,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        num_preprocessing_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        data_workdir: Optional[str] = None,
        cleanup_workdir_stage: Optional[str] = "test",
        splitting: Optional[SplittingStrategy] = None,
        pin_memory: Optional[bool] = None,
        force_preprocessing: Optional[bool] = False,
    ):
        """

        Args:
            datapath: Path to dataset.
            batch_size: (train) Batch size.
            placement_cutoff: Determines the distance between atoms considered as
                neighbors for placement.
            use_covalent_radii: If True, pairs inside the placement cutoff will be
                further filtered using the covalent radii from ase. In this way, the
                cutoff is for example smaller for carbon-hydrogen pairs than for
                carbon-carbon pairs. Two atoms will be considered as neighbors if the
                distance between them is 1. smaller than `placement_cutoff` and 2.
                smaller than the sum of the covalent radii of the two involved atom
                types times `covalent_radius_factor`.
            covalent_radius_factor: Allows coarse-grained control of the covalent radius
                criterion when assembling the placement neighborhood (see
                `use_covalent_radii`).
            num_train: Number of training examples.
            num_val: Number of validation examples.
            num_test: Number of test examples.
            split_file: Path to npz file with data partitions.
            format: Dataset format.
            load_properties: Subset of properties to load.
            remove_uncharacterized: Do not include uncharacterized molecules.
            val_batch_size: Validation batch size. If None, use test_batch_size, then
                batch_size.
            test_batch_size: Test batch size. If None, use val_batch_size,
                then batch_size.
            transforms: Transform applied to each system separately before batching.
            train_transforms: Overrides transform_fn for training.
            val_transforms: Overrides transform_fn for validation.
            test_transforms: Overrides transform_fn for testing.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers (overrides
                num_workers).
            num_test_workers: Number of test data loader workers (overrides
                num_workers).
            num_preprocessing_workers: Number of workers for one-time preprocessing
                during data setup (overrides num_workers).
            property_units: Dictionary from property to corresponding unit as a
                string (eV, kcal/mol, ...).
            distance_unit: Unit of the atom positions and cell as a
                string (Ang, Bohr, ...).
            data_workdir: Copy data here as part of setup, e.g. cluster scratch for
                faster performance.
            cleanup_workdir_stage: Determines after which stage to remove the data
                workdir.
            splitting: Method to generate train/validation/test partitions
                (default: RandomSplit)
            pin_memory: If true, pin memory of loaded data to GPU. Default: Will be
                set to true, when GPUs are used.
            force_preprocessing: If true, the list of disconnected structures is
                re-computed (instead of taking precomputed results from the metadata
                of the database if they exist).
        """
        super().__init__(
            datapath=datapath,
            batch_size=batch_size,
            placement_cutoff=placement_cutoff,
            use_covalent_radii=use_covalent_radii,
            covalent_radius_factor=covalent_radius_factor,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            split_file=split_file,
            format=format,
            load_properties=load_properties,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            num_workers=num_workers,
            num_val_workers=num_val_workers,
            num_test_workers=num_test_workers,
            num_preprocessing_workers=num_preprocessing_workers,
            property_units=property_units,
            distance_unit=distance_unit,
            data_workdir=data_workdir,
            cleanup_workdir_stage=cleanup_workdir_stage,
            splitting=splitting,
            pin_memory=pin_memory,
            force_preprocessing=force_preprocessing,
        )

        self.remove_uncharacterized = remove_uncharacterized

    def prepare_data(self):
        if not os.path.exists(self.datapath):
            property_unit_dict = {
                QM9Gen.A: "GHz",
                QM9Gen.B: "GHz",
                QM9Gen.C: "GHz",
                QM9Gen.mu: "D",
                QM9Gen.alpha: "a0^3",
                QM9Gen.homo: "Ha",
                QM9Gen.lumo: "Ha",
                QM9Gen.gap: "Ha",
                QM9Gen.r2: "a0^2",
                QM9Gen.zpve: "Ha",
                QM9Gen.U0: "Ha",
                QM9Gen.U: "Ha",
                QM9Gen.H: "Ha",
                QM9Gen.G: "Ha",
                QM9Gen.Cv: "cal/mol/K",
            }

            tmpdir = tempfile.mkdtemp("qm9")
            atomrefs = self._download_atomrefs(tmpdir)

            dataset = create_dataset(
                datapath=self.datapath,
                format=self.format,
                distance_unit="Ang",
                property_unit_dict=property_unit_dict,
                atomrefs=atomrefs,
            )

            if self.remove_uncharacterized:
                uncharacterized = self._download_uncharacterized(tmpdir)
            else:
                uncharacterized = None
            self._download_data(tmpdir, dataset, uncharacterized=uncharacterized)
            shutil.rmtree(tmpdir)
        else:
            dataset = load_dataset(self.datapath, self.format)
            if self.remove_uncharacterized and len(dataset) == 133885:
                raise AtomsDataModuleError(
                    "The dataset at the chosen location contains the uncharacterized 3054 molecules. "
                    + "Choose a different location to reload the data or set `remove_uncharacterized=False`!"
                )
            elif not self.remove_uncharacterized and len(dataset) < 133885:
                raise AtomsDataModuleError(
                    "The dataset at the chosen location does NOT contain the uncharacterized 3054 molecules. "
                    + "Choose a different location to reload the data or set `remove_uncharacterized=True`!"
                )

    def _download_uncharacterized(self, tmpdir):
        logging.info("Downloading list of uncharacterized molecules...")
        at_url = "https://ndownloader.figshare.com/files/3195404"
        tmp_path = os.path.join(tmpdir, "uncharacterized.txt")
        request.urlretrieve(at_url, tmp_path)
        logging.info("Done.")

        uncharacterized = []
        with open(tmp_path) as f:
            lines = f.readlines()
            for line in lines[9:-1]:
                uncharacterized.append(int(line.split()[0]))
        return uncharacterized

    def _download_atomrefs(self, tmpdir):
        logging.info("Downloading GDB-9 atom references...")
        at_url = "https://ndownloader.figshare.com/files/3195395"
        tmp_path = os.path.join(tmpdir, "atomrefs.txt")
        request.urlretrieve(at_url, tmp_path)
        logging.info("Done.")

        props = [QM9Gen.zpve, QM9Gen.U0, QM9Gen.U, QM9Gen.H, QM9Gen.G, QM9Gen.Cv]
        atref = {p: np.zeros((100,)) for p in props}
        with open(tmp_path) as f:
            lines = f.readlines()
            for z, l in zip([1, 6, 7, 8, 9], lines[5:10]):
                for i, p in enumerate(props):
                    atref[p][z] = float(l.split()[i + 1])
        atref = {k: v.tolist() for k, v in atref.items()}
        return atref

    def _download_data(
        self, tmpdir, dataset: BaseAtomsData, uncharacterized: List[int]
    ):
        logging.info(f"Downloading GDB-9 data to temporary directory {tmpdir}...")
        tar_path = os.path.join(tmpdir, "gdb9.tar.gz")
        raw_path = os.path.join(tmpdir, "gdb9_xyz")
        url = "https://ndownloader.figshare.com/files/3195389"

        request.urlretrieve(url, tar_path)
        logging.info("Done.")

        logging.info("Extracting files...")
        tar = tarfile.open(tar_path)
        tar.extractall(raw_path)
        tar.close()
        logging.info("Done.")

        logging.info("Parse xyz files...")
        ordered_files = sorted(
            os.listdir(raw_path), key=lambda x: (int(re.sub("\D", "", x)), x)
        )

        property_list = []

        irange = np.arange(len(ordered_files), dtype=int)
        if uncharacterized is not None:
            irange = np.setdiff1d(irange, np.array(uncharacterized, dtype=int) - 1)

        for i in tqdm(irange):
            xyzfile = os.path.join(raw_path, ordered_files[i])
            props = {}

            tmp = io.StringIO()
            with open(xyzfile, "r") as f:
                lines = f.readlines()
                l = lines[1].split()[2:]
                for pn, p in zip(dataset.available_properties, l):
                    props[pn] = np.array([float(p)])
                for line in lines:
                    tmp.write(line.replace("*^", "e"))

            tmp.seek(0)
            ats: Atoms = list(read_xyz(tmp, 0))[0]
            props[properties.Z] = ats.numbers
            props[properties.R] = ats.positions
            props[properties.cell] = ats.cell
            props[properties.pbc] = ats.pbc
            property_list.append(props)

        logging.info(f"Write atoms to db at {self.datapath}...")
        dataset.add_systems(property_list=property_list)
        logging.info("Done.")
