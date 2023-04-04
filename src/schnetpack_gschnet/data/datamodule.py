import os
import shutil
import logging
import operator
from typing import Optional, List, Dict, Tuple, Union, Iterable, Any
from tqdm import tqdm
from multiprocessing import Pool
from ase.db import connect
from ase.data import covalent_radii
from collections import deque
from ase.neighborlist import neighbor_list

import numpy as np
import fasteners

import torch

from schnetpack.data import (
    AtomsDataFormat,
    load_dataset,
    AtomsLoader,
    AtomsDataModule,
    AtomsDataModuleError,
    SplittingStrategy,
    RandomSplit,
)
from schnetpack_gschnet.data import gschnet_collate_fn

__all__ = ["GenerativeAtomsDataModule"]


class GenerativeAtomsDataModule(AtomsDataModule):
    """
    Base class for atoms datamodules for cG-SchNet.
    """

    def __init__(
        self,
        datapath: str,
        batch_size: int,
        placement_cutoff: float,
        use_covalent_radii: bool = True,
        covalent_radius_factor: float = 1.1,
        num_train: Union[int, float] = None,
        num_val: Union[int, float] = None,
        num_test: Optional[Union[int, float]] = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = None,
        load_properties: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 8,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        num_preprocessing_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        data_workdir: Optional[str] = None,
        cleanup_workdir_stage: Optional[str] = "test",
        splitting: Optional[SplittingStrategy] = None,
        pin_memory: Optional[bool] = None,
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
            num_train: Number of training examples (absolute or relative).
            num_val: Number of validation examples (absolute or relative).
            num_test: Number of test examples (absolute or relative).
            split_file: Path to npz file with data partitions.
            format: Dataset format.
            load_properties: Subset of properties to load.
            val_batch_size: Validation batch size. If None, use test_batch_size, then
                batch_size.
            test_batch_size: Vest batch size. If None, use val_batch_size, then
                batch_size.
            transforms: Preprocessing transform applied to each system separately
                before batching.
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
            distance_unit: Unit of the atom positions and cell as a string (Ang,
                Bohr, ...).
            data_workdir: Copy data here as part of setup, e.g. cluster scratch for
                faster performance.
            cleanup_workdir_stage: Determines after which stage to remove the data
                workdir.
            splitting: Method to generate train/validation/test partitions
                (default: RandomSplit)
            pin_memory: If true, pin memory of loaded data to GPU. Default: Will be
                set to true, when GPUs are used.

        """
        super().__init__(
            datapath=datapath,
            batch_size=batch_size,
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
            property_units=property_units,
            distance_unit=distance_unit,
            data_workdir=data_workdir,
            cleanup_workdir_stage=cleanup_workdir_stage,
            splitting=splitting,
            pin_memory=pin_memory,
        )

        self.placement_cutoff = placement_cutoff
        self.use_covalent_radii = use_covalent_radii
        self.covalent_radius_factor = covalent_radius_factor
        self.subset_idx = None
        self.registered_properties = None
        self.num_preprocessing_workers = self.num_workers
        if num_preprocessing_workers is not None:
            self.num_preprocessing_workers = num_preprocessing_workers

    def setup(self, stage: Optional[str] = None):
        if self.trainer.ckpt_path is not None and stage != "load_checkpoint":
            # skip setup, it will be called after restoring the checkpoint
            return
        # check whether data needs to be copied
        if self.data_workdir is None:
            datapath = self.datapath
        else:
            if not os.path.exists(self.data_workdir):
                os.makedirs(self.data_workdir, exist_ok=True)

            name = self.datapath.split("/")[-1]
            datapath = os.path.join(self.data_workdir, name)

            lock = fasteners.InterProcessLock(
                os.path.join(self.data_workdir, f"cache_{name}.lock")
            )

            with lock:
                # retry reading, in case other process finished in the meantime
                if not os.path.exists(datapath):
                    shutil.copy(self.datapath, datapath)

                # reset datasets in case they need to be reloaded
                self.dataset = None
                self._train_dataset = None
                self._val_dataset = None
                self._test_dataset = None

                # reset cleanup
                self._has_teardown_fit = False
                self._has_teardown_val = False
                self._has_teardown_test = False

        # create subset excluding disconnected molecules (according to placement_cutoff)
        if self.subset_idx is None:
            if self.use_covalent_radii:
                logging.info(
                    f"Setting up training data - "
                    f"checking connectivity of molecules using covalent "
                    f"radii from ASE with a factor of {self.covalent_radius_factor} "
                    f"and a maximum neighbor distance (i.e. placement cutoff) of "
                    f"{self.placement_cutoff}."
                )
            else:
                logging.info(
                    f"Setting up training data - "
                    f"checking connectivity of molecules using a placement cutoff "
                    f"of {self.placement_cutoff}"
                )
            dataset = load_dataset(
                datapath,
                self.format,
                property_units=self.property_units,
                distance_unit=self.distance_unit,
                load_properties=self.load_properties,
                load_structure=True,
            )
            subset = np.ones(len(dataset), dtype=bool)
            # check connectivity (in multiple threads if multiple workers are available)
            if self.num_preprocessing_workers > 0:
                datapaths = [dataset.datapath] * len(dataset)
                cutoffs = [self.placement_cutoff] * len(dataset)
                conversions = [dataset.distance_conversion] * len(dataset)
                use_covalent_radii = [self.use_covalent_radii] * len(dataset)
                cv_factor = [self.covalent_radius_factor] * len(dataset)
                idcs = range(len(dataset))
                arguments = zip(
                    idcs, datapaths, cutoffs, conversions, use_covalent_radii, cv_factor
                )
                with Pool(self.num_preprocessing_workers) as p:
                    for res in tqdm(
                        p.imap_unordered(check_connectivity, arguments),
                        total=len(dataset),
                    ):
                        if not res[0]:
                            subset[res[1]] = False
            else:
                for i in tqdm(range(len(dataset))):
                    res, _ = check_connectivity(
                        (
                            i,
                            dataset.datapath,
                            self.placement_cutoff,
                            dataset.distance_conversion,
                            self.use_covalent_radii,
                            self.covalent_radius_factor,
                        )
                    )
                    if not res:
                        subset[i] = False
            subset_idx = np.nonzero(subset)[0].tolist()
            n_disconnected = len(dataset) - len(subset_idx)
            if n_disconnected > 0:
                logging.info(
                    f"Done! Found {n_disconnected} disconnected structures among the "
                    f"{len(dataset)} molecules in the database. These structures will "
                    f"not be included in the training, validation, or test set. If "
                    f"you want to decrease this number, it might help to increase "
                    f"`placement_cutoff` (currently {self.placement_cutoff}) and (if "
                    f"`use_covalent_radii` is set) "
                    f"`covalent_radius_factor` (currently {self.covalent_radius_factor}"
                    f")."
                )
            else:
                logging.info(
                    f"Done! Found no disconnected structures, i.e. all {len(dataset)} "
                    f"molecules from the database will be used in the training, "
                    f"validation or test set."
                )
            self.subset_idx = list(subset_idx)

        # load dataset using the corresponding subset
        self.dataset = load_dataset(
            datapath,
            self.format,
            property_units=self.property_units,
            distance_unit=self.distance_unit,
            load_properties=self.load_properties,
            subset_idx=self.subset_idx,
        )

        # generate partitions
        self.load_partitions()

        # partition dataset
        self._train_dataset = self.dataset.subset(self.train_idx)
        self._val_dataset = self.dataset.subset(self.val_idx)
        self._test_dataset = self.dataset.subset(self.test_idx)

        self._setup_transforms()

    # storing placement cutoff information in split file
    def load_partitions(self):
        # TODO: handle IterDatasets
        # handle relative dataset sizes
        if self.num_train is not None and self.num_train <= 1.0:
            self.num_train = round(self.num_train * len(self.dataset))
        if self.num_val is not None and self.num_val <= 1.0:
            self.num_val = min(
                round(self.num_val * len(self.dataset)),
                len(self.dataset) - self.num_train,
            )
        if self.num_test is not None and self.num_test <= 1.0:
            self.num_test = min(
                round(self.num_test * len(self.dataset)),
                len(self.dataset) - self.num_train - self.num_val,
            )

        # split dataset
        if self.split_file is not None and os.path.exists(self.split_file):
            logging.info(
                f"Loading splits for training, validation and testing "
                f"from existing split file at {self.split_file}."
            )
            S = np.load(self.split_file)
            self.train_idx = S["train_idx"].tolist()
            self.val_idx = S["val_idx"].tolist()
            self.test_idx = S["test_idx"].tolist()
            if self.num_train and self.num_train != len(self.train_idx):
                raise AtomsDataModuleError(
                    f"Split file was given, but `num_train ({self.num_train}) != "
                    f"len(train_idx)` ({len(self.train_idx)})!"
                )
            if self.num_val and self.num_val != len(self.val_idx):
                raise AtomsDataModuleError(
                    f"Split file was given, but `num_val ({self.num_val}) != "
                    f"len(val_idx)` ({len(self.val_idx)})!"
                )
            if self.num_test and self.num_test != len(self.test_idx):
                raise AtomsDataModuleError(
                    f"Split file was given, but `num_test ({self.num_test}) != "
                    f"len(test_idx)` ({len(self.test_idx)})!"
                )
            if "placement_cutoff" in S:
                if S["placement_cutoff"] != self.placement_cutoff:
                    raise AtomsDataModuleError(
                        f"Split file was given, but it was created with a placement "
                        f"cutoff of {S['placement_cutoff']} which is different from "
                        f"the selected placement cutoff of {self.placement_cutoff}. "
                        f"Please specify another split file, change `placement_cutoff`"
                        f", or start with a new split."
                    )
            else:
                raise AtomsDataModuleError(
                    "Split file was given, but it was created without "
                    '"placement cutoff". Please specify another split file or start '
                    "with a new split."
                )
        else:
            if not self.num_train or not self.num_val:
                raise AtomsDataModuleError(
                    "If no `split_file` is given, the sizes of the training and "
                    "validation partitions need to be set!"
                )
            if self.num_test:
                if self.num_train + self.num_val + self.num_test > len(self.dataset):
                    raise AtomsDataModuleError(
                        f"The chosen numbers of training, validation, and test points "
                        f"({self.num_train} + {self.num_val} + {self.num_test} = "
                        f"{self.num_train + self.num_val + self.num_test}) is larger "
                        f"than the number of points in the dataset ("
                        f"{len(self.dataset)}). Please choose lower numbers."
                    )
            else:
                if self.num_train + self.num_val > len(self.dataset):
                    raise AtomsDataModuleError(
                        f"The chosen numbers of training and validation points "
                        f"({self.num_train} + {self.num_val} = "
                        f"{self.num_train + self.num_val}) is larger than the "
                        f"number of points in the dataset ({len(self.dataset)}). "
                        f"Please choose lower numbers."
                    )

            self.train_idx, self.val_idx, self.test_idx = self.splitting.split(
                self.dataset, self.num_train, self.num_val, self.num_test
            )

            if self.split_file is not None:
                np.savez(
                    self.split_file,
                    train_idx=self.train_idx,
                    val_idx=self.val_idx,
                    test_idx=self.test_idx,
                    placement_cutoff=self.placement_cutoff,
                )

    def state_dict(self):
        return {
            "subset_idx": self.subset_idx,
            "load_properties": self.load_properties,
            "train_transforms": self.train_transforms,
            "val_transforms": self.val_transforms,
            "test_transforms": self.test_transforms,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.subset_idx = state_dict["subset_idx"]
        self._train_transforms = state_dict["train_transforms"]
        self._val_transforms = state_dict["val_transforms"]
        self._test_transforms = state_dict["test_transforms"]
        self.load_properties = state_dict["load_properties"]
        self.setup(stage="load_checkpoint")

    def register_properties(self, properties: List[str]):
        if self.dataset is None:
            # dataset not yet initialized, properties will be determined from checkpoint
            return
        if properties is not None and len(properties) > 0:
            available_properties = self.dataset.available_properties
            for p in properties:
                if p not in available_properties:
                    raise AtomsDataModuleError(
                        f"Property {p} required for conditioning not found in data "
                        f"base at {self.datapath}."
                    )
            # extend load_properties list
            if self.load_properties is None:
                self.load_properties = properties
            else:
                unique_props = set(properties).union(set(self.load_properties))
                self.load_properties = list(unique_props)

            # set load_properties in datasets
            self.dataset.load_properties = self.load_properties
            self._train_dataset.load_properties = self.load_properties
            self._val_dataset.load_properties = self.load_properties
            self._test_dataset.load_properties = self.load_properties

    # using other collate function for the dataloaders
    def train_dataloader(self) -> AtomsLoader:
        return AtomsLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=gschnet_collate_fn,
        )

    def val_dataloader(self) -> AtomsLoader:
        return AtomsLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_val_workers,
            pin_memory=True,
            collate_fn=gschnet_collate_fn,
        )

    def test_dataloader(self) -> AtomsLoader:
        return AtomsLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_test_workers,
            pin_memory=True,
            collate_fn=gschnet_collate_fn,
        )


def check_connectivity(args):
    # load molecule from data base
    (
        i,
        datapath,
        cutoff,
        distance_conversion,
        use_covalent_radii,
        covalent_radius_factor,
    ) = args
    with connect(datapath) as con:
        at = con.get(i + 1).toatoms()
    at.positions *= distance_conversion
    n_atoms = len(at.numbers)
    # create list of neighbors inside cutoff
    _idx_i, _idx_j, _r_ij = neighbor_list("ijd", at, cutoff, self_interaction=False)
    if use_covalent_radii:
        thresh = covalent_radii[at.numbers[_idx_i]] + covalent_radii[at.numbers[_idx_j]]
        idcs = np.nonzero(_r_ij <= (thresh * covalent_radius_factor))[0]
        _idx_i = _idx_i[idcs]
        _idx_j = _idx_j[idcs]
    n_nbh = np.bincount(_idx_i, minlength=n_atoms)
    # check if there are atoms without neighbors, i.e. disconnected atoms
    if np.sum(n_nbh == 0) > 0:
        return False, i
    # store where the neighbors in _idx_j of each center atom in _idx_i start
    # assuming that _idx_i is ordered
    start_idcs = np.append(np.zeros(1, dtype=int), np.cumsum(n_nbh))
    # check connectivity of atoms given the neighbor list
    seen = np.zeros(n_atoms, dtype=bool)
    seen[0] = True
    count = 1
    first_neighbors = _idx_j[start_idcs[0] : start_idcs[1]]
    seen[first_neighbors] = True
    count += len(first_neighbors)
    queue = deque(first_neighbors)
    while queue and count < n_atoms:
        atom = queue.popleft()
        for neighbor in _idx_j[start_idcs[atom] : start_idcs[atom + 1]]:
            if not seen[neighbor]:
                seen[neighbor] = True
                count += 1
                queue.append(neighbor)
    if count < n_atoms:
        return False, i  # there are disconnected parts (we did not visit every atom)
    else:
        return True, i  # everything is connected via some path
