import os
import shutil
import logging
import operator
from typing import Optional, List, Dict, Tuple, Union, Iterable, Any
from tqdm import tqdm

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
from schnetpack_gschnet.transform import ConnectivityCheck

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
            force_preprocessing: If true, the list of disconnected structures is
                re-computed (instead of taking precomputed results from the metadata
                of the database if they exist).
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
        self.force_preprocessing = force_preprocessing

    def setup(self, stage: Optional[str] = None):
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
                load_properties=[],
                distance_unit=self.distance_unit,
                load_structure=True,
            )
            # check for pre-computed disconnected structures in db metadata
            metadata = dataset.metadata
            if "disconnected_idx" not in metadata:
                metadata["disconnected_idx"] = {}
            precomputed = metadata["disconnected_idx"]
            key = (
                f"{self.placement_cutoff}/{self.use_covalent_radii}"
                f"/{self.covalent_radius_factor}"
            )
            if key in precomputed and not self.force_preprocessing:
                # obtain disconnected structures from metadata
                subset = np.ones(len(dataset), dtype=bool)
                subset[precomputed[key]] = False
                logging.info(
                    f"Using precomputed results stored in the metadata of the "
                    f"database. If you want to force the re-computation of "
                    f"disconnected structures, set force_preprocessing=True."
                )
            else:
                # iterate over the whole dataset to find disconnected molecules
                dataset.transforms = [
                    ConnectivityCheck(
                        self.placement_cutoff,
                        self.use_covalent_radii,
                        self.covalent_radius_factor,
                        return_inputs=False,
                    ),
                ]
                _batch_size = 100
                preprocessing_loader = AtomsLoader(
                    dataset,
                    batch_size=_batch_size,
                    num_workers=self.num_preprocessing_workers,
                    shuffle=False,
                    pin_memory=False,
                    collate_fn=lambda x: x,
                )
                subset = np.empty(len(dataset), dtype=bool)
                _iteration = 0
                with tqdm(total=len(dataset)) as pbar:
                    for connected_list in preprocessing_loader:
                        start = _iteration * _batch_size
                        end = start + len(connected_list)
                        subset[start:end] = connected_list
                        _iteration += 1
                        pbar.update(len(connected_list))
                # update metadata with found disconnected structures
                disconnected_idx = np.nonzero(~subset)[0].tolist()
                precomputed[key] = disconnected_idx
                dataset._set_metadata(metadata)
            # create subset without disconnected molecules
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
            if (
                "placement_cutoff" in S
                and "use_covalent_radii" in S
                and "covalent_radius_factor" in S
            ):
                if (
                    S["placement_cutoff"] != self.placement_cutoff
                    or S["use_covalent_radii"] != self.use_covalent_radii
                    or S["covalent_radius_factor"] != self.covalent_radius_factor
                ):
                    raise AtomsDataModuleError(
                        f"Split file was given, but it was created with placement_"
                        f"cutoff={S['placement_cutoff']}, use_covalent_radii="
                        f"{S['use_covalent_radii']}, and covalent_radius_factor="
                        f"{S['covalent_radius_factor']}, which is different from "
                        f"the current settings ({self.placement_cutoff}"
                        f"/{self.use_covalent_radii}/{self.covalent_radius_factor}). "
                        f"Please specify another split file, adjust the settings, "
                        f"or start with a new split."
                    )
            else:
                raise AtomsDataModuleError(
                    "Split file was given, but it is missing information about "
                    '"placement_cutoff", "use_covalent_radii", or "covalent_radius_'
                    'factor". Please specify another split file or start with a new '
                    "split."
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
                    use_covalent_radii=self.use_covalent_radii,
                    covalent_radius_factor=self.covalent_radius_factor,
                )

    def register_properties(self, properties: List[str]):
        if properties is not None:
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
