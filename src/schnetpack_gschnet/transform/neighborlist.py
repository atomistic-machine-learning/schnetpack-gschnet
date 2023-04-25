from typing import Dict, Optional, List, Tuple

import os
import shutil

# from numpy import lexsort
from dirsync import sync
import logging
import torch
import numpy as np
from collections import deque
from ase import Atoms
from ase.neighborlist import neighbor_list
from matscipy.neighbours import neighbour_list as msp_neighbor_list
from ase.data import covalent_radii
from schnetpack.transform.base import Transform
import fasteners
from schnetpack_gschnet import properties

logger = logging.getLogger(__name__)

__all__ = [
    "GeneralCachedNeighborList",
    "ConditionalGSchNetNeighborList",
    "MultipleNeighborListsTransform",
    "ASEMultipleNeighborLists",
    "TorchMultipleNeighborLists",
    "MultipleCountNeighbors",
    "ConnectivityCheck",
]


class CacheException(Exception):
    pass


class GeneralCachedNeighborList(Transform):
    """
    Dynamic caching of neighbor lists.
    This wraps a neighbor list and stores the results the first time it is called
    for a dataset entry with the pid provided by AtomsDataset. Particularly,
    for large systems, this speeds up training significantly.
    Note:
        The provided cache location should be unique to the used dataset. Otherwise,
        wrong neighborhoods will be provided. The caching location can be reused
        across multiple runs, by setting `keep_cache=True`.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        cache_path: str,
        neighbor_list: Transform,
        keep_cache: bool = False,
        cache_workdir: str = None,
    ):
        """
        Args:
            cache_path: Path of caching directory.
            neighbor_list: The neighbor list Transform to use.
            keep_cache: Keep cache at `cache_location` at the end of training, or copy
                built/updated cache there from `cache_workdir` (if set). A pre-existing
                cache at `cache_location` will not be deleted, while a temporary cache
                at `cache_workdir` will always be removed.
            cache_workdir: If this is set, the cache will be build here, e.g. a cluster
                scratch space for faster performance. An existing cache at
                `cache_location` is copied here at the beginning of training, and
                afterwards (if `keep_cache=True`) the final cache is copied to
                `cache_workdir`.
        """
        super().__init__()
        self.neighbor_list = neighbor_list
        self.keep_cache = keep_cache
        self.cache_path = cache_path
        self.cache_workdir = cache_workdir
        self.preexisting_cache = os.path.exists(self.cache_path)
        self.has_tmp_workdir = cache_workdir is not None

        os.makedirs(cache_path, exist_ok=True)

        if self.has_tmp_workdir:
            # cache workdir should be empty to avoid loading nbh lists from earlier runs
            if os.path.exists(cache_workdir):
                raise CacheException("The provided `cache_workdir` already exists!")

            # copy existing nbh lists to cache workdir
            if self.preexisting_cache:
                shutil.copytree(cache_path, cache_workdir)
            self.cache_location = cache_workdir
        else:
            # use cache_location to store and load neighborlists
            self.cache_location = cache_path

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cache_file = os.path.join(
            self.cache_location, f"cache_{inputs[properties.idx][0]}.pt"
        )

        # try to read cached NBL
        try:
            data = torch.load(cache_file)
            inputs.update(data)
        except IOError:
            # acquire lock for caching
            lock = fasteners.InterProcessLock(
                os.path.join(
                    self.cache_location, f"cache_{inputs[properties.idx][0]}.lock"
                )
            )
            with lock:
                # retry reading, in case other process finished in the meantime
                try:
                    data = torch.load(cache_file)
                    inputs.update(data)
                except IOError:
                    # now it is save to calculate and cache
                    inputs = self.neighbor_list(inputs)
                    # store idx_i, idx_j, offsets, and pairwise distances r_ij
                    data = {
                        properties.idx_i: inputs[properties.idx_i],
                        properties.idx_j: inputs[properties.idx_j],
                        properties.offsets: inputs[properties.offsets],
                        properties.r_ij: inputs[properties.r_ij],
                    }
                    # store additional cacheable data
                    if hasattr(self.neighbor_list, "additional_data"):
                        for data_name in self.neighbor_list.additional_data:
                            data[data_name] = inputs[data_name]
                    torch.save(data, cache_file)
                except Exception as e:
                    print(e)
        return inputs

    def teardown(self):
        if not self.keep_cache and not self.preexisting_cache:
            try:
                shutil.rmtree(self.cache_path)
            except:
                pass

        if self.cache_workdir is not None:
            if self.keep_cache:
                try:
                    sync(self.cache_workdir, self.cache_path, "sync")
                except:
                    pass

            try:
                shutil.rmtree(self.cache_workdir)
            except:
                pass


class ConditionalGSchNetNeighborList(Transform):
    """
    Class to compute the three neighbor lists resulting from the model, prediction, and
    atom placement cutoffs used in the cG-SchNet model.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        model_cutoff: float,
        prediction_cutoff: float,
        placement_cutoff: float,
        environment_provider: str = "matscipy",
        use_covalent_radii: bool = True,
        covalent_radius_factor: float = 1.1,
    ):
        """
        Args:
            model_cutoff: Determines which atoms pass messages to each other during
                feature extraction.
            prediction_cutoff: Determines which atoms are used to predict pairwise
                distances (i.e. which atoms are close enough to the focus such that
                they are utilized to predict their distance to the next atom).
            placement_cutoff: Determines which atoms are considered to be neighbors
                when sampling sequences of atom placements (i.e. which atoms can be
                placed given a focus atom).
            environment_provider: Can be "matscipy", "ase", or "torch" to use the
                matscipy, ASE, or custom torch implementation of neighbor list.
            use_covalent_radii: If True, pairs inside the placement cutoff will be
                further filtered using the covalent radii from ase. In this way, the
                cutoff is for example smaller for carbon-hydrogen pairs than for
                carbon-carbon pairs. Two atoms will be considered as neighbors if the
                distance between them is 1. smaller than `placement_cutoff` and 2.
                smaller than the sum of the covalent radii of the two involved atom
                types times `covalent_radius_factor`.
            covalent_radius_factor: Allows control of the covalent radius criterion
                when assembling the placement neighborhood (see `use_covalent_radii`).
        """
        super().__init__()
        if environment_provider == "torch":
            nbh_class = TorchMultipleNeighborLists
        elif environment_provider == "ase":
            nbh_class = ASEMultipleNeighborLists
        else:
            nbh_class = MatScipyMultipleNeighborLists
            if environment_provider.lower() != "matscipy":
                logging.info(
                    f'The specified environment provider "{environment_provider}" '
                    f'does not exist, please choose from "matscipy", "ase", and '
                    f'"torch". Using default provider "matscipy" now.'
                )
        # initialize transform for computation of the neighbor lists
        self.nbh_transform = nbh_class(
            cutoffs=[model_cutoff, prediction_cutoff, placement_cutoff],
            nbh_names=[
                properties.nbh_model,
                properties.nbh_prediction,
                properties.nbh_placement,
            ],
        )
        self.model_cutoff = model_cutoff
        self.prediction_cutoff = prediction_cutoff
        self.placement_cutoff = placement_cutoff
        # initialize transform for computation of the numbers of neighbors per center
        # atom in all lists
        self.nbh_count_transform = MultipleCountNeighbors(
            nbh_name_pairs=[
                (properties.nbh_model, properties.n_nbh_model),
                (properties.nbh_prediction, properties.n_nbh_prediction),
                (properties.nbh_placement, properties.n_nbh_placement),
            ],
            sorted=True,
        )
        # store the properties calculated here besides idx_i, idx_j etc.
        # (required for caching)
        self.additional_data = self.nbh_transform.additional_data + [
            properties.n_nbh,
            properties.n_nbh_model,
            properties.n_nbh_prediction,
            properties.n_nbh_placement,
        ]
        self.use_covalent_radii = use_covalent_radii
        self.covalent_radius_factor = covalent_radius_factor

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # compute all neighbor lists
        self.nbh_transform(inputs)
        # filter neighbor list of placement cutoff with covalent radii if desired
        if self.use_covalent_radii:
            self.filter_with_covalent_radii(inputs)
        # count number of neighbors per center atom in all lists
        self.nbh_count_transform(inputs)
        return inputs

    def filter_with_covalent_radii(self, inputs):
        nbh_idcs = inputs[properties.nbh_placement]
        idx_i = inputs[properties.idx_i][nbh_idcs]
        idx_j = inputs[properties.idx_j][nbh_idcs]
        r_ij = inputs[properties.r_ij][nbh_idcs]
        Z = inputs[properties.Z]
        thresh = torch.tensor((covalent_radii[Z[idx_i]] + covalent_radii[Z[idx_j]]))
        idcs = torch.where(r_ij <= (thresh * self.covalent_radius_factor))[0]
        inputs[properties.nbh_placement] = nbh_idcs[idcs]

    def check_cutoffs(self, model_cutoff, prediction_cutoff, placement_cutoff):
        if model_cutoff - self.model_cutoff != 0:
            raise ValueError(
                f"{model_cutoff}!={self.model_cutoff}, `model_cutoff` does not match"
                f"the value in the neighbor list transform."
            )
        if prediction_cutoff - self.prediction_cutoff != 0:
            raise ValueError(
                f"{prediction_cutoff}!={self.prediction_cutoff}, `prediction_cutoff` "
                f"does not match the value in the neighbor list transform."
            )
        if placement_cutoff - self.placement_cutoff != 0:
            raise ValueError(
                f"{placement_cutoff}!={self.placement_cutoff}, `placement_cutoff` "
                f"does not match the value in the neighbor list transform."
            )


class MultipleNeighborListsTransform(Transform):
    """
    Base class for computation of multiple neighbor lists using multiple cutoff values.
    Will store idx_i, idx_j, offsets, and absolute pairwise distances of neighbors for
    the largest provided cutoff and a list of indices to extract these properties for
    each provided cutoff.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self, cutoffs: List[float], nbh_names: List[str]):
        """
        Args:
            cutoffs: Multiple cutoff radii for neighbor search.
            nbh_names: Names of the neighborhoods that will be used to store the list
                of indices in the dictionary. The neighborhoods are defined by the
                cutoffs, i.e. the name in neighborhoods[i] corresponds to the
                neighborhood defined by the cutoff in cutoffs[i].
        """
        super().__init__()
        # sort cutoffs from large to small
        self._cutoffs, sorted_cutoffs_idx = torch.sort(
            torch.tensor(cutoffs), descending=True
        )
        # store names in the corresponding order
        self._nbh_names = [nbh_names[int(i)] for i in sorted_cutoffs_idx]
        self.additional_data = self._nbh_names

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        Z = inputs[properties.Z]
        R = inputs[properties.R]
        cell = inputs[properties.cell].view(3, 3)
        pbc = inputs[properties.pbc]

        # build neighbor list using largest cutoff
        cutoff = float(self._cutoffs[0])  # maximum cutoff (sorted in descending order)
        idx_i, idx_j, offset, dists = self._build_neighbor_list(Z, R, cell, pbc, cutoff)
        inputs[properties.idx_i] = idx_i.detach()
        inputs[properties.idx_j] = idx_j.detach()
        inputs[properties.offsets] = offset.detach()
        inputs[properties.r_ij] = dists.detach()

        # store lists of indices to idx_i, idx_j, offset, and r_ij for all cutoffs
        # leveraging the knowledge that they are descending
        inputs[self._nbh_names[0]] = torch.arange(len(idx_i))
        prev_pairs = inputs[self._nbh_names[0]]
        for cutoff, nbh_name in zip(self._cutoffs[1:], self._nbh_names[1:]):
            # it is sufficient to consider only the pairs within the previous cutoff
            # as the cutoffs are in descending order
            idcs = torch.nonzero(inputs[properties.r_ij][prev_pairs] <= cutoff)
            idcs = idcs.squeeze(-1)
            inputs[nbh_name] = prev_pairs[idcs]
            prev_pairs = inputs[nbh_name]

        return inputs

    def _build_neighbor_list(
        self,
        Z: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        cutoff: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Override with specific neighbor list implementation"""
        raise NotImplementedError


class MatScipyMultipleNeighborLists(MultipleNeighborListsTransform):
    """
    Neighborlist using the efficient implementation of the Matscipy package

    References:
        https://github.com/libAtoms/matscipy
    """

    def _build_neighbor_list(
        self, Z, positions, cell, pbc, cutoff, eps=1e-6, buffer=0.1
    ):
        at = Atoms(numbers=Z, positions=positions, cell=cell, pbc=pbc)

        # Add cell if none is present (if volume = 0)
        if at.cell.volume < eps:
            if len(Z) < 200:
                # for small structures, the simple, potentially quadratic neighbor search is faster
                # therefore we set a small dummy cell where many atoms are outside
                at.set_cell([1.0, 1.0, 1.0], scale_atoms=False)
            else:
                # for large structures, we compute a proper cell and make sure all atoms are inside
                # max values - min values along xyz augmented by small buffer for stability
                new_cell = np.ptp(at.positions, axis=0) + 0.1
                # set cell and center
                at.set_cell(new_cell, scale_atoms=False)
                at.center()

        # Compute neighborhood
        _idx_i, _idx_j, S, dists = msp_neighbor_list("ijSd", at, cutoff)
        # the results from matscipy are sorted by idx_i but the order of idx_j can be random
        # since ordered idx_j are needed for sampling of placement trajectories, we sort
        # them here
        # since idx_i and idx_j are symmetric and idx_i is already sorted,
        # we can simply sort idx_j in a stable manner and then switch idx_i and idx_j
        idx_j = torch.from_numpy(_idx_j.astype(int))
        _, order = torch.sort(idx_j, stable=True)
        idx_i = idx_j[order]
        idx_j = torch.from_numpy(_idx_i.astype(int))[order]
        # flip the shift vectors since we flipped i and j
        S = torch.from_numpy(-S).to(dtype=positions.dtype)[order]
        offset = torch.mm(S, cell)
        dists = torch.from_numpy(dists[order])
        return idx_i, idx_j, offset, dists


class ASEMultipleNeighborLists(MultipleNeighborListsTransform):
    """
    Calculate neighbor list using ASE.
    """

    def _build_neighbor_list(self, Z, positions, cell, pbc, cutoff):
        at = Atoms(numbers=Z, positions=positions, cell=cell, pbc=pbc)
        at.wrap()

        _idx_i, _idx_j, S, dists = neighbor_list(
            "ijSd", at, cutoff, self_interaction=False
        )
        # the results from ASE are sorted by idx_i but the order of idx_j can be random
        # since ordered idx_j are needed for sampling of placement trajectories, we sort
        # them here
        # first way: is to sort by idx_i primarily and use idx_j as the secondary
        # criterion with numpy.lexsort
        # order = lexsort(_idx_j, _idx_i)  # sort by idx_i (primary) and idx_j (second.)
        # idx_i = torch.from_numpy(_idx_i[order])
        # idx_j = torch.from_numpy(_idx_j[order])
        # S = torch.from_numpy(S[order]).to(dtype=positions.dtype)
        # second way: since idx_i and idx_j are symmetric and idx_i is already sorted,
        # we can simply sort idx_j in a stable manner and then switch idx_i and idx_j
        idx_j = torch.from_numpy(_idx_j)
        _, order = torch.sort(idx_j, stable=True)
        idx_i = idx_j[order]
        idx_j = torch.from_numpy(_idx_i)[order]
        # flip the shift vectors since we flipped i and j
        S = torch.from_numpy(-S).to(dtype=positions.dtype)[order]
        offset = torch.mm(S, cell)
        dists = torch.from_numpy(dists[order])
        return idx_i, idx_j, offset, dists


class TorchMultipleNeighborLists(MultipleNeighborListsTransform):
    """
    Environment provider making use of neighbor lists as implemented in TorchAni
    (https://github.com/aiqm/torchani/blob/master/torchani/aev.py).
    Supports cutoffs and PBCs and can be performed on either CPU or GPU.
    """

    def _build_neighbor_list(self, Z, positions, cell, pbc, cutoff):
        # Check if shifts are needed for periodic boundary conditions
        if torch.all(pbc == 0):
            shifts = torch.zeros(0, 3, device=cell.device, dtype=torch.long)
        else:
            shifts = self._get_shifts(cell, pbc, cutoff)
        idx_i, idx_j, offset, dists = self._get_neighbor_pairs(
            positions, cell, shifts, cutoff
        )

        # Create bidirectional id arrays, similar to what the ASE neighbor_list returns
        bi_idx_i = torch.cat((idx_i, idx_j), dim=0)
        bi_idx_j = torch.cat((idx_j, idx_i), dim=0)

        # Sort along second and first dimension (necessary for atom-wise pooling and
        # sampling of placement trajectories)
        sorted_idx_j = torch.argsort(bi_idx_j)
        _, sorted_idx_i = torch.sort(bi_idx_i[sorted_idx_j], stable=True)
        idx_i = bi_idx_i[sorted_idx_j][sorted_idx_i]
        idx_j = bi_idx_j[sorted_idx_j][sorted_idx_i]

        bi_offset = torch.cat((-offset, offset), dim=0)
        offset = bi_offset[sorted_idx_j][sorted_idx_i]
        offset = torch.mm(offset.to(cell.dtype), cell)

        bi_dists = torch.cat((dists, dists), dim=0)
        dists = bi_dists[sorted_idx_j][sorted_idx_i]

        return idx_i, idx_j, offset, dists

    def _get_neighbor_pairs(self, positions, cell, shifts, cutoff):
        """Compute pairs of atoms that are neighbors
        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)
        Arguments:
            positions (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
                defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            shifts (:class:`torch.Tensor`): tensor of shape (?, 3) storing shifts
        """
        num_atoms = positions.shape[0]
        all_atoms = torch.arange(num_atoms, device=cell.device)

        # 1) Central cell
        pi_center, pj_center = torch.combinations(all_atoms).unbind(-1)
        shifts_center = shifts.new_zeros(pi_center.shape[0], 3)

        # 2) cells with shifts
        # shape convention (shift index, molecule index, atom index, 3)
        num_shifts = shifts.shape[0]
        all_shifts = torch.arange(num_shifts, device=cell.device)
        shift_index, pi, pj = torch.cartesian_prod(
            all_shifts, all_atoms, all_atoms
        ).unbind(-1)
        shifts_outside = shifts.index_select(0, shift_index)

        # 3) combine results for all cells
        shifts_all = torch.cat([shifts_center, shifts_outside])
        pi_all = torch.cat([pi_center, pi])
        pj_all = torch.cat([pj_center, pj])

        # 4) Compute shifts and distance vectors
        shift_values = torch.mm(shifts_all.to(cell.dtype), cell)
        Rij_all = positions[pi_all] - positions[pj_all] + shift_values

        # 5) Compute distances, and find all pairs within cutoff
        distances = torch.norm(Rij_all, dim=1)
        in_cutoff = torch.nonzero(distances < cutoff, as_tuple=False)

        # 6) Reduce tensors to relevant components
        pair_index = in_cutoff.squeeze()
        atom_index_i = pi_all[pair_index]
        atom_index_j = pj_all[pair_index]
        offsets = shifts_all[pair_index]
        distances = distances[pair_index]

        return atom_index_i, atom_index_j, offsets, distances

    def _get_shifts(self, cell, pbc, cutoff):
        """Compute the shifts of unit cell along the given cell vectors to make it
        large enough to contain all pairs of neighbor atoms with PBC under
        consideration.
        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)
        Arguments:
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
            vectors defining unit cell:
            tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
                if pbc is enabled for that direction.
        Returns:
            :class:`torch.Tensor`: long tensor of shifts. the center cell and
                symmetric cells are not included.
        """
        reciprocal_cell = cell.inverse().t()
        inverse_lengths = torch.norm(reciprocal_cell, dim=1)

        num_repeats = torch.ceil(cutoff * inverse_lengths).long()
        num_repeats = torch.where(
            pbc, num_repeats, torch.Tensor([0], device=cell.device).long()
        )

        r1 = torch.arange(1, num_repeats[0] + 1, device=cell.device)
        r2 = torch.arange(1, num_repeats[1] + 1, device=cell.device)
        r3 = torch.arange(1, num_repeats[2] + 1, device=cell.device)
        o = torch.zeros(1, dtype=torch.long, device=cell.device)

        return torch.cat(
            [
                torch.cartesian_prod(r1, r2, r3),
                torch.cartesian_prod(r1, r2, o),
                torch.cartesian_prod(r1, r2, -r3),
                torch.cartesian_prod(r1, o, r3),
                torch.cartesian_prod(r1, o, o),
                torch.cartesian_prod(r1, o, -r3),
                torch.cartesian_prod(r1, -r2, r3),
                torch.cartesian_prod(r1, -r2, o),
                torch.cartesian_prod(r1, -r2, -r3),
                torch.cartesian_prod(o, r2, r3),
                torch.cartesian_prod(o, r2, o),
                torch.cartesian_prod(o, r2, -r3),
                torch.cartesian_prod(o, o, r3),
            ]
        )


class MultipleCountNeighbors(Transform):
    """
    Store the number of neighbors for each atom
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self, nbh_name_pairs: List[Tuple[str, str]], sorted: bool = True):
        """
        Args:
            nbh_name_pairs: Names of the neighborhoods and the corresponding counts
                that will be used to store the number of neighbors for each atom (per
                neighborhood)
            sorted: Set to false if chosen neighbor list yields unsorted center
                indices (idx_i).
        """
        super(MultipleCountNeighbors, self).__init__()
        self.nbh_name_pairs = nbh_name_pairs
        self.sorted = sorted

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        idx_i = inputs[properties.idx_i]

        if self.sorted:
            _, n_nbh = torch.unique_consecutive(idx_i, return_counts=True)
        else:
            n_nbh = torch.bincount(idx_i)
        inputs[properties.n_nbh] = n_nbh

        for nbh_name, count_name in self.nbh_name_pairs:
            _idcs = inputs[nbh_name]
            if len(_idcs) == len(idx_i):
                inputs[count_name] = inputs[properties.n_nbh]
            else:
                idx_i = inputs[properties.idx_i][inputs[nbh_name]]
                if self.sorted:
                    _, n_nbh = torch.unique_consecutive(idx_i, return_counts=True)
                else:
                    n_nbh = torch.bincount(idx_i)
                inputs[count_name] = n_nbh

        return inputs


class ConnectivityCheck(Transform):
    """
    Checks whether all atoms in the molecule are connected via some path given a
    cutoff value.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self, cutoff, use_covalent_radii, covalent_radius_factor, return_inputs=True
    ):
        """
        Args:
            cutoff: The cutoff used to compute neighbors of atoms in the molecule.
            use_covalent_radii: If True, pairs inside the cutoff will be further
                filtered using the covalent radii from ase. In this way, the cutoff
                is for example smaller for carbon-hydrogen pairs than for
                carbon-carbon pairs. Two atoms will be considered as neighbors if
                the distance between them is 1. smaller than `placement_cutoff` and
                2. smaller than the sum of the covalent radii of the two involved
                atom types times `covalent_radius_factor`.
            covalent_radius_factor: Allows control of the covalent radius criterion
                when assembling the placement neighborhood (see `use_covalent_radii`).
            return_inputs: If True, the result is a boolean torch tensor of length one
                and it is written into the inputs dictionary using the key "connected".
                If False, the result is returned as a simple boolean value.
        """
        super(ConnectivityCheck, self).__init__()
        self.cutoff = cutoff
        self.use_covalent_radii = use_covalent_radii
        self.covalent_radius_factor = covalent_radius_factor
        self.return_inputs = return_inputs

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # create ase.Atoms object
        at = Atoms(
            numbers=inputs[properties.Z],
            positions=inputs[properties.R],
            cell=inputs[properties.cell].view(3, 3),
            pbc=inputs[properties.pbc],
        )
        n_atoms = len(at.numbers)
        # Add cell if none is present (if volume = 0)
        if at.cell.volume < 1e-6:
            if n_atoms < 200:
                # for small structures, the simple, potentially quadratic neighbor search is faster
                # therefore we set a small dummy cell where many atoms are outside
                at.set_cell([1.0, 1.0, 1.0], scale_atoms=False)
            else:
                # for large structures, we compute a proper cell and make sure all atoms are inside
                # max values - min values along xyz augmented by small buffer for stability
                new_cell = np.ptp(at.positions, axis=0) + 0.1
                # set cell and center
                at.set_cell(new_cell, scale_atoms=False)
                at.center()

        # Compute neighborhood
        _idx_i, _idx_j, _r_ij = msp_neighbor_list("ijd", at, self.cutoff)
        if self.use_covalent_radii:
            thresh = (
                covalent_radii[at.numbers[_idx_i]] + covalent_radii[at.numbers[_idx_j]]
            )
            idcs = np.nonzero(_r_ij <= (thresh * self.covalent_radius_factor))[0]
            _idx_i = _idx_i[idcs]
            _idx_j = _idx_j[idcs]
        n_nbh = np.bincount(_idx_i, minlength=n_atoms)
        # check if there are atoms without neighbors, i.e. disconnected atoms
        if np.count_nonzero(n_nbh == 0) > 0:
            if self.return_inputs:
                inputs["connected"] = torch.tensor([False], dtype=torch.bool)
                return inputs
            else:
                return False
            # return False
        # store where the neighbors in _idx_j of each center atom in _idx_i start
        # assuming that _idx_i is ordered
        start_idcs = np.empty(n_nbh.size + 1, dtype=int)
        start_idcs[0] = 0
        start_idcs[1:] = np.cumsum(n_nbh)
        # check connectivity of atoms given the neighbor list
        unseen = np.ones(n_atoms, dtype=bool)
        unseen[0] = False
        count = 1
        queue = deque([0])
        while queue and count < n_atoms:
            atom = queue.popleft()
            neighbors = _idx_j[start_idcs[atom] : start_idcs[atom + 1]]
            for neighbor in neighbors:
                if unseen[neighbor]:
                    unseen[neighbor] = False
                    count += 1
                    queue.append(neighbor)
        connected = count == n_atoms  # molecule is connected if we saw all atoms
        if self.return_inputs:
            inputs["connected"] = torch.tensor([connected], dtype=torch.bool)
            return inputs
        else:
            return connected


def sort_j_parallel(n_nbhs, j):
    # sort the j indices block-wise (e.g. all j belonging to i=0, all belonging to
    # i=1 etc. in parallel)
    # n_nbhs is the number of neighbors per atom i
    # it is assumed that the first n_nbhs[0] j belong to i=0, the following n_nbhs[1]
    # entries in j belong to i=1 etc.
    max_entry = int(torch.amax(n_nbhs))
    # we create a temporary array where each row i will hold the j indices which are
    # neighbors to atom i since the number of j indices per row can be different, they
    # are padded with values larger than the largest i in this way, the padded values
    # will stay in the last columns of each row even after sorting
    temp = torch.full((len(n_nbhs), max_entry), len(n_nbhs), dtype=torch.long)
    # this mask tells us which values are actual j indices and which are padding
    mask = torch.arange(max_entry).reshape(1, -1) < n_nbhs.reshape(-1, 1)
    # fill temp array with j indices
    temp[mask] = j
    # sort rows
    s, idcs = torch.sort(temp, dim=1)
    # add offset to sorting indices (since these are local to each row)
    idcs[1:, :] += torch.cumsum(n_nbhs[:-1], dim=0).reshape(-1, 1)
    # return sorted j indices and the sorting indices
    return s[mask], idcs[mask]
