from typing import Dict, Optional, List, Tuple
import logging
import torch
from schnetpack.transform.base import Transform
from schnetpack_gschnet import properties

logger = logging.getLogger(__name__)

__all__ = [
    "BuildAtomsTrajectory",
    "BuildAtomsTrajectoryFromSubstructure",
]


class BuildAtomsTrajectory(Transform):
    """
    Takes a complete molecule, samples a random trajectory of atom placement steps for
    the structure, and then assembles the corresponding tensors of atom positions, atom
    types, neighborhoods of atoms, labels for the atom type and pairwise distances to
    predict etc. for all (or some randomly drawn) steps in the trajectory.
    It is assumed that the atoms in the complete molecule are ordered by their distance
    to the center of mass (use `OrderByDistanceToOrigin` transform before this one) and
    that the pairs `idx_i` and `idx_j` are ordered by `idx_i` (primary) and `idx_j`
    (secondary), i.e. the first neighbor j of i in `idx_j` is its neighbor closest to
    the origin (use e.g. `ConditionalGSchNetNeighborList` transform).
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        centered: bool = True,
        origin_type: int = 50,
        focus_type: int = 92,
        stop_type: int = 94,
        draw_random_samples: int = 0,
        sort_idx_i: bool = False,
    ):
        """
        Args:
            centered: Set True if the provided atom positions are centered (else they
                will be centered in this transform).
            origin_type: The atom type used to mark the origin token (needs to be
                distinct from actual atom types existing in the data).
            focus_type: The atom type used to mark the focus token (needs to be
                distinct from actual atom types existing in the data).
            stop_type: The atom type that shall be predicted by the model to mark an
                atom as finished, i.e. unavailable as focus (needs to be distinct from
                actual atom types existing in the data).
            draw_random_samples: Number of samples that are randomly drawn from the
                atom placement trajectory to train the model (set 0 to use all steps
                instead).
            sort_idx_i: Set true to sort the `idx_i` in the output of this transform in
                ascending order (otherwise they will be unordered).
        """
        super().__init__()
        self.centered = centered
        self.origin_type = origin_type
        self.focus_type = focus_type
        self.stop_type = stop_type
        self.draw_random_samples = draw_random_samples
        self.sort_idx_i = sort_idx_i
        # we have three lists that store different types of conditions
        # 1. the condition is shared by all atoms in the whole trajectory
        self.trajectory_conditions = []
        # 2. the conditions are shared by atoms of the same step (i.e. partial molecule)
        self.step_conditions = []
        # 3. each atom has its own condition
        self.atom_conditions = []

    def datamodule(self, value):
        pass

    def register_conditions(self, conditions):
        self.trajectory_conditions = conditions["trajectory"]
        self.step_conditions = conditions["step"]
        self.atom_conditions = conditions["atom"]

    def sample_atom_placement_trajectory(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # extract neighborhood information needed for atom placement from inputs
        # indices of the neighbors of the center atoms (inside placement cutoff)
        idx_j = inputs[properties.idx_j][inputs[properties.nbh_placement]]
        # number of neighbors for each atom i
        n_nbhs = inputs[properties.n_nbh_placement]
        # idx_i and idx_j are assumed to be ordered, first by idx_i, second by idx_j,
        # e.g.: idx_i = [0, 0, 1, 1, 1, 2, 3, 4]
        # and   idx_j = [1, 3, 0, 2, 4, 1, 0, 1]
        # for each atom i, store the index where its neighborhood starts (i.e. the
        # first entry where idx_i = i)
        start_idcs = (n_nbhs.cumsum(dim=0) - n_nbhs).long()
        # extract additional information
        n_atoms = int(inputs[properties.n_atoms])  # number of atoms in the molecule
        types = inputs[properties.Z]  # types of the atoms in the molecule

        # Store intermediate information (which atom is available as focus, which is
        # currently focused etc.). There are 2*n_atoms steps as we place each atom and
        # additionally need to mark each atom as finished
        # order in which atoms are placed
        order = torch.empty(n_atoms, dtype=torch.long)
        # mark atoms available as focus with 1 in this array
        avail = torch.zeros(n_atoms, dtype=torch.float)
        # mark atoms that have been placed with 0 in this array
        unplaced = torch.ones(n_atoms, dtype=torch.bool)
        # number of neighbors already placed for each atom
        n_nbhs_placed = torch.zeros(n_atoms, dtype=torch.long)
        # the atom focused at each step
        focus = torch.empty(n_atoms * 2, dtype=torch.long)
        # the type of the atom placed at each step
        pred_types = torch.empty(n_atoms * 2, dtype=torch.long)
        # number of atoms already placed
        n_placed = 0

        # start sampling of the atom placement trajectory
        # the first atom to be placed is always the one closest to the center of mass,
        # i.e. the first atom in the list since we assume they are ordered
        focus[0] = -1  # in the first step, there are no atoms that can be focused yet
        unplaced[0] = 0  # mark first atom as placed
        pred_types[0] = types[0]  # the type of the first atom
        order[0] = 0  # the first atom stays first in the new order
        avail[0] = 1  # for the following steps, the first atom is available as focus
        n_placed += 1  # we have placed only the first atom

        # now traverse the molecular graph to place remaining atoms (choosing the focus
        # randomly at each step)
        for i in range(1, n_atoms * 2):
            # choose new focus randomly
            focus[i] = torch.multinomial(avail, 1)[0]
            placed_atom = False

            # check if there are unplaced neighbors
            if n_nbhs_placed[focus[i]] < n_nbhs[focus[i]]:
                # n_nbhs_placed might be outdated, we need to check the neighbors
                # status in `unplaced`
                start_idx = start_idcs[focus[i]] + n_nbhs_placed[focus[i]]
                stop_idx = start_idcs[focus[i]] + n_nbhs[focus[i]]
                # extract potentially unplaced neighbors
                neighbor_idcs = idx_j[start_idx:stop_idx]
                # check status in "unplaced"
                unplaced_idcs = torch.nonzero(unplaced[neighbor_idcs])
                if len(unplaced_idcs) > 0:
                    # place the first unplaced neighbor
                    # get index of the first unplaced neighbor
                    next_atom = neighbor_idcs[unplaced_idcs[0][0]]
                    unplaced[next_atom] = 0  # mark as placed
                    pred_types[i] = types[next_atom]  # fill in type of the next atom
                    order[n_placed] = next_atom  # add next atom to the order
                    avail[next_atom] = 1  # mark next atom as available as focus
                    n_placed += 1  # increase the count of placed atoms
                    # update number of placed neighbors for focus since we might have
                    # skipped some previously placed neighbors from `neighbor_idcs`
                    # and we placed an unplaced neighbor
                    n_nbhs_placed[focus[i]] += unplaced_idcs[0][0] + 1
                    placed_atom = True
                else:
                    # all neighbors were placed, update number of placed neighbors
                    # for focus accordingly
                    n_nbhs_placed[focus[i]] = n_nbhs[focus[i]]

            if n_nbhs_placed[focus[i]] == n_nbhs[focus[i]] and not placed_atom:
                # all neighbors were already placed
                # fill in stop type as type to be predicted in this step
                pred_types[i] = self.stop_type
                # remove current focus from `avail` so it cannot be chosen again
                avail[focus[i]] = 0

        return order, focus, pred_types

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # center positions if they were not centered before
        if not self.centered:
            _mean = torch.mean(inputs[properties.R], dim=0, keepdim=True)
            inputs[properties.R] = inputs[properties.R] - _mean

        # import time
        # start_time = time.time()
        # sample atom placement trajectory (the new atom order and the focus and type
        # to predict at each step)
        order, focus, pred_types = self.sample_atom_placement_trajectory(inputs)
        # first_stop = time.time()-start_time

        # extract information from the complete molecule
        R = inputs[properties.R]
        Z = inputs[properties.Z]
        n_atoms = int(inputs[properties.n_atoms])
        idx_j = inputs[properties.idx_j]
        n_nbh = inputs[properties.n_nbh]
        r_ij = inputs[properties.r_ij]  # pairwise distances
        center_dists = torch.linalg.norm(R, dim=1)  # distances of atoms to the origin
        offsets = inputs[properties.offsets]
        # store where the neighbors of atom i start in ixd_j
        start_idcs = torch.cat((torch.zeros(1, dtype=torch.long), n_nbh.cumsum(dim=0)))
        # build masks that mark neighbors j inside model and prediction cutoff
        nbh_model_mask = torch.zeros(idx_j.size(), dtype=torch.bool)
        nbh_model_mask[inputs[properties.nbh_model]] = True
        nbh_prediction_mask = torch.zeros(idx_j.size(), dtype=torch.bool)
        nbh_prediction_mask[inputs[properties.nbh_prediction]] = True
        # store the new index of each atom i in the sampled placement order
        new_idcs = torch.empty(n_atoms, dtype=torch.long)
        new_idcs[order] = torch.arange(n_atoms)
        # store positions and types of atoms and tokens in the order they are placed
        # in the sampled trajectory
        R_ordered = torch.cat((torch.zeros(2, 3), R[order]), dim=0)
        Z_ordered = torch.cat(
            (
                torch.tensor([self.focus_type, self.origin_type], dtype=torch.long),
                Z[order],
            ),
            dim=0,
        )

        # randomly draw which steps from the trajectory are used for training
        random_steps = set()
        if self.draw_random_samples > 0:
            if n_atoms * 2 > self.draw_random_samples:
                random_steps = torch.multinomial(
                    torch.ones(n_atoms * 2), self.draw_random_samples, replacement=False
                )
                ordered_steps = random_steps.sort()[0]
                random_steps = set(random_steps.numpy())
            else:
                # the molecule is smaller than the number of steps we want to draw
                # therefore we can just use all steps
                ordered_steps = torch.arange(n_atoms * 2)
                random_steps = set(ordered_steps.numpy())

        # now that we have a complete molecule, a trajectory in which it can be built
        # and a list of steps in this trajectory the we want to use for training, we
        # need to build the data structures that hold the atoms at each step, their
        # types, their neighbors, the origin and focus tokens etc. (i.e. everything
        # required by the model to predict the next atom at each step)

        # initialize tensors which will be the new entries in inputs (e.g. the atoms
        # of all steps, their types etc.)
        # these will be filled successively in the following
        new_R = []  # positions
        new_Z = []  # types
        new_idx_i = []  # local center atom index i
        new_idx_j = []  # local neighbor atom index j
        new_r_ij = []  # distances between center and neighbor atoms i, j
        new_offsets = []  # offsets of neighbor atom j
        new_n_atoms = []  # global molecule index
        n_idx_i = []  # number of idcs in new_idx_i per step
        idx_i_offsets = []  # the offset of idcs in new_idx_i per step
        # at each step, the focus and its neighbors inside the prediction cutoff are
        # used to predict the type of the next atom and the pairwise distances to it
        # we store the required information as following
        pred_j = []  # indices of focus and its neighbors inside prediction cutoff
        pred_idx_m = []  # molecule index for atoms in pred_j
        n_pred_nbh = []  # number of atoms used for prediction in each step
        pred_dists = [torch.empty(0)]  # ground truth distances
        # the distances are only predicted in steps where the next type is not the stop
        # type, therefore only those entries from pred_j are needed
        pred_dists_idx = [torch.empty(0, dtype=torch.long)]  # indexes entries in pred_j
        n_pred_nbh_dists = []  # number of atoms for prediction of distances per step
        n_pred_j = 0  # overall number of idcs in pred_j

        # initialize new idx_i, idx_j etc. that hold the environment of atoms at each
        # step of the sampled atom placement trajectory (these can be build successively
        # when walking the trajectory step by step)
        # we separate the pairs that involve the focus token and those without the
        # focus token, without focus (can be gathered since they do not change):
        gathered_idx_i = []
        gathered_idx_j = []
        gathered_r_ij = []  # distances between atoms for all pairs without focus token
        gathered_offsets = []  # cell offsets for gathered_idx_j
        n_gathered = 0  # keep track of number of entries in gathered_idx_i etc.
        # with focus (replaced each step as the focus changes each step):
        # at the first step, start with the focus and origin at (0, 0, 0))
        # local atom index i (focus and origin are neighbors)
        focus_idx_i = torch.tensor([0, 1], dtype=torch.long)
        # local atom index j (focus and origin are neighbors)
        focus_idx_j = torch.tensor([1, 0], dtype=torch.long)
        # distances (between origin and focus)
        focus_r_ij = torch.zeros(2, dtype=torch.float)
        # cell offsets for focus_idx_j
        focus_offsets = torch.zeros((2, 3), dtype=torch.float)

        # if the placement of the first atom is part of the training data, put this step
        # into the data structures
        if self.draw_random_samples <= 0 or 0 in random_steps:
            new_R += [R_ordered[:2]]
            new_Z += [Z_ordered[:2]]
            new_idx_i += [focus_idx_i]
            new_idx_j += [focus_idx_j]
            new_r_ij += [focus_r_ij]
            new_offsets += [focus_offsets]
            new_n_atoms += [2]
            n_idx_i += [2]
            idx_i_offsets += [0]
            pred_j += [torch.tensor([0, 1], dtype=torch.long)]
            pred_idx_m += [torch.zeros(2, dtype=torch.long)]
            n_pred_nbh += [2]
            pred_dists += [torch.tensor([center_dists[order[0]]]).repeat(2)]
            pred_dists_idx += [torch.arange(2)]
            n_pred_nbh_dists += [2]
            n_pred_j += 2

        next_atom = order[0]  # used to store the atom placed in each step
        n_placed = 1  # number of atoms that have been placed so far
        cur_idx_i_offset = sum(new_n_atoms)  # sum of atoms in new_R (offset for idx_i)

        # build batch with steps from the placement trajectory
        # each atom needs to be placed and marked as finished, therefore we have
        # 2*n_atoms steps per trajectory
        for i in range(1, 2 * n_atoms):
            # 1. assemble neighborhoods for the current step
            # if a new atom was placed in the last step, add it to the neighborhood
            if next_atom != -1:
                start = start_idcs[next_atom]
                stop = start_idcs[next_atom + 1]
                next_atom_j = idx_j[start:stop]  # neighbors of the next atom
                # find the neighbors that were already placed (i.e. neighbors with
                # new index < number of atoms placed) and keep only those within the
                # model cutoff (add 2 to the indices since we prepend focus and origin)
                # neighbors of the next atom in new order
                new_next_atom_j = new_idcs[next_atom_j]
                # neighbors already placed
                placed_idcs = torch.where(new_next_atom_j < n_placed)[0] + start
                # placed neighbors inside the model cutoff
                placed_idcs = placed_idcs[nbh_model_mask[placed_idcs]]
                # for idx_j, extract the indices of those neighbors (in the new order)
                new_next_atom_j = new_idcs[idx_j[placed_idcs]] + 2
                # for idx_i, use the index of next atom (in the new order)
                new_next_atom_i = torch.full_like(
                    new_next_atom_j, fill_value=new_idcs[next_atom] + 2
                )
                # store distances of newly added i, j pairs
                next_r_ij = r_ij[placed_idcs]
                # store offsets of j in the newly added i, j pairs
                next_offsets = offsets[placed_idcs]
                # add connection between next atom and origin (which is connected to
                # all atoms in this implementation)
                origin_idx_i = torch.tensor(
                    [1, new_idcs[next_atom] + 2], dtype=torch.long
                )
                origin_idx_j = torch.tensor(
                    [new_idcs[next_atom] + 2, 1], dtype=torch.long
                )
                origin_r_ij = center_dists[next_atom].repeat(2)
                origin_offsets = torch.zeros((2, 3), dtype=torch.float)
                # stack everything (including i and j inverted to get symmetric list)
                gathered_idx_i += [new_next_atom_i, new_next_atom_j, origin_idx_i]
                gathered_idx_j += [new_next_atom_j, new_next_atom_i, origin_idx_j]
                gathered_r_ij += [next_r_ij.repeat(2), origin_r_ij]
                gathered_offsets += [next_offsets, -next_offsets, origin_offsets]
                n_gathered += (
                    len(new_next_atom_i) + len(new_next_atom_j) + len(origin_idx_i)
                )

            # add focus to neighborhood
            cur_focus = focus[i]  # focus in this step
            start = start_idcs[cur_focus]
            stop = start_idcs[cur_focus + 1]
            focus_atom_j = idx_j[start:stop]  # neighbors of the focus
            # find the neighbors that were already placed (i.e. neighbors with
            # new index < number of atoms placed)
            # neighbors of the focus atom in new order
            new_focus_atom_j = new_idcs[focus_atom_j]
            # neighbors already placed
            placed_idcs = torch.where(new_focus_atom_j < n_placed)[0] + start
            # keep the placed neighbors which are inside the model cutoff and which
            # are inside the prediction cutoff
            # inside model cutoff
            placed_idcs_model = placed_idcs[nbh_model_mask[placed_idcs]]
            # placed neighbors of focus inside model cutoff
            new_focus_atom_j = new_idcs[idx_j[placed_idcs_model]] + 2
            # new index of focus token
            new_focus_atom_i = torch.zeros_like(new_focus_atom_j)
            # distances of newly added i, j pairs
            focus_r_ij = r_ij[placed_idcs_model]
            # offsets of j in the newly added i, j pairs
            focus_offsets = offsets[placed_idcs_model]
            # inside prediction cutoff
            placed_idcs_prediction = placed_idcs[nbh_prediction_mask[placed_idcs]]
            # neighbors of focus inside prediction cutoff
            new_pred_j = new_idcs[idx_j[placed_idcs_prediction]] + 2
            # origin and focus token/atom
            focus_pred_j = torch.tensor(
                [0, 1, new_idcs[cur_focus] + 2], dtype=torch.long
            )
            # add connection between focus and origin (which is connected to all atoms
            # in this implementation) and between focus token and focus atom
            origin_idx_i = torch.tensor([1, 0, new_idcs[cur_focus] + 2, 0])
            origin_idx_j = torch.tensor([0, 1, 0, new_idcs[cur_focus] + 2])
            origin_r_ij = torch.cat([center_dists[cur_focus].repeat(2), torch.zeros(2)])
            origin_offsets = torch.zeros((4, 3), dtype=torch.float)
            # stack everything (including i and j inverted to get symmetric list)
            focus_idx_i = [new_focus_atom_i, new_focus_atom_j, origin_idx_i]
            focus_idx_j = [new_focus_atom_j, new_focus_atom_i, origin_idx_j]
            focus_r_ij = [focus_r_ij.repeat(2), origin_r_ij]
            focus_offsets = [focus_offsets, -focus_offsets, origin_offsets]
            # print(f"focus: {new_idcs[cur_focus]+2}")
            # print(f"idx_i: {focus_idx_i}")
            # print(f"gathered_idx_i: {gathered_idx_i}")

            # 2. check whether an atom is placed in this step
            if pred_types[i] != self.stop_type:
                next_atom = order[n_placed]
            else:
                next_atom = -1

            # 3. prepare data of this step for inputs dictionary (if the step shall
            # be used in the training)
            if self.draw_random_samples <= 0 or i in random_steps:
                n_steps = len(new_n_atoms)  # number of steps before adding this one
                new_R += [R[None, cur_focus], R_ordered[1 : n_placed + 2]]
                new_Z += [Z_ordered[: n_placed + 2]]
                if self.sort_idx_i:
                    sorted_idx_i, sorted_order = torch.cat(
                        focus_idx_i + gathered_idx_i, dim=0
                    ).sort()
                    new_idx_i += [sorted_idx_i]
                    new_idx_j += [
                        torch.cat(focus_idx_j + gathered_idx_j, dim=0)[sorted_order]
                    ]
                    new_r_ij += [
                        torch.cat(focus_r_ij + gathered_r_ij, dim=0)[sorted_order]
                    ]
                    new_offsets += [
                        torch.cat(focus_offsets + gathered_offsets, dim=0)[sorted_order]
                    ]
                else:
                    new_idx_i += focus_idx_i + gathered_idx_i
                    new_idx_j += focus_idx_j + gathered_idx_j
                    new_r_ij += focus_r_ij + gathered_r_ij
                    new_offsets += focus_offsets + gathered_offsets
                idx_i_offsets += [cur_idx_i_offset]
                n_idx_i += [
                    len(focus_idx_i[0])
                    + len(focus_idx_i[1])
                    + len(focus_idx_i[2])
                    + n_gathered
                ]
                new_n_atoms += [n_placed + 2]
                pred_j += [
                    focus_pred_j + cur_idx_i_offset,
                    new_pred_j + cur_idx_i_offset,
                ]
                n_prediction_atoms = len(focus_pred_j) + len(new_pred_j)
                pred_idx_m += [
                    torch.full((n_prediction_atoms,), n_steps, dtype=torch.long)
                ]
                n_pred_nbh += [n_prediction_atoms]
                if next_atom != -1:
                    # compute and store distances between new atom and the atoms
                    # used for prediction
                    new_dists = torch.linalg.norm(
                        R[None, next_atom] - R_ordered[new_pred_j], dim=1
                    )
                    focus_dist = torch.linalg.norm(R[next_atom] - R[cur_focus])
                    origin_dist = center_dists[next_atom]
                    pred_dists += [
                        torch.tensor([focus_dist, origin_dist, focus_dist]),
                        new_dists,
                    ]
                    # update the indices in pred_dists_idx accordingly
                    pred_dists_idx += [
                        torch.arange(n_pred_j, n_pred_j + n_prediction_atoms)
                    ]
                    n_pred_nbh_dists += [len(pred_dists_idx[-1])]
                # update counts of indices in pred_j and in idx_i
                n_pred_j += n_prediction_atoms
                cur_idx_i_offset += new_n_atoms[-1]

            # 4. increase count of placed atoms if an atom is placed in this step
            if next_atom != -1:
                n_placed += 1

        # second_stop = time.time()-start_time-first_stop
        # write the assembled data of the trajectory into inputs dictionary
        inputs[properties.R] = torch.cat(new_R, dim=0)
        inputs[properties.Z] = torch.cat(new_Z, dim=0)
        inputs[properties.n_atoms] = torch.tensor(new_n_atoms, dtype=torch.long)
        idx_i_offsets = torch.repeat_interleave(
            torch.tensor(idx_i_offsets, dtype=torch.long), torch.tensor(n_idx_i)
        )
        inputs[properties.idx_i] = torch.cat(new_idx_i, dim=0) + idx_i_offsets
        inputs[properties.idx_j] = torch.cat(new_idx_j, dim=0) + idx_i_offsets
        inputs[properties.r_ij] = torch.cat(new_r_ij, dim=0)
        inputs[properties.offsets] = torch.cat(new_offsets)
        inputs[properties.pred_idx_j] = torch.cat(pred_j, dim=0)
        inputs[properties.pred_idx_m] = torch.cat(pred_idx_m, dim=0)
        inputs[properties.n_pred_nbh] = torch.tensor(n_pred_nbh, dtype=torch.long)
        inputs[properties.pred_r_ij] = torch.cat(pred_dists, dim=0)
        inputs[properties.pred_r_ij_idcs] = torch.cat(pred_dists_idx, dim=0)
        if self.draw_random_samples <= 0:
            # all prediction steps included
            inputs[properties.pred_Z] = pred_types
        else:
            # only a few prediction steps included
            inputs[properties.pred_Z] = pred_types[ordered_steps]
        inputs[properties.next_Z] = torch.repeat_interleave(
            inputs[properties.pred_Z][inputs[properties.pred_Z] != self.stop_type],
            torch.tensor(n_pred_nbh_dists, dtype=torch.long),
        )
        # store updated n_nbh
        if self.sort_idx_i:
            inputs[properties.n_nbh] = torch.unique_consecutive(
                inputs[properties.idx_i], return_counts=True
            )[1]
        else:
            inputs[properties.n_nbh] = torch.bincount(inputs[properties.idx_i])
        # repeat remaining molecule-wise entries in inputs for each step
        if self.draw_random_samples != 1:
            n_steps = len(new_n_atoms)
            for key in [properties.cell, properties.idx]:
                if key in inputs:
                    s = inputs[key].size()
                    inputs[key] = inputs[key].expand(n_steps, *s).squeeze()
        # repeat conditions such that they can be embedded
        if len(inputs[properties.pred_idx_j]) > 1:
            # repeat trajectory-wise conditions for each atom in the trajectory
            for cond in self.trajectory_conditions:
                s = inputs[cond].size()
                inputs[cond] = inputs[cond].expand(n_pred_j, *s).squeeze()
        # remove outdated/unused information from inputs
        for key in [
            properties.pbc,
            properties.nbh_prediction,
            properties.nbh_model,
            properties.nbh_placement,
            properties.n_nbh_prediction,
            properties.n_nbh_model,
            properties.n_nbh_placement,
        ]:
            if key in inputs:
                inputs.pop(key)

        # last_stop = time.time()-start_time-first_stop-second_stop
        # print('-------------')
        # print(f'Needed {first_stop:.5f}+{second_stop:.5f}+{last_stop:.5f}='
        # f'{first_stop+second_stop+last_stop:.5f}s')
        return inputs


class BuildAtomsTrajectoryFromSubstructure(Transform):
    """
    Takes a complete molecule, samples a random trajectory of atom placement steps
    starting from a fixed substructure, and then assembles the corresponding tensors
    of atom positions, atom types, neighborhoods of atoms, labels for the atom type
    and pairwise distances to predict etc. for all (or some randomly drawn) steps in
    the trajectory.
    It is assumed that the pairs `idx_i` and `idx_j` are ordered by `idx_i` (use
    e.g. `ConditionalGSchNetNeighborList` transform).
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
            self,
            focus_type: int = 92,
            stop_type: int = 94,
            draw_random_samples: int = 0,
            sort_idx_i: bool = False,
            mark_substructure_as_finished: bool = True,
    ):
        """
        Args:
            origin_type: The atom type used to mark the origin token (needs to be
                distinct from actual atom types existing in the data).
            focus_type: The atom type used to mark the focus token (needs to be
                distinct from actual atom types existing in the data).
            stop_type: The atom type that shall be predicted by the model to mark an
                atom as finished, i.e. unavailable as focus (needs to be distinct from
                actual atom types existing in the data).
            draw_random_samples: Number of samples that are randomly drawn from the
                atom placement trajectory to train the model (set 0 to use all steps
                instead).
            sort_idx_i: Set true to sort the `idx_i` in the output of this transform in
                ascending order (otherwise they will be unordered).
            mark_substructure_as_finished: If true, atoms inside the substructure that
                are not connected to any atoms outside of the substructure are.
                automatically marked as finished, i.e. the model will not use them as
                training data to learn to predict stop.
        """
        super().__init__()
        self.focus_type = focus_type
        self.stop_type = stop_type
        self.draw_random_samples = draw_random_samples
        self.sort_idx_i = sort_idx_i
        self.mark_substructure_as_finished = mark_substructure_as_finished
        # we have three lists that store different types of conditions
        # 1. the condition is shared by all atoms in the whole trajectory
        self.trajectory_conditions = []
        # 2. the conditions are shared by atoms of the same step (i.e. partial molecule)
        self.step_conditions = []
        # 3. each atom has its own condition
        self.atom_conditions = []

    def datamodule(self, value):
        pass

    def register_conditions(self, conditions):
        self.trajectory_conditions = conditions["trajectory"]
        self.step_conditions = conditions["step"]
        self.atom_conditions = conditions["atom"]

    def extract_substructure_neighborhood(
            self,
            n_atoms: int,
            substructure_idcs: torch.Tensor,
            idx_i: torch.Tensor,
            idx_j: torch.Tensor,
            return_complement: Optional[bool] = False,
    ):
        """
        Extracts the neighborhood (i.e. center atoms idx_i, neighbors idx_j, and the
        number of neighbors per center atom n_nbhs) of all atoms that are inside a
        given substructure.
        Can also be used to extract the complementary neighborhood (i.e. all the
        remaining neighborhood pairs which are not completely contained inside the
        substructure).

        Args:
            substructure_idcs: Indices of atoms that are part of the substructure.
                If the input is a torch.Tensor of dtype torch.bool, it is interpreted
                as a mask that marks atoms inside the substructure with True.
            idx_i: Neighborlist with center atoms.
            idx_j: Neighborlist with neighbor atoms.
            return_complement: Set true to return the complementary neighborhood,
                i.e. the remaining pairs not completely contained inside the
                provided substructure.

        Returns:
            Center atoms idx_i, neighbors idx_j, and number of neighbors of center
            atoms n_nbhs for the extracted substructure (or its complement).
        """
        if substructure_idcs.dtype != torch.bool:
            # make a mask that marks atoms that are part of the substructure
            in_substructure = torch.zeros(n_atoms, dtype=torch.bool)
            in_substructure[substructure_idcs] = 1
        else:
            # mask was provided as input
            in_substructure = substructure_idcs
        # mark which atoms in idx_i and idx_j are part of the substructure
        idx_i_mask = in_substructure[idx_i]
        idx_j_mask = in_substructure[idx_j]
        # check which pairs are completely inside the substructure
        remaining_pairs = torch.logical_and(idx_i_mask, idx_j_mask)
        if return_complement:
            # take the complement, i.e. pairs that are not completely inside
            remaining_pairs = ~remaining_pairs
        # only keep the remaining pairs
        idx_i = idx_i[remaining_pairs]
        idx_j = idx_j[remaining_pairs]
        # count number of neighbors per center atom
        # assumes that idx_i is sorted!
        n_nbhs = torch.zeros(n_atoms, dtype=torch.long)
        idcs, counts = torch.unique_consecutive(idx_i, return_counts=True)
        n_nbhs[idcs] = counts
        return idx_i, idx_j, n_nbhs

    def sample_atom_placement_trajectory(
            self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # number of atoms in the molecule
        n_atoms = int(inputs[properties.n_atoms])
        # types of the atoms in the molecule
        types = inputs[properties.Z]
        # indices of atoms that are inside the pre-defined substructure
        substructure_idcs = inputs[properties.substructure_idcs]
        if len(substructure_idcs) > 0:
            # Only keep parts of the neighborhood that are not completely contained
            # inside the pre-defined substructure. Only these atoms shall be part
            # of the sampled atom placement trajectory
            idx_i, idx_j, n_nbhs =  self.extract_substructure_neighborhood(
                n_atoms,
                substructure_idcs,
                inputs[properties.idx_i][inputs[properties.nbh_placement]],
                inputs[properties.idx_j][inputs[properties.nbh_placement]],
                return_complement=True,
            )
        else:
            # the substructure is empty, we can take the full neighborhood
            # indices of the center atoms
            idx_i = inputs[properties.idx_i][inputs[properties.nbh_placement]]
            # indices of the neighbors of the center atoms (inside placement cutoff)
            idx_j = inputs[properties.idx_j][inputs[properties.nbh_placement]]
            # number of neighbors per center atom
            n_nbhs = inputs[properties.n_nbh_placement]

        # There are up to 2*n_atoms steps in the trajectory as we place each atom and
        # additionally need to mark each atom as finished. This number is reduced by
        # the number of atoms in the substructure (as these are already placed) and
        # potentially the number of atoms inside the substructure that do not have
        # any neighbors outside (if we decide to mark these automatically as
        # finished).
        n_steps = (n_atoms * 2) - len(substructure_idcs)
        if self.mark_substructure_as_finished:
            finished_in_substructure = n_nbhs==0
            n_steps -= torch.sum(finished_in_substructure)
        # draw which of the steps shall be predicted by the model
        if (self.draw_random_samples > 0 and
                self.draw_random_samples < n_steps):
            # draw randomly
            chosen_steps, _ = torch.multinomial(
                torch.ones(n_steps),
                self.draw_random_samples,
                replacement=False,
            ).sort()
        else:
            # use all steps
            chosen_steps = torch.arange(n_steps)
        chosen_counter = 0  # will store which step we have processed last

        # Store intermediate information (which atom is available as focus, which is
        # currently focused etc.).
        # order in which atoms are placed
        order = torch.empty(n_atoms, dtype=torch.long)
        # mark atoms available as focus with 1 in this array
        avail = torch.zeros(n_atoms, dtype=torch.float)
        # mark atoms that have been placed with 0 in this array
        unplaced = torch.ones(n_atoms, dtype=torch.bool)
        # number of neighbors already placed for each atom
        n_nbhs_placed = torch.zeros(n_atoms, dtype=torch.long)
        # the atom focused at each step chosen for prediction
        foci = torch.empty(len(chosen_steps), dtype=torch.long)
        # the type of the atom placed at each step chosen for prediction
        pred_types = torch.empty(len(chosen_steps), dtype=torch.long)
        # the number of atoms already placed at each step chosen for prediction
        n_placed_so_far = torch.empty(len(chosen_steps), dtype=torch.long)

        # start sampling of the atom placement trajectory
        if len(substructure_idcs) > 0:
            # note atoms in the substructure as already placed etc.
            n_placed = len(substructure_idcs)  # substructure is already placed
            order[:n_placed] = substructure_idcs  # add substructure to order
            avail[substructure_idcs] = 1  # atoms in substructure are available
            if self.mark_substructure_as_finished:
                # remove atoms that do not have neighbors outside of substructure
                avail[finished_in_substructure] = 0
            unplaced[substructure_idcs] = 0  # mark atoms in substructure as placed
            start_step = 0  # we did no step in the trajectory, loop from 0
        else:
            # if the there is no pre-defined substructure, randomly place a first atom
            first_atom = torch.randint(0, n_atoms, [1])[0]
            n_placed = 1  # we have placed only the first atom
            order[0] = first_atom  # add the first atom to order
            avail[first_atom] = 1  # the first atom is now available as focus
            unplaced[first_atom] = 0  # mark first atom as placed
            start_step = 1  # we already did one step in the trajectory, loop from 1
            if chosen_steps[0] == 0:
                # if the first step has been chosen to be predicted, add info
                foci[0] = -1  # add dummy for "no focus" at first step
                pred_types[0] = types[first_atom]  # add type of the first atom
                n_placed_so_far[0] = 0  # there were no atoms placed before
                chosen_counter += 1  # increase the counter
        # for each center atom i, store the index where its neighborhood starts
        start_idcs = (n_nbhs.cumsum(dim=0) - n_nbhs).long()

        # now traverse the molecular graph to place remaining atoms (choosing the focus
        # randomly at each step)
        for i in range(start_step, n_steps):

            # end loop if all steps have been sampled
            if chosen_counter >= len(chosen_steps):
                break

            # choose new focus randomly
            focus = torch.multinomial(avail, 1)[0]
            placed_atom = False

            # check if there are unplaced neighbors
            if n_nbhs_placed[focus] < n_nbhs[focus]:
                # n_nbhs_placed might be outdated, we need to check the neighbors
                # status in `unplaced`
                start_idx = start_idcs[focus]
                stop_idx = start_idcs[focus] + n_nbhs[focus]
                # extract potentially unplaced neighbors
                neighbor_idcs = idx_j[start_idx:stop_idx]
                # check status in "unplaced"
                unplaced_idcs = torch.nonzero(unplaced[neighbor_idcs])
                if len(unplaced_idcs) > 0:
                    # place a random unplaced neighbor
                    random_idx = torch.randint(0, len(unplaced_idcs), [1])[0]
                    next_atom = neighbor_idcs[unplaced_idcs[random_idx]]
                    unplaced[next_atom] = 0  # mark as placed
                    avail[next_atom] = 1  # mark next atom as available as focus
                    order[n_placed] = next_atom  # add next atom to the order
                    # update number of placed neighbors for focus since we might have
                    # skipped some previously placed neighbors from `neighbor_idcs`
                    # and we placed an unplaced neighbor
                    n_nbhs_placed[focus] = (len(neighbor_idcs) -
                                             len(unplaced_idcs) + 1)
                    # store type of the next atom
                    pred_type = types[next_atom]
                    placed_atom = True
                else:
                    # all neighbors were placed, update number of placed neighbors
                    # for focus accordingly
                    n_nbhs_placed[focus] = n_nbhs[focus]

            if n_nbhs_placed[focus] == n_nbhs[focus] and not placed_atom:
                # all neighbors were already placed
                # store stop type as type to be predicted in this step
                pred_type = self.stop_type
                # remove current focus from `avail` so it cannot be chosen again
                avail[focus] = 0

            if i == chosen_steps[chosen_counter]:
                # if the current step has been chosen to be predicted
                # add the focus to the list of foci
                foci[chosen_counter] = focus
                # add the type to be predicted to the list
                pred_types[chosen_counter] = pred_type
                # store how many atoms were placed before this step
                n_placed_so_far[chosen_counter] = n_placed
                # and increase the counter
                chosen_counter += 1

            if placed_atom:
                # increase the count of placed atoms
                n_placed += 1

        # make sure that all atoms are included in order
        if n_placed < n_atoms:
            missing_atoms = torch.nonzero(unplaced)[:, 0]
            order[n_placed:] = missing_atoms

        return order, foci, pred_types, n_placed_so_far

    def forward(
            self,
            inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # prepare tokens
        if self.focus_type is not None:
            token_types = torch.tensor([self.focus_type], dtype=torch.long)
        else:
            token_types = torch.tensor([], dtype=torch.long)
        n_tokens = len(token_types)

        # sample atom placement trajectory (the new atom order, the focus, the type
        # to predict at each step, and the number of atoms available for prediction)
        order, focus, pred_types, n_placed_so_far = \
            self.sample_atom_placement_trajectory(inputs)
        n_steps = len(focus)  # number of steps to be predicted from the trajectory

        # store the new index of each atom given by the sampled trajectory
        n_atoms = int(inputs[properties.n_atoms])
        new_idcs = torch.zeros(n_atoms, dtype=torch.long)
        new_idcs[order] = torch.arange(n_atoms)

        # store focus at each step using the new indices of atoms
        focus_old_index = focus
        focus = new_idcs[focus]
        if focus_old_index[0] == -1:
            # we have a very first step without any atoms placed
            focus[0] = -1

        # load positions and types of atoms and tokens in the order they are placed
        # in the sampled trajectory (tokens are prepended as first entries)
        R = torch.cat(
            (
                torch.zeros(n_tokens, 3),
                inputs[properties.R][order],
            ),
            dim=0,
        )
        Z = torch.cat(
            (
                token_types[:n_tokens],
                inputs[properties.Z][order],
            ),
            dim=0,
        )
        # since we prepend the tokens, the number of atoms and the indices in order
        # etc. have to be shifted accordingly
        n_atoms += n_tokens  # we have n_tokens more atoms in the structure
        focus += n_tokens  # each index is shifted by n_tokens
        new_idcs += n_tokens  # each index is shifted by n_tokens
        n_placed_so_far += n_tokens  # we have placed n_tokens more atoms

        # Sort neighborhood list by ascending i (with respect to new atom order):
        n_nbh = inputs[properties.n_nbh]
        start_idcs = n_nbh.cumsum(dim=0)-n_nbh
        nbh_list_order = torch.cat(
            [torch.arange(start_idcs[i], start_idcs[i]+n_nbh[i]) for i in order],
            dim=0,
        )
        # load neighborlist using the new atom indicies from the sampled order
        idx_i = new_idcs[inputs[properties.idx_i]][nbh_list_order]
        idx_j = new_idcs[inputs[properties.idx_j]][nbh_list_order]
        r_ij = inputs[properties.r_ij][nbh_list_order]
        offsets = inputs[properties.offsets][nbh_list_order]
        # build n_nbh with correct order and 0 as first entry for token
        n_nbh = torch.cat([torch.zeros(n_tokens, dtype=torch.long), n_nbh[order]])
        start_idcs = n_nbh.cumsum(dim=0)-n_nbh
        start_idcs = torch.cat([start_idcs, start_idcs[-1:]+n_nbh[-1:]], dim=0)

        # load information about cutoff-specific neighborhoods
        extract_nbh_model = False
        extract_nbh_prediction = False
        if len(inputs[properties.nbh_model]) < len(idx_i):
            extract_nbh_model = True
            nbh_model_mask = torch.zeros(idx_j.size(), dtype=torch.bool)
            nbh_model_mask[inputs[properties.nbh_model]] = True
            nbh_model_mask = nbh_model_mask[nbh_list_order]
        if len(inputs[properties.nbh_prediction]) < len(idx_i):
            nbh_prediction_mask = torch.zeros(idx_j.size(), dtype=torch.bool)
            nbh_prediction_mask[inputs[properties.nbh_prediction]] = True
            nbh_prediction_mask = nbh_prediction_mask[nbh_list_order]

        # set up empty data structures that will be filled with our trajectory
        # some lists have to be filled with empty tensors as a starting point to
        # prevent errors when concatenating results in edge cases (e.g. when
        # predicting only the type of the first atom using only the focus)
        empty = torch.empty(0, dtype=torch.float)
        empty_long = torch.empty(0, dtype=torch.long)
        new_R = []  # positions
        new_Z = []  # types
        new_idx_i = [empty_long]  # local center atom index i
        new_idx_j = [empty_long]  # local neighbor atom index j
        new_r_ij = [empty]  # distances between center and neighbor atoms i, j
        new_off = [empty]  # offsets of neighbor atom j
        # number of neighbors in the gathered neighborlist at each step
        n_ij_pairs = torch.zeros(n_steps, dtype=torch.long)
        # number of neighbors of the focus token at each step
        n_foc_j_pairs = torch.zeros(n_steps, dtype=torch.long)
        # we can re-use neighborlists computed at each step in following steps
        # so we store them separately (except for the focus since it changes)
        gathered_idx_i = []
        gathered_idx_j = []
        gathered_r_ij = []
        gathered_off = []
        # at each step, the focus and its neighbors inside the prediction cutoff are
        # used to predict the type of the next atom and the pairwise distances to it
        # we store the required information as following
        pred_idx_j = []  # indices of focus and neighbors inside prediction cutoff
        # entries in pred_idx_j needed for distance predictions
        pred_r_ij_idcs = [empty_long]
        # ground truth distances to be predicted by the model
        pred_r_ij = [empty]
        # types of the new atom repeated per distance prediction
        next_Z = [empty_long]
        # number of neighbors used for prediction at each step
        n_pred_nbh = torch.zeros(n_steps, dtype=torch.long)

        # iterate over steps to build the required tensors
        for cur_step in range(n_steps):
            cur_focus = focus[cur_step]
            n_cur_placed = n_placed_so_far[cur_step]
            if cur_step == 0:
                n_prev_placed = 0
            else:
                n_prev_placed = n_placed_so_far[cur_step-1]

            # 1. Build neighborhood lists for this prediction step:
            # extend neighborhood with atoms from new block
            # to this end, only include neighbors with index smaller than
            # the number of placed atoms at this point
            start = start_idcs[n_prev_placed]  # start of block with new atoms
            end = start_idcs[n_cur_placed]  # end of block with new atoms
            if end > start:
                block_j = idx_j[start:end]
                included = block_j < n_cur_placed
                if extract_nbh_model:
                    # the model cutoff does not contain all pairs in the list, we
                    # have to adjust the block accordingly
                    included = torch.logical_and(
                        nbh_model_mask[start:end],
                        included,
                    )
                included_j = block_j[included]
                # also find neighbors that were part of a previous block, as these
                # have to be added in both combinations (as i,j and j,i)
                previous = included_j < n_prev_placed
                # add the corresponding pairs
                included_i = idx_i[start:end][included]
                included_r_ij = r_ij[start:end][included]
                included_off = offsets[start:end][included]
                gathered_idx_i += [included_i, included_j[previous]]
                gathered_idx_j += [included_j, included_i[previous]]
                gathered_r_ij += [included_r_ij, included_r_ij[previous]]
                gathered_off += [included_off, -included_off[previous]]
                n_ij_pairs[cur_step] += len(included_j) + previous.sum()
            # add the gathered neighborhood to nbh lists
            new_idx_i += gathered_idx_i
            new_idx_j += gathered_idx_j
            new_r_ij += gathered_r_ij
            new_off += gathered_off
            if cur_step > 0:
                n_ij_pairs[cur_step] += n_ij_pairs[cur_step-1]
            # extend neighborhood with pairs from focus
            start = start_idcs[cur_focus]
            end = start + n_nbh[cur_focus]
            if end > start:
                foc_j = idx_j[start:end]
                included = foc_j < n_cur_placed
                # store which neighbors of the focus are used for prediction
                if extract_nbh_prediction:
                    # the prediction cutoff does not contain all pairs in
                    # the list, we have to adjust the block accordingly
                    pred_j_mask = torch.logical_and(
                        nbh_prediction_mask[start:end],
                        included,
                    )
                else:
                    pred_j_mask = included.clone()
                pred_j = foc_j[pred_j_mask]
                if n_tokens > 0:
                    # add i,j for focus token
                    if extract_nbh_model:
                        # the model cutoff does not contain all pairs in
                        # the list, we have to adjust the block accordingly
                        included = torch.logical_and(
                            nbh_model_mask[start:end],
                            included,
                        )
                    included_j = foc_j[included]
                    included_i = torch.zeros_like(included_j)
                    included_r_ij = r_ij[start:end][included]
                    included_off = offsets[start:end][included]
                    foc_pair = torch.tensor([0, cur_focus], dtype=torch.long)
                    foc_dist = torch.zeros(2, dtype=torch.long)
                    foc_off = torch.zeros((2, 3), dtype=torch.float)
                    new_idx_i += [included_i, included_j, foc_pair]
                    new_idx_j += [included_j, included_i, foc_pair[[-1, -2]]]
                    new_r_ij += [included_r_ij, included_r_ij, foc_dist]
                    new_off += [included_off, -included_off, foc_off]
                    n_foc_j_pairs[cur_step] = 2*len(included_j) + 2
            else:
                pred_j = torch.empty(0, dtype=torch.long)

            # 2. build other tensors for positions, types etc.
            # add focus token
            if n_tokens > 0:
                new_R += [R[None, cur_focus]]
                new_Z += [token_types]
                pred_idx_j += [torch.zeros(n_tokens, dtype=torch.long)]
            # add focus atom to prediction neighborhood
            if cur_focus != 0:
                pred_idx_j += [focus[cur_step:cur_step+1]]
                n_pred_nbh[cur_step] += 1
            # add other atoms
            new_R += [R[n_tokens:n_cur_placed]]
            new_Z += [Z[n_tokens:n_cur_placed]]
            pred_idx_j += [pred_j]
            n_pred_nbh[cur_step] += len(pred_j) + n_tokens
            # store information for prediction of pairwise atom distances
            # if a new atom is placed in this step
            if pred_types[cur_step] != self.stop_type and cur_focus != 0:
                _foc_idcs = [cur_focus,]*2 if n_tokens > 0 else [cur_focus,]
                pred_r_ij += [
                    torch.linalg.norm(
                        R[None, n_cur_placed] - R[_foc_idcs, :],
                        dim=1,
                    ),
                    torch.linalg.norm(
                        R[None, n_cur_placed] - R[pred_j],
                        dim=1,
                    ),
                ]
                n_prev_pred_nbh = n_pred_nbh[:cur_step].sum()
                pred_r_ij_idcs += [
                    torch.arange(n_prev_pred_nbh, n_prev_pred_nbh+n_pred_nbh[cur_step]),
                ]
                next_Z += [
                    torch.full(
                        (n_pred_nbh[cur_step],),
                        pred_types[cur_step],
                        dtype=torch.long,
                    ),
                ]

        # assemble tensors from gathered data
        inputs[properties.R] = torch.cat(new_R, dim=0)
        inputs[properties.Z] = torch.cat(new_Z, dim=0)
        inputs[properties.n_atoms] = n_placed_so_far
        atom_idx_offset = n_placed_so_far.cumsum(dim=0) - n_placed_so_far
        nbh_idx_offsets = torch.repeat_interleave(
            atom_idx_offset,
            n_ij_pairs + n_foc_j_pairs,
        )
        inputs[properties.idx_i] = torch.cat(new_idx_i, dim=0) + nbh_idx_offsets
        inputs[properties.idx_j] = torch.cat(new_idx_j, dim=0) + nbh_idx_offsets
        inputs[properties.r_ij] = torch.cat(new_r_ij, dim=0)
        inputs[properties.offsets] = torch.cat(new_off, dim=0)
        pred_idx_offsets = torch.repeat_interleave(
            atom_idx_offset,
            n_pred_nbh,
            )
        inputs[properties.pred_idx_j] = (torch.cat(pred_idx_j, dim=0)
                                         + pred_idx_offsets)
        inputs[properties.pred_idx_m] = torch.repeat_interleave(
            torch.arange(n_steps), n_pred_nbh
        )
        inputs[properties.n_pred_nbh] = n_pred_nbh
        inputs[properties.pred_r_ij] = torch.cat(pred_r_ij, dim=0)
        inputs[properties.pred_r_ij_idcs] = torch.cat(pred_r_ij_idcs, dim=0)
        inputs[properties.pred_Z] = pred_types
        inputs[properties.next_Z] = torch.cat(next_Z, dim=0)
        # store updated n_nbh
        inputs[properties.n_nbh] = torch.bincount(
            inputs[properties.idx_i],
            minlength=n_placed_so_far.sum()
        )
        # repeat remaining molecule-wise entries in inputs for each step
        if self.draw_random_samples != 1:
            for key in [properties.cell, properties.idx]:
                if key in inputs:
                    s = inputs[key].size()
                    inputs[key] = inputs[key].expand(n_steps, *s).squeeze()
        # repeat conditions such that they can be embedded
        n_pred_j = len(inputs[properties.pred_idx_j])
        if n_pred_j > 1:
            # repeat trajectory-wise conditions for each atom in the trajectory
            for cond in self.trajectory_conditions:
                s = inputs[cond].size()
                inputs[cond] = inputs[cond].expand(n_pred_j, *s).squeeze()
        # remove outdated/unused information from inputs
        for key in [
            properties.pbc,
            properties.nbh_prediction,
            properties.nbh_model,
            properties.nbh_placement,
            properties.n_nbh_prediction,
            properties.n_nbh_model,
            properties.n_nbh_placement,
        ]:
            if key in inputs:
                inputs.pop(key)

        return inputs
