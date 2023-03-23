from typing import Dict, Optional, List, Tuple
import logging
import torch
from schnetpack.transform.base import Transform
from schnetpack_gschnet import properties

logger = logging.getLogger(__name__)

__all__ = [
    "BuildAtomsTrajectory",
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
                ascending order (otherwise they will be unordered)
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
