import torch
import math
from schnetpack_gschnet import properties
from schnetpack.nn.scatter import scatter_add
from torch.functional import F
from ase.visualize import view
from ase import Atoms

__all__ = ["generate_molecules", "generate_molecules_debug"]


def generate_molecules(
    model: torch.nn.Module,
    n_molecules: int,
    max_n_atoms: int,
    grid_distance_min: float,
    grid_spacing: float,
    conditions: dict,
    device: torch.device,
    t: float = 0.1,
    grid_batch_size: float = 0,
):
    """
    Generate a batch of molecules.

    Args:
        model: The model used to predict atom types and pairwise distances at each step.
        n_molecules: The number of molecules that shall be generated.
        max_n_atoms: The maximum number of atoms in generated molecules.
        grid_distance_min: The minimum distance covered by the 3d grid when placing
            atoms.
        grid_spacing: The spacing of bins of the 3d grid in one dimension (e.g. a
            spacing of 0.05 means each bin is a cube where each side has a length of
            0.05).
        conditions: The target property values used as conditions for the generated
            molecules. These need to be given in a nested dictionary, where the highest
            level signifies the type of the condition (i.e. `trajectory`, `step`, or
            `atom`) and each nested dictionary contains key-value-pairs where the key
            is the name of the property and the value is the target value.
            For example, in order to condition the generation on a gap value of 4.0
            and relative atomic energy of -0.1, one would provide
            `{'trajectory': {'gap': 4.0, 'relative_atomic_energy': -0.1}}`. Each
            condition the model was trained on needs to be specified, otherwise a
            ValueError is raised.
        device: The device used by pytorch (e.g. `cuda` or `cpu`).
        t: The temperature term that controls the spread of probability mass on the 3d
            grid. Higher values lead to increased randomness, whereas smaller values
            decrease randomness (by making the distribution flatter or more peaky,
            respectively).
        grid_batch_size: Determines for how many already placed atoms the 3d grid
            probability is computed at the same time. Set to 0 to compute for all atoms
            at once.
            Since the computation of many 3d grids at once requires a lot of memory,
            it is a bottleneck to the total number of molecules that can be generated
            in one batch (i.e. n_molecules). By setting grid_batch_size (e.g. to the
            same value as n_molecules), it is possible to use larger values of
            n_molecules, as the memory requirement for computation of the 3d grid and
            of the model forward pass is aligned.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
            1. The positions of atoms as a batch (n_molecules x max_n_atoms x 3).
            2. The types of atoms as a batch (n_molecules x max_n_atoms).
            3. A list with tuples where each tuple contains the index of a finished
            molecule and its length (i.e. number of atoms in that molecule). This means
            that one can use positions[tuple[0], :tuple[1]] to get the positions of a
            finished molecule (and proceed accordingly with the types).
            Caution: The list can be shorter than n_molecules, as it only contains
            tuples for molecules that where marked as finished by the model. Molecules
            where the generation was not finished after placing max_n_atoms are not
            contained in this list. Accordingly, rows in the returned positions and
            types that are not referenced in the list are the results of an
            unfinished/failed generation attempt.
    """
    # ================================ initialization ================================

    # check if all conditions required by the model are provided
    condition_names = model.get_condition_names()
    for condition_type in condition_names:
        for condition_name in condition_names[condition_type]:
            if (
                conditions[condition_type] is None
                or condition_name not in conditions[condition_type]
            ):
                raise ValueError(
                    f"The condition `{condition_name}` is required by the model but "
                    f'not  provided in `conditions["{condition_type}"]`!'
                )
    # initialize tensors for positions and atom types generated molecules
    R = torch.zeros((n_molecules, max_n_atoms + 2, 3), device=device)
    Z = torch.zeros((n_molecules, max_n_atoms + 3), dtype=torch.long, device=device)
    # the first two types are the focus and origin type
    Z[:, [0, 1]] = torch.tensor(
        [model.focus_type, model.origin_type], dtype=torch.long, device=device
    )
    # initialize tensors for the pairwise distances and neighborhoods (connectivity)
    # neighborhood inside the model cutoff is gathered as idx_m, idx_i, idx_j pairs
    gathered_mij_idcs = torch.zeros((3, 0), dtype=torch.long, device=device)
    # also gather distances between atoms at idx_i/j
    gathered_r_ij = torch.zeros(0, device=device)
    # neighborhood inside prediction cutoff is stored in connectivity matrices where
    # the rows do not contain origin and focus token but the columns do
    nbh_prediction = torch.zeros(
        (n_molecules, max_n_atoms, max_n_atoms + 2), dtype=bool, device=device
    )
    # origin token, focus token, and focus atom are always used for prediction
    # therefore we set their values to 1 in the connectivity matrices
    nbh_prediction[:, :, :2] = 1  # set origin token and focus token
    torch.diagonal(nbh_prediction, offset=2, dim1=1, dim2=2)[:] = 1  # set focus atom
    # we will also store the number of atoms used for prediction (changes each step)
    n_nbh_prediction = torch.zeros(n_molecules, dtype=torch.long, device=device)
    # initialize tensors that tell which of the molecules are finished
    unfinished_molecules = torch.ones(n_molecules, dtype=bool, device=device)
    finished_list = []  # records the order in which molecules where finished
    # initialize tensors that tell which atoms are available as focus
    available_atoms = torch.ones((n_molecules, max_n_atoms), device=device)
    # initialize tensor for the current focus atom in each molecule
    focus = torch.zeros(n_molecules, dtype=torch.long, device=device)
    # initialize 3d grid of candidate positions for atom placement
    grid_3d = get_3d_grid(
        distance_min=grid_distance_min,
        distance_max=float(model.placement_cutoff),
        grid_spacing=grid_spacing,
    ).to(device)
    # initialize "1d" grid that only extends into x-direction for the very first step
    # where we sample the position of the first atom only from the focus and origin
    grid_1d = torch.zeros((model.n_distance_bins, 3), device=device)
    grid_1d[:, 0] = torch.linspace(
        model.distance_min, model.distance_max, model.n_distance_bins
    )

    # ==================== assemble batches for the network input ====================

    def build_batch(i, mask):
        # assemble the inputs required by the network in the `i`-th step using the
        # molecules selected in `mask`
        inputs = {}
        _i = i + 2  # index including focus and origin token
        n_mols = int(torch.sum(mask))  # number of molecules in the batch
        # atom positions and types
        inputs[properties.R] = R[mask, :_i].flatten(end_dim=-2)
        inputs[properties.Z] = Z[mask, :_i].flatten()  # extract and flatten atom types
        # number of atoms per molecule
        inputs[properties.n_atoms] = torch.full((n_mols,), fill_value=_i, device=device)
        # neighborhood defined by model cutoff (center atoms i and neighbors j)
        # extract idx_m, idx_i, idx_j pairs and r_ij of molecules selected in mask
        gathered_mask = mask[gathered_mij_idcs[0]]
        mij_idcs = gathered_mij_idcs[:, gathered_mask]
        r_ij = gathered_r_ij[gathered_mask]
        # find idx_i/j pairs where idx_i = focus and prepare focus token idx_i/j pairs
        if i > 0:
            focus_mask = mij_idcs[1] == (focus + 2)[mij_idcs[0]]
            focus_mij = mij_idcs[:, focus_mask]
            focus_mij[1] = 0  # idx_i of focus is zero
            focus_r_ij = r_ij[focus_mask]
            # also add pair of focus token and focus atom
            token_atom_pairs = torch.cat(
                [
                    torch.where(mask)[0][None],
                    torch.zeros((1, n_mols), dtype=torch.long, device=device),
                    focus[None, mask] + 2,
                ],
                dim=0,
            )
            focus_mij = torch.cat([focus_mij, token_atom_pairs], dim=-1)
            focus_r_ij = torch.cat(
                [focus_r_ij, torch.zeros(n_mols, device=device)], dim=0
            )
        else:  # in the first step there are only origin and focus, both at (0, 0, 0)
            focus_mij = torch.zeros((3, n_mols), dtype=torch.long, device=device)
            focus_mij[0] = torch.arange(n_mols)  # each molecule has exactly one pair
            focus_mij[2] = 1  # idx_j for origin is one
            focus_r_ij = torch.zeros(n_mols, device=device)  # dist between tokens=zero
        # append to other mij_idcs and r_ij
        mij_idcs = torch.cat([mij_idcs, focus_mij, focus_mij[[0, 2, 1]]], dim=-1)
        r_ij = torch.cat([r_ij, focus_r_ij, focus_r_ij], dim=-1)
        # get offset for idx_i and idx_j (i.e. local idx_m * (n_atoms+n_tokens))
        local_idx_m = torch.empty(n_molecules, dtype=torch.long, device=device)
        local_idx_m[mask] = torch.arange(n_mols, device=device)
        offset = local_idx_m[mij_idcs[0]] * _i
        inputs[properties.idx_i] = mij_idcs[1] + offset
        inputs[properties.idx_j] = mij_idcs[2] + offset
        inputs[properties.r_ij] = r_ij
        inputs[properties.offsets] = torch.zeros((len(mij_idcs[0]), 3), device=device)
        # extract neighborhood of current focus (these atoms are used for prediction)
        focus_nbh = torch.where(nbh_prediction[mask, focus[mask], :_i])
        inputs[properties.pred_idx_m] = focus_nbh[0]
        _, inputs[properties.n_pred_nbh] = torch.unique_consecutive(
            focus_nbh[0], return_counts=True
        )
        n_nbh_prediction[mask] = inputs[properties.n_pred_nbh]
        inputs[properties.pred_idx_j] = focus_nbh[1] + (focus_nbh[0] * _i)
        inputs["local_" + properties.pred_idx_j] = focus_nbh[1]
        # repeat conditions of type "trajectory" for every atom in focus_nbh
        for condition_name in condition_names["trajectory"]:
            condition = torch.tensor(
                conditions["trajectory"][condition_name], device=device
            )
            s = condition.size()
            inputs[condition_name] = condition.expand(len(focus_nbh[0]), *s).squeeze()
        return inputs

    # ============================= atom placement loop =============================

    for i in range(max_n_atoms + 1):
        # current index in tensors with focus and origin tokens (e.g. R and Z)
        _i = i + 2

        # =================== 1. sample the type of the next atom ===================

        # mask telling which molecules the next type needs to be predicted for
        mol_mask = unfinished_molecules.clone()
        # keeps track of which molecules were finished in this step
        finished_this_step = torch.zeros(n_molecules, dtype=bool, device=device)
        # will store the sampled next atom type
        next_types = Z[:, _i]
        # stores representation extracted by the model (for distance predictions)
        representation = []
        # stores order of molecules in representation (it changes if the type prediction
        # is repeated for some molecules, i.e. when the stop type is predicted)
        mol_order = []
        # stores the indices of neighbors of the focus for distance predictions
        dist_pred_idx_j = []
        while mol_mask.sum() > 0:
            # sample focus atom (indexing starts after the tokens, i.e. 0 is R[:, 2])
            focus[mol_mask] = torch.multinomial(
                available_atoms[mol_mask, : max(i, 1)], 1
            ).squeeze()
            # set position of focus token (first entry in R) accordingly
            if i > 0:  # not required in first step (origin and focus at (0, 0, 0))
                R_focus = R[mol_mask, 2:][
                    torch.arange(mol_mask.sum(), device=device), focus[mol_mask]
                ]
                R[mol_mask, 0] = R_focus
                R[mol_mask, :_i] -= R_focus[:, None]  # center positions on focus
            # build inputs dictionary for network
            inputs = build_batch(i, mol_mask)
            # extract representation from network
            inputs = model.extract_atom_wise_features(inputs)
            inputs = model.extract_conditioning_features(inputs)
            # predict type distribution
            inputs = model.predict_type(inputs, use_log_probabilities=False)
            type_predictions = inputs[properties.distribution_Z]
            # sample the type of the next atom
            sampled_classes = torch.multinomial(type_predictions, 1).view(-1)
            next_types[mol_mask] = model.classes_to_types(sampled_classes)
            # mark focus as unavailable for molecules where the stop type was sampled
            sampled_stop = torch.logical_and(next_types == model.stop_type, mol_mask)
            if i > 0:  # not in the first step where the focus is on the origin
                available_atoms[sampled_stop, focus[sampled_stop]] = 0
            # check if a molecule is finished now (no atoms available as focus)
            finished_this_step[sampled_stop] = 0 == torch.sum(
                available_atoms[sampled_stop, :i], dim=-1
            )
            # store representations of atoms in molecules where a proper next type was
            # predicted (i.e. not the stop type) for prediction of distances later on
            relevant_mols = ~sampled_stop[mol_mask]
            relevant_atoms = torch.repeat_interleave(
                relevant_mols, inputs[properties.n_pred_nbh]
            )
            representation += [inputs["representation"][relevant_atoms]]
            # also store the index of these molecules to keep track of the order
            mol_order += [torch.where(mol_mask)[0][torch.where(relevant_mols)[0]]]
            # and the index of the neighbors of the focus (in local format)
            dist_pred_idx_j += [
                inputs["local_" + properties.pred_idx_j][relevant_atoms]
            ]
            # repeat loop for the remaining molecules (where stop type was predicted)
            mol_mask = torch.logical_and(sampled_stop, ~finished_this_step)

        # mark the molecules which where finished this step
        unfinished_molecules[finished_this_step] = 0
        # store finished molecules in list (to keep the order they were finished in)
        finished_idcs = torch.where(finished_this_step)[0]
        finished_list += [(int(idx), i) for idx in finished_idcs]

        # stop if all molecules are finished or the maximum number of atoms is exceeded
        if torch.sum(unfinished_molecules) == 0 or i >= max_n_atoms:
            break

        # ================= 2. sample the position of the next atom =================

        # assemble inputs from stored representations and the sampled next type
        mol_order = torch.cat(mol_order)
        representation = torch.cat(representation, dim=0)
        inputs = {
            "representation": representation,
            properties.pred_r_ij_idcs: torch.arange(len(representation), device=device),
            properties.next_Z: torch.repeat_interleave(
                next_types[mol_order], n_nbh_prediction[mol_order]
            ),
        }
        # predict distances to atoms in the neighborhood of the focus with the model
        inputs = model.predict_distances(inputs, use_log_probabilities=True)
        prediction = inputs[properties.distribution_r_ij]
        # compute distances between these atoms and the candidate grid positions
        idx_m_local = torch.repeat_interleave(
            torch.arange(len(mol_order), device=device), n_nbh_prediction[mol_order]
        )
        idx_m = mol_order[idx_m_local]
        dist_pred_idx_j = torch.cat(dist_pred_idx_j)
        pos = R[idx_m, dist_pred_idx_j]  # positions of relevant atoms
        grid = grid_1d if i == 0 else grid_3d  # choose special grid in very first step
        # run calculations over positional grid in batches if desired (since the grids
        # require a lot of memory, this can help to speed up computations by allowing
        # larger batches of molecules for generation, i.e. the memory demands of the
        # model forward pass and the computations on the grid are aligned)
        if grid_batch_size <= 0:
            n_batches = 1
            atoms_per_grid_batch = len(idx_m)
        else:
            n_batches = math.ceil(len(idx_m) / grid_batch_size)
            atoms_per_grid_batch = grid_batch_size
        log_p_mol = torch.zeros((len(mol_order), len(grid)), device=device)
        for b in range(n_batches):
            start = b * atoms_per_grid_batch
            stop = (b + 1) * atoms_per_grid_batch
            dists = cdists(pos[start:stop], grid)  # pairwise distances
            # map distances to bins (of distance distribution from the network output)
            bins = model.dists_to_classes(dists, in_place=True)
            del dists
            # lookup of log probabilities of these distances in the network output
            log_p_atom = torch.gather(prediction[start:stop], dim=1, index=bins)
            del bins
            # multiply probabilities of all neighbors at candidate positions (add log_p)
            log_p_mol += scatter_add(
                log_p_atom, idx_m_local[start:stop], dim_size=len(mol_order), dim=0
            )
            del log_p_atom
        # normalize probabilities and use temperature parameter t
        # first normalizing, applying t and re-normalizing is equivalent to directly
        # applying t and normalizing thereafter but it is numerically more stable
        log_p_mol -= torch.logsumexp(log_p_mol, dim=-1, keepdim=True)
        if i > 0:  # do not use temperature parameter in first step
            log_p_mol /= t
            log_p_mol -= torch.logsumexp(log_p_mol, dim=-1, keepdim=True)
        # sample position of next atom for all molecules from candidates
        next_pos_idcs = torch.multinomial(log_p_mol.exp_(), 1).view(-1)
        del log_p_mol
        # store sampled positions of the next atom
        R[mol_order, _i] = grid[next_pos_idcs]

        # ====== 3. update model and prediction neighborhoods with the new atom ======

        # compute distances between new atom and previous atoms
        dists = torch.linalg.norm(R[mol_order, :_i] - R[mol_order, _i : _i + 1], dim=-1)
        # check which atoms are inside model cutoff of new atom
        # we set the distance to the origin to 0 to always have it inside the cutoff
        # and set the distance to the focus to a high value outside the cutoff since
        # it changes at each step and therefore is unknown at this point
        _dists = dists.clone()
        _dists[:, 0] = max(model.prediction_cutoff, model.model_cutoff) + 1
        _dists[:, 1] = 0
        in_model_cutoff = torch.where(_dists <= model.model_cutoff)
        # extract the corresponding distances and idx_m, idx_i, idx_j
        new_r_ij = dists[in_model_cutoff]  # distances (incl. correct dist to origin)
        new_mij = torch.cat(
            [
                mol_order[in_model_cutoff[0]][None],
                torch.full((1, len(in_model_cutoff[0])), fill_value=_i, device=device),
                in_model_cutoff[1][None],
            ],
            dim=0,
        )
        # add distances and indices to gathered model_cutoff neighborhood
        # add 2 times to get symmetric idx_i/j-pairs
        gathered_r_ij = torch.cat([gathered_r_ij, new_r_ij, new_r_ij], dim=0)
        gathered_mij_idcs = torch.cat(
            [gathered_mij_idcs, new_mij, new_mij[[0, 2, 1]]],  # with swapped idx_i/j
            dim=-1,
        )
        # check which atoms are inside prediction cutoff of the new atom (skip tokens)
        in_pred_cutoff = torch.where(dists[:, 2:] <= model.prediction_cutoff)
        in_pred_m = mol_order[in_pred_cutoff[0]]
        in_pred_j = in_pred_cutoff[1]
        # update row and column of new atom in nbh matrices for prediction
        # be aware that the rows do not contain origin and focus but the columns do
        nbh_prediction[:, i, 2:][in_pred_m, in_pred_j] = True  # update rows
        nbh_prediction[:, :, _i][in_pred_m, in_pred_j] = True  # update columns

    # center positions on origin
    R = R - R[:, 1:2]
    return R, Z, finished_list


def get_3d_grid(distance_min: float, distance_max: float, grid_spacing: float):
    # compute the number of bins in a 1d grid between -distance_max and distance_max
    n_bins = int(torch.ceil(torch.tensor(2 * distance_max / grid_spacing)) + 1)
    # get the coordinates of the 1d grid
    coordinates = torch.linspace(-distance_max, distance_max, n_bins)
    # use cartesian product to get 3d grid (cube)
    grid = torch.cartesian_prod(coordinates, coordinates, coordinates)
    # remove positions that are too far from or too close to the center (spherical grid)
    dists = torch.linalg.norm(grid, dim=1)
    mask = torch.logical_and(dists >= distance_min, dists <= distance_max)
    return grid[mask]


def cdists(pos_1, pos_2):
    """
    Calculates the pairwise Euclidean distances between two sets of atom positions.
    Uses inplace operations to minimize memory demands.
    Args:
        pos_1 (:class:`torch.Tensor`): atom positions of shape (n_atoms_1 x n_dims)
        pos_2 (:class:`torch.Tensor`): atom positions of shape (n_atoms_2 x n_dims)
    Returns:
        :class:`torch.Tensor`: distance matrix (n_atoms_1 x n_atoms_2)
    """
    return F.relu(
        torch.sum((pos_1[:, None] - pos_2[None]).pow_(2), -1), inplace=True
    ).sqrt_()


def generate_molecules_debug(
    model: torch.nn.Module,
    max_n_atoms: int,
    grid_distance_min: float,
    grid_spacing: float,
    conditions: dict,
    device: torch.device,
    t: float = 0.1,
    print_progress: bool = True,
    view_progress: bool = True,
):
    """
    This is a simpler implementation of the molecule generation procedure designed for
    debugging the process. It only samples a single molecule (instead of a batch of
    molecules) and allows to view the partial molecule at each steps as well as printing
    some information.

    Args:
        model: The model used to predict atom types and pairwise distances at each step.
        max_n_atoms: The maximum number of atoms in generated molecules.
        grid_distance_min: The minimum distance covered by the 3d grid when placing
            atoms.
        grid_spacing: The spacing of bins of the 3d grid in one dimension (e.g. a
            spacing of 0.05 means each bin is a cube where each side has a length of
            0.05).
        conditions: The target property values used as conditions for the generated
            molecules. These need to be given in a nested dictionary, where the highest
            level signifies the type of the condition (i.e. `trajectory`, `step`, or
            `atom`) and each nested dictionary contains key-value-pairs where the key
            is the name of the property and the value is the target value.
            For example, in order to condition the generation on a gap value of 4.0
            and relative atomic energy of -0.1, one would provide
            `{'trajectory': {'gap': 4.0, 'relative_atomic_energy': -0.1}}`. Each
            condition the model was trained on needs to be specified, otherwise a
            ValueError is raised.
        device: The device used by pytorch (e.g. `cuda` or `cpu`).
        t: The temperature term that controls the spread of probability mass on the 3d
            grid. Higher values lead to increased randomness, whereas smaller values
            decrease randomness (by making the distribution flatter or more peaky,
            respectively).
        print_progress: Set True to print focus, type distribution and sampled type
            at each step
        view_progress: Set True to display the (partial) molecule at each step

    Returns:
        Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
            1. The positions of atoms as a batch (1 x max_n_atoms x 3).
            2. The types of atoms as a batch (1 x max_n_atoms).
            3. A list with a tuple that contains the index of the finished molecule (0)
            and its length (i.e. number of atoms in that molecule). This means
            that one can use positions[tuple[0], :tuple[1]] to get the positions of the
            finished molecule (and proceed accordingly with the types).
            Caution: The list can be empty, as it only contains a tuple if the molecule
            was marked as finished by the model. Molecules where the generation was not
            finished after placing max_n_atoms are not contained in this list.
            Accordingly, rows in the returned positions and types that are not
            referenced in the list are the results of an unfinished/failed generation
            attempt.
    """
    # check if all conditions required by the model are provided
    condition_names = model.get_condition_names()
    for condition_type in condition_names:
        for condition_name in condition_names[condition_type]:
            if (
                conditions[condition_type] is None
                or condition_name not in conditions[condition_type]
            ):
                raise ValueError(
                    f"The condition `{condition_name}` is required by the model but "
                    f'not  provided in `conditions["{condition_type}"]`!'
                )

    n_tokens = 2
    # initialize tensors for positions and type
    R = torch.zeros((max_n_atoms + n_tokens, 3)).to(device)
    Z = torch.zeros(max_n_atoms + n_tokens + 1, dtype=torch.long).to(device)
    # the first two atoms are the focus and the origin token
    Z[0] = model.focus_type
    Z[1] = model.origin_type

    offsets = torch.zeros((max_n_atoms, 3)).to(device)
    large_distance = max(model.prediction_cutoff, model.model_cutoff) + 1.0
    ats = []
    # initialize 3d grid of candidate positions for atom placement
    grid_3d = get_3d_grid(
        distance_min=grid_distance_min,
        distance_max=float(model.placement_cutoff),
        grid_spacing=grid_spacing,
    ).to(device)
    # initialize "1d" grid that only extends into x-direction for the very first step
    # where we sample the position of the first atom only from the focus and origin
    grid_1d = torch.zeros((model.n_distance_bins, 3), device=device)
    grid_1d[:, 0] = torch.linspace(
        model.distance_min, model.distance_max, model.n_distance_bins
    )

    # sample new atoms
    available_atoms = torch.ones(max_n_atoms + n_tokens).to(device)
    available_atoms[0] = 0  # the focus cannot be chosen as focus
    for i in range(n_tokens, max_n_atoms + n_tokens + 1):
        if print_progress:
            print(
                f"\nStep {i-n_tokens}:\n"
                f"------------------------------------------------------------------"
            )

        # sample atom type
        sampled_proper_type = False  # set true if type != stop was sampled
        while not sampled_proper_type:
            # choose focus randomly
            focus = torch.multinomial(available_atoms[:i], 1)
            R[0] = R[focus]
            # center positions on focus
            R -= R[0:1].clone()
            # compute neighborhood
            all_dists = cdists(R[:i], R[:i])
            dists = all_dists.clone()
            # the origin is connected to everything -> set distances to 0
            dists[1] = 0
            dists[:, 1] = 0
            # atoms are not self connected -> set diagonal to large distance
            dists.fill_diagonal_(large_distance)
            # extract i,j pairs and corresponding distances
            idx_i, idx_j = torch.where(dists <= model.model_cutoff)
            r_ij = all_dists[idx_i, idx_j]
            # extract which atoms are used for predictions (origin is always used)
            dists = all_dists.clone()
            dists[1] = 0
            dists[:, 1] = 0
            _, pred_j = torch.where(dists[focus] <= model.prediction_cutoff)
            pred_m = torch.zeros(len(pred_j), dtype=torch.long).to(device)
            n_pred_nbh = torch.tensor([len(pred_j)], dtype=torch.long).to(device)
            # build inputs
            inputs = {}
            inputs[properties.R] = R[:i]
            inputs[properties.Z] = Z[:i]
            inputs[properties.n_atoms] = torch.tensor(i, dtype=torch.long).to(device)
            inputs[properties.idx_i] = idx_i
            inputs[properties.idx_j] = idx_j
            inputs[properties.r_ij] = r_ij
            inputs[properties.offsets] = offsets[:i]
            inputs[properties.cell] = torch.zeros((3, 3)).to(device)
            inputs[properties.pred_idx_j] = pred_j
            inputs[properties.pred_idx_m] = pred_m
            inputs[properties.n_pred_nbh] = n_pred_nbh
            inputs[properties.n_nbh] = torch.unique_consecutive(
                idx_i, return_counts=True
            )[1]
            # extract conditions for inputs
            for cond_name in conditions["trajectory"]:
                cond = torch.tensor(conditions["trajectory"][cond_name], device=device)
                s = cond.size()
                inputs[cond_name] = cond.expand(len(pred_j), *s).squeeze()
            # predict type
            inputs = model.extract_atom_wise_features(inputs)
            inputs = model.extract_conditioning_features(inputs)
            inputs = model.predict_type(inputs, use_log_probabilities=False)
            # sample type
            next_class = torch.multinomial(inputs[properties.distribution_Z], 1)
            next_type = model.classes_to_types(next_class)
            if print_progress:
                print(f"Current focus: {focus}")
                print(f"Type distribution: {inputs[properties.distribution_Z]}")
                print(f"Sampled type: {next_type}\n")
            if next_type != model.stop_type:
                # sampled a proper type -> fill Z
                Z[i] = next_type
                sampled_proper_type = True
                if i == n_tokens:
                    available_atoms[1] = 0  # origin cannot be focused anymore
            else:
                # sampled stop -> remove current focus from available_atoms
                available_atoms[focus] = 0.0
                if available_atoms[:i].sum() <= 0:
                    R -= R[1:2].clone()
                    if view_progress:
                        view(ats)
                    return R[None], Z[None], [(0, i - n_tokens)]

        # sample position of the next atom
        if i == n_tokens:
            grid = grid_1d
        elif i < max_n_atoms + n_tokens:
            grid = grid_3d
        else:
            return R, Z, []  # molecule not finished
        # first get probabilities of pairwise distances from model
        inputs_dists = {
            "representation": inputs["representation"],
            properties.pred_r_ij_idcs: torch.arange(len(pred_j), device=device),
            properties.next_Z: torch.full(
                (len(pred_j),),
                fill_value=int(next_type),
                dtype=torch.long,
                device=device,
            ),
        }
        inputs_dists = model.predict_distances(inputs_dists, True)
        predictions = inputs_dists[properties.distribution_r_ij]
        # then compute the distances between the atoms and grid positions
        dists = cdists(R[pred_j], grid)
        # and assign distances to bins in the model output
        bins = model.dists_to_classes(dists)
        # finally look up the probabilities of distances on the model predictions
        log_p_atom = torch.gather(predictions, dim=1, index=bins)
        # and aggregate grids for individual atoms to get 3d distribution
        log_p_mol = torch.sum(log_p_atom, dim=0)
        # normalize over grid and sample position
        log_p_mol -= torch.logsumexp(log_p_mol, dim=-1)
        if i > n_tokens:  # do not use temperature parameter in first step
            log_p_mol /= t
            log_p_mol -= torch.logsumexp(log_p_mol, dim=-1)
        # sample position of next atom from candidates
        next_pos_idx = torch.multinomial(log_p_mol.exp_(), 1)
        # store sampled positions of the next atom
        R[i] = grid[next_pos_idx]

        if view_progress:
            ats += [
                Atoms(
                    positions=(R[n_tokens : i + 1].clone() - R[1:2])
                    .detach()
                    .cpu()
                    .numpy(),
                    numbers=Z[n_tokens : i + 1].detach().cpu().numpy(),
                )
            ]

    return R, Z, []
