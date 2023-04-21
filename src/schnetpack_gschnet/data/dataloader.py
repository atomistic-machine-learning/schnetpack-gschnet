import torch
from schnetpack_gschnet import properties

__all__ = ["gschnet_collate_fn"]


def gschnet_collate_fn(batches):
    """
    Build batch with all the trajectories of different molecules from individual
    trajectories of atom placements.

    Args:
        batches (list): each entry is a (partial) molecules making up a trajectory
            of atom placement steps

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems (consisting of all
            trajectories from the input)
    """
    elem = batches[0]
    idx_keys = {
        properties.idx_i,
        properties.idx_j,
        properties.pred_idx_j,
        properties.pred_idx_m,
        properties.pred_r_ij_idcs,
    }

    molecules_per_trajectory = torch.tensor(
        [len(d[properties.n_atoms]) for d in batches], dtype=torch.long
    )
    atoms_per_trajectory = torch.tensor(
        [len(d[properties.Z]) for d in batches], dtype=torch.long
    )
    pred_atoms_per_trajectory = torch.tensor(
        [len(d[properties.pred_idx_j]) for d in batches], dtype=torch.long
    )

    coll_batch = {}
    for key in elem:
        if key not in idx_keys:
            coll_batch[key] = torch.cat([d[key] for d in batches], 0)
        # elif key in idx_keys:
        #     coll_batch[key + "_local"] = torch.cat([d[key] for d in batches], 0)

    # calculate molecule index for each atom (each partial molecule has its own index)
    idx_m = torch.repeat_interleave(
        torch.arange(int(torch.sum(molecules_per_trajectory))),
        repeats=coll_batch[properties.n_atoms],
        dim=0,
    )
    coll_batch[properties.idx_m] = idx_m
    # calculate trajectory index for each atom (all partial molecules in the same
    # trajectory share an index)
    idx_t = torch.repeat_interleave(
        torch.arange(len(batches)), repeats=atoms_per_trajectory, dim=0
    )
    coll_batch[properties.idx_t] = idx_t

    # update idx keys which require specific offset calculations
    # 1. offset for atom-based indices
    seg_t_atoms = torch.empty(len(batches) + 1, dtype=torch.long)
    seg_t_atoms[0] = 0
    seg_t_atoms[1:] = torch.cumsum(atoms_per_trajectory, dim=0)
    # 2. offset for molecule-based indices
    seg_t_molecules = torch.empty(len(batches) + 1, dtype=torch.long)
    seg_t_molecules[0] = 0
    seg_t_molecules[1:] = torch.cumsum(molecules_per_trajectory, dim=0)
    # 3. offset for prediction-atom-based indices (e.g. pred_r_ij_idcs which refer to
    # pred_idx_j)
    seg_t_pred_atoms = torch.empty(len(batches) + 1, dtype=torch.long)
    seg_t_pred_atoms[0] = 0
    seg_t_pred_atoms[1:] = torch.cumsum(pred_atoms_per_trajectory, dim=0)
    for key in idx_keys:
        if key in elem.keys():
            if key == properties.pred_idx_m:
                coll_batch[key] = torch.cat(
                    [d[key] + off for d, off in zip(batches, seg_t_molecules)], dim=0
                )
            elif key == properties.pred_r_ij_idcs:
                coll_batch[key] = torch.cat(
                    [d[key] + off for d, off in zip(batches, seg_t_pred_atoms)], dim=0
                )
            else:
                coll_batch[key] = torch.cat(
                    [d[key] + off for d, off in zip(batches, seg_t_atoms)], dim=0
                )

    return coll_batch
