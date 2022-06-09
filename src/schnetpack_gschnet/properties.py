"""
Keys to access structure properties.

Note: Had to be moved out of Structure class for TorchScript compatibility

"""
from schnetpack.properties import *

## structure
#: absolute pairwise distances between center atom i and neighbor atom j
r_ij: Final[str] = "_rij"
#: indices of trajectories (i.e. all atom placement steps of one system share the index)
idx_t: Final[str] = "_idx_t"

#: lists of indices to extract different neighborhoods from idx_i and idx_j given by
# different cutoffs
#: neighborhood from SchNet model cutoff
nbh_model: Final[str] = "_nbh_model"
#: neighborhood from prediction cutoff (i.e. atoms used to predict distances)
nbh_prediction: Final[str] = "_nbh_prediction"
#: neighborhood from placement cutoff (i.e. atoms considered as neighbors during
# placement)
nbh_placement: Final[str] = "_nbh_placement"

#: number of neighbors for each atom in model cutoff
n_nbh_model: Final[str] = "_n_nbh_model"
#: number of neighbors for each atom in prediction cutoff
n_nbh_prediction: Final[str] = "_n_nbh_prediction"
#: number of neighbors for each atom in placement cutoff
n_nbh_placement: Final[str] = "_n_nbh_placement"

## information required for prediction
# the type is predicted at every step (half of the time, the stop type is predicted to
# mark the focus as finished)
#: the types to predict (in n_atoms*2 steps)
pred_Z: Final[str] = "_pred_Z"
#: indices of the focus atom and its neighbors inside the prediction cutoff
pred_idx_j: Final[str] = "_pred_idx_j"
#: indices of the prediction step the atoms in pred_idx_j belong to
pred_idx_m: Final[str] = "_pred_idx_m"
#: number of atoms used for prediction in each step (i.e. count of pred_idx_m)
n_pred_nbh: Final[str] = "_n_pred_nbh"
# distances are only predicted in half of the steps (when the predicted type is not
# the stop type)
#: indices in pred_idx_j that are used to predict distances
pred_r_ij_idcs: Final[str] = "_pred_r_ij_idcs"
#: the distances between the corresponding atoms and the new atom
pred_r_ij: Final[str] = "_pred_r_ij"

#: like pred_Z, but each type is repeated n_pred_nbh times (for embedding of next types)
next_Z: Final[str] = "_next_Z"

#: distribution predicted by the model (type of the next atom)
distribution_Z: Final[str] = "_distribution_Z"
#: distribution predicted by the model (pairwise distances)
distribution_r_ij: Final[str] = "_distribution_r_ij"

## properties (for conditioning)
composition: Final[str] = "composition"  #: the atomic composition
relative_atomic_energy: Final[str] = "relative_atomic_energy"
