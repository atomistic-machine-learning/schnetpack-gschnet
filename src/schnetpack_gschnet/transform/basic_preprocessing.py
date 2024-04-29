from typing import Dict, Optional, List, Tuple
import logging
import torch
from schnetpack.transform.base import Transform
from schnetpack_gschnet import properties

logger = logging.getLogger(__name__)

__all__ = [
    "OrderByDistanceToOrigin",
    "GetComposition",
    "GetRelativeAtomicEnergy",
    "GetRandomSubstructure",
]


class OrderByDistanceToOrigin(Transform):
    """
    Order atoms by their distance to the origin.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # compute distances of atoms to (0, 0, 0)
        center_dists = torch.linalg.norm(inputs[properties.R], dim=1)
        # order by that distance
        _, order = center_dists.sort()
        # store ordered positions and types
        inputs[properties.R] = inputs[properties.R][order]
        inputs[properties.Z] = inputs[properties.Z][order]
        return inputs


class GetNumHeavyAtoms(Transform):
    """
    Extracts the number of heavy atoms in a molecule.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # count number of hydrogen atoms
        n_Hs = torch.count_nonzero(inputs[properties.Z] == 1)
        # the number of heavy atoms is the number of atoms - number of hydrogens
        inputs["_n_heavy_atoms"] = inputs[properties.n_atoms] - n_Hs
        return inputs


class GetComposition(Transform):
    """
    Extracts the number of atoms in a molecule that correspond to types from a given
    list of atom types.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        atom_types: List[int],
    ):
        """
        Args:
            atom_types: List of atom types.
        """
        super().__init__()
        self.register_buffer("atom_types", torch.tensor(atom_types, dtype=torch.long))
        self.register_buffer("_min_length", (torch.max(self.atom_types) + 1).long())

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        counts = torch.bincount(inputs[properties.Z], minlength=self._min_length)
        inputs[properties.composition] = counts[self.atom_types]

        return inputs


class GetRelativeAtomicEnergy(Transform):
    """
    Computes the relative atomic energy of a molecule

    .. math::

        E^Z - \hat{E}^Z

    where :math:`E^Z` denotes the energy per atom of the structure and
    :math:`\hat{E}^Z` denotes the expected energy per atom of other molecules in the
    training data that share the same atomic composition. Consequently, a negative
    relative atomic energy indicates comparatively low energy whereas a positive value
    indicates comparatively high energy relative to similar structures in the training
    data set. We compute :math:`\hat{E}^Z` with linear regression from the atomic
    concentration. The weights and bias are learned using the training data or can
    optionally be supplied.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        atom_types: List[int],
        target_energy_name: str,
        regression_weights: Optional[List[float]] = None,
        regression_bias: Optional[float] = None,
        max_train_points: Optional[int] = None,
    ):
        """
        Args:
            atom_types: List of all atom types occurring in the data set.
            target_energy_name: Name of the energy in the data set.
            regression_weights: List of weights for the regression (mapping atomic
                concentration to energy per atom).
            regression_bias: Bias for the regression.
            max_train_points: The maximum number of points used from the training data
                in order to learn the regression weights and bias (if they are not
                provided). Set None to use all training data points.
        """
        super().__init__()
        self.register_buffer("atom_types", torch.tensor(atom_types, dtype=torch.long))
        self.register_buffer("_min_length", (torch.max(self.atom_types) + 1).long())
        self.target_energy_name = target_energy_name
        self.max_train_points = max_train_points
        if regression_weights:
            self.register_buffer(
                "regression_weights",
                torch.tensor(regression_weights, dtype=torch.float32),
            )
        if regression_bias:
            self.register_buffer(
                "regression_bias", torch.tensor(regression_bias, dtype=torch.float32)
            )

    def datamodule(self, value):
        if not (
            hasattr(self, "regression_weights") and hasattr(self, "regression_bias")
        ):
            weights, bias = self.lstsq_regression(value)
            if not hasattr(self, "regression_weights"):
                self.register_buffer("regression_weights", weights)
            else:
                self.regression_weights = weights
            if not hasattr(self, "regression_bias"):
                self.register_buffer("regression_bias", bias)
            else:
                self.regression_bias = bias

    def lstsq_regression(self, datamodule):
        train_dataset = datamodule.train_dataset
        if self.max_train_points:
            idcs = torch.randperm(len(train_dataset))[: self.max_train_points]
            n_points = len(idcs)
            logger.info(
                f"Calculating weights for regression for relative atomic energy using "
                f"{n_points} randomly sampled points from the training set..."
            )
        else:
            n_points = len(train_dataset)
            idcs = torch.arange(n_points)
            logger.info(
                f"Calculating weights for regression for relative atomic energy using "
                f"all {n_points} points from the training set."
            )
        # read regression variables x (concentration of atoms in molecules +1 column
        # for bias) and regression targets y (energy per atom of molecules)
        x = torch.ones((n_points, len(self.atom_types) + 1))
        y = torch.empty((n_points, 1))
        idx = 0
        for inputs in train_dataset.iter_properties(
            load_properties=[self.target_energy_name],
            load_structure=True,
            indices=idcs.tolist(),
        ):
            x[idx, :-1] = self.get_concentration(inputs)
            y[idx] = inputs[self.target_energy_name] / inputs[properties.n_atoms]
            idx += 1
        # compute LSTSQ solution to obtain weights and bias
        weights = torch.linalg.lstsq(x, y, rcond=1e-8, driver="gelsd").solution
        regression_weights = weights[:-1, 0]
        regression_bias = weights[-1, 0]

        return regression_weights, regression_bias

    def get_concentration(self, inputs):
        counts = torch.bincount(inputs[properties.Z], minlength=self._min_length)

        return counts[self.atom_types] / inputs[properties.n_atoms]

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        concentration = self.get_concentration(inputs)
        predicted_energy_per_atom = (
            self.regression_weights.dot(concentration) + self.regression_bias
        )
        energy_per_atom = inputs[self.target_energy_name] / inputs[properties.n_atoms]
        inputs[properties.relative_atomic_energy] = (
            energy_per_atom - predicted_energy_per_atom
        )

        return inputs


class GetRandomSubstructure(Transform):
    """
    Uniformly randomly samples a certain percentage of the atoms in the molecule as
    pre-defined substructure. Caution, there are no constraints that the atoms have
    to be connected.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        percentage: Optional[float] = 0.5,
    ):
        """
        Args:
            percentage: Number in [0, 1) that determines the percentage of atoms
                that are part of the randomly sampled substructure.
        """
        self.percentage = percentage
        super().__init__()

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # number of atoms in the molecule
        n_atoms_mol = int(inputs[properties.n_atoms])
        # number of atoms in the substructure
        n_atoms_substructure = int(self.percentage * n_atoms_mol)
        # make sure that at least one atom is not part of the substructure
        if n_atoms_mol == n_atoms_substructure:
            n_atoms_substructure -= 1
        # sample substructure
        if n_atoms_substructure == 0:
            # set empty list, i.e. no substructure
            inputs[properties.substructure_idcs] = torch.tensor(
                [],
                dtype=torch.long,
            )
        else:
            # randomly draw substructure
            inputs[properties.substructure_idcs] = torch.multinomial(
                torch.ones(n_atoms_mol),
                n_atoms_substructure,
                replacement=False,
            )
        return inputs
