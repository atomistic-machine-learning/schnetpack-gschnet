from __future__ import annotations

from typing import Dict, Optional, List, Callable, Union, Sequence

from schnetpack.transform import Transform
from schnetpack.model import AtomisticModel
from schnetpack.nn.activations import shifted_softplus
from schnetpack_gschnet.schnet import SchNet
from schnetpack.nn import build_mlp
from schnetpack.nn.scatter import scatter_add
from schnetpack.nn.radial import GaussianRBF

from schnetpack_gschnet import properties
import math
import torch
import torch.nn as nn
from torch.nn.functional import pad

__all__ = [
    "ConditionalGenerativeSchNet",
    "ConditioningModule",
    "ConditionEmbedding",
    "ScalarConditionEmbedding",
    "CompositionEmbedding",
    "VectorialConditionEmbedding",
]


class ConditionalGenerativeSchNet(AtomisticModel):
    """
    cG-SchNet class that sequentially applies a conditioning module and a
    representation module. Then, it predicts the type of the next atom as well as its
    pairwise distances to preceding atoms inside a prediction cutoff.
    """

    def __init__(
        self,
        representation: SchNet,
        atom_types: List[int],
        origin_type: int,
        focus_type: int,
        stop_type: int,
        model_cutoff: float,
        prediction_cutoff: float,
        placement_cutoff: float,
        conditioning: ConditioningModule = None,
        type_prediction_n_layers: int = 5,
        type_prediction_n_hidden: List[int] = None,
        type_prediction_activation: Callable = shifted_softplus,
        distance_prediction_n_layers: int = 5,
        distance_prediction_n_hidden: List[int] = None,
        distance_prediction_activation: Callable = shifted_softplus,
        distance_prediction_n_bins: int = 301,
        distance_prediction_min_dist: float = 0.0,
        distance_prediction_max_dist: float = 15.0,
        postprocessors: Optional[List[Transform]] = None,
        input_dtype_str: str = "float32",
        do_postprocessing: bool = False,
        average_type_distributions: bool = False,
        input_modules: List[nn.Module] = None,
    ):
        """
        Args:
            representation: The SchNet module that builds the representation from the
                inputs.
            atom_types: The atom types (i.e. nuclear charges) that shall be available
                in the type distributions predicted by the model).
            stop_type: The stop type (predicted if no other atom should be placed close
                to the current focus, needs to be distinct from the types in
                `atom_types`).
            model_cutoff: cutoff value used in the SchNet model (determines which atoms
                pass messages to each other during feature extraction).
            prediction_cutoff: determines which atoms are used to predict pairwise
                distances (i.e. which atoms are close enough to the focus such that
                they are utilized to predict their distance to the next atom).
            placement_cutoff: determines which atoms are considered to be neighbors
                when sampling sequences of atom placements (i.e. which atoms can be
                placed given a focus atom) and thus the range of the grid around the
                focus atom during generation
            conditioning: Module that embeds the conditions, e.g. the composition or a
                target property value. Set None to train an unconditional model.
            type_prediction_n_layers: Number of layers in the type prediction network.
            type_prediction_n_hidden: List with number of neurons in the hidden layers
                of the type prediction network. Set None to have a block structure where
                each hidden layer has as many neurons as the number of inputs in the
                first layer (i.e. number of features from SchNet + number of features
                from the conditioning module).
            type_prediction_activation: Activation function used in the type prediction
                network after all but the last layer.
            distance_prediction_n_layers: Number of layers in the distance prediction
                network.
            distance_prediction_n_hidden: List with number of neurons in the hidden
                layers of the distance prediction network. Set None to have a block
                structure where each hidden layer has as many neurons as the number of
                inputs in the first layer (i.e. number of features from SchNet + number
                of features from the conditioning module).
            distance_prediction_activation: Activation function used in the distance
                prediction network after all but the last layer.
            distance_prediction_n_bins: Number of bins (i.e. output neurons) for the
                distance prediction network.
            distance_prediction_min_dist: Minimum distance covered by the bins of the
                discretized distance distribution.
            distance_prediction_max_dist: Maximum distance covered by the bins of the
                discretized distance distribution (all larger distances are mapped to
                the last bin).
            postprocessors: Post-processing transforms that may be initialized using the
                `datamodule`, but are not applied during training.
            input_dtype_str: The dtype of real inputs as string.
            do_postprocessing: If true, post-processing is applied.
            average_type_distributions: Determines how the distribution of the type of
                the next atom is computed from all the distributions predicted by
                individual atoms. In any case, the individual distributions are first
                normalized with a softmax.
                If true, the average of individual distributions is taken as the
                prediction.
                If false, the individual distributions are instead multiplied
                element-wise and then normalized by taking the softmax again, which
                leads to sharper distributions compared to the averaging, i.e. it
                further suppresses small probabilities and increases large ones.
            input_modules: Modules that are applied before representation, e.g. to
                modify input or add additional tensors for response properties.
        """
        super().__init__(
            postprocessors=postprocessors,
            input_dtype_str=input_dtype_str,
            do_postprocessing=do_postprocessing,
        )
        if stop_type in atom_types:
            raise ValueError(
                f"The stop type {stop_type} needs to be distinct from the types in "
                f"atom types ({atom_types})."
            )
        self.representation = representation
        self.register_buffer("atom_types", torch.tensor(atom_types, dtype=torch.long))
        self.register_buffer("origin_type", torch.tensor(origin_type, dtype=torch.long))
        self.register_buffer("focus_type", torch.tensor(focus_type, dtype=torch.long))
        self.register_buffer("stop_type", torch.tensor(stop_type, dtype=torch.long))
        self.register_buffer(
            "_all_types",  # atom types+stop type (all types that shall be predicted)
            torch.tensor(atom_types + [stop_type], dtype=torch.long),
        )
        self.register_buffer(
            "_type_to_class_map",
            -torch.ones(torch.amax(self._all_types) + 1, dtype=torch.long),
        )
        self._type_to_class_map[self._all_types] = torch.arange(len(self._all_types))

        self.register_buffer("model_cutoff", torch.tensor(model_cutoff))
        self.register_buffer("prediction_cutoff", torch.tensor(prediction_cutoff))
        self.register_buffer("placement_cutoff", torch.tensor(placement_cutoff))

        self.conditioning_module = conditioning
        self.register_buffer("n_conditional_features", torch.zeros(1, dtype=torch.long))
        if self.conditioning_module is not None:
            self.n_conditional_features += self.conditioning_module.n_features
        n_features = representation.n_atom_basis + int(self.n_conditional_features)
        self.register_buffer("n_features", torch.tensor(n_features, dtype=torch.long))

        self.register_buffer(
            "n_distance_bins",
            torch.tensor(distance_prediction_n_bins, dtype=torch.long),
        )
        self.register_buffer("distance_min", torch.tensor(distance_prediction_min_dist))
        self.register_buffer("distance_max", torch.tensor(distance_prediction_max_dist))
        self.register_buffer(
            "distance_bin_width",
            (self.distance_max - self.distance_min) / (self.n_distance_bins - 1),
        )

        # initialize type and distance prediction networks
        if type_prediction_n_hidden is None:
            type_prediction_n_hidden = n_features
        self.type_prediction_net = build_mlp(
            n_in=n_features,
            n_out=len(self._all_types),
            n_hidden=type_prediction_n_hidden,
            n_layers=type_prediction_n_layers,
            activation=type_prediction_activation,
        )
        self.next_type_embedding = nn.Embedding(
            representation.embedding.num_embeddings,
            representation.n_atom_basis,
            padding_idx=0,
        )
        if distance_prediction_n_hidden is None:
            distance_prediction_n_hidden = n_features
        self.distance_prediction_net = build_mlp(
            n_in=n_features,
            n_out=distance_prediction_n_bins,
            n_hidden=distance_prediction_n_hidden,
            n_layers=distance_prediction_n_layers,
            activation=distance_prediction_activation,
        )

        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.collect_derivatives()
        self.average_type_distributions = average_type_distributions

        self.input_modules = nn.ModuleList(input_modules)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # apply input modules
        for m in self.input_modules:
            inputs = m(inputs)
        # extract atom-wise features from placed atoms
        inputs = self.extract_atom_wise_features(inputs)
        # extract conditioning features from the conditions
        inputs = self.extract_conditioning_features(inputs)
        # predict type of the next atom
        inputs = self.predict_type(inputs)
        # predict pairwise distances between existing atoms and the next atom
        inputs = self.predict_distances(inputs)

        # apply postprocessing (if enabled)
        inputs = self.postprocess(inputs)
        return inputs

    def extract_atom_wise_features(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # compute representation
        inputs = self.representation(inputs)
        # store representation of all neighbors within prediction cutoff
        nbh = inputs[properties.pred_idx_j]
        inputs["representation"] = inputs["scalar_representation"][nbh]
        return inputs

    def extract_conditioning_features(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # compute conditional feature vector and concat it with the atom-wise features
        if self.conditioning_module is not None:
            conditional_features = self.conditioning_module(inputs)
            inputs["representation"] = torch.cat(
                (inputs["representation"], conditional_features), dim=-1
            )
        return inputs

    def predict_type(
        self, inputs: Dict[str, torch.Tensor], use_log_probabilities: bool = True
    ) -> Dict[str, torch.Tensor]:
        # get predictions for the type of the next atom (from all atoms within the
        # prediction cutoff)
        predictions = self.type_prediction_net(inputs["representation"])
        if self.average_type_distributions:
            # normalize these to get distributions
            predictions = self.softmax(predictions)
            # sum the distributions of all neighbors
            predictions = scatter_add(
                predictions,
                inputs[properties.pred_idx_m],
                len(inputs[properties.n_pred_nbh]),
                0,
            )
            # normalize by dividing by the number of neighbors
            distribution = predictions / inputs[properties.n_pred_nbh][..., None]
            if use_log_probabilities:
                inputs[properties.distribution_Z] = torch.log(distribution + 1e-12)
            else:
                inputs[properties.distribution_Z] = distribution
        else:
            # normalize these to get log distributions
            predictions = self.logsoftmax(predictions)
            # multiply the distributions of all neighbors
            predictions = scatter_add(
                predictions,
                inputs[properties.pred_idx_m],
                len(inputs[properties.n_pred_nbh]),
                0,
            )
            # normalize with softmax again
            if use_log_probabilities:
                inputs[properties.distribution_Z] = self.logsoftmax(predictions)
            else:
                inputs[properties.distribution_Z] = self.softmax(predictions)
        return inputs

    def predict_distances(
        self, inputs: Dict[str, torch.Tensor], use_log_probabilities: bool = True
    ) -> Dict[str, torch.Tensor]:
        # extract representations of atoms where pairwise distances shall be predicted
        representation = inputs["representation"][inputs[properties.pred_r_ij_idcs]]
        # embed the type of the next atom
        next_atom_features = self.next_type_embedding(inputs[properties.next_Z])
        # pad the embedding (needs to be multiplied with the representation without
        # changing the conditional features)
        if self.n_conditional_features > 0:
            next_atom_features = pad(
                next_atom_features, (0, self.n_conditional_features), "constant", 1.0
            )
        # multiply embedding with representation
        representation = representation * next_atom_features
        # get predictions for the pairwise distances to the next atom (of each neighbor
        # within the prediction cutoff)
        predictions = self.distance_prediction_net(representation)
        # normalize and return (log) distribution
        if use_log_probabilities:
            inputs[properties.distribution_r_ij] = self.logsoftmax(predictions)
        else:
            inputs[properties.distribution_r_ij] = self.softmax(predictions)
        return inputs

    def get_required_data_properties(self):
        if self.conditioning_module is None:
            return None
        else:
            return self.conditioning_module.required_data_properties

    def get_condition_names(self):
        if self.conditioning_module is None:
            return {"trajectory": [], "step": [], "atom": []}
        else:
            return self.conditioning_module.condition_names

    def classes_to_types(self, classes: torch.Tensor):
        return self._all_types[classes]

    def types_to_classes(self, types: torch.Tensor):
        return self._type_to_class_map[types]

    def dists_to_classes(self, dists: torch.Tensor, in_place: bool = False):
        if not in_place:
            classes = (
                (dists - self.distance_min + (self.distance_bin_width / 2))
                / self.distance_bin_width
            ).long()
            classes = torch.clamp(classes, min=0, max=self.n_distance_bins - 1)
        else:
            offset = -self.distance_min + (self.distance_bin_width / 2)
            dists.add_(offset).div_(self.distance_bin_width).clamp_(
                min=0.0, max=self.n_distance_bins - 1
            )
            classes = dists.long()
        return classes

    def get_available_atom_types(self):
        return self._all_types[:-1]


class ConditioningModule(nn.Module):
    """
    Module for computation of the conditional features for cG-SchNet. First, each
    condition is embedded individually. Then, the embeddings are concatenated and
    mapped to the conditional features vector using a fully connected network.
    """

    def __init__(
        self,
        condition_embeddings: List[ConditionEmbedding],
        n_features: int,
        n_layers: int,
        activation: Callable = shifted_softplus,
    ):
        """
        Args:
            condition_embeddings: Layers that embed individual conditions.
            n_features: Number of features in the conditional features vector.
            n_layers: Number of layers in the fully connected network.
            activation: Activation function used in the fully connected network after
                all but the last layer.
        """
        super().__init__()
        self.condition_embeddings = nn.ModuleList(condition_embeddings)
        self.n_features = n_features

        n_in = 0
        self.condition_names = {
            "trajectory": [],  # trajectory-wise conditions
            "step": [],  # step-wise conditions
            "atom": [],  # atom-wise conditions
        }
        self.required_data_properties = []
        # retrieve information from condition embeddings
        for emb in condition_embeddings:
            # get names of properties and store them in corresponding list
            self.condition_names[emb.condition_type] += [emb.condition_name]
            # get properties that need to be loaded from the data set
            self.required_data_properties += emb.required_data_properties
            # count the number of features in all the embeddings to get the input
            # dimension for the dense network
            n_in += emb.n_features

        # initialize dense network that mixes embedded conditions
        self.dense_net = build_mlp(
            n_in=n_in,
            n_out=self.n_features,
            n_hidden=self.n_features,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # embed all conditions
        emb_features = []
        for emb in self.condition_embeddings:
            emb_features += [emb(inputs)]
        # concatenate the features
        emb_features = torch.cat(emb_features, dim=-1)
        # mix the concatenated features
        conditional_features = self.dense_net(emb_features)
        return conditional_features


class ConditionEmbedding(nn.Module):
    """
    Base class for all condition embeddings.
    The base class ensures that each module is compatible with the ConditioningModule,
    that the required property values from the data set are loaded, and that the shape
    of the conditions in the inputs is as required (e.g. the condition is repeated for
    each atom in each molecule in a trajectory if the condition is trajectory-wise,
    i.e. if it remains constant over all steps and atoms in the trajectory such as, for
    example, a target composition).

    To implement a new ConditionEmbedding, override the foward method. It should take
    property values of conditions from the inputs dictionary and somehow embed them
    into feature vectors with n_features entries. The return value should be a
    torch.tensor with the vectors representing the embedded conditions.
    """

    def __init__(
        self,
        condition_name: str,
        n_features: int,
        required_data_properties: Optional[List[str]] = [],
        condition_type: str = "trajectory",
    ):
        """
        Args:
            condition_name: The name of the condition (e.g. `_composition`).
            n_features: The number of features in the embedding vector.
            required_data_properties: Names of the properties that need to be loaded
                from the data set in order to compute the condition (e.g. `energy` if
                the energy is required).
            condition_type: The type of the condition, either `trajectory`, `step`, or
                `atom` for trajectory-wise, step-wise, or atom-wise conditions,
                respectively.
        """
        super().__init__()
        if condition_type not in ["trajectory", "step", "atom"]:
            raise ValueError(
                f"`condition_type` is {condition_type} but needs to be `trajectory`, "
                f"`step`, or `atom` for trajectory-wise, step-wise, or atom-wise "
                f"conditions, respectively."
            )
        self.condition_name = condition_name
        self.condition_type = condition_type
        self.n_features = n_features
        self.required_data_properties = required_data_properties

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError


class ScalarConditionEmbedding(ConditionEmbedding):
    """
    An embedding network for scalar conditions. The property value is first expanded
    with Gaussian radial basis functions (rbfs) and this vector is then mapped to the
    final embedding with a fully connected neural network.
    """

    def __init__(
        self,
        condition_name: str,
        condition_min: float,
        condition_max: float,
        grid_spacing: float,
        n_features: int,
        n_layers: int,
        activation: Callable = shifted_softplus,
        required_data_properties: Optional[List[str]] = [],
        condition_type: str = "trajectory",
    ):
        """
        Args:
            condition_name: The name of the condition (e.g. `_energy`).
            condition_min: Minimum value of the scalar condition (center of first
                Gaussian rbf).
            condition_max: Maximum value of the scalar condition (center of the last
                Gaussian rbf if `grid_spacing` is fitting).
            grid_spacing: Distance between the centers of the Gaussian rbfs. If
                condition_max-condition_min/grid_spacing is not an integer, the number
                of rbfs is rounded up such that the last rbf center lays beyond
                `condition_max`.
            n_features: The number of features in the final embedding vector.
            n_layers: The number of layers in the fully connected network that maps
                from the Gaussian rbf expansion to the final embedding vector.
            activation: Activation function used after all but the last layer of the
                fully connected network.
            required_data_properties: Names of the properties that need to be loaded
                from the data set in order to compute the condition (e.g. `energy` if
                the energy is required).
            condition_type: The type of the condition, either `trajectory`, `step`, or
                `atom` for trajectory-wise, step-wise, or atom-wise conditions,
                respectively.
        """
        super().__init__(
            condition_name, n_features, required_data_properties, condition_type
        )
        # compute the number of rbfs
        n_rbf = math.ceil((condition_max - condition_min) / grid_spacing) + 1
        # compute the position of the last rbf
        _max = condition_min + grid_spacing * (n_rbf - 1)
        # initialize Gaussian rbf expansion network
        self.gaussian_expansion = GaussianRBF(
            n_rbf=n_rbf, cutoff=_max, start=condition_min
        )
        # initialize fully connected network
        self.dense_net = build_mlp(
            n_in=n_rbf,
            n_out=n_features,
            n_hidden=n_features,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # get the scalar condition value
        scalar_condition = inputs[self.condition_name]
        # expand the scalar value with Gaussian rbfs
        expanded_condition = self.gaussian_expansion(scalar_condition)
        # feed through fully connected network
        embedded_condition = self.dense_net(expanded_condition)
        return embedded_condition


class VectorialConditionEmbedding(ConditionEmbedding):
    """
    An embedding network for vectorial conditions (e.g. a fingerprint). The vector is
    mapped to the final embedding with a fully connected neural network.
    """

    def __init__(
        self,
        condition_name: str,
        n_in: int,
        n_features: int,
        n_layers: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        activation: Callable = shifted_softplus,
        required_data_properties: Optional[List[str]] = [],
        condition_type: str = "trajectory",
    ):
        """
        Args:
            condition_name: The name of the condition (e.g. `_fingerprint`).
            n_in: The number of features in the input vector (i.e. of the condition).
            n_features: The number of features in the final embedding vector.
            n_layers: The number of layers in the fully connected network that maps
                from the input vector to the final embedding vector.
            n_hidden: The number of features in each hidden layer.
                If an integer, the same number of features is used for all hidden layers
                resulting in a rectangular network.
                If None, the number of features is divided by two after each layer
                (starting with n_in) resulting in a pyramidal network.
            activation: Activation function used after all but the last layer of the
                fully connected network.
            required_data_properties: Names of the properties that need to be loaded
                from the data set in order to compute the condition (e.g. `fingerprint`
                if the fingerprint is required).
            condition_type: The type of the condition, either `trajectory`, `step`, or
                `atom` for trajectory-wise, step-wise, or atom-wise conditions,
                respectively.
        """
        super().__init__(
            condition_name, n_features, required_data_properties, condition_type
        )
        # initialize fully connected network
        self.dense_net = build_mlp(
            n_in=n_in,
            n_out=n_features,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # get the vectorial condition value
        vectorial_condition = inputs[self.condition_name]
        # feed through fully connected network
        embedded_condition = self.dense_net(vectorial_condition)
        return embedded_condition


class CompositionEmbedding(ConditionEmbedding):
    """
    Module that embeds atomic compositions for conditioning.
    A learnable atom type embedding is used to embed each atom type individually. The
    individual embeddings are weighted with the concentration of the corresponding type
    in the target composition and subsequently stacked. The stacked vector is processed
    by a fully connected neural network to obtain an embedding of the target
    concentration. In order to account for the total number of atoms in the target
    composition, it is embedded using a ScalarConditionEmbedding module and then
    concatenated with the embedding of the concentration to obtain the final
    composition embedding.
    """

    def __init__(
        self,
        atom_types: List[int],
        n_atom_basis: int,
        n_features_concentration: int,
        n_layers_concentration: int,
        n_features_n_atoms: int,
        n_layers_n_atoms: int,
        condition_min_n_atoms: float,
        condition_max_n_atoms: float,
        grid_spacing_n_atoms: float,
        activation_concentration: Callable = shifted_softplus,
        activation_n_atoms: Callable = shifted_softplus,
        skip_h: bool = False,
    ):
        """
        Args:
            atom_types: List of atom types.
            n_atom_basis: Number of features in the embedding of individual atom types.
            n_features_concentration: The number of features in the concentration
                embedding vector.
            n_layers_concentration: The number of layers in the fully connected network
                that maps from weighted, stacked individual atom type embeddings to the
                concentration embedding.
            n_features_n_atoms: The number of features in the embedding of the total
                number of atoms.
            n_layers_n_atoms: The number of layers in the fully connected network that
            maps from the Gaussian rbf
                expansion of the number of atoms to the embedding vector.
            condition_min_n_atoms: Minimum value for the Gaussian rbf expansion of the
                number of atoms.
            condition_max_n_atoms: Maximum value for the Gaussian rbf expansion of the
                number of atoms.
            grid_spacing_n_atoms: Distance between the centers of the Gaussian rbfs
                (see ScalarConditionEmbedding for details).
            activation_concentration: Activation function used after all but the last
                layer of the fully connected network for the concentration embedding.
            activation_n_atoms: Activation function used after all but the last layer
                of the fully connected network for the embedding of the number of atoms.
            skip_h: Set True to ignore hydrogen in the composition embedding.
        """
        super().__init__(
            condition_name=properties.composition,
            n_features=n_features_concentration,
            required_data_properties=[],
            condition_type="trajectory",
        )
        self.register_buffer("atom_types", torch.tensor(atom_types, dtype=torch.long))
        self.register_buffer("type_mask", torch.ones(len(atom_types), dtype=bool))
        if skip_h:
            self.type_mask[self.atom_types == 1] = 0
        # initialize atom type embedding
        self.type_embedding = nn.Embedding(
            max(atom_types) + 1, n_atom_basis, padding_idx=0
        )
        # compute the number of inputs to the fully connected network
        n_in = n_atom_basis * int(torch.sum(self.type_mask))
        # initialize fully connected network
        self.dense_net = build_mlp(
            n_in=n_in,
            n_out=n_features_concentration,
            n_hidden=n_features_concentration,
            n_layers=n_layers_concentration,
            activation=activation_concentration,
        )
        # initialize scalar condition embedding for the number of atoms
        self.n_atoms_embedding_net = ScalarConditionEmbedding(
            condition_name="composition_n_atoms",
            condition_type="trajectory",
            condition_min=condition_min_n_atoms,
            condition_max=condition_max_n_atoms,
            grid_spacing=grid_spacing_n_atoms,
            n_features=n_features_n_atoms,
            n_layers=n_layers_n_atoms,
            activation=activation_n_atoms,
        )
        # store the total number of features (after concatenating the n_atoms embedding
        # and the concentration embedding)
        self.n_features = n_features_concentration + n_features_n_atoms

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # embed each atom type individually
        type_embeddings = self.type_embedding(self.atom_types[self.type_mask])
        # extract composition
        composition = inputs[properties.composition][..., self.type_mask]
        # calculate the total number of atoms in the composition
        inputs["composition_n_atoms"] = torch.sum(composition, dim=-1)
        # divide by the total number to get the concentration of each atom type
        concentration = composition / inputs["composition_n_atoms"][..., None]
        # weight individual type embeddings by multiplying them with the concentration
        type_embeddings = concentration[..., None] * type_embeddings[None, ...]
        # stack weighted individual type embeddings
        type_embeddings = type_embeddings.flatten(start_dim=1)
        # feed through fully connected network
        type_embeddings = self.dense_net(type_embeddings)
        # get embedding of the number of atoms
        n_atoms_embeddings = self.n_atoms_embedding_net(inputs)
        # stack both vectors
        composition_embedding = torch.cat((type_embeddings, n_atoms_embeddings), dim=-1)
        return composition_embedding
