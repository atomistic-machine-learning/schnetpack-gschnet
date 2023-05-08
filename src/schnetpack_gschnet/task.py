from pathlib import Path
from typing import Optional, Dict, List, Type, Any, Union

import pytorch_lightning as pl
import torch
from torch import nn as nn
from schnetpack_gschnet.transform import (
    BuildAtomsTrajectory,
    GeneralCachedNeighborList,
    ConditionalGSchNetNeighborList,
)
from schnetpack_gschnet.model import ConditionalGenerativeSchNet
from schnetpack_gschnet import properties
from schnetpack import AtomisticTask

__all__ = ["ConditionalGenerativeSchNetTask"]


class ConditionalGenerativeSchNetTask(AtomisticTask):
    """
    Defines a generative learning task with cG-SchNet.

    """

    def __init__(
        self,
        model: ConditionalGenerativeSchNet,
        dist_label_width_factor: float = 0.1,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
    ):
        """
        Args:
            model: The cG-SchNet model.
            dist_label_width_factor: Scaling for the width of the Gaussian expansion to
                obtain distance distributions from the ground-truth distances.
                Decreasing the width factor leads to peakier distributions while
                increasing it increases the noise. If set to values smaller than 0.01,
                the distances will be one-hot encoded and treated as classes, i.e. the
                negative log likelihood loss is used instead of KL divergence.
            optimizer_cls: Type of torch optimizer,e.g. torch.optim.Adam.
            optimizer_args: Dict of optimizer keyword arguments.
            scheduler_cls: Type of torch learning rate scheduler.
            scheduler_args: Dict of scheduler keyword arguments
            scheduler_monitor: Name of metric to be observed for ReduceLROnPlateau.
        """
        super().__init__(
            model=model,
            outputs=[],
            optimizer_cls=optimizer_cls,
            optimizer_args=optimizer_args,
            scheduler_cls=scheduler_cls,
            scheduler_args=scheduler_args,
            scheduler_monitor=scheduler_monitor,
            warmup_steps=0,
        )
        # initialize loss function for atom type predictions
        self.type_loss_fn = nn.NLLLoss(reduction="mean")
        # initialize loss function for distance predictions
        if dist_label_width_factor >= 0.01:
            self.dists_loss_fn = nn.KLDivLoss(reduction="batchmean")
            gaussian_centers = torch.linspace(
                model.distance_min, model.distance_max, model.n_distance_bins
            )
            gaussian_width = dist_label_width_factor * model.distance_bin_width
            self.register_buffer("gaussian_centers", gaussian_centers[None])
            self.register_buffer("gaussian_width", gaussian_width)
        else:
            self.dists_loss_fn = nn.NLLLoss(reduction="mean")
            self.gaussian_centers = None
            self.gaussian_width = None

    def setup(self, stage=None):
        # register which properties from the data base are required for conditioning
        dm = self.trainer.datamodule
        load_properties = self.model.get_required_data_properties()
        dm.register_properties(load_properties)
        # register the conditions in the BuildAtomsTrajectory transform and check that
        # the cutoffs in the ConditionalGSchNetNeighborList transform are matching
        # the cutoffs registered in the model
        conditions = self.model.get_condition_names()
        for tfs in [dm.train_transforms, dm.val_transforms, dm.test_transforms]:
            for tf in tfs:
                if isinstance(tf, BuildAtomsTrajectory):
                    tf.register_conditions(conditions)
                if isinstance(tf, GeneralCachedNeighborList):
                    tf = tf.neighbor_list
                if isinstance(tf, ConditionalGSchNetNeighborList):
                    tf.check_cutoffs(
                        self.model.model_cutoff,
                        self.model.prediction_cutoff,
                        self.model.placement_cutoff,
                    )
        if stage == "fit":
            self.model.initialize_transforms(dm)

    def load_state_dict(self, state_dict: Dict[str, Any], **kwargs) -> None:
        # make sure that cutoff values have not been changed
        for name, val1 in [
            ("model_cutoff", self.model.model_cutoff),
            ("prediction_cutoff", self.model.prediction_cutoff),
            ("placement_cutoff", self.model.placement_cutoff),
        ]:
            val2 = state_dict[f"model.{name}"]
            if val2 != val1:
                raise ValueError(
                    f"The {name} in the checkpoint is different from the {name} in "
                    f"the config ({val2:.2f}!={val1:.2f}). You cannot change the "
                    f"{name}. Please set it to {val2:.2f} or train a new model."
                )
        # load checkpoint
        super().load_state_dict(state_dict, **kwargs)

    def loss_fn(self, pred, batch, return_individual_losses=False):
        # calculate loss on type predictions (NLL loss using atomic types as classes)
        type_labels = batch[properties.pred_Z]
        class_labels = self.model.types_to_classes(type_labels)  # obtain class labels
        type_loss = self.type_loss_fn(pred[properties.distribution_Z], class_labels)
        # calculate loss on distance predictions
        dists_labels = batch[properties.pred_r_ij]
        if len(dists_labels) > 0:
            # only if steps with distance predictions are in the batch
            if self.gaussian_centers is not None:
                # compute distance distributions for KLD loss using Gaussian expansion
                dists_labels = torch.clamp(
                    dists_labels,
                    min=self.model.distance_min,
                    max=self.model.distance_max,
                )[:, None]
                dists_labels = torch.exp(
                    -(1 / self.gaussian_width)
                    * (dists_labels - self.gaussian_centers) ** 2
                )
                dists_labels /= torch.sum(dists_labels, dim=-1, keepdim=True)
            else:
                # or compute class labels (i.e. in which bin they fall) for NLL loss
                dists_labels = self.model.dists_to_classes(dists_labels)
            dist_loss = self.dists_loss_fn(
                pred[properties.distribution_r_ij], dists_labels
            )
        else:
            dist_loss = 0.0

        if return_individual_losses:
            return type_loss + dist_loss, type_loss, dist_loss
        else:
            return type_loss + dist_loss

    def training_step(self, batch, batch_idx):
        pred = self.predict_without_postprocessing(batch)
        loss = self.loss_fn(pred, batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)
        pred = self.predict_without_postprocessing(batch)
        loss = self.loss_fn(pred, batch, return_individual_losses=True)
        self.log("val_loss", loss[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss_type", loss[1], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_loss_dist", loss[2], on_step=False, on_epoch=True, prog_bar=False)
        return {"val_loss": loss[0]}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)
        pred = self.predict_without_postprocessing(batch)
        loss = self.loss_fn(pred, batch, return_individual_losses=True)
        self.log("test_loss", loss[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test_loss_type", loss[1], on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "test_loss_dist", loss[2], on_step=False, on_epoch=True, prog_bar=False
        )
        return {"test_loss": loss[0]}

    def extract_features(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        inputs = self.model.extract_atom_wise_features(inputs)
        inputs = self.model.extract_conditioning_features(inputs)
        return inputs

    def predict_type(
        self, inputs: Dict[str, torch.Tensor], use_log_probabilities: bool = False
    ) -> torch.Tensor:
        inputs = self.model.predict_type(inputs, use_log_probabilities)
        return inputs[properties.distribution_Z]

    def predict_distances(
        self, inputs: Dict[str, torch.Tensor], use_log_probabilities: bool = False
    ) -> torch.Tensor:
        inputs = self.model.predict_distances(inputs, use_log_probabilities)
        return inputs[properties.distribution_r_ij]
