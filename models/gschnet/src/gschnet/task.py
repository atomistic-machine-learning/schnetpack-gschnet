from pathlib import Path
from typing import Optional, Dict, List, Type, Any, Union

import pytorch_lightning as pl
import torch
from torch import nn as nn
from gschnet.transform import (
    BuildAtomsTrajectory,
    GeneralCachedNeighborList,
    ConditionalGSchNetNeighborList,
)
from gschnet import ConditionalGenerativeSchNet
import gschnet.properties as properties

__all__ = ["ConditionalGenerativeSchNetTask"]


class ConditionalGenerativeSchNetTask(pl.LightningModule):
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
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor

        self.grad_enabled = len(self.model.required_derivatives) > 0
        self.inference_mode = False

        self.type_loss_fn = nn.NLLLoss(reduction="mean")
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
            self.model.initialize_postprocessors(dm)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        results = self.model(inputs)
        return results

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
        pred = self(batch)
        loss = self.loss_fn(pred, batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)
        pred = self(batch)
        loss = self.loss_fn(pred, batch, return_individual_losses=True)
        self.log("val_loss", loss[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss_type", loss[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss_dist", loss[2], on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss[0]}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)
        pred = self(batch)
        loss = self.loss_fn(pred, batch, return_individual_losses=True)
        self.log("test_loss", loss[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_loss_type", loss[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_loss_dist", loss[2], on_step=False, on_epoch=True, prog_bar=True)
        return {"test_loss": loss[0]}

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )

        if self.scheduler_cls:
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)

            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            return [optimizer], [optimconf]
        else:
            return optimizer

    def to_torchscript(
        self,
        file_path: Optional[Union[str, Path]] = None,
        method: Optional[str] = "script",
        example_inputs: Optional[Any] = None,
        **kwargs,
    ) -> Union[torch.ScriptModule, Dict[str, torch.ScriptModule]]:
        imode = self.inference_mode
        self.inference_mode = True
        script = super().to_torchscript(file_path, method, example_inputs, **kwargs)
        self.inference_mode = imode
        return script

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
