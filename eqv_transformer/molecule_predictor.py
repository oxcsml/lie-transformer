from torch import nn
from attrdict import AttrDict

import torch
import torch.nn.functional as F


class MoleculePredictor(nn.Module):
    """This class implements forward pass through our model, including loss computation."""

    def __init__(self, predictor, task, ds_stats=None):
        """Builds the model.

        Args:
            predictor: callable that takes an input, e.g. a molecule, and returns a prediction of a property of the molecule.
            task: which of the prediction tasks to do
            ds_stats: normalisation mean and variance of the targets. If None, do no normalisation.
        Returns:
            a dictionary of outputs.
        """
        super().__init__()
        self.predictor = predictor
        self.task = task
        self.ds_stats = ds_stats

    def forward(self, inpt, compute_loss=False):

        o = AttrDict()

        o.prediction = self.predictor(inpt)

        # label = label.long()
        target = inpt[self.task]

        # import pdb;
        # pdb.set_trace()
        if self.ds_stats is not None:
            meadian, mad = self.ds_stats

        if self.ds_stats is not None:
            meadian, mad = self.ds_stats

            target_norm = (target - meadian) / mad
            prediction_actual = o.prediction * mad + meadian
            o.prediction_actual = prediction_actual

            o.loss = (o.prediction - target_norm).abs().mean()
            o.mae = (prediction_actual - target).abs().mean()
        else:
            o.loss = (o.prediction - target).abs().mean()
            o.mae = o.loss
        # import pdb;
        # pdb.set_trace()

        o.reports = AttrDict({"loss": o.loss, "mae": o.mae})

        return o
