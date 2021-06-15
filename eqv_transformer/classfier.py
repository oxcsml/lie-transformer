from torch import nn
from attrdict import AttrDict

import torch
import torch.nn.functional as F


class Classifier(nn.Module):
    """This class implements forward pass through our model, including loss computation."""

    def __init__(self, encoder):
        """Builds the model.

        Args:
            encoder: callable that takes an input, e.g. an image, and returns a representation.
        Returns:
            a dictionary of outputs.
        """
        super().__init__()
        self.encoder = encoder

    def forward(self, inpt, label=None):

        o = AttrDict()

        # compute model forward pass
        if isinstance(inpt, (list, tuple)):
            o.logits = self.encoder(*inpt)
        else:
            o.logits = self.encoder(inpt)

        # compute the predicted classes
        _, o.predicted = torch.max(o.logits, -1)

        if label is not None:
            label = label.long()

            o.loss = F.cross_entropy(o.logits.transpose(1, 2), label)
            o.acc = o.predicted.eq(label).float().mean(())

        o.reports = AttrDict({"loss": o.loss, "acc": o.acc})

        return o