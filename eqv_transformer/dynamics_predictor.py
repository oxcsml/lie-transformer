from torch import nn
from attrdict import AttrDict

import torch
import torch.nn.functional as F
from lie_conv.dynamicsTrainer import Partial
from torchdiffeq import odeint


class DynamicsPredictor(nn.Module):
    """This class implements forward pass through our model, including loss computation."""

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def _rollout_model(self, z0, ts, sys_params, tol=1e-4):
        """inputs [z0: (bs, z_dim), ts: (bs, T), sys_params: (bs, n, c)]
        outputs pred_zs: (bs, T, z_dim)"""
        dynamics = Partial(self.predictor, sysP=sys_params)
        zs = odeint(dynamics, z0, ts[0], rtol=tol, method="rk4")
        return zs.permute(1, 0, 2)

    def forward(self, data):
        (z0, sys_params, ts), true_zs = data

        pred_zs = self._rollout_model(z0, ts, sys_params)
        mse = (pred_zs - true_zs).pow(2).mean()

        o = AttrDict()

        o.prediction = pred_zs
        o.mse = mse
        o.loss = mse # loss wrt which we train the model
        o.reports = AttrDict({"mse": o.mse})

        return o
