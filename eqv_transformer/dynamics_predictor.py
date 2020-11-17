from torch import nn
from attrdict import AttrDict

import torch
import torch.nn.functional as F
from lie_conv.dynamicsTrainer import Partial
from torchdiffeq import odeint

from lie_conv.hamiltonian import SpringV, SpringH, HamiltonianDynamics, KeplerV, KeplerH


class DynamicsPredictor(nn.Module):
    """This class implements forward pass through our model, including loss computation."""

    def __init__(self, predictor, debug=False, task="spring"):
        super().__init__()
        self.predictor = predictor
        self.debug = debug
        self.task = task

        if self.debug:
            print("DynamicsPredictor is in DEBUG MODE.")

    def _rollout_model(self, z0, ts, sys_params, tol=1e-4):
        """inputs [z0: (bs, z_dim), ts: (bs, T), sys_params: (bs, n, c)]
        outputs pred_zs: (bs, T, z_dim)"""
        dynamics = Partial(self.predictor, sysP=sys_params)
        zs = odeint(dynamics, z0, ts[0], rtol=tol, method="rk4")
        return zs.permute(1, 0, 2)

    def forward(self, data):
        o = AttrDict()

        (z0, sys_params, ts), true_zs = data

        pred_zs = self._rollout_model(z0, ts, sys_params)
        mse = (pred_zs - true_zs).pow(2).mean()

        if self.debug:
            if self.task == "spring":
                # currently a bit inefficient to do the below?
                with torch.no_grad():
                    (z0, sys_params, ts), true_zs = data

                    z = z0
                    m = sys_params[..., 0]  # assume the first component encodes masses
                    D = z.shape[-1]  # of ODE dims, 2*num_particles*space_dim
                    q = z[:, : D // 2].reshape(*m.shape, -1)
                    p = z[:, D // 2 :].reshape(*m.shape, -1)
                    V_pred = self.predictor.compute_V((q, sys_params))

                    k = sys_params[..., 1]
                    V_true = SpringV(q, k)

                    mse_V = (V_pred - V_true).pow(2).mean()

                    # dynamics
                    dyn_tz_pred = self.predictor(ts, z0, sys_params)

                    H = lambda t, z: SpringH(
                        z, sys_params[..., 0].squeeze(-1), sys_params[..., 1].squeeze(-1)
                    )
                    dynamics = HamiltonianDynamics(H, wgrad=False)
                    dyn_tz_true = dynamics(ts, z0)

                    mse_dyn = (dyn_tz_true - dyn_tz_pred).pow(2).mean()

            if self.task == "nbody":
                # currently a bit inefficient to do the below?
                with torch.no_grad():
                    (z0, sys_params, ts), true_zs = data

                    z = z0
                    m = sys_params[..., 0]  # assume the first component encodes masses
                    D = z.shape[-1]  # of ODE dims, 2*num_particles*space_dim
                    q = z[:, : D // 2].reshape(*m.shape, -1)
                    p = z[:, D // 2 :].reshape(*m.shape, -1)
                    V_pred = self.predictor.compute_V((q, sys_params))

                    V_true = KeplerV(q, m)

                    mse_V = (V_pred - V_true).pow(2).mean()

                    # dynamics
                    dyn_tz_pred = self.predictor(ts, z0, sys_params)

                    H = lambda t, z: KeplerH(
                        z, sys_params[..., 0].squeeze(-1)
                    )
                    dynamics = HamiltonianDynamics(H, wgrad=False)
                    dyn_tz_true = dynamics(ts, z0)

                    mse_dyn = (dyn_tz_true - dyn_tz_pred).pow(2).mean()


            o.mse_dyn = mse_dyn
            o.mse_V = mse_V

        o.prediction = pred_zs
        o.mse = mse
        o.loss = mse  # loss wrt which we train the model

        if self.debug:
            o.reports = AttrDict({"mse": o.mse, "mse_V": o.mse_V, "mse_dyn": o.mse_dyn})
        else:
            o.reports = AttrDict({"mse": o.mse})

        return o
