# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable
import torch
from torch import nn


class IMA_SPS_safe(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            ell_star: float = 0.0,
            lambd: float = 1.0,
            M: float = 1.0,
            weight_decay: float = 0.0,
            ):
        """
        Implements the IMA-SPS_safe optimizer, the safeguarded Polyak-type step size for
        the Iterate Moving Average (IMA) momentum update rule.

        \eta_t = [f_i(x^t) - \ell_i^* + \lambda_t <g_i^t, x^t - x^{t-1}>]_+ / max{\|g_i^t\|^2, M}
        z^{t+1} = z^t - \eta_t * g_i^t
        x^{t+1} = (\lambda x^t + z^{t+1}) / (\lambda + 1)

        Setting M <= 0 at construction enables an auto-init of M to the first \|g_i^t\|^2.

        Arguments:
            params (iterable):
                Iterable of parameters to optimize or dicts defining parameter groups.
            ell_star (float):
                The lower bound \ell_i^* of the (mini-batch) loss. Typically 0 for non-negative losses.
            lambd (float):
                The IMA averaging parameter \lambda >= 0. Equivalent to SHB momentum via
                \beta = \lambda / (1 + \lambda).
            M (float):
                The safeguard threshold M on \|g_i^t\|^2. If M <= 0, auto-init from the first step.
            weight_decay (float):
                L2 weight-decay coefficient.
        """

        defaults = {"ell_star": ell_star, "lambd": lambd, "M": M, "weight_decay": weight_decay}
        super().__init__(params, defaults)

        self.lambd = lambd
        self.ell_star = ell_star
        self.M = M
        self.weight_decay = weight_decay
        self.number_steps = 0

        self.ss = 0.0
        self.grad_norm = 0.0

    def step(self, loss=None):
        """
        Performs a single optimization step.

        Parameters
        ----------
        loss : torch.tensor
            The loss tensor. Use this when the backward step has already been performed. By default None.

        Returns
        -------
        (Stochastic) Loss function value.
        """

        self.number_steps += 1
        _norm = 0.
        _dot = 0.

        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data.detach()
                state = self.state[p]

                if self.number_steps == 1:
                    state['z'] = p.detach().clone().to(p.device)

                z = state['z']
                _dot += torch.sum(torch.mul(grad, z - p.data))
                _norm += torch.sum(torch.mul(grad, grad))

        if self.M <= 0:
            ima_sps = (max(loss.item() - self.ell_star + _dot, 0) / _norm).item()
            self.M = _norm
        else:
            ima_sps = (max(loss.item() - self.ell_star + _dot, 0) / max(self.M, _norm)).item()

        self.grad_norm = torch.sqrt(_norm)
        self.ss = ima_sps

        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data.detach()
                state = self.state[p]

                z = state['z']
                z.add_(grad, alpha=-ima_sps)
                p.data.mul_(self.lambd / (1 + self.lambd)).add_(other=z, alpha=1 / (1 + self.lambd))

        return loss
