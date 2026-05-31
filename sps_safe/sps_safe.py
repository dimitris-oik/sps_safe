# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable
import torch
from torch import nn


class SPS_safe(torch.optim.Optimizer):
    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            ell_star: float = 0.0,
            M: float = 1.0,
            weight_decay: float = 0.0,
            ):
        """
        Implements the SPS_safe optimizer for the Stochastic Subgradient Method.

        \gamma_t = (f_i(x^t) - \ell_i^*) / max{\|g_i^t\|^2, M}
        x^{t+1} = x^t - \gamma_t * g_i^t

        Arguments:
            params (iterable):
                Iterable of parameters to optimize or dicts defining parameter groups.
            ell_star (float):
                The lower bound \ell_i^* of the (mini-batch) loss. Typically 0 for non-negative losses.
            M (float):
                The safeguard threshold M > 0 on \|g_i^t\|^2.
            weight_decay (float):
                L2 weight-decay coefficient.
        """

        defaults = {"ell_star": ell_star, "M": M, "weight_decay": weight_decay}
        super().__init__(params, defaults)

        self.ell_star = ell_star
        self.M = M
        self.weight_decay = weight_decay

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

        self.grad_norm = self.compute_grad_terms().item()

        spsm = ((loss - self.ell_star) / max(self.M, self.grad_norm ** 2)).item()
        self.ss = spsm

        for group in self.param_groups:
            for p in group['params']:
                p.data.mul_(1 - self.weight_decay * spsm)
                p.data.add_(other=p.grad.data.detach(), alpha=-spsm)

        return loss

    @torch.no_grad()
    def compute_grad_terms(self):
        grad_norm = 0.
        for group in self.param_groups:
            for p in group['params']:
                g = p.grad.data
                grad_norm += torch.sum(torch.mul(g, g))

        grad_norm = torch.sqrt(grad_norm)
        return grad_norm
