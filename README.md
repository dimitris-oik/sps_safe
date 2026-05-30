# SPS_safe

**Safeguarded Polyak step sizes for non-smooth stochastic optimization — robust to vanishing gradients in deep networks.**

[![Paper: ICML 2026](https://img.shields.io/badge/Paper-ICML%202026-1f6feb)](https://arxiv.org/abs/2512.02342)
[![arXiv](https://img.shields.io/badge/arXiv-2512.02342-b31b1b)](https://arxiv.org/abs/2512.02342)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

`SPS_safe` is the official PyTorch implementation of the optimizers proposed in:

> **Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization: Robust Performance Without Small (Sub)Gradients**
> Dimitris Oikonomou, Nicolas Loizou. *ICML 2026.*

The package provides two `torch.optim.Optimizer` subclasses — `SPS_safe` and `IMA_SPS_safe` — that adapt the Polyak step size for **convex, Lipschitz, non-smooth** objectives without requiring interpolation or oracle access to $f_i(x^*)$. Instead of capping the step size from above (as in SPS$_{\max}$), they place a single safeguard $M$ on the **denominator** $\\|g_i^t\\|^2$, which (i) prevents step-size blow-up when subgradients vanish, (ii) keeps the rule genuinely adaptive — never collapsing to a constant update — and (iii) admits an equivalent interpretation as an **adaptive clipped subgradient method** (Proposition 2.1 of the paper).

---

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [The algorithm](#the-algorithm)
- [API reference](#api-reference)
- [Experiments](#experiments)
- [Citation](#citation)
- [License](#license)

---

## Installation

From source:

```bash
git clone https://github.com/dimitris-oik/sps_safe.git
cd sps_safe
pip install -e .
```

Requirements: `torch`, `numpy`, `scipy` (only for the numpy experiments), Python 3.8+.

---

## Quick start

Both optimizers need the **(mini-batch) loss value** inside `.step()`. The only change to a standard PyTorch training loop is passing `loss=` to `.step()`:

```python
import torch
from sps_safe import SPS_safe

model     = MyModel()
criterion = torch.nn.CrossEntropyLoss()
optimizer = SPS_safe(model.parameters(), ell_star=0.0, M=1.0, weight_decay=5e-4)

for x, y in loader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step(loss=loss)               # <-- the only change vs. SGD/Adam
```

For the momentum variant (Stochastic Heavy Ball / Iterate Moving Average with the safeguarded Polyak step):

```python
from sps_safe import IMA_SPS_safe

optimizer = IMA_SPS_safe(
    model.parameters(),
    ell_star=0.0,
    lambd=1.0,           # IMA averaging param; equivalent SHB momentum is beta = lambd / (1 + lambd)
    M=1.0,               # set M <= 0 to auto-initialise from the first ||g||^2
    weight_decay=0.0,
)
```

---

## The algorithm

Given a mini-batch loss $f_i$ with stochastic subgradient $g_i^t \in \partial f_i(x^t)$, **SPS_safe** sets

$$\gamma_t = \frac{f_i(x^t) - \ell_i^*}{\max\\{\\|g_i^t\\|^2,\ M\\}}, \qquad x^{t+1} = x^t - \gamma_t g_i^t.$$

The single hyper-parameter $M > 0$ replaces *both* the clip $\gamma_b$ of SPS$_{\max}$ and the oracle values $f_i(x^*)$ required by SPS$^*$ / IMA-SPS.

For momentum, **IMA-SPS_safe** uses the Iterate Moving Average form of Stochastic Heavy Ball (Sebbouh et al., 2021) with a safeguarded Polyak step on $z$:

$$\eta_t = \frac{\big[f_i(x^t) - \ell_i^* + \lambda_t \langle g_i^t,\ x^t - x^{t-1}\rangle\big]_+}{\max\\{\\|g_i^t\\|^2,\ M\\}},$$

$$z^{t+1} = z^t - \eta_t g_i^t, \qquad x^{t+1} = \frac{\lambda x^t + z^{t+1}}{\lambda + 1}.$$

The averaging parameter $\lambda \ge 0$ corresponds to SHB momentum $\beta = \lambda / (1 + \lambda)$.

### Why safeguard the denominator?

For SPS$_{\max}$, the upper bound $\gamma_b$ is hit far more often than is comfortable in DNN training (§2.2 of the paper): on a ResNet-20 / CIFAR-10 sweep, $\gamma_b$ was selected in **31.8%** of iterations with the constant rule and in **98.45%** of iterations with the smoothed variant — meaning the method is, in practice, behaving like a hand-tuned learning-rate schedule rather than a Polyak step. `SPS_safe` is genuinely adaptive at every step regardless of $M$.

### Connection to gradient clipping

**Proposition 2.1.** *SSM with `SPS_safe` and $M = c^2$ is algebraically equivalent to Clipped SSM with the adaptive step size $\tilde\gamma_t = \frac{f_i(x^t) - \ell_i^*}{c \cdot \max\\{c,\ \\|g_i^t\\|\\}}$.* This gives the first theoretical guarantees for a Polyak-style clipped subgradient method.

### Theoretical guarantees

| Setting | Method | Rate (Cesàro / last iterate) |
|---|---|---|
| Convex, Lipschitz, non-smooth | `SPS_safe` | $O(T^{-1/2})$ to a neighborhood (Theorem 3.1) |
| Convex, Lipschitz + momentum | `IMA_SPS_safe` | $O(T^{-1/2})$ Cesàro **and** last-iterate (Theorems 3.4, 3.5) |
| Interpolated ($\sigma^2 = 0$) | both | neighborhood collapses; exact convergence |

All results hold **without** the interpolation assumption and **without** oracle access to $f_i(x^*)$ — both of which were required by prior Polyak-type step sizes in the non-smooth setting (Loizou et al., 2021; Garrigos et al., 2023; Gower et al., 2025).

---

## API reference

### `SPS_safe(params, ell_star=0.0, M=1.0, weight_decay=0.0)`

| Argument | Type | Default | Description |
|---|---|---|---|
| `params` | iterable | — | Parameters to optimize. |
| `ell_star` | float | `0.0` | Lower bound $\ell_i^*$ on the mini-batch loss. Typically `0.0` for non-negative losses. |
| `M` | float | `1.0` | Safeguard threshold on $\\|g_i^t\\|^2$ in the denominator. |
| `weight_decay` | float | `0.0` | L2 weight-decay coefficient. |

### `IMA_SPS_safe(params, ell_star=0.0, lambd=1.0, M=1.0, weight_decay=0.0)`

| Argument | Type | Default | Description |
|---|---|---|---|
| `params` | iterable | — | Parameters to optimize. |
| `ell_star` | float | `0.0` | Lower bound $\ell_i^*$ on the mini-batch loss. |
| `lambd` | float | `1.0` | IMA averaging parameter $\lambda \ge 0$. Equivalent SHB momentum is $\beta = \lambda / (1 + \lambda)$. |
| `M` | float | `1.0` | Safeguard threshold on $\\|g_i^t\\|^2$. Pass a value $\le 0$ to auto-initialise from the first $\\|g_i^t\\|^2$. |
| `weight_decay` | float | `0.0` | L2 weight-decay coefficient. |

Both classes expose the standard `torch.optim.Optimizer` interface. The only non-standard requirement is that `.step()` is called as `optimizer.step(loss=loss)`.

---

## Experiments

The [`numpy_exps/`](numpy_exps/) directory reproduces the §4.1 plots that empirically verify the convergence guarantees on non-smooth convex benchmarks:

- [`numpy_exps/loss.py`](numpy_exps/loss.py) — `SVM` (hinge loss) and `PhaseRetrieval` objectives.
- [`numpy_exps/methods.py`](numpy_exps/methods.py) — every step-size selection compared in the paper: `sgd`, `sgd_sps_max` (Loizou et al., 2021), `sgd_sps_plus` (Garrigos et al., 2023), `sgd_sps_m` (`SPS_safe`), and the IMA variants `ima`, `ima_sps_plus`, `ima_sps_m`, plus the last-iterate variants used in Theorem 3.5.
- [`numpy_exps/exps_svm.ipynb`](numpy_exps/exps_svm.ipynb) — SVM figures.
- [`numpy_exps/exps_phase.ipynb`](numpy_exps/exps_phase.ipynb) — Phase retrieval figures.

For the deep-learning experiments (ResNet-20 / ResNet-32 on CIFAR-10 / CIFAR-100), the full hyper-parameter protocol is described in §4.2 and Appendix C.1 of the paper. The training script is intentionally not shipped — `SPS_safe` and `IMA_SPS_safe` are drop-in `torch.optim.Optimizer`s and integrate with any standard PyTorch training loop (see [Quick start](#quick-start)).

The full paper (theory + extended experiments) is checked into this repository as [`paper.md`](paper.md).

---

## Citation

If you use this code or build on the method, please cite:

```bibtex
@inproceedings{oikonomou2026safeguarded,
  title = {Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization: Robust Performance Without Small (Sub)Gradients},
  author = {Oikonomou, Dimitris and Loizou, Nicolas},
  booktitle = {ICML},
  year = {2026},
}
```

---

## License

Released under the [MIT License](LICENSE).
