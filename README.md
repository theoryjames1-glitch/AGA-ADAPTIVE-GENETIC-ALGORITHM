# AGA-ADAPTIVE-GENETIC-ALGORITHM

### AGA.py
```python
# -*- coding: utf-8 -*-
"""
AGA.py — Adaptive Genetic Algorithm (PyTorch)
A control/DSP reinterpretation of evolution for optimizer coefficients.

Provides:
- AGAConfig: hyperparameters & guards
- AGAAgent: single adaptive optimizer (updates θ and adapts α, μ, σ)
- AGAEnsemble: multi-agent with coefficient consensus (recombination)

Run this file directly to see a quadratic demo with pretty logs.


Adaptive Genetic Algorithm (AGA) — Theory (compact, pasteable)
==============================================================

Definition
----------
AGA is a control- and signal-processing reinterpretation of "evolution"
applied to optimizer *coefficients* (not genomes). The evolving object is a
small vector of differentiable, bounded coefficients that steer parameter
updates. Evolution proceeds as a Markov process of coefficient updates driven
by filtered performance signals, with optional ensemble consensus standing in
for recombination.

State, Signals, and Guards
--------------------------
State at step t:
  x_t = [theta_t, alpha_t, mu_t, sigma_t]
  Constraints: alpha_min <= alpha_t <= alpha_max, 0 <= mu_t < 1, 0 <= sigma_t <= sigma_max

Observations (from loss l_t = l(theta_t; z_t) and reward r_t = -l_t or task reward):
  Delta_l_t = l_t - l_{t-1}
  Delta_r_t = r_t - r_{t-1}
  v_t       = EMA_tau[(l_t - EMA_tau[l])^2]    # volatility proxy (bias-reduced EW variance)

Signals (normalized features):
  surprise   = Delta_l_t / sqrt(v_t + eps)
  trend      = Delta_r_t
  volatility = sqrt(v_t)
  s_t        = [surprise, trend, volatility, 1]^T

Gradients & momentum proxy (base optimizer skeleton):
  g_t = ∇_theta l(theta_t)
  m_{t+1} = mu_t * m_t + (1 - mu_t) * g_t

Law 1 — Variation (Exploration-as-Dither)
-----------------------------------------
Inject small, state-dependent Gaussian dither into both parameters and,
optionally, into coefficient updates (via log-space noise terms):

  theta_{t+1} = theta_t - alpha_t * m_{t+1} + sigma_t * N(0, I)

Law 2 — Selection (Feedback Filtering of Coefficients)
------------------------------------------------------
Coefficients are updated by smooth, differentiable maps modulated by s_t
through a (possibly learned) feature map phi(s_t). Multiplicative updates in
log-space preserve positivity/scale, logistic keeps mu in [0, 1).

Let k_alpha, k_mu, k_sigma be gain vectors (same length as phi(s_t)). Let
clip_[lo, hi](.) enforce bounds. Let tau_* be log-space noise scales.

  alpha_{t+1} = clip_[alpha_min, alpha_max](
                   alpha_t * exp( k_alpha^T phi(s_t) + tau_alpha * N(0,1) ) )

  mu_{t+1}    = (1 - eps_mu) * sigmoid( k_mu^T phi(s_t) + tau_mu * N(0,1) )
                # keep strictly < 1 to avoid exploding momentum accumulator

  sigma_{t+1} = clip_[0, sigma_max](
                   sigma_t * exp( k_sigma^T phi(s_t) + tau_sigma * N(0,1) ) )

Resonance Gate (stability–plasticity control, optional):
  Let novelty exceed volatility by a threshold kappa, and include reward trend:

    rho_t = sigmoid( c0 + c1 * max(|Delta_l_t| - kappa * sqrt(v_t), 0) + c2 * Delta_r_t )

  Then shrink or expand selection gains by rho_t:
    k_alpha <- rho_t * k_alpha,  k_mu <- rho_t * k_mu,  k_sigma <- rho_t * k_sigma

Per-step budgets (anti-chatter):
  Optionally clamp Delta(log alpha) and Delta(log sigma) to small bounds to
  prevent chatter and improve robustness on non-stationary data.

Law 3 — Recombination (Low-Pass Consensus, Optional Ensemble)
-------------------------------------------------------------
Run N agents j=1..N in parallel with diverse seeds/initializations. Compute
soft performance weights from current losses (lower is better):
  w_t^(j) ∝ exp( -beta * (l_t^(j) - min_i l_t^(i)) ),  sum_j w_t^(j) = 1

Apply a consensus/low-pass filter to coefficients (not parameters):
  c_{t+1}^{(j)} <- (1 - gamma_t) * c_{t+1}^{(j)} + gamma_t * sum_i w_t^(i) * c_{t+1}^{(i)}
  for c in {alpha, mu, sigma}, then re-clip to legal bounds.

Markov Property and Well-Posedness
----------------------------------
Given x_t and s_t (constructed from a finite window of past losses/rewards),
the distribution of x_{t+1} is fully determined, so {x_t} is a (possibly
time-inhomogeneous) Markov process with compact constraints:
  C = [alpha_min, alpha_max] × [0, 1) × [0, sigma_max]

Specializations (Unification)
-----------------------------
* SGD:            k_alpha = k_mu = k_sigma = 0,  sigma_t ≡ 0
* Momentum/Adam:  choose phi to include g_t^2 or v_t; adapt alpha in log-space;
                  fix or adapt mu; sigma_t ≡ 0
* Bandits/RL:     include Delta_r_t in phi so positive reward trends increase
                  alpha/sigma under novelty and damp them under noise

Stability Sketch (Robbins–Monro)
--------------------------------
For L-smooth convex l, bounded gradient variance, and bounded coefficients with
  sum_t E[alpha_t]   = ∞,
  sum_t E[alpha_t^2] < ∞,
  sum_t E[sigma_t^2] < ∞,
the standard stochastic approximation machinery gives convergence of theta_t to
a minimizer in expectation. The resonance gate helps maintain these summability
conditions under sustained volatility and non-stationarity.

Continuous-Time View (optional SDE/ODE limit)
---------------------------------------------
As Δt → 0:

  d theta  = -alpha(t) * m(t) dt + sigma(t) dW_t
  dot alpha = alpha * ( k_alpha^T phi(s) )
  dot mu    = (1 - mu) * sigmoid( k_mu^T phi(s) ) - mu
  dot sigma = sigma * ( k_sigma^T phi(s) )

Telemetry & Practical Notes
---------------------------
* gamma_t = alpha_t * ||g_t|| is a useful scale/gauge; keeping gamma_t in a
  target band improves robustness across tasks with different gradient scales.
* Multiplicative updates (log-space) + per-step budgets + post-consensus
  re-clipping provide strong safety defaults without hand-crafted schedules.
* All maps are differentiable, enabling meta-learning of gains, features, and
  time-constants end-to-end, and analysis with control-theoretic tools.

Slogan
------
Adaptive Evolution = Learning as Resonant Feedback.
Not survival of the fittest — stability of the adaptive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

# A loss function may return just a scalar loss Tensor,
# or a tuple (loss, grad) where grad has the same shape as theta.
LossFn = Callable[[Tensor], Union[Tensor, Tuple[Tensor, Tensor]]]


# -------------------- utils --------------------
def _clip_scalar(x: float, lo: float, hi: float) -> float:
    return float(max(min(x, hi), lo))


class _EMAStat:
    """Exponential moving mean & bias-reduced variance (Welford-style EW)."""

    def __init__(self, beta: float):
        self.beta = beta
        self.mean: Optional[float] = None
        self.var: float = 0.0  # EW variance around moving mean

    def update(self, x: float):
        if self.mean is None:
            self.mean = float(x)
            self.var = 0.0
            return
        b = self.beta
        m_prev = self.mean
        m_new = b * m_prev + (1.0 - b) * x
        # bias-reduced EW variance: product of deviations around old/new means
        self.var = b * self.var + (1.0 - b) * (x - m_prev) * (x - m_new)
        self.mean = m_new


# -------------------- config --------------------
@dataclass
class AGAConfig:
    # Coefficient bounds
    alpha_min: float = 1e-5
    alpha_max: float = 1.0
    sigma_max: float = 1.0
    eps_mu: float = 0.0  # small damping for logistic map

    # Noise (std in log-space for multiplicative maps)
    tau_alpha: float = 0.0
    tau_mu: float = 0.0
    tau_sigma: float = 0.0

    # Selection gains (dot with phi(s_t)); defaults match s=[surprise, Δr, volatility, 1]
    k_alpha: List[float] = field(default_factory=lambda: [0.0, +0.50, -0.30, 0.0])
    k_mu:    List[float] = field(default_factory=lambda: [0.0, +0.25, -0.10, -0.25])
    k_sigma: List[float] = field(default_factory=lambda: [+0.25, +0.25, +0.25, -0.20])

    # Resonance gate ρ parameters
    c0: float = 0.0
    c1: float = 0.75
    c2: float = 0.50
    kappa: float = 1.0

    # EMA for loss stats
    ema_beta: float = 0.9
    eps_var: float = 1e-8

    # Ensemble consensus (recombination)
    gamma_consensus: float = 0.25
    beta_weight: float = 2.0  # softmax inverse temp for weights

    # Switches
    adapt_alpha: bool = True
    adapt_mu: bool = True
    adapt_sigma: bool = True
    param_noise: bool = True

    # Optional grad clipping
    clip_grad_norm: Optional[float] = None

    # Per-step Δlog budgets (None disables)
    max_delta_log_alpha: Optional[float] = 0.5
    max_delta_log_sigma: Optional[float] = 0.5

    # Feature map φ: Tensor[4] -> Tensor[k]
    phi: Optional[Callable[[Tensor], Tensor]] = None

    # Device/dtype defaults
    dtype: torch.dtype = torch.float64
    device: Union[str, torch.device] = "cpu"


# -------------------- agent --------------------
class AGAAgent:
    """
    Single-agent Adaptive Genetic Algorithm (AGA) in PyTorch.
    State x_t = [theta, alpha, mu, sigma]; coefficients evolve by differentiable maps.
    """

    def __init__(
        self,
        dim: int,
        config: AGAConfig = AGAConfig(),
        seed: Optional[int] = None,
        init_theta: Optional[Tensor] = None,
        init_alpha: float = 1e-2,
        init_mu: float = 0.0,
        init_sigma: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.cfg = config
        self.device = torch.device(device or config.device)
        self.dtype = dtype or config.dtype

        # RNG
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(int(seed))

        # Parameters
        if init_theta is None:
            self.theta = torch.zeros(dim, dtype=self.dtype, device=self.device, requires_grad=True)
        else:
            self.theta = init_theta.detach().to(device=self.device, dtype=self.dtype).requires_grad_(True)

        self.alpha = float(init_alpha)
        self.mu = float(init_mu)
        self.sigma = float(init_sigma)
        self.m = torch.zeros_like(self.theta)  # momentum proxy

        # Stats & history
        self._loss_ema = _EMAStat(beta=self.cfg.ema_beta)
        self._prev_loss: Optional[float] = None
        self._prev_reward: Optional[float] = None

        # Gains on device
        self.k_alpha = torch.as_tensor(self.cfg.k_alpha, dtype=self.dtype, device=self.device)
        self.k_mu    = torch.as_tensor(self.cfg.k_mu,    dtype=self.dtype, device=self.device)
        self.k_sigma = torch.as_tensor(self.cfg.k_sigma, dtype=self.dtype, device=self.device)

    # ---- internal helpers ----
    def _budgeted(self, d: float, budget: Optional[float]) -> float:
        if budget is None:
            return float(d)
        return float(max(min(d, +budget), -budget))

    # ---- main step ----
    def step(self, loss_fn: LossFn) -> Dict[str, float]:
        """
        One AGA step: compute loss/grad, update EMA → signals, adapt α/μ/σ, then update θ.
        loss_fn(theta) -> loss  OR  (loss, grad)
        """
        # --- forward & gradient ---
        out = loss_fn(self.theta)
        if isinstance(out, (tuple, list)):
            loss, grad = out
        else:
            loss = out
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite loss encountered")
            grad = torch.autograd.grad(loss, self.theta, retain_graph=False, create_graph=False, allow_unused=False)[0]

        # Guards (common)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite loss encountered")
        if grad is None or grad.shape != self.theta.shape:
            raise ValueError(f"grad is None or wrong shape: {None if grad is None else grad.shape} vs {self.theta.shape}")
        if not torch.isfinite(grad).all().item():
            raise FloatingPointError("Non-finite gradient encountered")

        # Optional grad clipping
        if self.cfg.clip_grad_norm is not None:
            gnorm_val = torch.linalg.norm(grad)
            gnorm = float(gnorm_val.item())
            if gnorm > self.cfg.clip_grad_norm and gnorm > 0.0:
                grad = grad * (self.cfg.clip_grad_norm / gnorm)

        # --- stats & signals ---
        loss_f = float(loss.item())
        self._loss_ema.update(loss_f)
        mean = self._loss_ema.mean if self._loss_ema.mean is not None else loss_f
        v = max(self._loss_ema.var, 0.0)

        dloss = 0.0 if self._prev_loss is None else (loss_f - self._prev_loss)
        reward = -loss_f
        dr = 0.0 if self._prev_reward is None else (reward - self._prev_reward)
        self._prev_loss = loss_f
        self._prev_reward = reward

        surprise = dloss / (v + self.cfg.eps_var) ** 0.5
        volatility = v ** 0.5

        s = torch.tensor([surprise, dr, volatility, 1.0], dtype=self.dtype, device=self.device)
        phi = self.cfg.phi or (lambda x: x)
        φs = phi(s)
        if φs.ndim != 1:
            raise ValueError(f"phi(s) must be 1D, got shape {tuple(φs.shape)}")

        klen = φs.numel()
        if self.k_alpha.numel() != klen or self.k_mu.numel() != klen or self.k_sigma.numel() != klen:
            raise ValueError(
                f"Mismatch: phi(s) has length {klen}, but k_alpha/k_mu/k_sigma are "
                f"{self.k_alpha.numel()}/{self.k_mu.numel()}/{self.k_sigma.numel()}"
            )

        # Resonance gate ρ ∈ (0,1)
        novelty_over_vol = max(abs(dloss) - self.cfg.kappa * (v + self.cfg.eps_var) ** 0.5, 0.0)
        rho = float(torch.sigmoid(torch.tensor(self.cfg.c0 + self.cfg.c1 * novelty_over_vol + self.cfg.c2 * dr)).item())

        # --- coefficient updates ---
        # α (multiplicative, log-space)
        if self.cfg.adapt_alpha:
            noise = self.cfg.tau_alpha * torch.randn((), generator=self.gen, device=self.device).item()
            delta_log_alpha = rho * float(torch.dot(self.k_alpha, φs).item()) + noise
            delta_log_alpha = self._budgeted(delta_log_alpha, self.cfg.max_delta_log_alpha)
            self.alpha = _clip_scalar(self.alpha * float(torch.exp(torch.tensor(delta_log_alpha)).item()),
                                      self.cfg.alpha_min, self.cfg.alpha_max)

        # μ (logistic to [0,1))
        if self.cfg.adapt_mu:
            noise = self.cfg.tau_mu * torch.randn((), generator=self.gen, device=self.device).item()
            mu_raw = rho * float(torch.dot(self.k_mu, φs).item()) + noise
            mu_raw = float(max(min(mu_raw, 20.0), -20.0))
            self.mu = (1.0 - self.cfg.eps_mu) * float(torch.sigmoid(torch.tensor(mu_raw)).item())
            self.mu = min(max(self.mu, 0.0), 1.0 - 1e-6)

        # σ (multiplicative, nonnegative)
        if self.cfg.adapt_sigma:
            noise = self.cfg.tau_sigma * torch.randn((), generator=self.gen, device=self.device).item()
            delta_log_sigma = rho * float(torch.dot(self.k_sigma, φs).item()) + noise
            delta_log_sigma = self._budgeted(delta_log_sigma, self.cfg.max_delta_log_sigma)
            self.sigma = _clip_scalar(self.sigma * float(torch.exp(torch.tensor(delta_log_sigma)).item()),
                                      0.0, self.cfg.sigma_max)

        # --- momentum & parameter update ---
        self.m = self.mu * self.m + (1.0 - self.mu) * grad
        step_vec = -self.alpha * self.m

        if self.cfg.param_noise and self.sigma > 0.0:
            noise_theta = torch.randn_like(self.theta, generator=self.gen) * self.sigma
        else:
            noise_theta = torch.zeros_like(self.theta)

        with torch.no_grad():
            self.theta.add_(step_vec + noise_theta)
        self.theta.requires_grad_(True)

        # Telemetry
        gnorm = float(torch.linalg.norm(grad).item())
        steplen = float(torch.linalg.norm(step_vec).item())
        gamma = float(self.alpha * gnorm)

        return {
            "loss": loss_f,
            "mean_loss": float(mean),
            "var_loss": float(v),
            "alpha": float(self.alpha),
            "mu": float(self.mu),
            "sigma": float(self.sigma),
            "rho": float(rho),
            "grad_norm": gnorm,
            "step_norm": steplen,
            "gamma": gamma,
        }

    # Optional alias
    update = step


# -------------------- ensemble --------------------
class AGAEnsemble:
    """
    Run N agents in parallel and apply coefficient consensus (recombination).
    Only coefficients (alpha, mu, sigma) are averaged; parameters remain diverse.
    """

    def __init__(self, agents: List[AGAAgent], config: Optional[AGAConfig] = None, seeds: Optional[List[int]] = None):
        assert len(agents) >= 2, "Ensemble requires at least two agents."
        self.agents = agents
        self.cfg = config or agents[0].cfg
        if seeds is not None:
            assert len(seeds) == len(agents), "seeds length must match number of agents"
            for a, s in zip(self.agents, seeds):
                a.gen = torch.Generator(device=a.device)
                a.gen.manual_seed(int(s))

    def step(self, loss_fn: LossFn) -> Dict[str, float]:
        logs = [agent.step(loss_fn) for agent in self.agents]
        losses = torch.tensor([lg["loss"] for lg in logs], dtype=torch.float64)

        # Soft performance weights (lower loss => higher weight)
        shifted = losses - torch.min(losses)
        weights = torch.exp(-self.cfg.beta_weight * shifted)
        weights = weights / (weights.sum() + 1e-12)

        # Weighted average of coefficients
        alphas = torch.tensor([a.alpha for a in self.agents], dtype=torch.float64)
        mus    = torch.tensor([a.mu    for a in self.agents], dtype=torch.float64)
        sigmas = torch.tensor([a.sigma for a in self.agents], dtype=torch.float64)

        bar_alpha = float(torch.dot(weights, alphas).item())
        bar_mu    = float(torch.dot(weights, mus).item())
        bar_sigma = float(torch.dot(weights, sigmas).item())

        # Consensus filter + enforce bounds after mixing
        for a in self.agents:
            a.alpha = (1.0 - self.cfg.gamma_consensus) * a.alpha + self.cfg.gamma_consensus * bar_alpha
            a.mu    = (1.0 - self.cfg.gamma_consensus) * a.mu    + self.cfg.gamma_consensus * bar_mu
            a.sigma = (1.0 - self.cfg.gamma_consensus) * a.sigma + self.cfg.gamma_consensus * bar_sigma

            a.alpha = _clip_scalar(a.alpha, a.cfg.alpha_min, a.cfg.alpha_max)
            a.mu    = min(max(a.mu, 0.0), 1.0 - 1e-6)
            a.sigma = _clip_scalar(a.sigma, 0.0, a.cfg.sigma_max)

        return {
            "mean_loss": float(torch.mean(losses).item()),
            "min_loss": float(torch.min(losses).item()),
            "max_loss": float(torch.max(losses).item()),
            "bar_alpha": bar_alpha,
            "bar_mu": bar_mu,
            "bar_sigma": bar_sigma,
        }


# -------------------- demo --------------------
def _pretty_log(t: int, log: Dict[str, float], err: Optional[float] = None):
    print(
        f"{t:04d} | loss {log['loss']:.6f} | mean {log['mean_loss']:.6f} | var {log['var_loss']:.3e} "
        f"| α {log['alpha']:.4f} | μ {log['mu']:.3f} | σ {log['sigma']:.4f} "
        f"| γ {log['gamma']:.3e}" + (f" | ‖θ-θ*‖ {err:.3e}" if err is not None else "")
    )


if __name__ == "__main__":
    # Small quadratic demo: minimize 0.5*θ^T A θ - b^T θ
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--print-every", type=int, default=25)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = args.device
    dtype = torch.float64

    A = torch.tensor([[3.0, 0.0], [0.0, 1.0]], dtype=dtype, device=device)
    b = torch.tensor([1.0, -2.0], dtype=dtype, device=device)
    theta_star = torch.linalg.solve(A, b)

    def quad_loss(theta: Tensor) -> Tensor:
        return 0.5 * (theta @ (A @ theta)) - (b @ theta)

    cfg = AGAConfig(
        alpha_min=1e-5, alpha_max=0.2, sigma_max=0.05, kappa=1.0, ema_beta=0.95,
        tau_alpha=0.0, tau_mu=0.0, tau_sigma=0.0, param_noise=False,
        max_delta_log_alpha=0.25, max_delta_log_sigma=0.25,
        device=device, dtype=dtype,
        # For a crisp demo you can freeze adaptation:
        # adapt_alpha=False, adapt_mu=False, adapt_sigma=False,
    )
    agent = AGAAgent(dim=2, config=cfg, seed=42, init_alpha=0.05, init_mu=0.0, init_sigma=0.0,
                     device=device, dtype=dtype)

    print(" step |        loss |       mean |       var    |    alpha |  mu  |  sigma |     gamma |   param_err")
    print("------+-------------+------------+--------------+----------+------+--------+-----------+------------")
    for t in range(1, args.steps + 1):
        log = agent.step(quad_loss)
        if (t % args.print_every) == 0 or t in (1, args.steps):
            err = torch.linalg.norm(agent.theta.detach() - theta_star).item()
            _pretty_log(t, log, err)
```

### AGATest.py

```python
# tests/test_AGA.py
import math
import pytest
import torch

from AGA import AGAAgent, AGAConfig, AGAEnsemble

DTYPE = torch.float64
DEVICE = "cpu"


# ---------- helpers ----------
def quad(dim=2, A_diag=(3.0, 1.0), b_vals=(1.0, -2.0), *, device=DEVICE, dtype=DTYPE):
    """Device/dtype-aware diagonal quadratic:
       loss = 0.5 * theta^T diag(A_diag) theta - b^T theta
       grad = diag(A_diag) theta - b
    """
    A = torch.diag(torch.tensor(A_diag[:dim], dtype=dtype, device=device))
    b = torch.tensor(b_vals[:dim], dtype=dtype, device=device)

    def loss_only(theta: torch.Tensor) -> torch.Tensor:
        return 0.5 * (theta @ (A @ theta)) - (b @ theta)

    def loss_and_grad(theta: torch.Tensor):
        loss = 0.5 * (theta @ (A @ theta)) - (b @ theta)
        grad = A @ theta - b
        return loss, grad

    theta_star = torch.linalg.solve(A, b)
    return loss_only, loss_and_grad, theta_star


# ---------- tests ----------
def test_quadratic_converges_baseline_nonadaptive():
    """Freeze adaptation so we get a crisp convergence assertion (< 1e-2 error)."""
    loss_only, _, theta_star = quad(device=DEVICE, dtype=DTYPE)
    cfg = AGAConfig(
        adapt_alpha=False, adapt_mu=False, adapt_sigma=False,
        alpha_min=1e-5, alpha_max=0.2, sigma_max=0.0,
        ema_beta=0.95, param_noise=False,
        dtype=DTYPE, device=DEVICE,
    )
    agent = AGAAgent(dim=2, config=cfg, seed=123, init_alpha=0.05, init_mu=0.0, init_sigma=0.0)
    errs, losses = [], []
    for _ in range(250):
        log = agent.step(loss_only)
        losses.append(log["loss"])
        err = torch.linalg.norm(agent.theta.detach() - theta_star).item()
        errs.append(err)

    assert losses[-1] < losses[0], "final loss should be less than initial loss"
    assert errs[-1] < errs[0], "final parameter error should be less than initial"
    assert errs[-1] < 1e-2, f"final parameter error too large: {errs[-1]:.3e}"


def test_adaptive_makes_progress():
    """With adaptation on, we at least expect progress (looser assertion)."""
    loss_only, _, _ = quad(device=DEVICE, dtype=DTYPE)
    cfg = AGAConfig(
        alpha_min=1e-5, alpha_max=0.2, sigma_max=0.0,
        ema_beta=0.95, param_noise=False,
        tau_alpha=0.0, tau_mu=0.0, tau_sigma=0.0,
        max_delta_log_alpha=0.25, max_delta_log_sigma=0.25,
        dtype=DTYPE, device=DEVICE,
    )
    agent = AGAAgent(dim=2, config=cfg, seed=123, init_alpha=0.05, init_mu=0.0, init_sigma=0.0)
    losses = []
    for _ in range(250):
        losses.append(agent.step(loss_only)["loss"])
    assert losses[-1] < losses[0]


def test_bounds_and_clamps_and_gamma_present():
    """Ensure coefficients remain within bounds and gamma is reported."""
    loss_only, _, _ = quad(device=DEVICE, dtype=DTYPE)
    cfg = AGAConfig(
        alpha_min=1e-5, alpha_max=0.1, sigma_max=0.1,
        max_delta_log_alpha=0.1, max_delta_log_sigma=0.1,
        tau_alpha=0.0, tau_mu=0.0, tau_sigma=0.0,
        ema_beta=0.95, param_noise=False,
        dtype=DTYPE, device=DEVICE,
    )
    agent = AGAAgent(dim=2, config=cfg, seed=0, init_alpha=0.05)
    for _ in range(120):
        log = agent.step(loss_only)
        assert cfg.alpha_min <= agent.alpha <= cfg.alpha_max
        assert 0.0 <= agent.mu < 1.0
        assert 0.0 <= agent.sigma <= cfg.sigma_max
        assert "gamma" in log and log["gamma"] >= 0.0


def test_consensus_reclip():
    """Start some agents outside bounds; after consensus they should be clipped and within bounds."""
    loss_only, _, _ = quad(device=DEVICE, dtype=DTYPE)
    cfg = AGAConfig(
        alpha_min=1e-5, alpha_max=0.05, sigma_max=0.05,
        ema_beta=0.9, param_noise=False,
        dtype=DTYPE, device=DEVICE,
    )
    agents = [
        AGAAgent(dim=2, config=cfg, seed=1, init_alpha=0.05),
        AGAAgent(dim=2, config=cfg, seed=2, init_alpha=0.10),  # above max initially
        AGAAgent(dim=2, config=cfg, seed=3, init_alpha=1e-6),  # below min initially
    ]
    ens = AGAEnsemble(agents, config=cfg)
    for _ in range(60):
        summary = ens.step(loss_only)
        for a in ens.agents:
            assert cfg.alpha_min <= a.alpha <= cfg.alpha_max
            assert 0.0 <= a.mu < 1.0
            assert 0.0 <= a.sigma <= cfg.sigma_max
        assert "bar_alpha" in summary


def test_phi_hook_identity_and_mismatch():
    calls = {"n": 0}

    def phi(s: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        out = s.clone()
        out[1] = 2.0 * out[1]  # simple feature tweak
        return out

    cfg = AGAConfig(phi=phi, ema_beta=0.9, param_noise=False, dtype=DTYPE, device=DEVICE)
    agent = AGAAgent(dim=2, config=cfg, seed=0)
    loss_only, _, _ = quad(device=DEVICE, dtype=DTYPE)
    agent.step(loss_only)
    assert calls["n"] == 1, "phi() should be called once"

    # Now a deliberate mismatch: phi returns 5-dim, but k_* are length-4
    def phi_bad(s: torch.Tensor) -> torch.Tensor:
        return torch.cat([s, torch.tensor([0.0], dtype=s.dtype, device=s.device)], dim=0)

    cfg_bad = AGAConfig(phi=phi_bad, dtype=DTYPE, device=DEVICE)
    agent_bad = AGAAgent(dim=2, config=cfg_bad)
    with pytest.raises(ValueError):
        agent_bad.step(loss_only)


def test_nonfinite_guard_raises():
    cfg = AGAConfig(param_noise=False, dtype=DTYPE, device=DEVICE)
    agent = AGAAgent(dim=2, config=cfg, seed=0)

    class LossMaker:
        def __init__(self):
            self.t = 0
        def __call__(self, theta: torch.Tensor):
            self.t += 1
            if self.t == 5:
                # return a NaN loss to trigger the guard
                return torch.tensor(float("nan"), dtype=DTYPE, device=theta.device)
            # simple squared norm loss (no grad provided, uses autograd)
            return 0.5 * torch.dot(theta, theta)

    lm = LossMaker()  # <-- instantiate once
    with pytest.raises(FloatingPointError):
        for _ in range(10):
            agent.step(lm)


def test_tuple_loss_grad_path():
    _, loss_and_grad, _ = quad(device=DEVICE, dtype=DTYPE)
    cfg = AGAConfig(param_noise=False, dtype=DTYPE, device=DEVICE)
    agent = AGAAgent(dim=2, config=cfg, seed=0)
    # Should run without error when (loss, grad) tuple is provided
    for _ in range(5):
        log = agent.step(loss_and_grad)
        assert "grad_norm" in log and log["grad_norm"] >= 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_single_step():
    device = "cuda"
    loss_only, _, _ = quad(device=device, dtype=torch.float64)
    cfg = AGAConfig(param_noise=False, device=device, dtype=torch.float64)
    agent = AGAAgent(dim=2, config=cfg, seed=0, device=device, dtype=torch.float64)
    log = agent.step(loss_only)
    assert math.isfinite(log["loss"])


def test_grad_clipping_applied():
    """Verify grad clipping limits the reported grad norm."""
    loss_only, _, _ = quad(device=DEVICE, dtype=DTYPE)
    cfg = AGAConfig(param_noise=False, dtype=DTYPE, device=DEVICE, clip_grad_norm=1.0,
                    adapt_alpha=False, adapt_mu=False, adapt_sigma=False)
    agent = AGAAgent(dim=2, config=cfg, seed=0, init_alpha=0.05)
    log = agent.step(loss_only)
    assert log["grad_norm"] <= 1.0 + 1e-12


def test_update_alias_calls_step():
    """Ensure .update() alias is present and returns a log dict like .step()."""
    loss_only, _, _ = quad(device=DEVICE, dtype=DTYPE)
    cfg = AGAConfig(param_noise=False, dtype=DTYPE, device=DEVICE,
                    adapt_alpha=False, adapt_mu=False, adapt_sigma=False)
    agent = AGAAgent(dim=2, config=cfg, seed=0, init_alpha=0.05)
    log1 = agent.step(loss_only)
    log2 = agent.update(loss_only)
    for k in ("loss", "alpha", "mu", "sigma", "gamma"):
        assert k in log1 and k in log2
```
