"""Continuous-observation local models, parametrized by the local ROC r(a).

A threshold region is identified by its false-alarm probability a in [0, 1]:
R_a = {y : LR(y) >= r'(a)}, with P(R_a | H0) = a and P(R_a | H1) = r(a).  A node with
regions (R_{a0}, R_{a1}) and arriving message distribution (f, d) outputs

    P_F = (1 - f) a0 + f a1,      P_D = (1 - d) r(a0) + d r(a1),

so rho_F = a1 - a0 and rho_D = r(a1) - r(a0): Theorem 1 holds verbatim (nothing in its
proof uses a finite alphabet).  The node thresholds tau(m) = eta f/d, eta (1-f)/(1-d) map
to a_m = astar(tau(m)), where astar is the inverse of r' (clipped to [0, 1]).

Models
------
GaussianShift(dprime)  : N(0,1) vs N(d',1); r(a) = Q(Q^{-1}(a) - d'),
                         r'(a) = exp(d' Q^{-1}(a) - d'^2/2),  astar(s) = Q(ln(s)/d' + d'/2)
PowerROC(beta)         : r(a) = a^beta, 0 < beta < 1.  Swerling-I square-law detection
                         with per-pulse SNR gamma gives beta = 1/(1+gamma); an exponential
                         shift-in-rate model Exp(l0) vs Exp(l1) gives beta = l1/l0.
                         astar(s) = min{1, (s/beta)^(1/(beta-1))}
LogisticShift(mu)      : Logistic(0,1) vs Logistic(mu,1) (the 'logistic control model').
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

Point = Tuple[float, float]


class ContinuousModel:
    name = "abstract"

    def r(self, a):               # local ROC
        raise NotImplementedError

    def rprime(self, a):          # slope of the ROC = likelihood ratio at the boundary
        raise NotImplementedError

    def astar(self, s):           # inverse of rprime, clipped to [0, 1]
        raise NotImplementedError

    # ------------------------------------------------------------ transfer
    def local_point(self, a: float) -> Point:
        return float(a), float(self.r(a))

    def node_output(self, a0: float, a1: float, f: float, d: float) -> Point:
        return float((1 - f) * a0 + f * a1), float((1 - d) * self.r(a0) + d * self.r(a1))

    def attenuation(self, a0: float, a1: float) -> Point:
        return float(a1 - a0), float(self.r(a1) - self.r(a0))

    def regions_from_eta(self, eta: float, f: float, d: float) -> Tuple[float, float]:
        """(a0, a1) induced by a node threshold eta at designed upstream point (f, d)."""
        if abs(f - d) < 1e-15:
            tau1 = tau0 = eta
        else:
            tau1 = np.inf if d == 0 else (0.0 if f == 0 else eta * f / d)
            tau0 = np.inf if d == 1 else (0.0 if f == 1 else eta * (1 - f) / (1 - d))
        return float(self.astar(tau0)), float(self.astar(tau1))

    def kappa(self, s: float) -> float:
        """Conjugate kappa(s) = max_a [r(a) - s a] (Proposition 2)."""
        a = self.astar(s)
        return float(self.r(a) - s * a)


class GaussianShift(ContinuousModel):
    def __init__(self, dprime: float):
        self.dprime = float(dprime)
        self.name = f"Gaussian(d'={dprime})"

    def r(self, a):
        a = np.clip(a, 0.0, 1.0)
        return norm.sf(norm.isf(a) - self.dprime)

    def rprime(self, a):
        t = norm.isf(np.clip(a, 1e-300, 1.0))
        return np.exp(self.dprime * t - self.dprime ** 2 / 2)

    def astar(self, s):
        if s <= 0:
            return 1.0
        if not np.isfinite(s):
            return 0.0
        return float(norm.sf(np.log(s) / self.dprime + self.dprime / 2))


class PowerROC(ContinuousModel):
    def __init__(self, beta: float = None, snr: float = None, rates: Tuple[float, float] = None):
        if beta is None:
            if snr is not None:
                beta = 1.0 / (1.0 + snr)
            elif rates is not None:
                beta = rates[1] / rates[0]
            else:
                raise ValueError("give beta, snr (Swerling I) or rates=(l0, l1) (exponential)")
        if not 0 < beta < 1:
            raise ValueError("beta must be in (0, 1)")
        self.beta = float(beta)
        self.name = f"Power(beta={self.beta:.3f})"

    def r(self, a):
        return np.clip(a, 0.0, 1.0) ** self.beta

    def rprime(self, a):
        a = np.clip(a, 1e-300, 1.0)
        return self.beta * a ** (self.beta - 1)

    def astar(self, s):
        if s <= 0:
            return 1.0
        if not np.isfinite(s):
            return 0.0
        if s <= self.beta:
            return 1.0
        return float((s / self.beta) ** (1.0 / (self.beta - 1)))


class LogisticShift(ContinuousModel):
    def __init__(self, mu: float):
        self.mu = float(mu)
        self.name = f"Logistic(mu={mu})"

    @staticmethod
    def _sf(x):
        return 1.0 / (1.0 + np.exp(x))          # P(Y >= x) for Logistic(0,1)

    @staticmethod
    def _isf(a):
        a = np.clip(a, 1e-300, 1 - 1e-16)
        return np.log((1 - a) / a)

    def r(self, a):
        t = self._isf(np.clip(a, 0.0, 1.0))
        return self._sf(t - self.mu)

    def rprime(self, a):
        t = self._isf(np.clip(a, 1e-300, 1.0))
        # LR of Logistic(mu,1) vs Logistic(0,1) at y = t
        f0 = np.exp(-t) / (1 + np.exp(-t)) ** 2
        f1 = np.exp(-(t - self.mu)) / (1 + np.exp(-(t - self.mu))) ** 2
        return f1 / f0

    def astar(self, s):
        """Closed form: the ROC is r(a) = e^mu a / (1 + (e^mu - 1) a), so
        r'(a) = e^mu / (1 + (e^mu - 1) a)^2 and a*(s) = (sqrt(e^mu / s) - 1) / (e^mu - 1),
        clipped to [0, 1] (a* = 1 for s <= e^-mu, a* = 0 for s >= e^mu)."""
        if s <= 0:
            return 1.0
        if not np.isfinite(s):
            return 0.0
        e = np.exp(self.mu)
        return float(min(1.0, max(0.0, (np.sqrt(e / s) - 1.0) / (e - 1.0))))


# ----------------------------------------------------------------- chains
@dataclass(frozen=True)
class ContStrategy:
    a1: float                                  # node-1 false-alarm rate
    pairs: Tuple[Tuple[float, float], ...]     # (a0, a1) of nodes 2..n


def evaluate_chain_cont(models: Sequence[ContinuousModel], strat: ContStrategy) -> List[Point]:
    f, d = models[0].local_point(strat.a1)
    pts = [(f, d)]
    for m, (a0, a1) in zip(models[1:], strat.pairs):
        f, d = m.node_output(a0, a1, f, d)
        pts.append((f, d))
    return pts


def strategy_from_etas(models: Sequence[ContinuousModel], a1: float, etas: Sequence[float]) -> ContStrategy:
    """Regions along the chain induced by node thresholds eta_2..eta_n (node 1 at a1)."""
    f, d = models[0].local_point(a1)
    pairs = []
    for m, eta in zip(models[1:], etas):
        a0, a1_ = m.regions_from_eta(eta, f, d)
        pairs.append((a0, a1_))
        f, d = m.node_output(a0, a1_, f, d)
    return ContStrategy(a1, tuple(pairs))


def chain_factors_cont(models, strat: ContStrategy):
    return [m.attenuation(a0, a1) for m, (a0, a1) in zip(models[1:], strat.pairs)]


def pbpo_cont(models: Sequence[ContinuousModel], lam: float, a_grid=None, eta_grid=None,
              init: ContStrategy = None, rng=None, max_sweeps: int = 50):
    """Lagrangian PBPO (max P_D - lam P_F) for continuous nodes on grids: node 1 over a
    grid of local false-alarm rates, nodes k >= 2 over a grid of thresholds eta_k (other
    nodes keep their eta and re-derive their regions)."""
    rng = np.random.default_rng() if rng is None else rng
    a_grid = np.linspace(0.001, 0.999, 400) if a_grid is None else a_grid
    eta_grid = np.exp(np.linspace(-7, 7, 281)) if eta_grid is None else eta_grid
    n = len(models)
    if init is None:
        a1 = float(rng.choice(a_grid)); etas = list(rng.choice(eta_grid, size=n - 1))
    else:
        a1, etas = init
    obj = lambda s: (lambda pt: pt[1] - lam * pt[0])(evaluate_chain_cont(models, s)[-1])
    best = obj(strategy_from_etas(models, a1, etas))
    for sweep in range(1, max_sweeps + 1):
        changed = False
        for k in range(n):
            if k == 0:
                for a in a_grid:
                    v = obj(strategy_from_etas(models, a, etas))
                    if v > best + 1e-12:
                        best, a1, changed = v, float(a), True
            else:
                for eta in eta_grid:
                    new = list(etas); new[k - 1] = float(eta)
                    v = obj(strategy_from_etas(models, a1, new))
                    if v > best + 1e-12:
                        best, etas, changed = v, new, True
        if not changed:
            break
    s = strategy_from_etas(models, a1, etas)
    return s, evaluate_chain_cont(models, s)[-1], sweep, (a1, etas)


def pbpo_cont_exact(models: Sequence[ContinuousModel], lam: float, init: ContStrategy,
                    max_sweeps: int = 5000, tol: float = 1e-13):
    """Lagrangian PBPO (max P_D - lam P_F) with exact best responses.  With the regions of the
    other nodes fixed, the operating point of the chain is affine in the output (f_k, d_k) of
    node k, with slopes A_F = prod_{j>k} rho_F,j and A_D = prod_{j>k} rho_D,j (Theorem 1), so
    node k maximizes A_D d_k - lam A_F f_k: its best regions are the likelihood-ratio regions
    of eq. (3) with eta_k = lam A_F / A_D, that is a_m = astar(tau_k(m)), and a = astar(eta_1)
    for node 1.  The nodes are updated in turn, each against the current regions of the
    others, until no false-alarm rate moves by more than tol.  A node whose message is ignored
    downstream (A_F or A_D not positive) keeps its regions, and so does the region of a node
    for a message that cannot occur.  Like every person-by-person optimization, the iteration
    can stop at a fixed point that is not the optimum (for instance one in which a node ignores
    its message, so that the nodes before it no longer matter): compare several starting points.
    Returns (strategy, operating point, sweeps, converged)."""
    n = len(models)
    a1, pairs = float(init.a1), [(float(a0), float(b1)) for a0, b1 in init.pairs]
    moved = np.inf
    for sweep in range(1, max_sweeps + 1):
        moved = 0.0
        for k in range(n):
            AF = AD = 1.0
            for m, (a0, b1) in zip(models[k + 1:], pairs[k:]):
                rf, rd = m.attenuation(a0, b1)
                AF *= rf; AD *= rd
            if not (AF > 0 and AD > 0):
                continue
            eta = lam * AF / AD
            if k == 0:
                new = float(models[0].astar(eta))
                moved = max(moved, abs(new - a1)); a1 = new
            else:
                f, d = evaluate_chain_cont(models[:k], ContStrategy(a1, tuple(pairs[:k - 1])))[-1]
                if max(f, d) < 1e-15:              # message 1 cannot occur: its region is free and is kept
                    new = (float(models[k].astar(eta)), pairs[k - 1][1])
                elif min(f, d) > 1 - 1e-15:        # message 0 cannot occur
                    new = (pairs[k - 1][0], float(models[k].astar(eta)))
                else:
                    new = models[k].regions_from_eta(eta, f, d)
                moved = max(moved, abs(new[0] - pairs[k - 1][0]), abs(new[1] - pairs[k - 1][1])); pairs[k - 1] = new
        if moved < tol:
            break
    s = ContStrategy(a1, tuple(pairs))
    return s, evaluate_chain_cont(models, s)[-1], sweep, bool(moved < tol)
