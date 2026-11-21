"""Discrete-observation tandem chain: local models, decision regions, forward recursion.

Conventions (they follow the paper, Section II)
------------------------------------------------
* Node k observes y in {0, ..., K-1} with P(y|H0) = p0, P(y|H1) = p1.
* A local likelihood-ratio threshold rule has decision region {y : LR(y) >= tau}.
  Distinct regions are indexed by a *level* j in {0, ..., K}: region_j consists of
  the observations whose LR rank (ascending) is >= j, i.e. the K - j observations
  with the largest likelihood ratios.  Level 0 is the whole alphabet, level K the
  empty set.  Tied likelihood ratios are kept together (a threshold cannot split
  them), so only levels at group boundaries are realizable.
* Node 1 uses one region (level j1).  Node k >= 2 uses a pair of regions
  (R_k(0), R_k(1)) = (level j0, level j1); with a single node threshold eta_k and
  an informative upstream message (d > f) one gets tau_k(1) < tau_k(0), hence
  j1 <= j0 and R_k(1) ⊇ R_k(0), eq. (3) of the paper.
* Forward recursion, eqs. (4)-(5):
      P_F^(k) = (1 - P_F^(k-1)) F[j0] + P_F^(k-1) F[j1]
      P_D^(k) = (1 - P_D^(k-1)) D[j0] + P_D^(k-1) D[j1]
  where F[j] = P(region_j | H0), D[j] = P(region_j | H1).
* Attenuation factors, eqs. (10)-(11): rho_F = F[j1] - F[j0], rho_D = D[j1] - D[j0].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


class DiscreteModel:
    """A node's discrete observation model."""

    def __init__(self, p0: Sequence[float], p1: Sequence[float], name: str = ""):
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        if p0.shape != p1.shape or p0.ndim != 1:
            raise ValueError("p0 and p1 must be 1-D arrays of the same length")
        if not (np.all(p0 > 0) and np.all(p1 >= 0)):
            raise ValueError("p0 must be strictly positive, p1 non-negative")
        if abs(p0.sum() - 1) > 1e-9 or abs(p1.sum() - 1) > 1e-9:
            raise ValueError("p0 and p1 must each sum to one")
        self.p0 = p0
        self.p1 = p1
        self.K = len(p0)
        self.name = name
        lr = p1 / p0
        order = np.argsort(lr, kind="stable")
        self.order = order                      # observation index by ascending LR
        self.lr_sorted = lr[order]
        # tail sums: F[j] = P(rank >= j | H0), j = 0..K
        self.F = np.concatenate([np.cumsum(p0[order][::-1])[::-1], [0.0]])
        self.D = np.concatenate([np.cumsum(p1[order][::-1])[::-1], [0.0]])
        # realizable levels: ranks at which the LR value changes (group boundaries)
        levels = [0]
        for j in range(1, self.K):
            if self.lr_sorted[j] > self.lr_sorted[j - 1] * (1 + 1e-12):
                levels.append(j)
        levels.append(self.K)
        self.levels = levels

    # ------------------------------------------------------------------ basics
    def __repr__(self) -> str:
        return f"DiscreteModel(p0={self.p0.tolist()}, p1={self.p1.tolist()})"

    def local_points(self) -> List[Point]:
        """Operating points (P_F, P_D) of the K+1 local threshold rules of node 1."""
        return [(float(self.F[j]), float(self.D[j])) for j in self.levels]

    def region_probs(self, j: int) -> Point:
        return float(self.F[j]), float(self.D[j])

    def level_from_tau(self, tau: float) -> int:
        """Level of the region {y : LR(y) >= tau}: the number of observations with LR < tau."""
        if not np.isfinite(tau):
            return self.K if tau > 0 else 0
        return int(np.searchsorted(self.lr_sorted, tau, side="left"))

    # --------------------------------------------------------- node transfer
    def node_output(self, j0: int, j1: int, f: float, d: float) -> Point:
        """Output of a node with fixed regions (level j0 for message 0, level j1 for
        message 1) when the arriving message distribution is (f, d), eqs. (4)-(5)."""
        PF = (1 - f) * self.F[j0] + f * self.F[j1]
        PD = (1 - d) * self.D[j0] + d * self.D[j1]
        return float(PF), float(PD)

    def attenuation(self, j0: int, j1: int) -> Point:
        """(rho_F, rho_D) of eqs. (10)-(11).  Negative values arise only for non-nested
        region pairs (j1 > j0), which the single-threshold class produces only when the
        designed upstream message is anti-informative (d < f)."""
        return float(self.F[j1] - self.F[j0]), float(self.D[j1] - self.D[j0])

    def intercepts(self, j0: int, j1: int) -> Point:
        """(c_F, c_D) = (P(R(0)|H0), P(R(0)|H1)); the node map is x -> c + rho * x."""
        return float(self.F[j0]), float(self.D[j0])


# ------------------------------------------------------------- region pairs
def _taus(f: float, d: float, eta: float) -> Tuple[float, float]:
    """tau(1) = eta f/d, tau(0) = eta (1-f)/(1-d) with the degenerate-case conventions
    described in realizable_pairs."""
    if abs(f - d) < 1e-15:            # uninformative (or constant) message
        return eta, eta
    tau1 = np.inf if d == 0 else (0.0 if f == 0 else eta * f / d)
    tau0 = np.inf if d == 1 else (0.0 if f == 1 else eta * (1 - f) / (1 - d))
    return tau1, tau0


def realizable_pairs(model: DiscreteModel, f: float, d: float,
                     with_eta: bool = False):
    """All distinct region pairs (j0, j1) a node can realize with a single threshold
    eta in (0, inf) when its designed upstream operating point is (f, d).

    Degenerate conventions: if f == d (the message carries no information, including a
    constant message) the node ignores the message, tau(0) = tau(1) = eta; if d == 0 < f
    the message 1 is impossible under H1 and R(1) is empty; if f == 0 < d the message 1
    is impossible under H0 and R(1) is the whole alphabet.

    Returns a list of (j0, j1) in order of increasing eta (optionally with a
    representative eta for each pair).
    """
    lrs = [model.lr_sorted[j] for j in model.levels if j < model.K]   # distinct LR values
    breaks = set()
    for ell in lrs:
        if 0 < f < 1 and 0 < d < 1 and abs(f - d) >= 1e-15:
            breaks.add(ell * d / f)
            breaks.add(ell * (1 - d) / (1 - f))
        elif abs(f - d) < 1e-15:
            breaks.add(ell)
        else:
            if 0 < d and 0 < f:
                breaks.add(ell * d / f)
            if d < 1 and f < 1:
                breaks.add(ell * (1 - d) / (1 - f))
    breaks = sorted(b for b in breaks if np.isfinite(b) and b > 0)
    if not breaks:
        probes = [1.0]
    else:
        probes = [breaks[0] / 2] + [(a + b) / 2 for a, b in zip(breaks[:-1], breaks[1:])] + [breaks[-1] * 2]
        # also probe exactly at breakpoints: the rule uses >= so the region changes there
        probes = sorted(set(probes) | set(breaks))
    out, seen = [], set()
    for eta in probes:
        tau1, tau0 = _taus(f, d, eta)
        j1, j0 = model.level_from_tau(tau1), model.level_from_tau(tau0)
        if (j0, j1) not in seen:
            seen.add((j0, j1))
            out.append(((j0, j1), eta) if with_eta else (j0, j1))
    return out


def all_nested_pairs(model: DiscreteModel):
    """Every nested region pair (j0, j1) with R(1) ⊇ R(0), i.e. j1 <= j0.  This class is
    larger than the single-threshold class; it is used only for benchmarks."""
    return [(j0, j1) for j0 in model.levels for j1 in model.levels if j1 <= j0]


# ---------------------------------------------------------------- strategies
@dataclass(frozen=True)
class Strategy:
    """A deterministic chain strategy in region form: level of node 1 and the region
    pair (j0, j1) of every downstream node, in chain order."""
    j1: int
    pairs: Tuple[Tuple[int, int], ...]

    @property
    def n(self) -> int:
        return 1 + len(self.pairs)

    def upstream_rules(self) -> tuple:
        """Rules of nodes 1..n-1 (what must coincide for case (i) of Proposition 4)."""
        return (self.j1,) + tuple(self.pairs[:-1])

    def terminal_rule(self):
        return self.pairs[-1] if self.pairs else self.j1


def evaluate_from(models: Sequence[DiscreteModel], pairs: Sequence[Tuple[int, int]],
                  start: Point) -> List[Point]:
    """Forward recursion through nodes with fixed region pairs, starting from an
    arriving message distribution `start`.  models[i] and pairs[i] describe the
    i-th node of the sub-chain.  Returns the output point of every node."""
    f, d = start
    out = []
    for m, (j0, j1) in zip(models, pairs):
        f, d = m.node_output(j0, j1, f, d)
        out.append((f, d))
    return out


def evaluate_chain(models: Sequence[DiscreteModel], strat: Strategy) -> List[Point]:
    """Operating points of nodes 1..n for a deterministic strategy (node 1 uses its local
    rule; downstream nodes use their fixed region pairs)."""
    if len(models) != strat.n:
        raise ValueError("one model per node is required")
    p1 = models[0].region_probs(strat.j1)
    return [p1] + evaluate_from(models[1:], strat.pairs, p1)


def chain_factors(models: Sequence[DiscreteModel], strat: Strategy):
    """Per-node (rho_F, rho_D) for nodes 2..n of a strategy."""
    return [m.attenuation(j0, j1) for m, (j0, j1) in zip(models[1:], strat.pairs)]


def random_model(K: int, rng: np.random.Generator, alpha: float = 1.0) -> DiscreteModel:
    """Node distributions drawn from a symmetric Dirichlet(alpha) prior on the K-point
    simplex (alpha = 1 is the uniform distribution on the simplex), as in Section IV."""
    p0 = rng.dirichlet(alpha * np.ones(K))
    p1 = rng.dirichlet(alpha * np.ones(K))
    return DiscreteModel(p0, p1)


def quantized_gaussian(K: int, dprime: float, lo: float = -3.0, hi: float = 3.0) -> DiscreteModel:
    """A structured family: a Gaussian shift-in-mean observation (N(0,1) vs N(d',1))
    quantized to K equiprobable-under-H0-like bins on [lo, hi] (outer bins unbounded)."""
    from scipy.stats import norm
    edges = np.concatenate([[-np.inf], np.linspace(lo, hi, K - 1), [np.inf]])
    p0 = np.diff(norm.cdf(edges))
    p1 = np.diff(norm.cdf(edges - dprime))
    p0 = np.maximum(p0, 1e-12); p1 = np.maximum(p1, 1e-12)
    return DiscreteModel(p0 / p0.sum(), p1 / p1.sum(), name=f"qGauss(K={K},d'={dprime})")


def quantized_gaussian_cuts(dprime: float, cuts) -> DiscreteModel:
    """A Gaussian shift-in-mean observation (N(0,1) vs N(d',1)) quantized at the given cut
    points: len(cuts) + 1 levels, ordered by the observation and hence by the likelihood ratio."""
    from scipy.stats import norm
    cuts = np.sort(np.asarray(cuts, dtype=float))
    edges = np.concatenate([[-np.inf], cuts, [np.inf]])
    p0, p1 = np.diff(norm.cdf(edges)), np.diff(norm.cdf(edges - dprime))
    return DiscreteModel(p0 / p0.sum(), p1 / p1.sum(),
                         name=f"qGauss{len(cuts) + 1}(d'={dprime:.4f}, cuts " + ", ".join(f"{c:.4f}" for c in cuts) + ")")
