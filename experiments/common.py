"""Shared helpers for the experiment scripts: model families, random configurations,
NP-optimal designs, and result I/O.  Every script fixes its random seed."""
from __future__ import annotations

import json
import os
import sys
from typing import List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tandem_sim import *                                                   # noqa: E402
from tandem_sim.design import (distinct_points, strategies_at, hull_by_pbpo, enumerate_nested_arrays,  # noqa: E402
                               hull_of_array, realizations_of, min_switch_realizations, align_pair,
                               switching_nodes)
from tandem_sim.design import _decode as decode_strategy  # noqa: E402
from tandem_sim.desync import worst_single_shift  # noqa: E402
from tandem_sim.discrete import random_model, quantized_gaussian, realizable_pairs  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

RUNNING = DiscreteModel([0.8, 0.1, 0.1], [0.3, 0.1, 0.6], name="running(K=3)")
VIOLATING = DiscreteModel([0.45, 0.50, 0.05], [0.35, 0.55, 0.10], name="violating(K=3)")


# ------------------------------------------------------------ model families
def family_models(family: str, n: int, rng: np.random.Generator, identical: bool = True):
    """Discrete model(s) for an n-node chain.
    family: 'running' | 'dirichlet3' | 'dirichlet5' | 'dirichlet10' | 'qgauss1' | 'qgauss2'
    identical=True draws one model shared by all nodes (as in the paper's running model);
    False draws one model per node."""
    def one():
        if family == "running":
            return RUNNING
        if family.startswith("dirichlet"):
            return random_model(int(family[len("dirichlet"):]), rng)
        if family.startswith("qgauss"):
            dprime = float(family[len("qgauss"):])
            return quantized_gaussian(8, dprime)
        raise ValueError(family)
    if identical:
        m = one()
        return [m] * n
    return [one() for _ in range(n)]


# ------------------------------------------------------ random configurations
def random_strategy(models: Sequence[DiscreteModel], rng: np.random.Generator,
                    exclude_trivial: bool = True) -> Strategy:
    """A random deterministic strategy: node 1 uniform over its (non-trivial) rules, each
    downstream node uniform over the region pairs realizable at its designed upstream
    point (the single-threshold class).  'Non-trivial' excludes the never/always-fire
    rules at node 1 and the message-ignoring / observation-ignoring pairs downstream
    when possible, so that every node is an actual detector."""
    m1 = models[0]
    levels = [j for j in m1.levels if (not exclude_trivial) or 0 < j < m1.K]
    j = int(rng.choice(levels))
    f, d = m1.region_probs(j)
    pairs = []
    for m in models[1:]:
        cands = realizable_pairs(m, f, d)
        if exclude_trivial:
            nt = [(a, b) for (a, b) in cands if not (a == b) and not (a == m.K and b == 0)]
            cands = nt or cands
        pr = cands[rng.integers(len(cands))]
        pairs.append(pr)
        f, d = m.node_output(pr[0], pr[1], f, d)
    return Strategy(j, tuple(pairs))


def random_pair(models, rng, differ_prob: float = 1.0):
    """Two random strategies; with probability differ_prob each node of B is re-drawn
    (otherwise it copies A's rule at that node)."""
    A = random_strategy(models, rng)
    B = random_strategy(models, rng)
    if differ_prob < 1.0:
        pairs = list(B.pairs)
        j1 = B.j1 if rng.random() < differ_prob else A.j1
        for i in range(len(pairs)):
            if rng.random() >= differ_prob:
                pairs[i] = A.pairs[i]
        B = Strategy(j1, tuple(pairs))
    return A, B


# ----------------------------------------------------------- optimal designs
def np_design(models: Sequence[DiscreteModel], alpha: float, exhaustive_max_n: int = 5,
              lambdas=None, restarts: int = 20, seed: int = 0, realization: str = "min_switch",
              compute_G: bool = False, compute_S: bool = False):
    """Dependent-randomization design at budget alpha: supporting vertices A, B of the upper
    hull and the mixing probability p.  Exhaustive for short chains (all threshold
    strategies), Lagrangian-PBPO hull otherwise.

    realization = 'min_switch' (default): the two vertices are realized by the pair of
    strategies with the fewest switching nodes among nodes 1..n-1, ties broken by the
    smallest worst-case shift over single-node failures (exhaustive search for short
    chains; for PBPO designs, a node whose rule does not affect the operating point takes
    the rule of the other strategy); 'first': the first single-threshold strategies
    enumerated (no selection among the realizations; kept for comparison).
    G is the best deterministic point of the single-threshold family at alpha (computed for
    short chains only when compute_G is set).
    S is the best design in which a single node randomizes with a private coin and the other
    nodes keep one rule (design.single_node_optimum; short chains, when compute_S is set).
    Returns dict(sA, sB, p, A, B, C, G, S, hull, deterministic_at_alpha, classes, switching, ties)."""
    n = len(models)
    info = {}
    S = None
    if n <= exhaustive_max_n:
        arr, codes, tables = enumerate_nested_arrays(models)
        hull = hull_of_array(arr)
        sp = support_pair(hull, alpha)
        items = enumerate_system_points(models) if (compute_G or realization == "first") else None
        G = deterministic_optimum([pt for pt, _ in items], alpha) if compute_G else None
        S = single_node_optimum(arr, codes, tables, alpha) if compute_S else None
        if realization == "first":
            sA, sB = strategies_at(items, sp.A)[0], strategies_at(items, sp.B)[0]
        elif sp.deterministic:
            sA = sB = decode_strategy(realizations_of(arr, codes, sp.A)[0], tables)
        else:
            sA, sB, _ = min_switch_realizations(arr, codes, tables, sp.A, sp.B, info=info,
                                                score=lambda a, b: worst_single_shift(models, a, b, sp.p))
    else:
        lambdas = np.exp(np.linspace(-5, 6, 45)) if lambdas is None else lambdas
        hull, items = hull_by_pbpo(models, lambdas, restarts=restarts, seed=seed)
        sp = support_pair(hull, alpha)
        sA, sB = strategies_at(items, sp.A)[0], strategies_at(items, sp.B)[0]
        if realization == "min_switch" and not sp.deterministic:
            sA, sB = align_pair(models, sA, sB)
        G = deterministic_optimum([pt for pt, _ in items], alpha)
    return {"sA": sA, "sB": sB, "p": sp.p, "A": sp.A, "B": sp.B, "C": sp.C, "G": G, "S": S, "hull": hull,
            "deterministic_at_alpha": sp.deterministic, "classes": classify_pair(sA, sB),
            "switching": switching_nodes(sA, sB), "ties": info}


# -------------------------------------------------------------------- I/O
def save_json(name: str, obj) -> str:
    path = os.path.join(RESULTS, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=_default)
    return path


def load_json(name: str):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Strategy,)):
        return {"j1": o.j1, "pairs": list(o.pairs)}
    if isinstance(o, frozenset):
        return sorted(o)
    return str(o)


def summarize(x) -> dict:
    x = np.asarray(x, float)
    if x.size == 0:
        return {}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p10": float(np.percentile(x, 10)), "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)), "p90": float(np.percentile(x, 90)),
            "min": float(x.min()), "max": float(x.max())}
