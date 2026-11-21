"""Design of the deterministic strategy set and of dependent randomization.

* enumerate_system_points : every deterministic system operating point of an n-node
  chain (exhaustive; feasible for n <= 5 with small alphabets).
* upper_hull / support_pair : the randomized ROC (upper concave envelope) and the two
  supporting strategies A, B with mixing probability p at a false-alarm budget alpha.
* classify_pair : the dichotomy of Proposition 4 (terminal-node vs upstream-involving)
  and the region-matched immunity of Corollary 1.
* single_node_optimum : the best design in which one node randomizes with a private coin
  and every other node keeps one rule; it needs no common draw (the reference S of Sec. V).
* enumerate_nested_arrays / min_switch_realizations / align_pair : a hull vertex usually
  has several threshold realizations (a node whose message is ignored downstream may use
  any rule; a region for a message that cannot occur is free; identical sensors can be
  placed differently).  The realization decides which nodes switch between A and B, hence
  which nodes can lose synchronization.  Designs use the pair of realizations with the
  fewest switching nodes, ties broken by the smallest worst-case shift: exhaustive search
  over all threshold strategies for short chains, alignment of the irrelevant nodes
  otherwise.
* pbpo : person-by-person optimization (Algorithm S1 of the supplement) with restarts,
  constrained (max P_D s.t. P_F <= alpha) or Lagrangian (max P_D - lambda P_F).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .discrete import (DiscreteModel, Point, Strategy, all_nested_pairs, evaluate_chain,
                       evaluate_from, realizable_pairs)


# ------------------------------------------------------------- enumeration
def enumerate_from(models: Sequence[DiscreteModel], start: Point,
                   pair_class: str = "eta") -> List[Tuple[Point, Tuple[Tuple[int, int], ...]]]:
    """All (output point, region pairs) of a sub-chain whose first node receives the
    message distribution `start`.  pair_class = 'eta' (single-threshold class, the
    strategy family of the paper) or 'nested' (all nested pairs; benchmarks only)."""
    results: List[Tuple[Point, Tuple[Tuple[int, int], ...]]] = []

    def rec(i: int, point: Point, pairs: Tuple[Tuple[int, int], ...]):
        if i == len(models):
            results.append((point, pairs))
            return
        m = models[i]
        cands = realizable_pairs(m, *point) if pair_class == "eta" else all_nested_pairs(m)
        for (j0, j1) in cands:
            rec(i + 1, m.node_output(j0, j1, *point), pairs + ((j0, j1),))

    rec(0, start, ())
    return results


def enumerate_system_points(models: Sequence[DiscreteModel],
                            pair_class: str = "eta") -> List[Tuple[Point, Strategy]]:
    """All deterministic system operating points of the chain, with their strategies."""
    out: List[Tuple[Point, Strategy]] = []
    m1 = models[0]
    for j in m1.levels:
        start = m1.region_probs(j)
        for point, pairs in enumerate_from(models[1:], start, pair_class):
            out.append((point, Strategy(j, pairs)))
    return out


def distinct_points(items: Sequence[Tuple[Point, object]], tol: float = 1e-12) -> List[Point]:
    pts = sorted({(round(x / tol) * tol, round(y / tol) * tol) for (x, y), _ in items})
    return [(float(x), float(y)) for x, y in pts]


# ------------------------------------------------------------------- hull
def upper_hull(points: Sequence[Point]) -> List[Point]:
    """Vertices of the upper concave envelope of a point set in [0,1]^2, from the
    left-most to the right-most point (the randomized ROC).  (0,0) and (1,1) are
    always added, since they are achievable (never / always declare H1)."""
    pts = sorted(set([(0.0, 0.0), (1.0, 1.0)] + [(float(x), float(y)) for x, y in points]))
    hull: List[Point] = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            cross = (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)
            if cross >= -1e-15:          # p makes a right turn or is collinear: pop
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def hull_value(hull: Sequence[Point], alpha: float) -> float:
    """Value of the upper envelope at P_F = alpha (linear interpolation on the hull)."""
    for (x1, y1), (x2, y2) in zip(hull[:-1], hull[1:]):
        if x1 <= alpha <= x2:
            if x2 == x1:
                return max(y1, y2)
            return y1 + (y2 - y1) * (alpha - x1) / (x2 - x1)
    raise ValueError("alpha outside the hull range")


@dataclass
class SupportPair:
    A: Point
    B: Point
    p: float            # probability of A
    C: Point            # mixed point, P_F = alpha
    deterministic: bool  # True when a hull vertex sits exactly at alpha (p = 1)


def support_pair(hull: Sequence[Point], alpha: float, tol: float = 1e-12) -> SupportPair:
    """Supporting hull vertices A (P_F <= alpha) and B (P_F >= alpha) and the mixing
    probability p such that p*A + (1-p)*B has P_F = alpha, eqs. (7)-(8)."""
    for (x1, y1), (x2, y2) in zip(hull[:-1], hull[1:]):
        if x1 - tol <= alpha <= x2 + tol:
            if abs(x1 - alpha) <= tol:
                return SupportPair((x1, y1), (x1, y1), 1.0, (x1, y1), True)
            if abs(x2 - alpha) <= tol:
                return SupportPair((x2, y2), (x2, y2), 1.0, (x2, y2), True)
            p = (x2 - alpha) / (x2 - x1)
            C = (alpha, p * y1 + (1 - p) * y2)
            return SupportPair((x1, y1), (x2, y2), float(p), C, False)
    raise ValueError("alpha outside the hull range")


def deterministic_optimum(points: Sequence[Point], alpha: float) -> Point:
    """Best deterministic point with P_F <= alpha."""
    feas = [p for p in points if p[0] <= alpha + 1e-12]
    return max(feas, key=lambda q: q[1]) if feas else (0.0, 0.0)


def strategies_at(items: Sequence[Tuple[Point, Strategy]], point: Point,
                  tol: float = 1e-9) -> List[Strategy]:
    """All strategies realizing a given system point (a hull vertex may have several
    threshold realizations, Sec. III-D)."""
    return [s for (x, y), s in items if abs(x - point[0]) <= tol and abs(y - point[1]) <= tol]


# ------------------------------------------------- realizations of a vertex
def enumerate_nested_arrays(models: Sequence[DiscreteModel]):
    """Every threshold strategy of the chain in array form: node 1 any level, node k >= 2
    any nested region pair (j0 >= j1, i.e. R(1) contains R(0)).  This family contains the
    single-threshold family of the paper and has the same upper hull; it is the family in
    which the realizations of a hull vertex are searched.

    Returns (pts, codes, tables): pts (N, 2) system points; codes (N, n) rule indices
    (column 0 indexes models[0].levels, column k-1 indexes all_nested_pairs(models[k-1]));
    tables decodes them.  Rows are in lexicographic (enumeration) order."""
    m1 = models[0]
    lv = np.array(m1.levels)
    f, d = m1.F[lv].astype(float), m1.D[lv].astype(float)
    codes = np.arange(len(lv), dtype=np.int16)[:, None]
    tables: List[list] = [list(m1.levels)]
    for m in models[1:]:
        pairs = all_nested_pairs(m)
        j0 = np.array([q[0] for q in pairs]); j1 = np.array([q[1] for q in pairs])
        f = ((1 - f)[:, None] * m.F[j0][None, :] + f[:, None] * m.F[j1][None, :]).ravel()
        d = ((1 - d)[:, None] * m.D[j0][None, :] + d[:, None] * m.D[j1][None, :]).ravel()
        codes = np.concatenate([np.repeat(codes, len(pairs), axis=0),
                                np.tile(np.arange(len(pairs), dtype=np.int16), codes.shape[0])[:, None]], axis=1)
        tables.append(pairs)
    return np.stack([f, d], axis=1), codes, tables


def hull_of_array(pts: np.ndarray, tol: float = 1e-12) -> List[Point]:
    """Upper hull of an (N, 2) array of points (rounded to tol, as distinct_points does)."""
    q = np.unique(np.round(pts / tol) * tol, axis=0)
    return upper_hull([(float(x), float(y)) for x, y in q])


def _decode(row, tables) -> Strategy:
    return Strategy(int(tables[0][int(row[0])]), tuple(tuple(tables[k][int(row[k])]) for k in range(1, len(tables))))


def realizations_of(pts: np.ndarray, codes: np.ndarray, point: Point, tol: float = 1e-9) -> np.ndarray:
    """Rule-index rows of every enumerated strategy whose system point is `point`."""
    mask = (np.abs(pts[:, 0] - point[0]) <= tol) & (np.abs(pts[:, 1] - point[1]) <= tol)
    return codes[mask].astype(np.int64)


def min_switch_realizations(pts: np.ndarray, codes: np.ndarray, tables, A: Point, B: Point,
                            tol: float = 1e-9, score=None, max_ties: int = 200,
                            info: Optional[dict] = None) -> Tuple[Strategy, Strategy, int]:
    """Realizations of the vertices A and B used by a design.

    Criterion 1: the fewest switching nodes among nodes 1..n-1, the nodes that can lose
    synchronization (the final node holds the reference draw).  The search is exhaustive
    over the enumerated strategies.
    Criterion 2 (ties): the smallest score(sA, sB); the experiments pass the largest shift
    ||W - C|| over single-node failures, so that a region that is never used under its own
    strategy, or the choice among equivalent placements of identical sensors, is settled
    the way a designer would settle it.  At most max_ties tied pairs per agreement set are
    scored, in enumeration order.
    Criterion 3: a final node that does not switch; then the enumeration order.

    Returns (sA, sB, number of switching nodes among 1..n-1).  `info`, if given, receives
    the number of tied pairs scored and whether the cap was reached."""
    RA, RB = realizations_of(pts, codes, A, tol), realizations_of(pts, codes, B, tol)
    if len(RA) == 0 or len(RB) == 0:
        raise ValueError("vertex not found among the enumerated strategies")
    n = codes.shape[1]
    up = n - 1
    base = max(len(t) for t in tables) + 1
    for size in range(up, -1, -1):
        cands, capped = set(), False
        for T in combinations(range(up), size):
            w = np.zeros(n, dtype=np.int64)
            for pos, t in enumerate(T):
                w[t] = base ** pos
            kA, kB = RA @ w, RB @ w
            common = np.intersect1d(kA, kB)
            if not common.size:
                continue
            count = 0
            for ia in np.flatnonzero(np.isin(kA, common)):
                for ib in np.flatnonzero(kB == kA[ia]):
                    cands.add((int(ia), int(ib)))
                    count += 1
                    if count >= max_ties:
                        break
                if count >= max_ties:
                    capped = True
                    break
        if cands:
            best = None
            for ia, ib in sorted(cands):
                sa, sb = _decode(RA[ia], tables), _decode(RB[ib], tables)
                val = 0.0 if score is None else float(score(sa, sb))
                key = (round(val, 12), int(RA[ia][-1] != RB[ib][-1]), ia, ib)
                if best is None or key < best[0]:
                    best = (key, sa, sb)
            if info is not None:
                info.update({"ties": len(cands), "capped": capped, "score": best[0][0]})
            return best[1], best[2], up - size
    raise RuntimeError("unreachable")


def _ignoring_node(models: Sequence[DiscreteModel], s: Strategy, tol: float = 1e-12) -> Tuple[int, Optional[int]]:
    """Largest k >= 2 such that node k does not depend on its incoming message, either
    because its two regions coincide or because the incoming message is constant; returns
    (k, level of the region it then uses), or (1, None) if there is none.  The nodes
    before k do not affect the system point."""
    pts = evaluate_chain(models, s)
    best: Tuple[int, Optional[int]] = (1, None)
    for k in range(2, s.n + 1):
        j0, j1 = s.pairs[k - 2]
        f, d = pts[k - 2]
        if j0 == j1:
            best = (k, j0)
        elif f <= tol and d <= tol:           # message always 0
            best = (k, j0)
        elif f >= 1 - tol and d >= 1 - tol:   # message always 1
            best = (k, j1)
    return best


def align_pair(models: Sequence[DiscreteModel], sA: Strategy, sB: Strategy) -> Tuple[Strategy, Strategy]:
    """Re-realize two strategies so that no node switches needlessly: a node whose rule
    does not affect its own strategy's system point (it lies before a node that ignores
    its message) takes the rule of the other strategy.  The system points are unchanged.
    Used for designs that are too long to enumerate; for short chains
    min_switch_realizations searches all realizations."""
    n = sA.n
    rules = {}
    cut = {}
    for name, s in (("A", sA), ("B", sB)):
        k, lvl = _ignoring_node(models, s)
        r = [s.j1] + [tuple(q) for q in s.pairs]
        if k >= 2:
            r[k - 1] = (lvl, lvl)
        rules[name], cut[name] = r, k
    for node in range(1, n + 1):
        freeA, freeB = node < cut["A"], node < cut["B"]
        if freeA and not freeB:
            rules["A"][node - 1] = rules["B"][node - 1]
        elif freeB:
            rules["B"][node - 1] = rules["A"][node - 1]
    out = tuple(Strategy(int(rules[x][0]), tuple(rules[x][1:])) for x in ("A", "B"))
    for s_old, s_new in zip((sA, sB), out):
        po, pn = evaluate_chain(models, s_old)[-1], evaluate_chain(models, s_new)[-1]
        if abs(po[0] - pn[0]) > 1e-9 or abs(po[1] - pn[1]) > 1e-9:
            raise RuntimeError("align_pair changed a system point")
    return out


def switching_nodes(sA: Strategy, sB: Strategy, include_last: bool = False) -> List[int]:
    """1-based indices of the nodes whose rule differs between the two strategies (the
    final node is left out unless include_last)."""
    n = sA.n
    last = n if include_last else n - 1
    return [k for k in range(1, last + 1)
            if (sA.j1 != sB.j1 if k == 1 else tuple(sA.pairs[k - 2]) != tuple(sB.pairs[k - 2]))]


def classify_pair(sa: Strategy, sb: Strategy) -> Dict[str, bool]:
    """Proposition 4 / Corollary 1 classification of a supporting pair.

    terminal_only  : identical rules at every upstream node (case (i), no sync needed)
    region_matched : identical downstream regions at every node k >= 2 (immune, Corollary 1)
    """
    terminal_only = sa.upstream_rules() == sb.upstream_rules()
    region_matched = tuple(sa.pairs) == tuple(sb.pairs)
    return {"terminal_only": terminal_only, "region_matched": region_matched,
            "susceptible": (not terminal_only) and (not region_matched)}


# ------------------------------------------- randomization at a single node
def single_node_optimum(pts: np.ndarray, codes: np.ndarray, tables, alpha: float, tol: float = 1e-12,
                        tie_tol: float = 1e-9) -> dict:
    """Best design in which a single node randomizes and every other node keeps one rule.

    Such a design needs no common draw: the randomizing node uses a private coin.  With the
    rules of the other nodes fixed, the system point is affine in the operating points of
    node k, so the designs in which only node k randomizes reach exactly the convex hull of
    the system points of the strategies that differ at node k alone, and the best of them at
    the budget alpha mixes two such strategies (or is deterministic).  The search is
    exhaustive over k, over the rules of the other nodes and over the two rules of node k,
    within the strategies of enumerate_nested_arrays (pts, codes, tables as returned there).

    The value lies between the deterministic optimum and the upper hull (dependent
    randomization), and it is a lower bound on what independent randomization at several
    nodes can reach (Papastavrou and Athans, 1995).

    Returns dict(PF, PD, node, nodes, sA, sB, q): sA is the strategy with the smaller
    false-alarm rate, used with probability q; node is the 1-based index of the randomizing
    node.  Several nodes may attain the same value: nodes lists every node at which a design
    of this kind reaches PD (to within tie_tol), and node, sA, sB, q describe the one with
    the smallest index.  If no mixture beats the best deterministic point, node is None,
    nodes is empty, sA = sB and q = 1."""
    shape = [len(t) for t in tables]
    n = len(shape)
    N = pts.shape[0]
    if N != int(np.prod(shape)):
        raise ValueError("pts is not the full enumeration of enumerate_nested_arrays")
    feas = np.flatnonzero(pts[:, 0] <= alpha + tol)
    if feas.size:
        i0 = int(feas[np.argmax(pts[feas, 1])])
        best = {"PF": float(pts[i0, 0]), "PD": float(pts[i0, 1]), "node": None, "rows": (i0, i0), "q": 1.0}
    else:                                   # cannot happen: the strategy that never declares H1 is enumerated
        raise ValueError("no deterministic strategy meets the constraint")
    P = pts.reshape(shape + [2])
    idx = np.arange(N).reshape(shape)
    per_node = []                           # the best value with node k as the only randomizing node
    for k in range(n):
        Q = np.moveaxis(P, k, n - 1).reshape(-1, shape[k], 2)        # groups x rules of node k x 2
        I = np.moveaxis(idx, k, n - 1).reshape(-1, shape[k])
        F, D = Q[:, :, 0], Q[:, :, 1]
        Fi, Fj = F[:, :, None], F[:, None, :]
        Di, Dj = D[:, :, None], D[:, None, :]
        ok = (Fi < alpha - tol) & (Fj > alpha + tol)                # a mixture is needed to reach P_F = alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.where(ok, Di + (Dj - Di) * (alpha - Fi) / (Fj - Fi), -np.inf)
        flat = val.reshape(val.shape[0], -1)
        g = int(np.argmax(flat.max(axis=1)))
        ij = int(np.argmax(flat[g]))
        per_node.append(float(flat[g, ij]))
        if flat[g, ij] > best["PD"] + 1e-12:
            i, j = divmod(ij, shape[k])
            best = {"PF": float(alpha), "PD": float(flat[g, ij]), "node": k + 1, "rows": (int(I[g, i]), int(I[g, j])),
                    "q": float((F[g, j] - alpha) / (F[g, j] - F[g, i]))}
    ra, rb = best.pop("rows")
    best["sA"], best["sB"] = _decode(codes[ra], tables), _decode(codes[rb], tables)
    best["nodes"] = [] if best["node"] is None else [k + 1 for k in range(n) if per_node[k] >= best["PD"] - tie_tol]
    return best


# ------------------------------------------------------------------- PBPO
def _objective(point: Point, alpha: Optional[float], lam: Optional[float]) -> float:
    """Constrained: lexicographic (feasible first, then P_D); Lagrangian: P_D - lam P_F."""
    if lam is not None:
        return point[1] - lam * point[0]
    if point[0] <= alpha + 1e-12:
        return 1.0 + point[1]
    return -point[0]      # infeasible: prefer the smaller false-alarm rate


def _regions_from_etas(models, j1, etas):
    """Regions induced by node thresholds eta_2..eta_n along the chain (the 'eta' mode:
    a downstream node re-derives tau(m) from its current upstream point)."""
    from .discrete import _taus
    f, d = models[0].region_probs(j1)
    pairs = []
    for m, eta in zip(models[1:], etas):
        tau1, tau0 = _taus(f, d, eta)
        j1_, j0_ = m.level_from_tau(tau1), m.level_from_tau(tau0)
        pairs.append((j0_, j1_))
        f, d = m.node_output(j0_, j1_, f, d)
    return tuple(pairs)


def pbpo(models: Sequence[DiscreteModel], alpha: Optional[float] = None,
         lam: Optional[float] = None, rng: Optional[np.random.Generator] = None,
         init: Optional[Strategy] = None, mode: str = "eta", max_sweeps: int = 100):
    """Person-by-person optimization (Algorithm S1).

    One node at a time, all other nodes fixed; a node update is accepted only on strict
    improvement; sweeps continue until no node changes.  The strategy space is finite and
    the objective is non-decreasing, so the procedure terminates.

    mode = 'eta'     : the other nodes keep their thresholds eta_k, and re-derive their
                       regions from the new upstream point (the paper's parametrization).
    mode = 'regions' : the other nodes keep their region pairs (fixed affine backward map).

    Returns (strategy, system point, number of sweeps, trace of objective values).
    """
    if (alpha is None) == (lam is None):
        raise ValueError("give exactly one of alpha (constrained) or lam (Lagrangian)")
    rng = np.random.default_rng() if rng is None else rng
    n = len(models)
    m1 = models[0]

    # ---- initialization: a random strategy (or the given one)
    if init is None:
        j1 = int(rng.choice(m1.levels))
        pairs = []
        etas = []
        f, d = m1.region_probs(j1)
        for m in models[1:]:
            cands = realizable_pairs(m, f, d, with_eta=True)
            (j0, j1_), eta = cands[rng.integers(len(cands))]
            pairs.append((j0, j1_)); etas.append(eta)
            f, d = m.node_output(j0, j1_, f, d)
        strat = Strategy(j1, tuple(pairs))
    else:
        strat = init
        etas = []
        f, d = m1.region_probs(strat.j1)
        for m, (j0, j1_) in zip(models[1:], strat.pairs):
            cands = dict(((pair, eta) for pair, eta in realizable_pairs(m, f, d, with_eta=True)))
            etas.append(cands.get((j0, j1_), 1.0))
            f, d = m.node_output(j0, j1_, f, d)

    def system_point(s: Strategy) -> Point:
        return evaluate_chain(models, s)[-1]

    best_val = _objective(system_point(strat), alpha, lam)
    trace = [best_val]
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        changed = False
        for k in range(n):
            # candidates for node k given the current upstream point
            if k == 0:
                cand_list = [(j, None) for j in m1.levels]
            else:
                pts = evaluate_chain(models, strat)
                f, d = pts[k - 1]
                cand_list = realizable_pairs(models[k], f, d, with_eta=True)
            for cand, eta in cand_list:
                if k == 0:
                    new_j1, new_pairs, new_etas = cand, list(strat.pairs), list(etas)
                else:
                    new_j1, new_pairs, new_etas = strat.j1, list(strat.pairs), list(etas)
                    new_pairs[k - 1] = cand
                    new_etas[k - 1] = eta
                if mode == "eta":
                    new_pairs = list(_regions_from_etas(models, new_j1, new_etas))
                cand_strat = Strategy(new_j1, tuple(new_pairs))
                val = _objective(system_point(cand_strat), alpha, lam)
                if val > best_val + 1e-12:
                    best_val, strat, etas, changed = val, cand_strat, new_etas, True
            trace.append(best_val)
        if not changed:
            break
    return strat, system_point(strat), sweeps, trace


def init_from_lambda(models: Sequence[DiscreteModel], lam: float) -> Strategy:
    """Deterministic initialization for the Lagrangian objective: node 1 applies the LRT
    at threshold lam and every downstream node uses eta_k = lam (regions re-derived along
    the chain).  For the terminal node eta_n = lam is exactly the optimal threshold given
    the upstream chain; for upstream nodes it is only a starting point."""
    j1 = models[0].level_from_tau(lam)
    return Strategy(j1, _regions_from_etas(models, j1, [lam] * (len(models) - 1)))


def pbpo_best(models, alpha=None, lam=None, restarts: int = 20, seed: int = 0, mode="eta"):
    """Best of several initializations (the lambda-initialization first when lam is given,
    then random ones).  Returns (strategy, point, stats) where stats holds the sweeps per
    restart and the fraction of restarts that reached the best value found.

    Single restarts reach the global optimum only part of the time (typically 20-60 %
    for mid-hull operating points of 4-5 node chains, less at the extremes), so 20 or
    more restarts are advisable, and exhaustive enumeration should be used to verify
    whenever n <= 5."""
    rng = np.random.default_rng(seed)
    runs = []
    if lam is not None:
        s, pt, sw, _tr = pbpo(models, lam=lam, init=init_from_lambda(models, lam), mode=mode)
        runs.append((s, pt, sw))
    while len(runs) < restarts:
        s, pt, sw, _tr = pbpo(models, alpha=alpha, lam=lam, rng=rng, mode=mode)
        runs.append((s, pt, sw))
    vals = [_objective(pt, alpha, lam) for _, pt, _ in runs]
    ibest = int(np.argmax(vals))
    hit = float(np.mean([abs(v - vals[ibest]) <= 1e-12 for v in vals]))
    return runs[ibest][0], runs[ibest][1], {"sweeps": [r[2] for r in runs], "hit_rate": hit}


def hull_by_pbpo(models, lambdas: Sequence[float], restarts: int = 5, seed: int = 0, mode="eta"):
    """Approximate the deterministic upper envelope for chains too long to enumerate:
    Lagrangian PBPO over a grid of multipliers, then the upper hull of the points found."""
    items = []
    for i, lam in enumerate(lambdas):
        s, pt, _ = pbpo_best(models, lam=lam, restarts=restarts, seed=seed + 1000 * i, mode=mode)
        items.append((pt, s))
    # the trivial strategies (never / always declare H1) realize (0,0) and (1,1)
    K = [m.K for m in models]
    items.append(((0.0, 0.0), Strategy(K[0], tuple((k, k) for k in K[1:]))))
    items.append(((1.0, 1.0), Strategy(0, tuple((0, 0) for _ in K[1:]))))
    return upper_hull([pt for pt, _ in items]), items
