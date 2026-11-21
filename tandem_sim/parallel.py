"""Parallel architecture (n local detectors reporting directly to a fusion center) for
the tandem-versus-parallel comparison of Section IV-D.

A deterministic parallel strategy = one threshold level per LD + a fusion rule.  Fusion
rules are the likelihood-ratio tests on the vector of local decisions (Chair-Varshney);
with identical LDs these are exactly the k-out-of-n rules.  The achievable set is
enumerated exhaustively (all LD level combinations x all fusion thresholds), the
randomized ROC is its upper hull, and dependent randomization switches all LDs and the
fusion center jointly between the two supporting strategies A and B with one common
draw.  Synchronization loss at LD i: LD i draws independently with the same marginal p.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .design import support_pair, upper_hull, deterministic_optimum
from .discrete import DiscreteModel, Point


@dataclass(frozen=True)
class ParallelStrategy:
    levels: Tuple[int, ...]          # threshold level of each LD
    fusion: frozenset                # set of decision patterns (as integers) mapped to u0 = 1

    def rule_of(self, i: int) -> int:
        return self.levels[i]


def _patterns(n: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(float)   # (2^n, n)


def _pattern_probs(bits: np.ndarray, F: np.ndarray, D: np.ndarray):
    p0 = np.prod(np.where(bits == 1, F[None, :], 1 - F[None, :]), axis=1)
    p1 = np.prod(np.where(bits == 1, D[None, :], 1 - D[None, :]), axis=1)
    return p0, p1


def system_point(models: Sequence[DiscreteModel], levels: Sequence[int], fusion: frozenset) -> Point:
    n = len(models)
    bits = _patterns(n)
    F = np.array([m.F[j] for m, j in zip(models, levels)])
    D = np.array([m.D[j] for m, j in zip(models, levels)])
    p0, p1 = _pattern_probs(bits, F, D)
    sel = np.array([k in fusion for k in range(2 ** n)])
    return float(p0[sel].sum()), float(p1[sel].sum())


def k_of_n_fusion(n: int, k: int) -> frozenset:
    bits = _patterns(n)
    return frozenset(int(i) for i in np.where(bits.sum(axis=1) >= k)[0])


def enumerate_parallel(models: Sequence[DiscreteModel], fusion_class: str = "cv"):
    """All deterministic system points (with strategies) of the parallel architecture.

    fusion_class = 'cv'   : every likelihood-ratio fusion rule for the given LD rules
                            (thresholds on the fusion LR; ties grouped)
                   'kofn' : k-out-of-n rules only (k = 1..n)
    """
    n = len(models)
    bits = _patterns(n)
    out: List[Tuple[Point, ParallelStrategy]] = []
    kofn = [k_of_n_fusion(n, k) for k in range(1, n + 1)]
    for levels in product(*[m.levels for m in models]):
        F = np.array([m.F[j] for m, j in zip(models, levels)])
        D = np.array([m.D[j] for m, j in zip(models, levels)])
        p0, p1 = _pattern_probs(bits, F, D)
        if fusion_class == "kofn":
            for fus in kofn:
                sel = np.fromiter((k in fus for k in range(2 ** n)), bool, 2 ** n)
                out.append(((float(p0[sel].sum()), float(p1[sel].sum())), ParallelStrategy(tuple(levels), fus)))
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.where(p0 > 0, p1 / np.where(p0 > 0, p0, 1), np.where(p1 > 0, np.inf, 1.0))
        order = np.argsort(-lr, kind="stable")          # descending fusion LR
        lr_sorted = lr[order]
        cum0 = np.cumsum(p0[order]); cum1 = np.cumsum(p1[order])
        # thresholds between distinct LR values: include the top-m patterns for every
        # m at a group boundary
        for m_ in range(1, 2 ** n):
            if lr_sorted[m_ - 1] > lr_sorted[m_] * (1 + 1e-12) or (np.isinf(lr_sorted[m_ - 1]) and not np.isinf(lr_sorted[m_])):
                fus = frozenset(int(i) for i in order[:m_])
                out.append(((float(cum0[m_ - 1]), float(cum1[m_ - 1])), ParallelStrategy(tuple(levels), fus)))
    out.append(((0.0, 0.0), ParallelStrategy(tuple(m.K for m in models), frozenset())))
    out.append(((1.0, 1.0), ParallelStrategy(tuple(0 for m in models), frozenset(range(2 ** n)))))
    return out


def canonical_form(models: Sequence[DiscreteModel], strat: ParallelStrategy):
    """Remove the arbitrary part of a parallel strategy.  An LD at level 0 or K is constant,
    so the fusion rule is used only on the patterns carrying that constant; the fusion set
    is replaced by its projection onto those patterns, which no longer depends on that LD.
    Returns (projected fusion set, tuple of the LDs the fusion rule ignores).  An ignored LD
    may use any threshold without changing the system point."""
    n = len(models)
    const = {i: (1 if strat.levels[i] == 0 else 0) for i in range(n) if strat.levels[i] in (0, models[i].K)}

    def proj(u: int) -> int:
        for i, c in const.items():
            u = (u | (1 << i)) if c else (u & ~(1 << i))
        return u

    phi = frozenset(u for u in range(2 ** n) if proj(u) in strat.fusion)
    ignored = tuple(i for i in range(n) if all(((u ^ (1 << i)) in phi) == (u in phi) for u in range(2 ** n)))
    return phi, ignored


def min_switch_parallel(models: Sequence[DiscreteModel], items, A: Point, B: Point, tol: float = 1e-9,
                        p: Optional[float] = None, max_ties: int = 200, info: Optional[dict] = None):
    """Realizations of the vertices A and B of the parallel system, chosen as for the chain.

    Criterion 1: the fewest LDs whose threshold differs (an LD that the fusion rule ignores
    takes the threshold of the other strategy).  Every enumerated realization of each
    vertex is considered.
    Criterion 2 (ties, when the mixing probability p is given): the smallest worst-case shift
    ||W - C|| over single-LD failures; at most max_ties tied pairs are scored.
    Criterion 3: the same fusion rule; then the enumeration order.
    Returns (sA, sB, switching LDs)."""
    n = len(models)

    def canon_list(point):
        seen, out = set(), []
        for pt, s in items:
            if abs(pt[0] - point[0]) <= tol and abs(pt[1] - point[1]) <= tol:
                phi, ign = canonical_form(models, s)
                lv = tuple(-1 if i in ign else s.levels[i] for i in range(n))
                if (lv, phi) not in seen:
                    seen.add((lv, phi)); out.append((lv, phi))
        return out

    CA, CB = canon_list(A), canon_list(B)
    if not CA or not CB:
        raise ValueError("vertex not found among the enumerated strategies")
    LA = np.array([c[0] for c in CA]); LB = np.array([c[0] for c in CB])
    diff = ((LA[:, None, :] != LB[None, :, :]) & (LA[:, None, :] >= 0) & (LB[None, :, :] >= 0)).sum(axis=2)
    dmin = int(diff.min())
    tied = [(int(i), int(j)) for i, j in zip(*np.nonzero(diff == dmin))]
    capped = len(tied) > max_ties
    tied = tied[:max_ties]

    def build(ia, ib):
        la, lb = list(CA[ia][0]), list(CB[ib][0])
        for i in range(n):
            if la[i] < 0 and lb[i] < 0:
                la[i] = lb[i] = models[i].K
            elif la[i] < 0:
                la[i] = lb[i]
            elif lb[i] < 0:
                lb[i] = la[i]
        return ParallelStrategy(tuple(la), CA[ia][1]), ParallelStrategy(tuple(lb), CB[ib][1])

    best = None
    for ia, ib in tied:
        sA, sB = build(ia, ib)
        sw = [i for i in range(n) if sA.levels[i] != sB.levels[i]]
        val = 0.0
        if p is not None:
            val = max([parallel_desync(models, sA, sB, p, i)["norm"] for i in sw], default=0.0)
        key = (round(val, 12), int(sA.fusion != sB.fusion), ia, ib)
        if best is None or key < best[0]:
            best = (key, sA, sB, sw)
    _, sA, sB, sw = best
    for s, point in ((sA, A), (sB, B)):
        pt = system_point(models, s.levels, s.fusion)
        if abs(pt[0] - point[0]) > 1e-9 or abs(pt[1] - point[1]) > 1e-9:
            raise RuntimeError("min_switch_parallel changed a system point")
    if info is not None:
        info.update({"ties": len(tied), "capped": capped, "score": best[0][0]})
    return sA, sB, sw


def parallel_design(models: Sequence[DiscreteModel], alpha: float, fusion_class: str = "cv",
                    realization: str = "min_switch"):
    """Randomized-ROC design of the parallel system at budget alpha: supporting strategies
    A, B, mixing probability p.  realization = 'min_switch' (default) uses the realizations
    of the two vertices with the fewest switching LDs, ties broken by the smallest worst-case
    shift; 'first' the first ones enumerated."""
    items = enumerate_parallel(models, fusion_class)
    pts = [pt for pt, _ in items]
    hull = upper_hull(pts)
    sp = support_pair(hull, alpha)
    def at(point):
        return [s for pt, s in items if abs(pt[0] - point[0]) <= 1e-9 and abs(pt[1] - point[1]) <= 1e-9]
    if realization == "first" or sp.deterministic:
        sA, sB = at(sp.A)[0], at(sp.B)[0]
    else:
        sA, sB, _ = min_switch_parallel(models, items, sp.A, sp.B, p=sp.p)
    return {"items": items, "hull": hull, "support": sp, "sA": sA, "sB": sB,
            "deterministic": deterministic_optimum(pts, alpha)}


def parallel_desync(models: Sequence[DiscreteModel], sA: ParallelStrategy, sB: ParallelStrategy,
                    p: float, i: int) -> Dict[str, object]:
    """Exact operating point when LD i (0-based) draws independently with marginal p while
    the other LDs and the fusion center follow the common draw."""
    strat = {"A": sA, "B": sB}
    W = np.zeros(2)
    for Z in ("A", "B"):
        pz = p if Z == "A" else 1 - p
        for Zi in ("A", "B"):
            pi = p if Zi == "A" else 1 - p
            levels = list(strat[Z].levels)
            levels[i] = strat[Zi].levels[i]
            pt = system_point(models, levels, strat[Z].fusion)
            W += pz * pi * np.asarray(pt)
    A = system_point(models, sA.levels, sA.fusion)
    B = system_point(models, sB.levels, sB.fusion)
    C = p * np.asarray(A) + (1 - p) * np.asarray(B)
    return {"W": tuple(W), "C": tuple(C), "shift": tuple(W - C), "norm": float(np.linalg.norm(W - C))}
