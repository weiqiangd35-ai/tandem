"""Exact-arithmetic check of Proposition 7 on two-node chains with likelihood-ratio ties.

Independent of the floating-point modules of the package.  The observation models have
integer weights, so equal likelihood ratios inside a node, and thresholds that fall exactly
on a likelihood-ratio value, are frequent; every operating point is a Fraction and every
comparison is exact.

Strategy class (Section II of the paper): N_1 applies an LRT {Lambda_1 >= eta}; the DFC
applies, for each message m, an LRT region {Lambda_2 >= t_m} with R(1) containing R(0).
For every edge of the upper hull of the deterministic strategies and every pair of
realizations of its two vertices in which the N_1 rules are distinct and non-trivial, the
function checks
  (i)  the DFC regions of the strategy with the larger f are contained in those of the other;
  (ii) alpha lies between L_F and U_F, i.e. the adjustment (19) meets the constraint
       (evaluated at p = 1/2; the two signs do not depend on p).
Pairs in which an N_1 rule is trivial are counted separately: there the region used for the
message that never occurs is free, and (ii) can fail (Sec. S2 of the supplementary material).
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction

import numpy as np


def _pmf(K: int, total: int, rng: np.random.Generator):
    """A strictly positive pmf on K levels with integer weights summing to `total`."""
    cuts = sorted(int(c) for c in rng.choice(np.arange(1, total), size=K - 1, replace=False))
    w = [b - a for a, b in zip([0] + cuts, cuts + [total])]
    return [Fraction(x, total) for x in w]


def _lrt_regions(p0, p1):
    """All regions {Lambda >= t}, from the empty set to the whole alphabet: (P(.|H0), P(.|H1))."""
    lr = [b / a for a, b in zip(p0, p1)]
    out = [(Fraction(0), Fraction(0))]
    for v in sorted(set(lr), reverse=True):
        S = [i for i, x in enumerate(lr) if x >= v]
        out.append((sum(p0[i] for i in S), sum(p1[i] for i in S)))
    return out


def _upper_hull(points):
    best = {}
    for f, d in points:
        if f not in best or d > best[f]:
            best[f] = d
    H = []
    for p in sorted(best.items()):
        while len(H) >= 2:
            (x1, y1), (x2, y2) = H[-2], H[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:      # H[-1] on or below the chord
                H.pop()
            else:
                break
        H.append(p)
    return H


def prop7_exact_check(n_models: int = 20000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    c = Counter()
    half = Fraction(1, 2)
    for _ in range(n_models):
        K1, K2 = int(rng.choice([2, 3, 4])), int(rng.choice([2, 3, 4]))
        t1, t2 = int(rng.choice([6, 8, 10, 12])), int(rng.choice([6, 8, 10, 12]))
        R1 = _lrt_regions(_pmf(K1, t1, rng), _pmf(K1, t1, rng))
        R2 = _lrt_regions(_pmf(K2, t2, rng), _pmf(K2, t2, rng))
        by_point = {}
        for j1, (f, d) in enumerate(R1):
            for i0 in range(len(R2)):
                for i1 in range(i0, len(R2)):
                    pt = ((1 - f) * R2[i0][0] + f * R2[i1][0], (1 - d) * R2[i0][1] + d * R2[i1][1])
                    by_point.setdefault(pt, []).append((j1, i0, i1))
        H = _upper_hull(by_point.keys())
        c["models"] += 1
        for a, b in zip(H[:-1], H[1:]):
            c["edges"] += 1
            alpha = half * (a[0] + b[0])
            for sA in by_point[a]:
                for sB in by_point[b]:
                    if sA[0] == sB[0]:
                        continue
                    fA, fB = R1[sA[0]][0], R1[sB[0]][0]
                    gA = lambda x: (1 - x) * R2[sA[1]][0] + x * R2[sA[2]][0]
                    gB = lambda x: (1 - x) * R2[sB[1]][0] + x * R2[sB[2]][0]
                    LF = half * (gB(fA) + gB(fB))
                    UF = half * (gA(fA) + gA(fB))
                    between = min(LF, UF) <= alpha <= max(LF, UF)
                    if sA[0] in (0, len(R1) - 1) or sB[0] in (0, len(R1) - 1):
                        c["pairs_with_a_trivial_N1_rule"] += 1
                        c["trivial_alpha_between_LF_and_UF"] += between
                        continue
                    hot, cold = (sA, sB) if fA > fB else (sB, sA)       # regions are the top-i groups
                    WF = (gA(fA) + gB(fA) + gA(fB) + gB(fB)) / 4
                    c["pairs"] += 1
                    c["regions_nested_as_in_prop7"] += hot[1] <= cold[1] and hot[2] <= cold[2]
                    c["alpha_between_LF_and_UF"] += between
                    c["one_dfc_ignores_its_message"] += sA[1] == sA[2] or sB[1] == sB[2]
                    c["PF_rises"] += WF > alpha
                    c["PF_falls"] += WF < alpha
                    if LF != UF:
                        q = (LF - alpha) / (LF - UF)
                        c["qstar_in_unit_interval_and_exact"] += (0 <= q <= 1) and ((1 - q) * LF + q * UF == alpha)
                    else:
                        c["LF_equals_UF"] += 1
                        c["LF_equals_UF_equals_alpha"] += LF == alpha
    return {k: int(v) for k, v in c.items()}


if __name__ == "__main__":
    import sys
    out = prop7_exact_check(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
    for k, v in out.items():
        print(f"{k:38s} {v}")
