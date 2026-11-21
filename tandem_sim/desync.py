"""Synchronization loss and its correction (Sections IV-A, IV-E, V of the paper and the
Proposition 6 of the revision), and the delayed-message variant of Section IV-F.

Model.  A dependent-randomization design switches the whole chain between two
deterministic strategies A and B by one common draw Z ~ Bernoulli(p) per decision
epoch.  A node in the set S "loses synchronization": it keeps the marginal p but draws
on its own, independently of Z and of the other desynchronized nodes.  Every node keeps
the decision regions it was designed with (its regions under A when it draws A, under B
when it draws B); what changes is the message distribution that actually arrives.

Everything here is exact: the desynchronized operating point is the probability-
weighted sum over the 2^(|S|+1) draw combinations.  Monte Carlo is never needed for a
given configuration; randomness enters only through the sampling of configurations.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .design import (enumerate_from, hull_value, support_pair, upper_hull, deterministic_optimum,
                     switching_nodes)
from .discrete import DiscreteModel, Point, Strategy, all_nested_pairs, evaluate_chain, realizable_pairs

Assignment = Tuple[str, ...]     # 'A' or 'B' per node


def evaluate_assignment(models: Sequence[DiscreteModel], sA: Strategy, sB: Strategy,
                        assign: Sequence[str]) -> List[Point]:
    """Forward recursion when node k uses the regions of strategy assign[k-1]."""
    strat = {"A": sA, "B": sB}
    j1 = strat[assign[0]].j1
    f, d = models[0].region_probs(j1)
    pts = [(f, d)]
    for k in range(1, len(models)):
        j0, j1_ = strat[assign[k]].pairs[k - 1]
        f, d = models[k].node_output(j0, j1_, f, d)
        pts.append((f, d))
    return pts


def _combos(n: int, p: float, S: Sequence[int], force: Optional[Dict[int, str]] = None):
    """All draw combinations: common draw Z plus one independent draw per node in S.
    `force` pins the draw of given nodes (1-based) to 'A' or 'B' with probability one
    (used for the terminal-node correction).  Yields (assignment, probability)."""
    S = sorted(set(S))
    force = force or {}
    for Z in ("A", "B"):
        pz = p if Z == "A" else 1 - p
        for bits in product("AB", repeat=len(S)):
            pr = pz
            assign = [Z] * n
            for k, b in zip(S, bits):
                pr *= p if b == "A" else 1 - p
                assign[k - 1] = b
            for k, b in force.items():
                assign[k - 1] = b
            yield tuple(assign), pr


def desync_point(models: Sequence[DiscreteModel], sA: Strategy, sB: Strategy, p: float,
                 S: Iterable[int], force: Optional[Dict[int, str]] = None,
                 node: Optional[int] = None):
    """Exact operating point under synchronization loss at the nodes in S (1-based).
    Returns (W, combos) where combos lists (assignment, probability, system point).
    `node` selects the output of node `node` instead of the system output."""
    n = len(models)
    idx = n - 1 if node is None else node - 1
    combos = []
    W = np.zeros(2)
    for assign, pr in _combos(n, p, list(S), force):
        if pr == 0:
            continue
        pt = evaluate_assignment(models, sA, sB, assign)[idx]
        combos.append((assign, pr, pt))
        W += pr * np.asarray(pt)
    return (float(W[0]), float(W[1])), combos


def sync_point(models, sA, sB, p) -> Point:
    A = evaluate_chain(models, sA)[-1]
    B = evaluate_chain(models, sB)[-1]
    return (p * A[0] + (1 - p) * B[0], p * A[1] + (1 - p) * B[1])


def worst_single_shift(models, sA, sB, p) -> float:
    """Largest shift ||W - C|| over the failures of one switching node among nodes 1..n-1
    (zero if no such node switches: the design is then immune)."""
    C = np.asarray(sync_point(models, sA, sB, p))
    out = 0.0
    for k in switching_nodes(sA, sB):
        W = np.asarray(desync_point(models, sA, sB, p, [k])[0])
        out = max(out, float(np.linalg.norm(W - C)))
    return out


def effective_point(models, sA, sB, p, S, node: int) -> Point:
    """Mean operating point of node `node` under the desynchronized configuration (the
    distribution a downstream node that randomizes on its own effectively faces)."""
    W, _ = desync_point(models, sA, sB, p, S, node=node)
    return W


# ---------------------------------------------------------------- Prop. 5
def prop5_two_node(models, sA, sB, p) -> Dict[str, float]:
    """Eq. (13): W_PF - alpha = p(1-p)(rho_F^A - rho_F^B)(f_B - f_A), and the P_D analogue."""
    fA, dA = models[0].region_probs(sA.j1)
    fB, dB = models[0].region_probs(sB.j1)
    rFA, rDA = models[1].attenuation(*sA.pairs[0])
    rFB, rDB = models[1].attenuation(*sB.pairs[0])
    return {"dPF": p * (1 - p) * (rFA - rFB) * (fB - fA),
            "dPD": p * (1 - p) * (rDA - rDB) * (dB - dA),
            "rho_F": (rFA, rFB), "rho_D": (rDA, rDB), "f": (fA, fB), "d": (dA, dB)}


def prop5_midchain(models, sA, sB, p, k: int, coord: int = 0) -> float:
    """Eq. (14) for a desynchronization at node k (1-based, k >= 2):
    p(1-p)[(H^A - H^B)(c^B - c^A) + (rho^B - rho^A)(H^A a_A - H^B a_B)],
    coord 0 for the P_F coordinate, 1 for P_D."""
    ptsA, ptsB = evaluate_chain(models, sA), evaluate_chain(models, sB)
    aA, aB = ptsA[k - 2][coord], ptsB[k - 2][coord]
    cA, cB = models[k - 1].intercepts(*sA.pairs[k - 2])[coord], models[k - 1].intercepts(*sB.pairs[k - 2])[coord]
    rA, rB = models[k - 1].attenuation(*sA.pairs[k - 2])[coord], models[k - 1].attenuation(*sB.pairs[k - 2])[coord]
    HA = float(np.prod([models[j].attenuation(*sA.pairs[j - 1])[coord] for j in range(k, len(models))]))
    HB = float(np.prod([models[j].attenuation(*sB.pairs[j - 1])[coord] for j in range(k, len(models))]))
    return p * (1 - p) * ((HA - HB) * (cB - cA) + (rB - rA) * (HA * aA - HB * aB))


# ------------------------------------------------ delayed message (Sec. IV-F)
def delayed_link_point(models, sA, sB, p, k: int) -> Point:
    """Exact operating point when the message on the link into node k (1-based, k >= 2)
    is delayed by one or more epochs while the hypothesis is unchanged.  The message then
    comes from an earlier epoch, so nodes 1..k-1 act as a block under an earlier common
    draw Z', independent of the current draw Z used by nodes k..n."""
    n = len(models)
    if not 2 <= k <= n:
        raise ValueError("k must satisfy 2 <= k <= n")
    W = np.zeros(2)
    for up, pu in (("A", p), ("B", 1 - p)):
        for dn, pd in (("A", p), ("B", 1 - p)):
            assign = [up] * (k - 1) + [dn] * (n - k + 1)
            W += pu * pd * np.asarray(evaluate_assignment(models, sA, sB, assign)[-1])
    return (float(W[0]), float(W[1]))


def delayed_link_identity(models, sA, sB, p, k: int, coord: int = 0) -> float:
    """Block form of eq. (13): W - C = p(1-p)(H^A - H^B)(x_B - x_A), where x_S is the rate
    of u_{k-1} = 1 under strategy S (false-alarm rate for coord 0, detection rate for
    coord 1) and H^S is the product of the attenuation factors of nodes k..n under S."""
    xA = evaluate_chain(models, sA)[k - 2][coord]
    xB = evaluate_chain(models, sB)[k - 2][coord]
    HA = float(np.prod([models[j].attenuation(*sA.pairs[j - 1])[coord] for j in range(k - 1, len(models))]))
    HB = float(np.prod([models[j].attenuation(*sB.pairs[j - 1])[coord] for j in range(k - 1, len(models))]))
    return p * (1 - p) * (HA - HB) * (xB - xA)


# ---------------------------------------------------------------- Prop. 6
def prop6_terms(models, sA, sB, p, S: Iterable[int]) -> Dict[str, object]:
    """Superposition identity and bound for several desynchronized nodes.

    Exact: W - C = sum_i T_i, with T_i = E[F(sigma^(i))] - E[F(sigma^(i-1))], sigma^(i)
    desynchronizing the first i nodes of S (in chain order).  T_i is the contribution
    of node k_i given that the nodes of S before it are already desynchronized.

    Bound (both coordinates):  |W - C| <= p(1-p) sum_{k in S} (|H_k^A| + |H_k^B|) Delta_k
                                      <= 2p(1-p) sum_{k in S} Delta_k prod_{j>k} max(|rho_j^A|, |rho_j^B|)
    with Delta_k = max_m |P(R_k^A(m)|H) - P(R_k^B(m)|H)| (Delta_1 = |f_A - f_B|) and
    H_k^S = prod_{j>k} rho_j^S.  For nested regions every rho_j lies in [0, 1] and the absolute
    values change nothing; they make the bound hold for any pair of regions (Theorem 1).
    """
    S = sorted(set(S))
    n = len(models)
    C = np.asarray(sync_point(models, sA, sB, p))
    terms = []
    prev = C
    for i in range(1, len(S) + 1):
        Wi, _ = desync_point(models, sA, sB, p, S[:i])
        Wi = np.asarray(Wi)
        terms.append(tuple(Wi - prev))
        prev = Wi
    W = prev
    bound_tight = np.zeros(2)
    bound_simple = np.zeros(2)
    for k in S:
        for coord in (0, 1):
            if k == 1:
                delta = abs(models[0].region_probs(sA.j1)[coord] - models[0].region_probs(sB.j1)[coord])
            else:
                m = models[k - 1]
                pa, pb = sA.pairs[k - 2], sB.pairs[k - 2]
                probs = m.F if coord == 0 else m.D
                delta = max(abs(probs[pa[0]] - probs[pb[0]]), abs(probs[pa[1]] - probs[pb[1]]))
            HA = float(np.prod([models[j].attenuation(*sA.pairs[j - 1])[coord] for j in range(k, n)]))
            HB = float(np.prod([models[j].attenuation(*sB.pairs[j - 1])[coord] for j in range(k, n)]))
            Hmax = float(np.prod([max(abs(models[j].attenuation(*sA.pairs[j - 1])[coord]),
                                      abs(models[j].attenuation(*sB.pairs[j - 1])[coord])) for j in range(k, n)]))
            bound_tight[coord] += p * (1 - p) * (abs(HA) + abs(HB)) * delta
            bound_simple[coord] += 2 * p * (1 - p) * Hmax * delta
    return {"C": tuple(C), "W": tuple(W), "shift": tuple(W - C), "terms": terms,
            "bound_tight": tuple(bound_tight), "bound_simple": tuple(bound_simple)}


# ------------------------------------------------------------- correction
def _conditional_points(models, sA, sB, p, S, n):
    """E[system point | Z, Q] over the independent draws of the nodes in S, for the
    common draw Z and the DFC rule Q in {A, B}."""
    S = sorted(set(S))
    out = {}
    for Z in ("A", "B"):
        for Q in ("A", "B"):
            E = np.zeros(2)
            for bits in product("AB", repeat=len(S)):
                pr = float(np.prod([p if b == "A" else 1 - p for b in bits])) if S else 1.0
                assign = [Z] * n
                for k, b in zip(S, bits):
                    assign[k - 1] = b
                assign[n - 1] = Q
                E += pr * np.asarray(evaluate_assignment(models, sA, sB, assign)[-1])
            out[(Z, Q)] = E
    return out


def terminal_adjust(models, sA, sB, p, alpha, S: Iterable[int] = (1,), tol: float = 1e-9) -> Dict[str, object]:
    """Terminal-node correction, eqs. (17)-(18) (closed form for n = 2: eqs. (19)-(20)).

    The DFC (node n) changes the probability with which it uses the strategy-A rule
    from p to q while staying coupled to the common draw Z as tightly as possible
    (monotone coupling: for q <= p it uses A only when Z = A, with probability q/p; for
    q >= p it uses A whenever Z = A and, when Z = B, with probability (q-p)/(1-p)).
    The nodes in S keep drawing independently with marginal p; every other node keeps
    Z.  At q = p the DFC follows Z exactly, so P(p) = W, the uncorrected point.

    With E_ZQ = E[system point | Z, Q] (expectation over the independent draws),
        q <= p:  P(q) = q E_AA + (p - q) E_AB + (1 - p) E_BB
        q >= p:  P(q) = p E_AA + (q - p) E_BA + (1 - q) E_BB
    P_F and P_D are piecewise affine in q with a kink at q = p, so the maximizer of P_D
    subject to P_F <= alpha is an endpoint of a feasible sub-interval of [0, p] or [p, 1].
    For n = 2 (no synchronized node besides the DFC) both pieces are the same line
    P(q) = (1-q) L + q U with L = p M_1 + (1-p) BB and U = p AA + (1-p) M_2, and
    q* = clip((L_F - alpha)/(L_F - U_F), 0, 1) in the nested case.
    """
    n = len(models)
    if n in set(S):
        raise ValueError("the DFC itself cannot be the desynchronized node here")
    E = _conditional_points(models, sA, sB, p, S, n)
    W = p * E[("A", "A")] + (1 - p) * E[("B", "B")]
    L = p * E[("A", "B")] + (1 - p) * E[("B", "B")]       # DFC always B
    U = p * E[("A", "A")] + (1 - p) * E[("B", "A")]       # DFC always A

    def P(q):
        if q <= p:
            return q * E[("A", "A")] + (p - q) * E[("A", "B")] + (1 - p) * E[("B", "B")]
        return p * E[("A", "A")] + (q - p) * E[("B", "A")] + (1 - q) * E[("B", "B")]

    # Feasible part of each piece, with a tolerance in P_F units: the constraint is often met
    # with equality at an end point (e.g. U_F = alpha when one strategy does not use the
    # desynchronized node), and alpha itself comes from hull points rounded to 1e-12.
    cands = []
    for lo, hi in ((0.0, p), (p, 1.0)):
        if abs(hi - lo) < 1e-15:
            continue
        Flo, Fhi = P(lo)[0], P(hi)[0]
        ok_lo, ok_hi = Flo <= alpha + tol, Fhi <= alpha + tol
        if ok_lo and ok_hi:
            feas = (lo, hi)
        elif ok_lo:
            feas = (lo, min(hi, max(lo, lo + (alpha - Flo) / (Fhi - Flo) * (hi - lo))))
        elif ok_hi:
            feas = (max(lo, min(hi, lo + (alpha - Flo) / (Fhi - Flo) * (hi - lo))), hi)
        else:
            feas = None
        if feas is not None:
            cands += [feas[0], feas[1]]
    if not cands:
        return {"feasible": False, "L": tuple(L), "U": tuple(U), "W": tuple(W),
                "dfc_switches": tuple(sA.pairs[-1]) != tuple(sB.pairs[-1])}
    q_star = max(cands, key=lambda q: P(q)[1])
    pt = P(q_star)
    return {"feasible": True, "q": float(q_star), "point": (float(pt[0]), float(pt[1])),
            "L": tuple(L), "U": tuple(U), "W": tuple(W),
            "E": {"".join(k): tuple(v) for k, v in E.items()},
            "dfc_switches": tuple(sA.pairs[-1]) != tuple(sB.pairs[-1])}


def _conjugate(hull: Sequence[Point], lam: float) -> float:
    return max(y - lam * x for x, y in hull)


def dfc_layer_benchmark(models, sA, sB, p, alpha, S: Iterable[int] = (1,),
                        pair_class: str = "eta", z_aware: bool = True) -> Dict[str, object]:
    """Best the DFC alone can do.  The DFC knows the common draw Z, hence which strategy
    the synchronized upstream nodes use; under each Z it faces the conditional mean
    arriving point x̄_Z (expectation over the independent draws of the nodes in S).
    It may choose, for each Z, any region pair (single-threshold class or all nested
    pairs) and randomize locally; the achievable set given Z is the upper hull H_Z of
    its points.  The benchmark maximizes p y_A + (1-p) y_B subject to
    p x_A + (1-p) x_B <= alpha over (x_Z, y_Z) in H_Z.  By concavity there is no duality
    gap, and the value equals min_lam [p kappa_A(lam) + (1-p) kappa_B(lam) + lam alpha]
    with kappa_Z the conjugate of H_Z; the minimum is at a hull slope.

    z_aware=False ignores Z: the DFC faces the overall mixture x̄ = p x̄_A + (1-p) x̄_B
    and uses one hull (this is the right benchmark for n = 2, where Z is the DFC's own
    draw; for n >= 3 it is weaker than the Z-aware one).
    """
    n = len(models)
    m = models[n - 1]
    S = sorted(set(S))

    def xbar_given(Z):
        E = np.zeros(2)
        for bits in product("AB", repeat=len(S)):
            pr = float(np.prod([p if b == "A" else 1 - p for b in bits])) if S else 1.0
            assign = [Z] * n
            for k, b in zip(S, bits):
                assign[k - 1] = b
            E += pr * np.asarray(evaluate_assignment(models, sA, sB, assign)[n - 2])
        return tuple(E)

    def hull_at(x):
        cands = realizable_pairs(m, *x) if pair_class == "eta" else all_nested_pairs(m)
        return upper_hull([m.node_output(j0, j1, *x) for (j0, j1) in cands])

    xA, xB = xbar_given("A"), xbar_given("B")
    if not z_aware or n == 2:
        xbar = (p * xA[0] + (1 - p) * xB[0], p * xA[1] + (1 - p) * xB[1])
        hull = hull_at(xbar)
        return {"xbar": xbar, "PD": hull_value(hull, alpha), "hull": hull, "z_aware": False}
    HA, HB = hull_at(xA), hull_at(xB)
    slopes = {0.0}
    for H in (HA, HB):
        for (x1, y1), (x2, y2) in zip(H[:-1], H[1:]):
            if x2 > x1:
                slopes.add((y2 - y1) / (x2 - x1))
    value = min(p * _conjugate(HA, lam) + (1 - p) * _conjugate(HB, lam) + lam * alpha for lam in slopes)
    return {"xbar_A": xA, "xbar_B": xB, "PD": float(value), "hull_A": HA, "hull_B": HB, "z_aware": True}


def downstream_benchmark(models, sA, sB, p, alpha, k: int, pair_class: str = "eta") -> Dict[str, object]:
    """Optimal correction when node k (1 <= k < n) draws independently with marginal p.
    Nodes 1..k-1 keep following the common draw Z, node k is fixed at its mixture, and
    nodes k+1..n, which know Z, are redesigned with dependent randomization among them:
    for each Z they face the mean arriving point x_Z at node k+1 and may use any point of
    the upper hull H_Z of the sub-chain k+1..n started at x_Z.  The value is
        max  p y_A + (1-p) y_B   s.t.  p x_A + (1-p) x_B <= alpha,  (x_Z, y_Z) in H_Z,
    computed through the dual (no gap, the sets are convex).  For k = 1 the two hulls
    coincide and this is full_downstream_benchmark; for k = n-1 it is the Z-aware
    DFC-layer benchmark."""
    n = len(models)
    if not 1 <= k < n:
        raise ValueError("k must satisfy 1 <= k < n")

    def x_given(Z):
        E = np.zeros(2)
        for b, pr in (("A", p), ("B", 1 - p)):
            assign = [Z] * n
            assign[k - 1] = b
            E += pr * np.asarray(evaluate_assignment(models, sA, sB, assign)[k - 1])
        return (float(E[0]), float(E[1]))

    hulls = {Z: upper_hull([pt for pt, _ in enumerate_from(models[k:], x_given(Z), pair_class)]) for Z in ("A", "B")}
    slopes = {0.0}
    for H in hulls.values():
        for (x1, y1), (x2, y2) in zip(H[:-1], H[1:]):
            if x2 > x1:
                slopes.add((y2 - y1) / (x2 - x1))
    value = min(p * _conjugate(hulls["A"], lam) + (1 - p) * _conjugate(hulls["B"], lam) + lam * alpha for lam in slopes)
    return {"PD": float(value), "x_A": x_given("A"), "x_B": x_given("B")}


def full_downstream_benchmark(models, sA, sB, p, alpha, pair_class: str = "eta") -> Dict[str, object]:
    """Optimal correction for a desynchronized node 1: node 1 is, for the rest of the
    chain, a fixed node at the mixture point p(f_A,d_A) + (1-p)(f_B,d_B); nodes 2..n are
    redesigned (exhaustively) with dependent randomization among them allowed."""
    fA, dA = models[0].region_probs(sA.j1)
    fB, dB = models[0].region_probs(sB.j1)
    xbar = (p * fA + (1 - p) * fB, p * dA + (1 - p) * dB)
    items = enumerate_from(models[1:], xbar, pair_class)
    pts = [pt for pt, _ in items]
    hull = upper_hull(pts)
    return {"xbar": xbar, "PD": hull_value(hull, alpha), "hull": hull,
            "deterministic": deterministic_optimum(pts, alpha),
            "support": support_pair(hull, alpha), "items": items}
