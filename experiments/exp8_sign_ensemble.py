"""Experiment 8 - direction of the P_F shift over every adjacent hull pair (Section IV-E).

Complements the alpha-sampled statistics of exp5: here every adjacent pair of vertices of
the upper hull of a two-node chain is a design (dependent randomization between the two
vertices), for N random Dirichlet(1) K-point models shared by both nodes.  The two edges
adjacent to the trivial strategies (never / always declare H_1) are excluded.

Main count: each pair is realized as in the designs of the paper (fewest switching nodes,
ties by the smallest shift) and classified as immune (N_1 does not switch, or the shift
is zero), P_F rises, or P_F falls; the sign of (13) does not depend on p.
For reference the script also counts every pair of single-threshold realizations of the
two vertices, with and without the trivial edges (the shares depend on these conventions).

Proposition 7 (two-node chains): for every susceptible design the script checks that both
N_1 rules are non-trivial, that the DFC regions of the strategy with the larger f are
contained in those of the other, and that the adjustment meets the constraint; it also
checks feasibility for every pair of nested realizations of the two vertices in which the
N_1 rules differ (free regions and trivial N_1 rules included).  Feasibility does not depend
on p: L_F <= alpha iff g_B(f_A) <= g_A(f_A), U_F <= alpha iff g_A(f_B) <= g_B(f_B).
The last block repeats the check of Proposition 7 in exact rational arithmetic on 20 000
chains with integer-weight models, in which likelihood-ratio ties are frequent
(tandem_sim/exact.py; independent of the floating-point code above).
Usage: python experiments/exp8_sign_ensemble.py [N] [K]      (default 5000, 3)
"""
import sys
from collections import Counter
import numpy as np
from common import *
from tandem_sim.discrete import random_model
from tandem_sim.exact import prop7_exact_check

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
rng = np.random.default_rng(0)
design, allpairs, allpairs_trivial = Counter(), Counter(), Counter()
edges_main = edges_all = 0
prop7 = Counter()
TOLF = 1e-9


def feasible(models, a, b):
    """min(L_F, U_F) <= alpha for the pair (a, b) of two-node strategies (any p)."""
    m1, m2 = models
    fA, fB = m1.F[a.j1], m1.F[b.j1]
    g = lambda st, x: (1 - x) * m2.F[st.pairs[0][0]] + x * m2.F[st.pairs[0][1]]
    return (g(b, fA) <= g(a, fA) + TOLF) or (g(a, fB) <= g(b, fB) + TOLF)

models_susceptible = 0
for _ in range(N):
    m = random_model(K, rng)
    models = [m, m]
    n_before = design["rise"] + design["fall"]
    arr, codes, tables = enumerate_nested_arrays(models)
    hull = hull_of_array(arr)
    items = enumerate_system_points(models)
    for a, b in zip(hull[:-1], hull[1:]):
        trivial = a == (0.0, 0.0) or b == (1.0, 1.0) or a[0] == b[0]
        edges_all += 1
        edges_main += not trivial
        if not trivial:
            sA, sB, _ = min_switch_realizations(arr, codes, tables, a, b,
                                                score=lambda x, y: worst_single_shift(models, x, y, 0.5))
            if sA.j1 == sB.j1:
                lab = "immune"
            else:
                d = prop5_two_node(models, sA, sB, 0.5)["dPF"]
                lab = "rise" if d > 1e-12 else "fall" if d < -1e-12 else "immune"
            design[lab] += 1
            if lab != "immune":
                prop7["susceptible designs"] += 1
                fA, fB = m.F[sA.j1], m.F[sB.j1]
                prop7["trivial N1 rule"] += (fA in (0.0, 1.0)) or (fB in (0.0, 1.0))
                hot, cold = (sA, sB) if fA > fB else (sB, sA)       # larger f <=> smaller DFC regions (levels >=)
                prop7["regions nested as in Prop. 7"] += all(h >= c for h, c in zip(hot.pairs[0], cold.pairs[0]))
                alpha = 0.5 * (a[0] + b[0])
                adj = terminal_adjust(models, sA, sB, 0.5, alpha)
                prop7["adjustment feasible"] += bool(adj["feasible"])
                prop7["P_F restored to alpha"] += bool(adj["feasible"]) and abs(adj["point"][0] - alpha) < 1e-9
            RA, RB = realizations_of(arr, codes, a), realizations_of(arr, codes, b)
            for ra in RA[:40]:
                for rb in RB[:40]:
                    x, y = decode_strategy(ra, tables), decode_strategy(rb, tables)
                    if x.j1 != y.j1 and worst_single_shift(models, x, y, 0.5) > TOLF:
                        prop7["susceptible realization pairs"] += 1
                        prop7["realization pairs feasible"] += feasible(models, x, y)
        for sA in strategies_at(items, a):
            for sB in strategies_at(items, b):
                if sA.j1 == sB.j1:
                    lab = "unchanged"
                else:
                    d = prop5_two_node(models, sA, sB, 0.5)["dPF"]
                    lab = "rise" if d > 1e-12 else "fall" if d < -1e-12 else "unchanged"
                allpairs_trivial[lab] += 1
                if not trivial:
                    allpairs[lab] += 1
    models_susceptible += (design["rise"] + design["fall"]) > n_before


def fr(c):
    t = sum(c.values())
    return {k: v / t for k, v in c.items()}


out = {"N_models": N, "K": K,
       "designs": {"edges": edges_main, "counts": dict(design), "fractions": fr(design)},
       "all_single_threshold_realization_pairs": {"edges": edges_main, "realization_pairs": sum(allpairs.values()),
                                                  "counts": dict(allpairs), "fractions": fr(allpairs)},
       "all_pairs_including_trivial_edges": {"edges": edges_all, "realization_pairs": sum(allpairs_trivial.values()),
                                             "counts": dict(allpairs_trivial), "fractions": fr(allpairs_trivial)}}
out["proposition_7"] = {k: int(v) for k, v in prop7.items()}
out["models_with_a_susceptible_edge"] = models_susceptible
out["proposition_7_exact_arithmetic"] = prop7_exact_check(20000, seed=0)
for k, v in out.items():
    print(k, v)
save_json("exp8_sign_ensemble.json", out)
