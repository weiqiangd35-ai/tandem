"""Regression tests: every number quoted in the submitted manuscript (TAES-2026-3140) that
the code can reproduce, plus the new worked numbers of the revision.

Run:  python -m pytest tests -q
"""
import numpy as np
import pytest

from tandem_sim import *
from tandem_sim.design import distinct_points, strategies_at, hull_value
from tandem_sim.desync import evaluate_assignment

RUN = DiscreteModel([0.8, 0.1, 0.1], [0.3, 0.1, 0.6])          # running model, Sec. III-C
VIO = DiscreteModel([0.45, 0.50, 0.05], [0.35, 0.55, 0.10])     # constraint-violating model, Sec. IV-E


def close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol)


# ----------------------------------------------------------------- Sec. III-C
def test_local_rules_and_lr():
    assert close(RUN.lr_sorted, [0.375, 1.0, 6.0])
    assert set(RUN.local_points()) == {(0, 0), (0.1, 0.6), (0.2, 0.7), (1, 1)}


def test_twelve_achievable_points():
    items = enumerate_system_points([RUN, RUN])
    pts = distinct_points(items)
    assert len(pts) == 12
    expected = {(0, 0), (0.01, 0.36), (0.02, 0.42), (0.04, 0.49), (0.1, 0.6), (0.11, 0.66),
                (0.12, 0.67), (0.19, 0.84), (0.2, 0.7), (0.28, 0.88), (0.36, 0.91), (1, 1)}
    assert all(any(close(p, e) for e in expected) for p in pts)


def test_alpha_017_terminal_node_regime():
    items = enumerate_system_points([RUN, RUN])
    pts = distinct_points(items)
    hull = upper_hull(pts)
    assert close(deterministic_optimum(pts, 0.17), (0.12, 0.67))
    sp = support_pair(hull, 0.17)
    assert close(sp.A, (0.11, 0.66)) and close(sp.B, (0.19, 0.84))
    assert close(sp.p, 0.25) and close(sp.C, (0.17, 0.795))
    sa, sb = strategies_at(items, sp.A)[0], strategies_at(items, sp.B)[0]
    assert classify_pair(sa, sb)["terminal_only"]            # same N_1 rule: immune
    assert close(0.795 - 0.67, 0.125)                         # Delta P_D = +0.125


def test_alpha_0075_upstream_involving_regime():
    items = enumerate_system_points([RUN, RUN])
    hull = upper_hull(distinct_points(items))
    sp = support_pair(hull, 0.075)
    assert close(sp.A, (0.04, 0.49)) and close(sp.B, (0.11, 0.66))
    assert close(sp.p, 0.5) and close(sp.C, (0.075, 0.575))
    sa, sb = strategies_at(items, sp.A)[0], strategies_at(items, sp.B)[0]
    assert classify_pair(sa, sb)["susceptible"]              # distinct N_1 rules


# ------------------------------------------------------------------ Sec. IV-C
def test_three_node_attenuation_example():
    models = [RUN] * 3
    sA = Strategy(1, ((2, 0), (2, 0)))     # N_1 at (0.2,0.7); R(0)={2}, R(1)=whole alphabet
    sB = Strategy(2, ((2, 0), (2, 0)))     # same downstream regions, N_1 at (0.1,0.6)
    sync = evaluate_chain(models, sA)
    assert close(sync, [(0.2, 0.7), (0.28, 0.88), (0.352, 0.952)])
    des = evaluate_assignment(models, sA, sB, ("B", "A", "A"))
    assert close(des, [(0.1, 0.6), (0.19, 0.84), (0.271, 0.936)])
    mags = [np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(sync, des)]
    assert close(mags, [0.1414, 0.0985, 0.0826], tol=6e-5)
    assert close(RUN.attenuation(2, 0), (0.9, 0.4))
    dF = [b[0] - a[0] for a, b in zip(sync, des)]
    dD = [b[1] - a[1] for a, b in zip(sync, des)]
    assert close(dF, [-0.10, -0.09, -0.081]) and close(dD, [-0.10, -0.04, -0.016])


def test_attenuation_factors_of_running_model():
    # R(0)={2}, R(1)={1,2} -> rho_F = rho_D = 0.1 ; R(0)={2}, R(1)=all -> rho_F = 0.9
    assert close(RUN.attenuation(2, 1), (0.1, 0.1))
    assert close(RUN.attenuation(2, 0)[0], 0.9)


# ------------------------------------------------------------------ Sec. IV-E
def test_prop5_running_model():
    items = enumerate_system_points([RUN, RUN])
    hull = upper_hull(distinct_points(items))
    sp = support_pair(hull, 0.075)
    sa, sb = strategies_at(items, sp.A)[0], strategies_at(items, sp.B)[0]
    out = prop5_two_node([RUN, RUN], sa, sb, sp.p)
    assert close(out["dPF"], -0.0025)
    W, _ = desync_point([RUN, RUN], sa, sb, sp.p, [1])
    assert close(W, (0.0725, 0.560))
    assert close(W[0] - 0.075, out["dPF"])


def test_prop5_constraint_violating_model():
    items = enumerate_system_points([VIO, VIO])
    hull = upper_hull(distinct_points(items))
    alpha = 0.21125                                   # p = 0.5 between A and B
    sp = support_pair(hull, alpha)
    assert close(sp.A, (0.0975, 0.19)) and close(sp.B, (0.325, 0.4575)) and close(sp.p, 0.5)
    sa, sb = strategies_at(items, sp.A)[0], strategies_at(items, sp.B)[0]
    assert close(prop5_two_node([VIO, VIO], sa, sb, sp.p)["rho_F"], (0.95, 0.50))
    assert close(prop5_two_node([VIO, VIO], sa, sb, sp.p)["dPF"], 0.05625)
    W, _ = desync_point([VIO, VIO], sa, sb, sp.p, [1])
    assert close(W, (0.2675, 0.371875))


def test_midchain_identity_reduces_to_two_node():
    items = enumerate_system_points([RUN, RUN])
    hull = upper_hull(distinct_points(items))
    sp = support_pair(hull, 0.075)
    sa, sb = strategies_at(items, sp.A)[0], strategies_at(items, sp.B)[0]
    # eq. (14) at k = 2 of a two-node chain is the terminal node's own desynchronization;
    # check it against the exact computation
    W, _ = desync_point([RUN, RUN], sa, sb, sp.p, [2])
    assert close(W[0] - 0.075, prop5_midchain([RUN, RUN], sa, sb, sp.p, 2, 0))
    assert close(W[1] - sp.C[1], prop5_midchain([RUN, RUN], sa, sb, sp.p, 2, 1))


# --------------------------------------------------------------------- Sec. V
def test_correction_running_model():
    items = enumerate_system_points([RUN, RUN])
    hull = upper_hull(distinct_points(items))
    sp = support_pair(hull, 0.075)
    sa, sb = strategies_at(items, sp.A)[0], strategies_at(items, sp.B)[0]
    c = terminal_adjust([RUN, RUN], sa, sb, sp.p, 0.075)
    assert c["feasible"]
    assert close(c["q"], 0.471, tol=5e-4)
    assert close(c["point"], (0.075, 0.5661), tol=1e-4)
    assert close(c["L"], (0.115, 0.665)) and close(c["U"], (0.03, 0.455))
    # here the q-adjustment is optimal at the DFC layer and for the whole downstream chain
    assert close(dfc_layer_benchmark([RUN, RUN], sa, sb, sp.p, 0.075)["PD"], c["point"][1])
    assert close(full_downstream_benchmark([RUN, RUN], sa, sb, sp.p, 0.075)["PD"], c["point"][1])


def test_correction_constraint_violating_model():
    items = enumerate_system_points([VIO, VIO])
    hull = upper_hull(distinct_points(items))
    alpha = 0.21125
    sp = support_pair(hull, alpha)
    sa, sb = strategies_at(items, sp.A)[0], strategies_at(items, sp.B)[0]
    c = terminal_adjust([VIO, VIO], sa, sb, sp.p, alpha)
    assert c["feasible"]
    assert close(c["L"], (0.2, 0.30625)) and close(c["U"], (0.335, 0.4375))
    assert close(c["q"], 1 / 12)
    assert close(c["point"], (0.21125, 0.3171875))
    bench = dfc_layer_benchmark([VIO, VIO], sa, sb, sp.p, alpha)["PD"]
    assert bench >= c["point"][1] and bench - c["point"][1] < 2e-4   # near-optimal, not optimal


# ------------------------------------------------------------------- Prop. 6
def test_prop6_identity_and_bound_random_chains():
    rng = np.random.default_rng(1)
    for trial in range(20):
        K = int(rng.integers(3, 6))
        n = int(rng.integers(3, 7))
        models = [random_model_(K, rng) for _ in range(n)]
        sA, sB = random_pair_(models, rng)
        p = float(rng.uniform(0.1, 0.9))
        S = sorted(rng.choice(np.arange(1, n + 1), size=int(rng.integers(1, min(4, n) + 1)), replace=False).tolist())
        out = prop6_terms(models, sA, sB, p, S)
        assert close(np.sum(out["terms"], axis=0), out["shift"])
        for c in (0, 1):
            assert abs(out["shift"][c]) <= out["bound_tight"][c] + 1e-12
            assert out["bound_tight"][c] <= out["bound_simple"][c] + 1e-12


def random_model_(K, rng):
    from tandem_sim.discrete import random_model
    return random_model(K, rng)


def random_pair_(models, rng):
    from tandem_sim.discrete import realizable_pairs
    def rand_strategy():
        m1 = models[0]
        j = int(rng.choice(m1.levels))
        f, d = m1.region_probs(j)
        pairs = []
        for m in models[1:]:
            cands = realizable_pairs(m, f, d)
            pr = cands[rng.integers(len(cands))]
            pairs.append(pr)
            f, d = m.node_output(pr[0], pr[1], f, d)
        return Strategy(j, tuple(pairs))
    return rand_strategy(), rand_strategy()


# ------------------------------------------------- delayed message (Sec. IV-F)
def test_delayed_link_block_identity_random_chains():
    """A message delayed on the link into node k desynchronizes the block N_1..N_{k-1}; the
    exact mixture agrees with the block form of eq. (13) in both coordinates.  The case
    k = n is the loss of the common draw at the final LD itself."""
    from tandem_sim.desync import delayed_link_point, delayed_link_identity
    rng = np.random.default_rng(5)
    for trial in range(30):
        K = int(rng.integers(3, 6))
        n = int(rng.integers(2, 8))
        models = [random_model_(K, rng) for _ in range(n)]
        sA, sB = random_pair_(models, rng)
        p = float(rng.uniform(0.1, 0.9))
        C = sync_point(models, sA, sB, p)
        for k in range(2, n + 1):
            W = delayed_link_point(models, sA, sB, p, k)
            for c in (0, 1):
                assert abs((W[c] - C[c]) - delayed_link_identity(models, sA, sB, p, k, c)) < 1e-12
        # a delay on the link into N_2 is the desynchronization of N_1 alone
        assert close(delayed_link_point(models, sA, sB, p, 2), desync_point(models, sA, sB, p, [1])[0], 1e-12)
        # a delay on the link into N_n is the loss of the draw at the final LD itself
        assert close(delayed_link_point(models, sA, sB, p, n), desync_point(models, sA, sB, p, [n])[0], 1e-12)


# ------------------------------------------- Prop. 3, exact certificate (Sec. S3)
def test_prop3_exact_rational_certificate():
    """Exact rational enumeration of the two-knot candidates: deterministic envelope 91/160
    at alpha = 3/40 and 29/50 at alpha = 2/25, below the hull chord by 1/160 and 1/140."""
    import os, sys
    from fractions import Fraction as Fr
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    import exp12_certificate as cert
    R, w = cert.envelope(Fr(3, 40))
    assert R == Fr(91, 160) and w == (Fr(1, 5), Fr(7, 160), Fr(1, 5))
    assert cert.chord(Fr(3, 40)) == Fr(23, 40) and cert.chord(Fr(3, 40)) - R == Fr(1, 160)
    R2, _ = cert.envelope(Fr(2, 25))
    assert R2 == Fr(29, 50) and cert.chord(Fr(2, 25)) - R2 == Fr(1, 140)
    assert cert.brute_force(Fr(3, 40), N=80) <= R and cert.brute_force(Fr(2, 25), N=80) <= R2
    # the candidate set is the 3 x 4 x 4 = 48 two-knot combinations before feasibility
    assert len(cert.candidates(Fr(3, 40))) <= 48


# ---------------------------------------------------------------------- PBPO
def test_pbpo_fixed_point_and_restarts():
    """The exhaustive optimum is a fixed point of PBPO (candidate generation is
    consistent), and enough restarts reach it on small random chains."""
    from tandem_sim.design import pbpo
    from tandem_sim.discrete import random_model
    rng = np.random.default_rng(7)
    for trial in range(8):
        K, n = 3, 4
        models = [random_model(K, rng) for _ in range(n)]
        alpha = float(rng.uniform(0.05, 0.3))
        items = enumerate_system_points(models)
        pts = distinct_points(items)
        best = deterministic_optimum(pts, alpha)
        sbest = strategies_at(items, best)[0]
        s0, pt0, sw0, _ = pbpo(models, alpha=alpha, init=sbest)
        assert close(pt0, best) and sw0 == 1


def test_lagrangian_pbpo_reaches_most_hull_vertices():
    """Hull vertices (the supporting points of dependent randomization) are found by the
    Lagrangian PBPO with 20 restarts in about 90 % of the cases on random 4-node chains;
    the misses sit at the extreme ends of the hull (P_F ~ 1e-4 or ~ 1)."""
    from tandem_sim.design import _objective
    from tandem_sim.discrete import random_model
    rng = np.random.default_rng(7)
    reached = total = 0
    for trial in range(4):
        models = [random_model(3, rng) for _ in range(4)]
        pts = distinct_points(enumerate_system_points(models))
        hull = upper_hull(pts)
        for i in range(1, len(hull) - 1):
            (x0, y0), (x1, y1), (x2, y2) = hull[i - 1], hull[i], hull[i + 1]
            lam = 0.5 * ((y1 - y0) / (x1 - x0) + (y2 - y1) / (x2 - x1))
            target = max(y - lam * x for x, y in pts)
            s, pt, stats = pbpo_best(models, lam=lam, restarts=20, seed=trial)
            assert _objective(pt, None, lam) <= target + 1e-9          # never above the true envelope
            reached += abs(_objective(pt, None, lam) - target) <= 1e-9
            total += 1
    assert reached / total >= 0.85, (reached, total)


# ------------------------------------------- realization convention (Sec. III-D)
def _design(models, alpha, **kw):
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    import common
    return common.np_design(models, alpha, **kw)


def test_three_node_examples_under_the_design_convention():
    """Three-node running model.  alpha = 0.078: N_2 is the only upstream node that switches and
    the rule of the final LD switches (Sec. V-C); alpha = 0.1675: N_1 and N_2 switch and the
    final LD keeps its rule (Sec. V-D)."""
    from tandem_sim.desync import downstream_benchmark
    models = [RUN] * 3
    des = _design(models, 0.078, compute_G=True)
    A, B, p = des["sA"], des["sB"], des["p"]
    assert des["switching"] == [2] and A.j1 == B.j1 and tuple(A.pairs[-1]) != tuple(B.pairs[-1])
    assert close(des["A"], (0.038, 0.588)) and close(des["B"], (0.118, 0.768)) and close(p, 0.5)
    assert close(des["C"], (0.078, 0.678)) and close(des["G"], (0.072, 0.637))
    W, _ = desync_point(models, A, B, p, [2])
    assert close(W, (0.10775, 0.6465))
    c = terminal_adjust(models, A, B, p, 0.078, S=(2,))
    assert c["feasible"] and close(c["L"], (0.1945, 0.852)) and close(c["U"], (0.021, 0.441))
    assert close(c["q"], 0.1165 / 0.1735) and close(c["point"], (0.078, 0.852 - 0.411 * 0.1165 / 0.1735))
    dfc = dfc_layer_benchmark(models, A, B, p, 0.078, S=(2,))["PD"]
    assert close(dfc, 0.58239, tol=5e-6) and close(downstream_benchmark(models, A, B, p, 0.078, 2)["PD"], dfc)
    assert dfc < des["G"][1]                      # here every correction at the final LD ends below G

    des = _design(models, 0.1675)
    A, B, p = des["sA"], des["sB"], des["p"]
    assert des["switching"] == [1, 2] and tuple(A.pairs[-1]) == tuple(B.pairs[-1])
    assert close(des["C"], (0.1675, 0.83))
    for k in (1, 2):
        W, _ = desync_point(models, A, B, p, [k])
        assert close(W, (0.16525, 0.824))
        c = terminal_adjust(models, A, B, p, 0.1675, S=(k,))
        assert not c["dfc_switches"] and close(c["L"], c["U"])        # nothing to adjust


def test_design_convention_never_switches_more_nodes_than_the_first_realization():
    from tandem_sim.design import switching_nodes
    from tandem_sim.discrete import random_model
    rng = np.random.default_rng(3)
    for trial in range(25):
        models = [random_model(3, rng) for _ in range(3)]
        alpha = float(rng.uniform(0.03, 0.3))
        new = _design(models, alpha)
        old = _design(models, alpha, realization="first")
        assert close(new["A"], old["A"]) and close(new["B"], old["B"]) and close(new["p"], old["p"])
        if new["deterministic_at_alpha"]:
            continue
        for s, pt in ((new["sA"], new["A"]), (new["sB"], new["B"])):
            assert close(evaluate_chain(models, s)[-1], pt)             # same hull vertices
        assert len(new["switching"]) <= len(switching_nodes(old["sA"], old["sB"]))


# ------------------------------------------------------------------ Prop. 7
def test_prop7_two_node_adjustment_always_restores_the_constraint():
    """Two-node chains, every non-trivial hull edge, design realization: if N_1 switches, both
    N_1 rules are non-trivial, the DFC regions of the strategy with the larger f are contained
    in those of the other, alpha lies between L_F and U_F for every p, and (19)-(20) hold."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    import common
    from tandem_sim.discrete import random_model
    rng = np.random.default_rng(8)
    checked = 0
    for trial in range(150):
        K = int(rng.integers(3, 6))
        models = [random_model(K, rng), random_model(K, rng)]
        arr, codes, tables = common.enumerate_nested_arrays(models)
        hull = common.hull_of_array(arr)
        for a, b in zip(hull[:-1], hull[1:]):
            if a == (0.0, 0.0) or b == (1.0, 1.0) or a[0] == b[0]:
                continue
            sA, sB, _ = common.min_switch_realizations(arr, codes, tables, a, b,
                                                       score=lambda x, y: common.worst_single_shift(models, x, y, 0.5))
            if sA.j1 == sB.j1 or common.worst_single_shift(models, sA, sB, 0.5) < 1e-9:
                continue
            fA, fB = models[0].F[sA.j1], models[0].F[sB.j1]
            assert 0 < fA < 1 and 0 < fB < 1
            hot, cold = (sA, sB) if fA > fB else (sB, sA)
            assert all(h >= c for h, c in zip(hot.pairs[0], cold.pairs[0]))      # level >= : smaller region
            for p in (0.2, 0.5, 0.8):
                alpha = p * a[0] + (1 - p) * b[0]
                c = terminal_adjust(models, sA, sB, p, alpha)
                L, U, W = c["L"], c["U"], c["W"]
                assert min(L[0], U[0]) <= alpha + 1e-12 <= max(L[0], U[0]) + 2e-12
                assert c["feasible"] and abs(c["point"][0] - alpha) < 1e-9
                q = (L[0] - alpha) / (L[0] - U[0])
                assert -1e-9 <= q <= 1 + 1e-9 and abs(q - c["q"]) < 1e-7
                cost = W[1] - (U[1] - L[1]) / (U[0] - L[0]) * (W[0] - alpha)
                assert abs(cost - c["point"][1]) < 1e-9
                if W[0] <= alpha:
                    assert c["point"][1] >= W[1] - 1e-12
            checked += 1
    assert checked >= 20


def test_downstream_benchmark_limits():
    """k = 1: the full-downstream benchmark; k = n - 1: the DFC-layer benchmark."""
    from tandem_sim.desync import downstream_benchmark
    from tandem_sim.discrete import random_model
    rng = np.random.default_rng(12)
    done = 0
    while done < 12:
        models = [random_model(3, rng) for _ in range(3)]
        alpha = float(rng.uniform(0.03, 0.3))
        des = _design(models, alpha)
        if des["deterministic_at_alpha"] or not des["switching"]:
            continue
        A, B, p = des["sA"], des["sB"], des["p"]
        for k in des["switching"]:
            d = downstream_benchmark(models, A, B, p, alpha, k)["PD"]
            if k == 1:
                assert abs(d - full_downstream_benchmark(models, A, B, p, alpha)["PD"]) < 1e-9
            if k == 2:
                assert abs(d - dfc_layer_benchmark(models, A, B, p, alpha, S=(2,))["PD"]) < 1e-9
            q = terminal_adjust(models, A, B, p, alpha, S=(k,))
            if q["feasible"]:
                assert q["point"][1] <= d + 1e-9          # no correction at the final LD beats the benchmark
        done += 1


def test_nested_class_has_the_same_upper_hull_as_the_single_threshold_class():
    """The realization search runs over all strategies with nested level pairs; its upper hull
    is the hull of the single-threshold class of Sec. S1 (the optimal rule of a node for a
    linear objective has the form (3))."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
    import common
    from tandem_sim.discrete import random_model
    rng = np.random.default_rng(21)
    for trial in range(40):
        K = int(rng.integers(3, 5))
        n = int(rng.integers(2, 4))
        models = [random_model(K, rng) for _ in range(n)]
        arr, codes, tables = common.enumerate_nested_arrays(models)
        h_nested = common.hull_of_array(arr)
        h_eta = upper_hull(distinct_points(enumerate_system_points(models)))
        assert len(h_nested) == len(h_eta) and close(h_nested, h_eta, 1e-9)


def test_prop7_exact_arithmetic_with_ties():
    """Proposition 7 in exact rational arithmetic on integer-weight models (likelihood-ratio ties
    inside a node and at the thresholds are frequent): for every pair of realizations of a hull
    edge with distinct non-trivial N_1 rules the DFC regions are nested as stated and alpha lies
    between L_F and U_F; with a trivial N_1 rule the free region can break (ii), which is why the
    proposition excludes it."""
    from tandem_sim.exact import prop7_exact_check
    c = prop7_exact_check(1500, seed=0)
    assert c["pairs"] > 3000
    assert c["regions_nested_as_in_prop7"] == c["pairs"]
    assert c["alpha_between_LF_and_UF"] == c["pairs"]
    assert c["qstar_in_unit_interval_and_exact"] + c["LF_equals_UF"] == c["pairs"]
    assert c["LF_equals_UF_equals_alpha"] == c["LF_equals_UF"]
    assert c["trivial_alpha_between_LF_and_UF"] < c["pairs_with_a_trivial_N1_rule"]


def test_single_node_randomization_of_the_worked_examples():
    """The reference S of Sections III-D and V (one node randomizes with a private coin, no common draw):
    0.56875 = 91/160 (final LD), 0.3199 (N_1), 0.6543 (final LD) and 0.8275 (N_2) in the four examples; in the
    two-node running model at alpha = 3/40 the common draw adds exactly the 1/160 of Proposition 3."""
    from fractions import Fraction
    from tandem_sim.design import single_node_optimum, hull_of_array
    cases = (([RUN, RUN], 0.075, 0.56875, 2, 0.49, 0.575), ([VIO, VIO], 0.21125, 0.31986, 1, 0.19, 0.32375),
             ([RUN] * 3, 0.078, 0.65432, 3, 0.637, 0.678), ([RUN] * 3, 0.1675, 0.8275, 2, 0.796, 0.83))
    for models, alpha, S_D, node, G_D, C_D in cases:
        pts, codes, tables = enumerate_nested_arrays(models)
        s = single_node_optimum(pts, codes, tables, alpha)
        assert abs(s["PD"] - S_D) < 5e-6 and s["node"] == node and abs(s["PF"] - alpha) < 1e-12
        assert close(deterministic_optimum([tuple(p) for p in pts], alpha)[1], G_D)
        assert close(support_pair(hull_of_array(pts), alpha).C[1], C_D)
        assert G_D < s["PD"] < C_D
    pts, codes, tables = enumerate_nested_arrays([RUN, RUN])
    s = single_node_optimum(pts, codes, tables, 0.075)
    assert close(evaluate_chain([RUN, RUN], s["sA"])[-1], (0.04, 0.49)) and close(evaluate_chain([RUN, RUN], s["sB"])[-1], (0.12, 0.67))
    assert s["sA"].j1 == s["sB"].j1                       # N_1 keeps the rule with (P_F, P_D) = (0.2, 0.7)
    assert Fraction(23, 40) - Fraction(91, 160) == Fraction(1, 160) and close(0.575 - s["PD"], 1 / 160)
    # a design whose supporting pair differs at the final LD only is itself of this kind: S = C at alpha = 0.17
    assert close(single_node_optimum(pts, codes, tables, 0.17)["PD"], 0.795)
