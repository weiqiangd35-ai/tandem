"""Unit checks of the parallel and continuous modules and of the correction mechanism."""
import numpy as np

from tandem_sim import *
from tandem_sim.continuous import (GaussianShift, PowerROC, LogisticShift, ContStrategy, strategy_from_etas,
                                   evaluate_chain_cont, chain_factors_cont, pbpo_cont_exact)
from tandem_sim.design import distinct_points, strategies_at
from tandem_sim.parallel import enumerate_parallel, k_of_n_fusion, system_point, parallel_design, parallel_desync

RUN = DiscreteModel([0.8, 0.1, 0.1], [0.3, 0.1, 0.6])


def test_parallel_k_of_n_by_hand():
    pf = 1 - (0.9 ** 4 + 4 * 0.1 * 0.9 ** 3)
    pd = 1 - (0.4 ** 4 + 4 * 0.6 * 0.4 ** 3)
    assert np.allclose(system_point([RUN] * 4, [2] * 4, k_of_n_fusion(4, 2)), (pf, pd))


def test_parallel_cv_contains_k_of_n_for_identical_lds():
    cv = {(round(x, 10), round(y, 10)) for (x, y), _ in enumerate_parallel([RUN] * 4, "cv")}
    kn = {(round(x, 10), round(y, 10)) for (x, y), _ in enumerate_parallel([RUN] * 4, "kofn")}
    assert kn <= cv


def test_paper_parallel_wrong_rule_shift_0_0428():
    """Sec. IV-D of the submitted manuscript: 'in the parallel system with a 2-out-of-4
    fusion rule, the system operating point shifts by 0.0428' - reproduced as the shift
    when all four LDs use (0.2, 0.7) and one LD switches to (0.1, 0.6)."""
    fus = k_of_n_fusion(4, 2)
    A = system_point([RUN] * 4, [1] * 4, fus)
    Ap = system_point([RUN] * 4, [2, 1, 1, 1], fus)
    assert abs(np.hypot(Ap[0] - A[0], Ap[1] - A[1]) - 0.0428) < 5e-4


def test_parallel_desync_consistency():
    des = parallel_design([RUN] * 4, 0.25, "kofn")
    sp = des["support"]
    out = parallel_desync([RUN] * 4, des["sA"], des["sB"], sp.p, 0)
    assert np.allclose(out["C"], sp.C)
    assert out["norm"] >= 0


def test_continuous_inverse_slopes():
    for m in (GaussianShift(1.0), GaussianShift(2.5), PowerROC(snr=3.0), PowerROC(rates=(2.0, 0.5)), LogisticShift(2.0)):
        for a in (0.01, 0.1, 0.5, 0.9):
            assert abs(m.astar(m.rprime(a)) - a) < 1e-8
        assert abs(m.r(0.0)) < 1e-12 and abs(m.r(1.0) - 1) < 1e-12


def test_continuous_transfer_is_exact_product():
    g = GaussianShift(1.5)
    models = [g] * 6
    rng = np.random.default_rng(0)
    for _ in range(10):
        A = strategy_from_etas(models, float(rng.uniform(0.01, 0.5)), np.exp(rng.uniform(-3, 3, 5)))
        B = type(A)(float(rng.uniform(0.01, 0.5)), A.pairs)
        pA, pB = evaluate_chain_cont(models, A), evaluate_chain_cont(models, B)
        d = np.array([(b[0] - a[0], b[1] - a[1]) for a, b in zip(pA, pB)])
        fac = chain_factors_cont(models, A)
        pred = d[0] * np.cumprod([[1, 1]] + [[f[0], f[1]] for f in fac], axis=0)
        assert np.abs(d - pred).max() < 1e-12
        assert all(0 <= f[0] <= 1 and 0 <= f[1] <= 1 for f in fac)


def test_exact_pbpo_of_a_continuous_chain():
    """Five identical nodes, lambda = 2: the fixed point of the closed-form best responses.  The reference
    factors rho_F of nodes 2..5 come from a separate implementation written from the formulas of Sec. S3."""
    lam = 2.0
    ref = {"gauss": [0.4947, 0.6006, 0.6488, 0.6713], "swerling": [0.9909, 0.9909, 0.9909, 0.9909], "logistic": [0.4434, 0.5728, 0.6006, 0.5213]}
    for key, m in (("gauss", GaussianShift(1.0)), ("swerling", PowerROC(snr=3.0)), ("logistic", LogisticShift(2.0))):
        models = [m] * 5
        s, pt, sweeps, converged = pbpo_cont_exact(models, lam, ContStrategy(0.3, ((0.2, 0.6),) * 4))
        assert converged
        assert np.abs(np.array([f[0] for f in chain_factors_cont(models, s)]) - ref[key]).max() < 1e-4
        # a fixed point: one more pass of best responses moves nothing
        s2 = pbpo_cont_exact(models, lam, s, max_sweeps=1)[0]
        assert abs(s2.a1 - s.a1) < 1e-10 and np.abs(np.array(s2.pairs) - np.array(s.pairs)).max() < 1e-10
        # and no single false-alarm rate can be moved to increase P_D - lambda P_F
        par = [s.a1] + [x for pr in s.pairs for x in pr]
        for i in range(len(par)):
            for h in (-1e-3, 1e-3):
                q = list(par); q[i] = min(1.0, max(0.0, q[i] + h))
                p2 = evaluate_chain_cont(models, ContStrategy(q[0], tuple((q[1 + 2 * j], q[2 + 2 * j]) for j in range(4))))[-1]
                assert p2[1] - lam * p2[0] <= pt[1] - lam * pt[0] + 1e-12


def test_terminal_adjust_recovers_W_at_q_equal_p_three_nodes():
    """With the coupled mechanism the uncorrected point is P(p) = W for n = 3."""
    models = [RUN] * 3
    items = enumerate_system_points(models)
    hull = upper_hull(distinct_points(items))
    sp = support_pair(hull, 0.078)
    A, B = strategies_at(items, sp.A)[0], strategies_at(items, sp.B)[0]
    out = terminal_adjust(models, A, B, sp.p, 0.078)
    W, _ = desync_point(models, A, B, sp.p, [1])
    assert np.allclose(out["W"], W)
    assert out["feasible"] and out["point"][0] <= 0.078 + 1e-12
    assert dfc_layer_benchmark(models, A, B, sp.p, 0.078)["PD"] >= out["point"][1] - 1e-12
    assert full_downstream_benchmark(models, A, B, sp.p, 0.078)["PD"] >= dfc_layer_benchmark(models, A, B, sp.p, 0.078)["PD"] - 1e-12


def test_logistic_closed_form_matches_root_finding():
    """Logistic shift model: r(a) = e^mu a / (1 + (e^mu - 1) a) and
    a*(s) = (sqrt(e^mu / s) - 1) / (e^mu - 1), clipped to [0, 1]."""
    from scipy.optimize import brentq
    from tandem_sim.continuous import LogisticShift
    for mu in (0.5, 1.0, 2.0, 3.0):
        m = LogisticShift(mu)
        e = np.exp(mu)
        for a in np.linspace(1e-4, 1 - 1e-4, 201):
            assert abs(float(m.r(a)) - e * a / (1 + (e - 1) * a)) < 1e-12
        for s in np.exp(np.linspace(-mu - 1, mu + 1, 101)):
            g = lambda a: np.log(m.rprime(a)) - np.log(s)
            ref = 0.0 if g(1e-12) <= 0 else 1.0 if g(1 - 1e-12) >= 0 else brentq(g, 1e-12, 1 - 1e-12, xtol=1e-14)
            assert abs(ref - m.astar(s)) < 1e-9


def test_quantized_gaussian_cuts():
    """Three-level quantization of N(0,1) versus N(d',1) used by exp2 --family=qgauss3:LO:HI:
    cell probabilities from the normal distribution function, levels ordered by the likelihood ratio."""
    from math import erf, sqrt
    from tandem_sim.discrete import quantized_gaussian_cuts
    Phi = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    m = quantized_gaussian_cuts(1.0, [1.2, 0.3])                      # the cut points are sorted
    assert np.allclose(m.p0, [Phi(0.3), Phi(1.2) - Phi(0.3), 1 - Phi(1.2)], atol=1e-12)
    assert np.allclose(m.p1, [Phi(-0.7), Phi(0.2) - Phi(-0.7), 1 - Phi(0.2)], atol=1e-12)
    lr = np.asarray(m.p1) / np.asarray(m.p0)
    assert lr[0] < lr[1] < lr[2]
    rng = np.random.default_rng(5)
    for _ in range(200):
        d = rng.uniform(0.3, 3.0)
        q = quantized_gaussian_cuts(d, rng.uniform(-1.0, d + 1.0, size=2))
        assert abs(sum(q.p0) - 1) < 1e-12 and abs(sum(q.p1) - 1) < 1e-12 and min(q.p0) > 0 and min(q.p1) > 0
        r = np.asarray(q.p1) / np.asarray(q.p0)
        assert r[0] < r[1] < r[2]
