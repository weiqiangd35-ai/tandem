"""Experiment 6 - Proposition 2 for parametric local ROCs (Section III-B, Sec. S3).

For two-node chains with Gaussian, Swerling-I (power) and logistic local ROCs we compute
  (a) the number of local maxima of Phi_lambda(f) (Proposition 2) for 400 values of lambda
      log-spaced in [1e-2, 1e2] (600 values in [1e-2, 1e4] when the two nodes differ) on two
      grids of f: 999 uniform values, and 801 values uniform in logit(f) on [-14, 14], which
      resolve the very small f that are optimal at small alpha when N_1 is much more
      informative than N_2.  One maximum for every lambda means that the contact set is a
      single point and the envelope is concave;
  (b) the deterministic envelope R_det(alpha) and its concavity gap hull(R_det) - R_det on
      alpha in [0.01, 0.30] (step 2e-3).  For each alpha the maximization over f starts from
      the union of a uniform (399 values) and a logit-spaced (401 values) grid and is refined
      by four successive 41-point zooms around each of the three best grid maxima; for each f
      the maximization over a_0 (concave) uses 201 points and three 41-point zooms.
kappa and a*(s) use the closed forms of continuous.py (Gaussian: a* = Q(ln s/d' + d'/2);
power ROC: a* = min{1, (s/beta)^(1/(beta-1))}; logistic: a* = (sqrt(e^mu/s) - 1)/(e^mu - 1)).
Usage: python experiments/exp6_prop2_parametric.py
"""
import time
import numpy as np
from scipy.stats import norm
from common import *
from tandem_sim.continuous import GaussianShift, PowerROC, LogisticShift
from tandem_sim.design import upper_hull, hull_value


# ------------------------------------------------------------------ envelope
def inner_max(fs, d, alpha, r2, N_A=201, refine=41, levels=3):
    """max over a_0 of (1-d) r2(a_0) + d r2(a_1) with a_1 = (alpha - (1-f) a_0)/f, for every f
    of the array fs (concave in a_0): N_A points on the feasible interval, then `levels`
    zooms of `refine` points within one step of the best point."""
    lo = np.maximum(0.0, (alpha - fs) / (1 - fs)); hi = np.minimum(1.0, alpha / (1 - fs))
    ok = lo <= hi
    lo = np.where(ok, lo, 0.0); hi = np.where(ok, hi, 0.0)
    best = np.full(len(fs), -1.0)
    n = N_A
    for _ in range(levels + 1):
        u = np.linspace(0.0, 1.0, n)[None, :]
        A0 = lo[:, None] + (hi - lo)[:, None] * u
        A1 = (alpha - (1 - fs)[:, None] * A0) / fs[:, None]
        val = (1 - d)[:, None] * r2(A0) + d[:, None] * r2(np.clip(A1, 0, 1))
        k = val.argmax(axis=1)
        best = np.maximum(best, val[np.arange(len(fs)), k])
        step = (hi - lo) / (n - 1)
        c = A0[np.arange(len(fs)), k]
        lo, hi = np.maximum(lo, c - step), np.minimum(hi, c + step)
        n = refine
    best[~ok] = -1.0
    return best


def envelope2(r1, r2, alphas, N_U=401, N_L=401, zoom=4, ncand=3):
    fs = np.unique(np.concatenate([np.linspace(0.0, 1.0, N_U)[1:-1],
                                   1.0 / (1.0 + np.exp(-np.linspace(-14.0, 14.0, N_L)))]))
    d = r1(fs)
    out = np.zeros(len(alphas))
    for ia, alpha in enumerate(alphas):
        val = inner_max(fs, d, alpha, r2)
        best = max(float(val.max()), float(r2(alpha)))        # r2(alpha): N_2 alone
        loc = [i for i in range(len(fs)) if (i == 0 or val[i] >= val[i - 1]) and (i == len(fs) - 1 or val[i] >= val[i + 1])]
        loc.sort(key=lambda i: -val[i])
        for i in loc[:ncand]:
            lo, hi = fs[max(i - 1, 0)], fs[min(i + 1, len(fs) - 1)]
            for _ in range(zoom):
                g = np.linspace(lo, hi, 41)
                v = inner_max(g, r1(g), alpha, r2)
                j = int(v.argmax()); best = max(best, float(v[j]))
                lo, hi = g[max(j - 1, 0)], g[min(j + 1, 40)]
        out[ia] = best
    return out


def gap(m1, m2, alphas):
    R = envelope2(m1.r, m2.r, alphas)
    hull = upper_hull([(float(a), float(v)) for a, v in zip(alphas, R)])
    H = np.array([hull_value(hull, float(a)) for a in alphas])
    return float((H - R).max()), float(alphas[(H - R).argmax()])


# --------------------------------------------------------------- Phi_lambda
def kappa_vec(m, s):
    """kappa(s) = max_a [r(a) - s a] for an array of s > 0 (same closed forms as m.astar)."""
    s = np.asarray(s, float)
    if isinstance(m, GaussianShift):
        a = norm.sf(np.log(s) / m.dprime + m.dprime / 2)
    elif isinstance(m, PowerROC):
        a = np.where(s <= m.beta, 1.0, (np.maximum(s, m.beta) / m.beta) ** (1.0 / (m.beta - 1)))
    elif isinstance(m, LogisticShift):
        e = np.exp(m.mu)
        a = np.clip((np.sqrt(e / s) - 1.0) / (e - 1.0), 0.0, 1.0)
    else:
        raise TypeError(m)
    return m.r(a) - s * a


def peaks(m1, m2, lambdas, fs):
    """Maximum over lambda of the number of local maxima of Phi_lambda(f) on the grid fs."""
    d = m1.r(fs)
    keep = (d > 0) & (d < 1)            # at the extreme logit values d can round to 0 or 1
    fs, d = fs[keep], d[keep]
    worst = 1
    for lam in lambdas:
        vals = (1 - d) * kappa_vec(m2, lam * (1 - fs) / (1 - d)) + d * kappa_vec(m2, lam * fs / d)
        n = int(np.sum((vals[1:-1] > vals[:-2] + 1e-12) & (vals[1:-1] >= vals[2:] + 1e-12)))
        worst = max(worst, n)
    return worst


alphas = np.arange(0.01, 0.30 + 1e-9, 2e-3)
lam_equal = np.exp(np.linspace(np.log(1e-2), np.log(1e2), 400))
lam_unequal = np.exp(np.linspace(np.log(1e-2), np.log(1e4), 600))
f_uniform = np.linspace(0.001, 0.999, 999)
f_logit = 1.0 / (1.0 + np.exp(-np.linspace(-14.0, 14.0, 801)))
cases = []
for d1, d2 in [(0.5, 0.5), (1, 1), (1.5, 1.5), (2, 2), (3, 3), (4, 4), (1, 2), (2, 1), (0.5, 3), (3, 0.5)]:
    cases.append((f"Gaussian d'=({d1},{d2})", GaussianShift(d1), GaussianShift(d2), d1 == d2))
for s1, s2 in [(1, 1), (3, 3), (10, 10), (1, 10), (10, 1)]:
    cases.append((f"Swerling SNR=({s1},{s2})", PowerROC(snr=s1), PowerROC(snr=s2), s1 == s2))
for mu1, mu2 in [(1, 1), (2, 2), (3, 3), (1, 3)]:
    cases.append((f"logistic mu=({mu1},{mu2})", LogisticShift(mu1), LogisticShift(mu2), mu1 == mu2))

# kappa_vec against the scalar closed forms of the package
for _, m1, m2, _ in cases:
    ss = np.exp(np.linspace(-6, 6, 25))
    assert np.allclose(kappa_vec(m2, ss), [m2.kappa(float(x)) for x in ss], atol=1e-12)

out = []
t0 = time.time()
print("model                      max concavity gap   at alpha   max #peaks of Phi_lambda (uniform f / logit f)")
for name, m1, m2, equal in cases:
    g, a = gap(m1, m2, alphas)
    lams = lam_equal if equal else lam_unequal
    pk_u, pk_l = peaks(m1, m2, lams, f_uniform), peaks(m1, m2, lams, f_logit)
    out.append({"model": name, "max_gap": g, "alpha_at_max": a, "max_peaks": max(pk_u, pk_l),
                "max_peaks_uniform_f": pk_u, "max_peaks_logit_f": pk_l, "n_lambda": len(lams)})
    print(f"{name:26s} {g:.2e}            {a:.3f}      {pk_u} / {pk_l}     ({time.time()-t0:.0f}s)", flush=True)
print("largest gap over all cases:", max(o["max_gap"] for o in out))
save_json("exp6_prop2_parametric.json", out)
