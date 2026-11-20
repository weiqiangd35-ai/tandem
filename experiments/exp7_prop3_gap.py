"""Experiment 7 - the smooth counterexample of Proposition 3: concavity gap versus the
smoothing parameter epsilon, with all grid parameters stated (reviewer 2, AE).

Model.  Both local ROCs equal r_eps, the endpoint-normalized softmin (log-sum-exp)
smoothing of the piecewise-linear concave hull of the running model's local points
(0,0), (0.1,0.6), (0.2,0.7), (1,1):
    h_eps(a) = -eps ln sum_i exp(-l_i(a)/eps),  l_1 = 6a, l_2 = a + 1/2, l_3 = 3a/8 + 5/8,
    r_eps(a) = (h_eps(a) - h_eps(0)) / (h_eps(1) - h_eps(0)).
Deterministic envelope (two-node chain, Lemma 1 reduction):
    R_det(alpha) = max_{f, a0, a1 : (1-f) a0 + f a1 = alpha} (1-d) r(a0) + d r(a1),  d = r(f).
Numerics.  f on a uniform grid of N_F points in [0, 1]; for each f the feasible a0 is an
interval and the objective is concave in a0, maximized on a 401-point grid refined by
41 points within one grid step (accuracy about 1e-7); alpha on a uniform grid of step 5e-4 on
[0.01, 0.30]; the gap is hull(R_det) - R_det on that grid (hull = upper concave envelope).
Also reported: the Lagrangian check of Proposition 2 (two tied maxima of Phi_lambda at
lambda* for eps = 0.004) and the perturbation of the two interior knots (420 draws).

Knot perturbations.  For every draw the piecewise-linear model (eps = 0) is treated exactly:
by Corollary S1, whose two lemmas use only that the local ROC is concave and piecewise
linear, the deterministic envelope is attained at a strategy with two of (f, a0, a1) at
knots, and its concave hull has its vertices among the 64 strategies with all three at
knots.  The envelope is concave exactly when every edge of that hull joins two strategies
that differ in one coordinate by one knot, because one threshold moving along one linear
segment then traces the edge.  The smoothed model is computed at eps = 0.004 for every
draw, and at eps = 0.002 and 0.001 for the draws whose piecewise-linear envelope is
concave.  Its f grid is the uniform one plus 400 values within 12 transition widths
eps / (difference of slopes) of each knot: a perturbed knot is not on the uniform grid, and
at eps = 0 the maximizing f is a knot unless a0 and a1 both are (Corollary S1).
Its alpha grid covers [0, 1]: step 1e-3 up to 0.2, 5e-3 up to 0.4 and 0.05 beyond, and the
gap is taken against the concave hull of all these points.  (The hull over a window of
alpha lies on or below the hull over [0, 1]: a window that does not contain the whole
non-concave region loses a tangent point of the hull and returns a gap that is too small.)
The same grid gives, for the nominal smoothed model, the gaps that the smoothing creates
on the edges of the hull other than the one of the certificate (alpha outside [0.04, 0.11]).
Usage: python experiments/exp7_prop3_gap.py [--procs N]
       (about 2 hours on one core, nearly all of it for the knots; N > 1 distributes the
       draws over N forked processes and gives the same numbers)
       python experiments/exp7_prop3_gap.py --checks
       (about 10 minutes, after the full run: sensitivity of the largest smoothed gaps of the
       draws with a concave piecewise-linear envelope to the grids of alpha and f;
       results/exp7_checks.json)
"""
import sys
from itertools import product
import numpy as np
from scipy.optimize import minimize_scalar
from common import *
from tandem_sim.design import upper_hull, hull_value

LINES = [(6.0, 0.0), (1.0, 0.5), (3 / 8, 5 / 8)]


def make_r(eps, lines=LINES):
    def h(a):
        a = np.asarray(a, float)
        vals = np.stack([-(m * a + b) / eps for m, b in lines])
        mx = vals.max(axis=0)
        return -eps * (mx + np.log(np.exp(vals - mx).sum(axis=0)))
    h0, h1 = float(h(0.0)), float(h(1.0))
    return lambda a: (h(a) - h0) / (h1 - h0)


def r_pl(a, lines=LINES):
    a = np.asarray(a, float)
    return np.min([m * a + b for m, b in lines], axis=0)


def envelope(r, alphas, N_F=801, N_A=401, refine=41, fs=None):
    """Deterministic envelope R_det(alpha) of the two-node chain with local ROC r at both
    nodes: vectorized grid over (f, a0) followed by a local refinement around the best a0
    of every f (the objective is concave in a0 for fixed f).  fs: values of f in (0, 1), by
    default the uniform grid of N_F points."""
    if fs is None:
        fs = np.linspace(0.0, 1.0, N_F)[1:-1]        # f = 0, 1 handled separately
    d = r(fs)
    out = np.zeros(len(alphas))
    for ia, alpha in enumerate(alphas):
        lo = np.maximum(0.0, (alpha - fs) / (1 - fs)); hi = np.minimum(1.0, alpha / (1 - fs))
        ok = lo <= hi
        u = np.linspace(0.0, 1.0, N_A)[None, :]
        A0 = lo[:, None] + (hi - lo)[:, None] * u                     # (N_F, N_A)
        A1 = (alpha - (1 - fs)[:, None] * A0) / fs[:, None]
        val = (1 - d)[:, None] * r(A0) + d[:, None] * r(np.clip(A1, 0, 1))
        val[~ok, :] = -1
        k = val.argmax(axis=1)
        step = (hi - lo) / (N_A - 1)
        a_lo = np.maximum(lo, A0[np.arange(len(fs)), k] - step); a_hi = np.minimum(hi, A0[np.arange(len(fs)), k] + step)
        uu = np.linspace(0.0, 1.0, refine)[None, :]
        A0r = a_lo[:, None] + (a_hi - a_lo)[:, None] * uu
        A1r = (alpha - (1 - fs)[:, None] * A0r) / fs[:, None]
        valr = (1 - d)[:, None] * r(A0r) + d[:, None] * r(np.clip(A1r, 0, 1))
        valr[~ok, :] = -1
        out[ia] = max(valr.max(), float(r(alpha)))                     # f in {0,1}: single node
    return out


def gap_curve(r, alphas):
    R = envelope(r, alphas)
    hull = upper_hull([(float(a), float(v)) for a, v in zip(alphas, R)])
    H = np.array([hull_value(hull, float(a)) for a in alphas])
    return R, H, H - R


def knot_draw(rng):
    """One perturbation of the two interior knots (0.1, 0.6), (0.2, 0.7) by independent uniform
    factors in [0.95, 1.05]: the knot points and the lines (slope, intercept) of the segments."""
    p1 = (0.1 * rng.uniform(0.95, 1.05), 0.6 * rng.uniform(0.95, 1.05))
    p2 = (0.2 * rng.uniform(0.95, 1.05), 0.7 * rng.uniform(0.95, 1.05))
    pts = [(0.0, 0.0), p1, p2, (1.0, 1.0)]
    return pts, [((y2 - y1) / (x2 - x1), y1 - (y2 - y1) / (x2 - x1) * x1) for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:])]


def pl_envelope(alphas, K, r):
    """Exact deterministic envelope of the piecewise-linear model with knot set K (Corollary S1):
    the best strategy with two of (f, a0, a1) in K and the third solved from P_F = alpha."""
    al = np.asarray(alphas, float); one = np.ones_like(al)
    best = np.full(len(al), -np.inf)

    def update(f, a0, a1, ok):
        d = r(f)
        np.maximum(best, np.where(ok, (1 - d) * r(a0) + d * r(a1), -np.inf), out=best)

    for f, a0 in product(K, K):                     # a1 solved
        if f > 0:
            a1 = (al - (1 - f) * a0) / f
            update(f * one, a0 * one, np.clip(a1, 0, 1), (a1 >= -1e-15) & (a1 <= 1 + 1e-15))
    for f, a1 in product(K, K):                     # a0 solved
        if f < 1:
            a0 = (al - f * a1) / (1 - f)
            update(f * one, np.clip(a0, 0, 1), a1 * one, (a0 >= -1e-15) & (a0 <= 1 + 1e-15))
    for a0, a1 in product(K, K):                    # f solved
        if a1 != a0:
            f = (al - a0) / (a1 - a0)
            update(np.clip(f, 0, 1), a0 * one, a1 * one, (f >= -1e-15) & (f <= 1 + 1e-15))
    return best


def pl_gap(pts, lines, per_edge=400):
    """Piecewise-linear model with the given knots: largest gap between the concave hull and the
    deterministic envelope over [0, 1] (per_edge values of alpha on every hull edge), the alpha
    where it occurs, and whether every hull edge joins two strategies that differ in one
    coordinate by one knot."""
    K = [q[0] for q in pts]
    r = lambda a: r_pl(a, lines)
    S = list(product(range(4), repeat=3))           # (f, a0, a1) at knots
    P = []
    for s_ in S:
        f, a0, a1 = (K[i] for i in s_)
        d = float(r(f))
        P.append(((1 - f) * a0 + f * a1, (1 - d) * float(r(a0)) + d * float(r(a1))))
    hull = upper_hull(P)
    at = [[s_ for s_, q in zip(S, P) if abs(q[0] - v[0]) < 1e-12 and abs(q[1] - v[1]) < 1e-12] for v in hull]
    one_step = all(any(sum(abs(x - y) for x, y in zip(s_, t_)) == 1 for s_ in A for t_ in B) for A, B in zip(at[:-1], at[1:]))
    xs = np.unique(np.concatenate([np.linspace(u[0], v[0], per_edge) for u, v in zip(hull[:-1], hull[1:])]))
    G = np.array([hull_value(hull, float(x)) for x in xs]) - pl_envelope(xs, K, r)
    return float(G.max()), float(xs[G.argmax()]), bool(one_step)


def knot_f_grid(lines, eps, N_F=801, cluster=400, widths=12):
    """Uniform grid of f plus `cluster` values within `widths` transition widths of each knot."""
    fs = [np.linspace(0.0, 1.0, N_F)[1:-1]]
    for (m1, b1), (m2, b2) in zip(lines[:-1], lines[1:]):
        x, w = (b2 - b1) / (m1 - m2), eps / (m1 - m2)
        fs.append(np.linspace(max(1e-6, x - widths * w), min(1 - 1e-6, x + widths * w), cluster))
    return np.unique(np.concatenate(fs))


# alpha grid of the smoothed perturbed models: the whole interval [0, 1], fine where the hull has its short edges
AL_KNOT = np.round(np.concatenate([np.arange(0, 201) * 1e-3, 0.2 + np.arange(1, 41) * 5e-3, 0.4 + np.arange(1, 13) * 0.05]), 12)


def smooth_curve(lines, eps, al=AL_KNOT):
    """Gap of the smoothed model on the grid al (concave hull of the computed points minus the envelope)."""
    R = envelope(make_r(eps, lines), al, fs=knot_f_grid(lines, eps))
    hull = upper_hull([(float(a), float(v)) for a, v in zip(al, R)])
    return np.array([hull_value(hull, float(a)) - v for a, v in zip(al, R)])


def smooth_gap(lines, eps):
    """Largest gap of the smoothed model over [0, 1] and the alpha where it occurs."""
    G = smooth_curve(lines, eps)
    return float(G.max()), float(AL_KNOT[G.argmax()])


if "--checks" in sys.argv:
    # Grid sensitivity of the nine largest ratios gap / eps among the draws whose piecewise-linear envelope is concave
    # (stored results): (a) alpha refined to steps of 1e-4 within 0.012 of the maximum; for the three largest, (b) a
    # uniform alpha grid of step 1e-3 over [0, 1] and (c) 1601 and 3201 uniform values of f instead of 801.
    kp = load_json("exp7_prop3_gap.json")["knot_perturbation"]
    rng = np.random.default_rng(7)
    model = {t: knot_draw(rng)[1] for t in range(kp["draws"])}
    pos = {t: i for i, t in enumerate(kp["draw"])}
    cases = sorted(((kp[key][pos[t]] / eps, t, eps, key) for t, g in zip(kp["draw"], kp["gap_pl"]) if g <= 1e-12
                    for eps, key in ((1e-3, "gap_eps001"), (2e-3, "gap_eps002"), (4e-3, "gap"))), reverse=True)[:9]

    def max_gap(lines, eps, al, N_F=801):
        R = envelope(make_r(eps, lines), al, fs=knot_f_grid(lines, eps, N_F=N_F))
        hull = upper_hull([(float(a), float(v)) for a, v in zip(al, R)])
        return float(max(hull_value(hull, float(a)) - v for a, v in zip(al, R)))

    chk = []
    for rank, (ratio, t, eps, key) in enumerate(cases):
        a0 = kp["alpha_" + key][pos[t]]
        al = np.unique(np.round(np.concatenate([AL_KNOT, a0 + np.arange(-120, 121) * 1e-4]), 12)); al = al[(al >= 0) & (al <= 1)]
        row = {"draw": t, "eps": eps, "gap_over_eps": ratio, "alpha_refined": max_gap(model[t], eps, al) / eps}
        if rank < 3:
            row["alpha_uniform_1e-3"] = max_gap(model[t], eps, np.arange(0, 1001) * 1e-3) / eps
            row["f_1601"] = max_gap(model[t], eps, AL_KNOT, 1601) / eps; row["f_3201"] = max_gap(model[t], eps, AL_KNOT, 3201) / eps
        chk.append(row)
        print(f"draw {t}, eps = {eps}: gap / eps = {ratio:.4f}; alpha refined around the maximum: {row['alpha_refined']:.4f}"
              + (f"; uniform alpha grid of step 1e-3 on [0, 1]: {row['alpha_uniform_1e-3']:.4f}; 1601 and 3201 values of f: {row['f_1601']:.4f}, {row['f_3201']:.4f}" if rank < 3 else ""), flush=True)
    rel = lambda key: max(abs(r[key] / r["gap_over_eps"] - 1) for r in chk if key in r)
    summary = {"largest_ratio": chk[0]["gap_over_eps"], "largest_ratio_alpha_refined": max(r["alpha_refined"] for r in chk),
               "largest_relative_change": {key: rel(key) for key in ("alpha_refined", "alpha_uniform_1e-3", "f_1601", "f_3201")}}
    print("largest relative changes:", ", ".join(f"{key} {100 * v:.2f} %" for key, v in summary["largest_relative_change"].items()),
          f"; largest ratio after refinement of alpha: {summary['largest_ratio_alpha_refined']:.4f}")
    save_json("exp7_checks.json", {"cases": chk, **summary})
    sys.exit(0)

alphas = np.arange(0.01, 0.30 + 1e-9, 5e-4)
out = {"alpha_grid": "uniform step 5e-4 on [0.01, 0.30]", "f_grid": 801, "a0_search": "401-point grid on the feasible interval, refined by 41 points within one grid step"}

# piecewise-linear limit: certificate values 91/160 at 3/40 and 29/50 at 2/25
R0, H0, G0 = gap_curve(r_pl, alphas)
i = int(np.argmin(abs(alphas - 0.075))); j = int(np.argmin(abs(alphas - 0.08)))
print(f"eps = 0 (PL): R_det(0.075) = {R0[i]:.6f} (certificate 91/160 = {91/160:.6f}), gap {G0[i]:.6f} (>= 1/160 = {1/160:.6f});"
      f" R_det(0.08) = {R0[j]:.6f} (29/50 = 0.58), gap {G0[j]:.6f} (>= 1/140 = {1/140:.6f}); max gap {G0.max():.5f} at alpha = {alphas[G0.argmax()]:.4f}")
out["pl"] = {"R_0.075": R0[i], "gap_0.075": G0[i], "R_0.08": R0[j], "gap_0.08": G0[j], "max_gap": G0.max(), "argmax": alphas[G0.argmax()]}

rows = []
for eps in (1e-4, 5e-4, 1e-3, 2e-3, 4e-3, 7e-3, 1e-2, 1.5e-2, 2e-2, 3e-2, 4e-2, 5e-2):
    r = make_r(eps)
    R, H, G = gap_curve(r, alphas)
    k = int(G.argmax())
    rows.append({"eps": eps, "max_gap": float(G.max()), "alpha_at_max": float(alphas[k]),
                 "gap_0.075": float(G[i]), "R_0.075": float(R[i]), "hull_0.075": float(H[i])})
    print(f"eps = {eps:<7g}: max gap {G.max():.5f} at alpha = {alphas[k]:.4f};  gap(0.075) = {G[i]:.5f}  "
          f"(paper: 0.0066 at eps = 0.004 near alpha = 0.076; 0.0055 at 0.01; 0.0007 at 0.03)")
out["gap_vs_eps"] = rows

# Lagrangian check (Prop. 2) at eps = 0.004: Phi_lambda(f) and its peaks at lambda* ~ 2.5165
r = make_r(4e-3)
def kappa(s):
    res = minimize_scalar(lambda a: -(float(r(a)) - s * a), bounds=(0, 1), method="bounded", options={"xatol": 1e-10})
    return -res.fun, res.x
def Phi(lam, f):
    d = float(r(f))
    if f <= 0 or f >= 1 or d <= 0 or d >= 1:
        return kappa(lam)[0]
    return (1 - d) * kappa(lam * (1 - f) / (1 - d))[0] + d * kappa(lam * f / d)[0]
fs = np.linspace(0.001, 0.999, 999)
lam = 2.5165
vals = np.array([Phi(lam, f) for f in fs])
peaks = [k for k in range(1, len(fs) - 1) if vals[k] > vals[k - 1] and vals[k] >= vals[k + 1]]
print(f"Phi_lambda at lambda* = {lam}: local maxima at f = {[round(fs[k], 3) for k in peaks]}, values {[round(vals[k], 5) for k in peaks]}"
      f" (paper: tied maxima at f ~ 0.102 and 0.197, dip 0.0048)")
out["lagrangian_check"] = {"lambda": lam, "peaks_f": [float(fs[k]) for k in peaks], "peak_values": [float(vals[k]) for k in peaks],
                           "dip": float(vals[peaks[0]:peaks[-1] + 1].min()) if len(peaks) >= 2 else None}

# perturbation of the two interior knots (0.1, 0.6), (0.2, 0.7) by +-5 % (uniform); see the docstring
NOMINAL = [(0.0, 0.0), (0.1, 0.6), (0.2, 0.7), (1.0, 1.0)]
g0, a0_, s0 = pl_gap(NOMINAL, LINES)
assert abs(g0 - 1 / 140) < 1e-12 and abs(a0_ - 0.08) < 1e-9 and not s0      # the certificate: gap 1/140 at alpha = 2/25
N_KNOT = 420
rng = np.random.default_rng(7)
jobs = []
for t in range(N_KNOT):
    pts, lines = knot_draw(rng)
    if lines[0][0] > lines[1][0] > lines[2][0]:            # keep the concave-hull structure
        jobs.append((t, pts, lines))


def knot_job(job):
    """One draw: exact gap at eps = 0, smoothed gap at eps = 0.004 and, if the piecewise-linear envelope is concave, at 0.002 and 0.001."""
    t, pts, lines = job
    gpl, _, one = pl_gap(pts, lines)
    g4 = smooth_gap(lines, 4e-3)
    g2, g1 = (smooth_gap(lines, 2e-3), smooth_gap(lines, 1e-3)) if gpl <= 1e-12 else ((None, None), (None, None))
    return {"draw": t, "middle_slope": lines[1][0], "gap_pl": gpl, "one_step": one, "gap": g4[0], "gap_eps002": g2[0], "gap_eps001": g1[0],
            "alpha_gap": g4[1], "alpha_gap_eps002": g2[1], "alpha_gap_eps001": g1[1]}


PROCS = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 1
kn = {k: [] for k in ("draw", "middle_slope", "gap_pl", "one_step", "gap", "gap_eps002", "gap_eps001", "alpha_gap", "alpha_gap_eps002", "alpha_gap_eps001")}
if PROCS > 1:
    import multiprocessing
    pool = multiprocessing.get_context("fork").Pool(PROCS)
    results = pool.imap(knot_job, jobs, chunksize=1)        # in the order of the draws
else:
    results = map(knot_job, jobs)
for i, res in enumerate(results):
    for k in kn:
        kn[k].append(res[k])
    if (i + 1) % 20 == 0:
        print(f"  knot draw {i + 1}/{len(jobs)}", flush=True)
if PROCS > 1:
    pool.close(); pool.join()
gpl = np.array(kn["gap_pl"]); gaps = np.array(kn["gap"]); m2 = np.array(kn["middle_slope"]); z = gpl <= 1e-12
assert np.array_equal(z, np.array(kn["one_step"]))          # concave exactly when every hull edge is traced by one threshold
g2 = np.array([x for x in kn["gap_eps002"] if x is not None]); g1 = np.array([x for x in kn["gap_eps001"] if x is not None]); g4 = gaps[z]
print(f"knot perturbation +-5 %: {len(gpl)} models with a concave local ROC")
print(f"  eps = 0 (exact): gap >= 1e-3 in {np.mean(gpl >= 1e-3):.3f} ({int(np.sum(gpl >= 1e-3))}), zero in {np.mean(z):.3f} ({int(z.sum())}), "
      f"between in {int(np.sum(~z & (gpl < 1e-3)))} (smallest positive {gpl[~z].min():.2e}); max {gpl.max():.5f}")
print(f"  middle slope: concave envelope for every slope below {m2[~z].min():.3f} ({int(np.sum(m2 < m2[~z].min()))} draws) and for none above "
      f"{m2[z].max():.3f} ({int(np.sum(m2 > m2[z].max()))} draws); range of the slopes {m2.min():.2f} to {m2.max():.2f}")
print(f"  eps = 0.004: gap >= 1e-3 in {np.mean(gaps >= 1e-3):.3f} ({int(np.sum(gaps >= 1e-3))}), max {gaps.max():.5f}, min {gaps.min():.2e}; "
      f"largest |gap - gap at eps = 0| over the draws with a positive gap there: {np.abs(gaps - gpl)[~z].max():.2e}")
print(f"  draws with a concave piecewise-linear envelope: gap {g4.min():.2e} to {g4.max():.2e} at eps = 0.004 (median {np.median(g4):.2e}); "
      f"gap / eps from {(g4 / 4e-3).min():.3f} to {(g4 / 4e-3).max():.3f}; ratio of the gaps at 0.004 and 0.002: {(g4 / g2).min():.2f} to {(g4 / g2).max():.2f}; "
      f"at 0.002 and 0.001: {(g2 / g1).min():.2f} to {(g2 / g1).max():.2f}")
out["knot_perturbation"] = {"models": int(len(gpl)), "draws": N_KNOT, "f_grid": "801 uniform values and 400 within 12 transition widths of each knot",
                            "alpha_grid": "0 to 0.2 in steps of 1e-3, to 0.4 in steps of 5e-3, to 1 in steps of 0.05",
                            "pl_zero": int(z.sum()), "pl_ge_1e-3": int(np.sum(gpl >= 1e-3)), "pl_between": int(np.sum(~z & (gpl < 1e-3))),
                            "pl_max": float(gpl.max()), "pl_smallest_positive": float(gpl[~z].min()),
                            "slope_below_which_all_concave": float(m2[~z].min()), "draws_below": int(np.sum(m2 < m2[~z].min())),
                            "slope_above_which_none_concave": float(m2[z].max()), "draws_above": int(np.sum(m2 > m2[z].max())),
                            "eps004_ge_1e-3": int(np.sum(gaps >= 1e-3)), "eps004_max": float(gaps.max()), "eps004_min": float(gaps.min()),
                            "concave_eps004": [float(g4.min()), float(np.median(g4)), float(g4.max())],
                            "concave_gap_over_eps": [float((g4 / 4e-3).min()), float((g4 / 4e-3).max())],
                            "concave_ratio_004_002": [float((g4 / g2).min()), float((g4 / g2).max())],
                            "concave_ratio_002_001": [float((g2 / g1).min()), float((g2 / g1).max())],
                            **kn}

# the nominal smoothed model on the same alpha grid: the gap of the certificate lies in [0.04, 0.11]; the smoothing also
# creates gaps on the other edges of the hull, which the piecewise-linear model traces exactly.  The largest of them sits
# on a short edge (between the hull vertices at 0.02 and 0.04) and peaks sharply, so alpha is refined to steps of 1e-4
# within 0.012 of the maximum found on AL_KNOT (at eps = 0.001 the grid of step 1e-3 alone gives 7 % less).
other = []
for eps in (1e-3, 2e-3, 4e-3):
    G = smooth_curve(LINES, eps)
    away = (AL_KNOT < 0.04 - 1e-9) | (AL_KNOT > 0.11 + 1e-9)
    k = int(np.argmax(np.where(away, G, -1.0)))
    al = np.unique(np.round(np.concatenate([AL_KNOT, AL_KNOT[k] + np.arange(-120, 121) * 1e-4]), 12)); al = al[(al >= 0) & (al <= 1)]
    Gf = smooth_curve(LINES, eps, al)
    awayf = (al < 0.04 - 1e-9) | (al > 0.11 + 1e-9)
    kf = int(np.argmax(np.where(awayf, Gf, -1.0)))
    other.append({"eps": eps, "max_gap": float(G.max()), "alpha_at_max": float(AL_KNOT[G.argmax()]),
                  "max_gap_outside_0.04_0.11": float(Gf[kf]), "alpha_outside": float(al[kf]), "on_the_grid_of_the_knots": float(G[k])})
    print(f"nominal model, eps = {eps}: max gap {G.max():.5f} at alpha = {AL_KNOT[G.argmax()]:.3f}; outside [0.04, 0.11]: {Gf[kf]:.3e} at alpha = {al[kf]:.4f} "
          f"({Gf[kf] / eps:.3f} eps; {G[k] / eps:.3f} eps on the grid of step 1e-3)")
out["nominal_other_edges"] = other
save_json("exp7_prop3_gap.json", out)
