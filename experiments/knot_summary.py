"""Statistics of the knot perturbations of Proposition 3 quoted in Section III-B, Section S3
and the response letter, from the per-draw values stored by exp7_prop3_gap.py
(results/exp7_prop3_gap.json).  Writes results/exp7_knot_summary.json.

The classification at eps = 0 is also re-derived in exact rational arithmetic, independently
of the floating-point code of exp7: the knots of every draw are taken as the exact binary
fractions of the stored floats, the concave hull is built from the 64 strategies with all
three coordinates at knots, and at the quarter points of every hull edge the chord is
compared with the best of the candidates of Corollary S1 (two coordinates at knots, the
third solved from P_F = alpha).  The difference is exactly zero on every edge for the draws
classified as concave and positive on some edge for the others.

The gaps of the smoothed models are taken against the concave hull over the whole interval
of alpha (exp7).  For comparison the summary also gives the largest gap of the nominal
smoothed model away from the gap of the certificate (alpha outside [0.04, 0.11]), where its
piecewise-linear envelope follows the hull exactly: gaps of this kind come from the smoothing.
Usage: python experiments/knot_summary.py
"""
from fractions import Fraction as Fr
from itertools import product
import numpy as np
from common import load_json, save_json


def exact_gap(pts):
    """Largest chord-minus-envelope difference at the quarter points of the hull edges, as a Fraction."""
    P = [(Fr(x), Fr(y)) for x, y in pts]
    lines = [((y2 - y1) / (x2 - x1), y1 - (y2 - y1) / (x2 - x1) * x1) for (x1, y1), (x2, y2) in zip(P[:-1], P[1:])]
    K = [q[0] for q in P]
    r = lambda a: min(m * a + b for m, b in lines)
    value = lambda f, a0, a1: (1 - r(f)) * r(a0) + r(f) * r(a1)

    def envelope(alpha):
        cand = [(f, a0, (alpha - (1 - f) * a0) / f) for f, a0 in product(K, K) if f > 0]
        cand += [(f, (alpha - f * a1) / (1 - f), a1) for f, a1 in product(K, K) if f < 1]
        cand += [((alpha - a0) / (a1 - a0), a0, a1) for a0, a1 in product(K, K) if a1 != a0]
        return max(value(*w) for w in cand if all(0 <= x <= 1 for x in w))

    best = {}
    for f, a0, a1 in product(K, K, K):
        x, y = (1 - f) * a0 + f * a1, value(f, a0, a1)
        best[x] = max(y, best.get(x, y))
    hull = []
    for q in sorted(best.items()):
        while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (q[0] - hull[-2][0]) <= (q[1] - hull[-2][1]) * (hull[-1][0] - hull[-2][0]):
            hull.pop()                                 # the last vertex is on or below the chord
        hull.append(q)
    return max(y1 + w * (y2 - y1) - envelope(x1 + w * (x2 - x1))
               for (x1, y1), (x2, y2) in zip(hull[:-1], hull[1:]) for w in (Fr(1, 4), Fr(1, 2), Fr(3, 4)))


e7 = load_json("exp7_prop3_gap.json")
k = e7["knot_perturbation"]
gpl, g4, m2 = np.array(k["gap_pl"]), np.array(k["gap"]), np.array(k["middle_slope"])
z = gpl <= 1e-12                                   # concave piecewise-linear envelope
g2 = np.array([x for x in k["gap_eps002"] if x is not None]); g1 = np.array([x for x in k["gap_eps001"] if x is not None]); g4z = g4[z]
N = len(gpl)
assert N == k["models"] and len(g2) == len(g1) == int(z.sum()) and np.array_equal(z, np.array(k["one_step"]))
# the same draws as in exp7 (generator seeded with 7, four uniform factors per draw), in exact arithmetic
rng = np.random.default_rng(7); exact = {}
for t in range(max(k["draw"]) + 1):
    u = rng.uniform(0.95, 1.05, 4)
    exact[t] = [(0.0, 0.0), (0.1 * u[0], 0.6 * u[1]), (0.2 * u[2], 0.7 * u[3]), (1.0, 1.0)]
for t, slope in zip(k["draw"], k["middle_slope"]):
    (x1, y1), (x2, y2) = exact[t][1], exact[t][2]
    assert (y2 - y1) / (x2 - x1) == slope            # the draws are those of exp7
ge = [exact_gap(exact[t]) for t in k["draw"]]
assert min(ge) >= 0
exact_ok = all((g == 0) == bool(zz) for g, zz in zip(ge, z))
c = np.vstack([g1 / 1e-3, g2 / 2e-3, g4z / 4e-3])  # gap / eps at the three values of eps
EPS = (1e-3, 2e-3, 4e-3)
dz = np.array(k["draw"])[z]                         # the draws with a concave piecewise-linear envelope
ie, id_ = np.unravel_index(int(c.argmax()), c.shape)
az = np.vstack([np.array([x for x in k[key] if x is not None]) if key != "alpha_gap" else np.array(k[key])[z]
                for key in ("alpha_gap_eps001", "alpha_gap_eps002", "alpha_gap")])     # where the largest gap of each concave draw lies
nom = [{"eps": o["eps"], "gap_over_eps": o["max_gap_outside_0.04_0.11"] / o["eps"], "alpha": o["alpha_outside"], "max_gap": o["max_gap"],
        "gap_over_eps_on_the_grid_of_the_knots": o["on_the_grid_of_the_knots"] / o["eps"]} for o in e7["nominal_other_edges"]]
r21, r42 = g2 / g1, g4z / g2
prop = (np.abs(r21 - 2) <= 0.25) & (np.abs(r42 - 2) <= 0.25)       # the gap doubles with eps, to within an eighth
LO, HI = float(np.floor(100 * m2[~z].min()) / 100), float(np.ceil(100 * m2[z].max()) / 100)    # the two slope bounds, rounded inward
out = {"draws": N,
       "eps0": {"gap_ge_1e-3": int(np.sum(gpl >= 1e-3)), "concave": int(z.sum()), "between": int(np.sum(~z & (gpl < 1e-3))), "max": float(gpl.max())},
       "exact_rational_check": {"zero_on_every_edge": int(sum(g == 0 for g in ge)), "agrees_with_the_classification": bool(exact_ok),
                                "smallest_positive": float(min(g for g in ge if g > 0))},
       "middle_slope": {"all_concave_below": float(m2[~z].min()), "draws_below": int(np.sum(m2 < m2[~z].min())),
                        "none_concave_above": float(m2[z].max()), "draws_above": int(np.sum(m2 > m2[z].max())), "range": [float(m2.min()), float(m2.max())],
                        "rounded": {"below": LO, "draws_below": int(np.sum(m2 < LO)), "concave_below": int(np.sum(z & (m2 < LO))),
                                    "above": HI, "draws_above": int(np.sum(m2 > HI)), "concave_above": int(np.sum(z & (m2 > HI)))}},
       "eps004": {"gap_ge_1e-3": int(np.sum(g4 >= 1e-3)), "max": float(g4.max()),
                  "from_ge_to_below_1e-3": int(np.sum((gpl >= 1e-3) & (g4 < 1e-3))), "from_below_to_ge_1e-3": int(np.sum((gpl < 1e-3) & (g4 >= 1e-3)))},
       "concave": {"gap_eps004": [float(g4z.min()), float(np.median(g4z)), float(g4z.max())],
                   "gap_over_eps_max": float(c.max()), "gap_over_eps_median": [float(np.median(x)) for x in c],
                   "gap_over_eps_max_by_eps": [float(x.max()) for x in c], "gap_over_eps_min_by_eps": [float(x.min()) for x in c],
                   "largest_at": {"draw": int(dz[id_]), "eps": EPS[ie], "alpha": float(az[ie, id_])},
                   "draws_above_0.1_eps": int(np.sum(c.max(axis=0) > 0.1)), "alpha_of_the_gap": [float(az.min()), float(az.max())],
                   "proportional": int(prop.sum()), "ratios_of_the_proportional": [float(min(r21[prop].min(), r42[prop].min())), float(max(r21[prop].max(), r42[prop].max()))],
                   "not_proportional": {"draws": [int(x) for x in np.array(k["draw"])[z][~prop]], "middle_slopes": [float(x) for x in m2[z][~prop]],
                                        "largest_middle_slope": float(m2[z][~prop].max()), "smallest_middle_slope_of_the_others": float(m2[z][prop].min())}}}
out["nominal_other_edges"] = nom
out["alpha_grid"] = k["alpha_grid"]
save_json("exp7_knot_summary.json", out)
e0, ms, e4, cc = out["eps0"], out["middle_slope"], out["eps004"], out["concave"]
print(f"{N} perturbed models with a concave local ROC")
print(f"eps = 0 (exact enumeration, Corollary S1): gap >= 1e-3 in {e0['gap_ge_1e-3']} draws ({100*e0['gap_ge_1e-3']/N:.1f} %), "
      f"concave envelope in {e0['concave']} ({100*e0['concave']/N:.1f} %), a smaller positive gap in {e0['between']}; maximum {e0['max']:.4f}")
print(f"exact rational arithmetic at the quarter points of the hull edges: zero on every edge in {out['exact_rational_check']['zero_on_every_edge']} draws; "
      f"same classification: {exact_ok}; smallest positive value {out['exact_rational_check']['smallest_positive']:.2e}")
assert exact_ok
print(f"middle slope (nominally 1; {ms['range'][0]:.2f} to {ms['range'][1]:.2f} over the draws): concave envelope in all {ms['draws_below']} draws below "
      f"{ms['all_concave_below']:.3f} and in none of the {ms['draws_above']} above {ms['none_concave_above']:.3f}; with rounded bounds, "
      f"in all {ms['rounded']['draws_below']} below {ms['rounded']['below']:.2f} and in none of the {ms['rounded']['draws_above']} above {ms['rounded']['above']:.2f}")
assert ms["rounded"]["concave_below"] == ms["rounded"]["draws_below"] and ms["rounded"]["concave_above"] == 0
print(f"eps = 0.004: gap >= 1e-3 in {e4['gap_ge_1e-3']} draws ({100*e4['gap_ge_1e-3']/N:.1f} %), maximum {e4['max']:.4f}; "
      f"{e4['from_ge_to_below_1e-3']} draws fall below 1e-3 and {e4['from_below_to_ge_1e-3']} rise above it")
print(f"draws with a concave piecewise-linear envelope: gap at eps = 0.004 from {cc['gap_eps004'][0]:.2e} to {cc['gap_eps004'][2]:.2e} "
      f"(median {cc['gap_eps004'][1]:.2e}); gap / eps at most {cc['gap_over_eps_max']:.3f} over eps = 0.001, 0.002, 0.004 "
      f"(medians {', '.join(f'{x:.3f}' for x in cc['gap_over_eps_median'])}; maxima {', '.join(f'{x:.3f}' for x in cc['gap_over_eps_max_by_eps'])}; "
      f"largest in draw {cc['largest_at']['draw']} at eps = {cc['largest_at']['eps']}, alpha = {cc['largest_at']['alpha']:.3f}; above 0.1 eps at some eps in "
      f"{cc['draws_above_0.1_eps']} draws; the largest gap lies at alpha between {cc['alpha_of_the_gap'][0]:.3f} and {cc['alpha_of_the_gap'][1]:.3f})")
print(f"  proportional to eps (the gap doubles from 0.001 to 0.002 and from 0.002 to 0.004, to within an eighth) in {cc['proportional']} of them "
      f"(ratios {cc['ratios_of_the_proportional'][0]:.2f} to {cc['ratios_of_the_proportional'][1]:.2f}); the other {len(cc['not_proportional']['draws'])} "
      f"have the smallest middle slopes ({', '.join(f'{x:.3f}' for x in sorted(cc['not_proportional']['middle_slopes']))}; next {cc['not_proportional']['smallest_middle_slope_of_the_others']:.3f})")
print("nominal smoothed model outside [0.04, 0.11] (where the piecewise-linear envelope has no gap), alpha refined to steps of 1e-4 around the maximum: "
      + "; ".join(f"eps = {o['eps']}: {o['gap_over_eps']:.3f} eps at alpha = {o['alpha']:.4f} ({o['gap_over_eps_on_the_grid_of_the_knots']:.3f} eps on the grid of the knots)" for o in nom))
