"""Experiment 12 - exact rational enumeration behind Proposition 3 (supplement, Sec. S3).

At epsilon = 0 both local ROCs are the piecewise-linear concave hull r of the running
model's local points (0,0), (1/10,3/5), (1/5,7/10), (1,1), i.e. the pointwise minimum of
the lines 6a, a + 1/2, 3a/8 + 5/8, with knot set K = {0, 1/10, 1/5, 1}.  A deterministic
two-node strategy is (f, a0, a1) with
    P_F = (1 - f) a0 + f a1,      P_D = (1 - d) r(a0) + d r(a1),      d = r(f).
By Corollary S1 the deterministic envelope at a given alpha is attained at a point with at
least two of (f, a0, a1) in K.  The script enumerates these candidates in exact rational
arithmetic (three choices of the pair of coordinates at knots, 4 x 4 knot values each, the
third coordinate solved from P_F = alpha; 48 candidates before discarding the infeasible
ones), returns the envelope, and compares it with the chord of the randomized (concave)
envelope through the two achievable points (1/25, 49/100) and (11/100, 33/50).
A brute-force search on a fine rational grid confirms that no other strategy does better.
Usage: python experiments/exp12_certificate.py
"""
from fractions import Fraction as Fr
from itertools import product
from common import save_json

K = [Fr(0), Fr(1, 10), Fr(1, 5), Fr(1)]
LINES = [(Fr(6), Fr(0)), (Fr(1), Fr(1, 2)), (Fr(3, 8), Fr(5, 8))]      # (slope, intercept)


def r(a):
    return min(m * a + b for m, b in LINES)


def point(f, a0, a1):
    d = r(f)
    return (1 - f) * a0 + f * a1, (1 - d) * r(a0) + d * r(a1)


def candidates(alpha):
    """Strategies with at least two coordinates in K and P_F = alpha."""
    out = []
    for f, a0 in product(K, K):                     # a1 solved
        if f > 0:
            a1 = (alpha - (1 - f) * a0) / f
            if 0 <= a1 <= 1:
                out.append((f, a0, a1))
    for f, a1 in product(K, K):                     # a0 solved
        if f < 1:
            a0 = (alpha - f * a1) / (1 - f)
            if 0 <= a0 <= 1:
                out.append((f, a0, a1))
    for a0, a1 in product(K, K):                    # f solved
        if a1 != a0:
            f = (alpha - a0) / (a1 - a0)
            if 0 <= f <= 1:
                out.append((f, a0, a1))
    return out


def envelope(alpha):
    best = max(candidates(alpha), key=lambda w: point(*w)[1])
    return point(*best)[1], best


def chord(alpha):
    (x1, y1), (x2, y2) = (Fr(1, 25), Fr(49, 100)), (Fr(11, 100), Fr(33, 50))
    return y1 + (y2 - y1) / (x2 - x1) * (alpha - x1)


def brute_force(alpha, N=200):
    """max P_D over the grid f, a0 in {0, 1/N, ..., 1} with a1 solved from P_F = alpha."""
    best = Fr(0)
    for i in range(N + 1):
        f = Fr(i, N)
        for j in range(N + 1):
            a0 = Fr(j, N)
            if f == 0:
                if a0 != alpha:
                    continue
                a1 = Fr(0)
            else:
                a1 = (alpha - (1 - f) * a0) / f
                if not 0 <= a1 <= 1:
                    continue
            best = max(best, point(f, a0, a1)[1])
    return best


if __name__ == "__main__":
    # the two achievable points of the chord
    assert point(Fr(1, 5), Fr(0), Fr(1, 5)) == (Fr(1, 25), Fr(49, 100))
    assert point(Fr(1, 10), Fr(1, 10), Fr(1, 5)) == (Fr(11, 100), Fr(33, 50))
    out = {}
    for alpha in (Fr(3, 40), Fr(2, 25)):
        R, w = envelope(alpha)
        ncand = len(candidates(alpha))
        gap = chord(alpha) - R
        bf = brute_force(alpha)
        print(f"alpha = {alpha}: {ncand} feasible candidates; deterministic envelope {R} = {float(R):.6f} at (f, a0, a1) = "
              f"({w[0]}, {w[1]}, {w[2]}); hull chord {chord(alpha)}; gap {gap} = {float(gap):.6f}; "
              f"brute force on a 201 x 201 grid: {bf} = {float(bf):.6f} (<= envelope: {bf <= R})")
        out[str(alpha)] = {"envelope": str(R), "argmax": [str(x) for x in w], "chord": str(chord(alpha)), "gap": str(gap),
                           "feasible_candidates": ncand, "brute_force": str(bf)}
    assert envelope(Fr(3, 40))[0] == Fr(91, 160) and chord(Fr(3, 40)) - Fr(91, 160) == Fr(1, 160)
    assert envelope(Fr(2, 25))[0] == Fr(29, 50) and chord(Fr(2, 25)) - Fr(29, 50) == Fr(1, 140)
    save_json("exp12_certificate.json", out)
    print("certificate values confirmed: 91/160 at alpha = 3/40 (gap 1/160), 29/50 at alpha = 2/25 (gap 1/140)")
