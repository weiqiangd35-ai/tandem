"""Experiment 13 - a message delayed on the link into node k (Section IV-F, Table SVII).

If the message u_{k-1} reaches N_k one or more epochs late and the hypothesis has not
changed, N_k fuses a message that was formed under an earlier common draw Z', independent
of the current draw Z: the block N_1..N_{k-1} is desynchronized as a whole.  The exact
operating point is the mixture over (Z', Z), and the shift obeys the block form of eq. (13),

    W - C = p(1-p) (H^A - H^B) (x_B - x_A),

with x_S the rate of u_{k-1} = 1 under strategy S and H^S the product of the attenuation
factors of nodes k..n under S (both coordinates).  The script checks the identity against
the exact mixture on random chains and on NP-optimal designs, and reports the size of the
shift as a function of the position of the delayed link.
Usage: python experiments/exp13_delayed_link.py [--quick]
"""
import sys
import numpy as np
from common import *
from tandem_sim.desync import delayed_link_point, delayed_link_identity, sync_point, desync_point

QUICK = "--quick" in sys.argv
rng = np.random.default_rng(13)
N_RANDOM = 40 if QUICK else 200           # random strategy pairs per chain length
N_DESIGN = 10 if QUICK else 40            # random models for the NP-optimal designs (n = 3, 4)
rows = []


def record(ens, n, models, A, B, p):
    C = sync_point(models, A, B, p)
    for k in range(2, n + 1):
        W = delayed_link_point(models, A, B, p, k)
        idF = delayed_link_identity(models, A, B, p, k, 0)
        idD = delayed_link_identity(models, A, B, p, k, 1)
        rows.append({"ensemble": ens, "n": n, "k": k, "p": p, "shift_F": W[0] - C[0], "shift_D": W[1] - C[1],
                     "identity_F": idF, "identity_D": idD})


for n in (3, 5, 8):
    for _ in range(N_RANDOM):
        models = family_models("dirichlet3", n, rng, identical=False)
        A, B = random_pair(models, rng)
        record("random", n, models, A, B, float(rng.uniform(0.2, 0.8)))
for n in (3, 4):
    for _ in range(N_DESIGN):
        models = family_models("dirichlet3", n, rng, identical=False)
        for alpha in (0.05, 0.1, 0.2):
            des = np_design(models, alpha)
            if des["deterministic_at_alpha"]:
                continue
            record("design", n, models, des["sA"], des["sB"], des["p"])

err = max(max(abs(r["shift_F"] - r["identity_F"]), abs(r["shift_D"] - r["identity_D"])) for r in rows)
print(f"{len(rows)} cases ({sum(r['ensemble'] == 'random' for r in rows)} from random pairs, "
      f"{sum(r['ensemble'] == 'design' for r in rows)} from NP-optimal designs); "
      f"max |exact - identity| over both coordinates: {err:.2e}")
summary = {"cases": len(rows), "max_abs_error": err, "by_position": {}}
print("eight-node chains, random pairs: median |shift of P_F| by position k of the delayed link")
for k in range(2, 9):
    sel = [abs(r["shift_F"]) for r in rows if r["ensemble"] == "random" and r["n"] == 8 and r["k"] == k]
    summary["by_position"][k] = summarize(sel)
    print(f"  k = {k}: median {np.median(sel):.2e}, 90th percentile {np.percentile(sel, 90):.2e}")

# worked example: three-node running model at alpha = 0.1675 (N_1 and N_2 switch)
d5 = load_json("exp5_correction.json")
three = [t for t in d5["three_node"] if abs(t["alpha"] - 0.1675) < 1e-9][0]
sA = Strategy(three["sA"]["j1"], tuple(tuple(x) for x in three["sA"]["pairs"]))
sB = Strategy(three["sB"]["j1"], tuple(tuple(x) for x in three["sB"]["pairs"]))
models = [RUNNING] * 3
C = sync_point(models, sA, sB, three["p"])
ex = {}
for k in (2, 3):
    W = delayed_link_point(models, sA, sB, three["p"], k)
    ex[k] = {"W": W, "shift": (W[0] - C[0], W[1] - C[1]),
             "identity": (delayed_link_identity(models, sA, sB, three["p"], k, 0),
                          delayed_link_identity(models, sA, sB, three["p"], k, 1))}
    print(f"three-node running model, alpha = 0.1675, link into N{k} delayed: W = ({W[0]:.5f}, {W[1]:.5f}), "
          f"shift ({W[0]-C[0]:+.5f}, {W[1]-C[1]:+.5f}), identity ({ex[k]['identity'][0]:+.5f}, {ex[k]['identity'][1]:+.5f})")
W1 = desync_point(models, sA, sB, three["p"], [1])[0]
print(f"  (N1 alone desynchronized: W = ({W1[0]:.5f}, {W1[1]:.5f}); identical to a delayed link into N2)")
summary["three_node_example"] = ex
summary["rows"] = rows
save_json("exp13_delayed_link.json", summary)
