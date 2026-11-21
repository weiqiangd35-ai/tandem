"""Experiment 11 - epoch-by-epoch Monte Carlo simulation of synchronization loss.

Every desynchronized operating point in the paper is computed exactly, as a
probability-weighted sum over the draw combinations.  This script checks that computation
with an independent simulation that does not use the recursion: in every epoch it draws
the hypothesis-conditional observations of all nodes, the common draw Z ~ Bernoulli(p),
an independent draw for every desynchronized node, and passes the one-bit messages down
the chain with each node applying the decision region of the strategy it is itself using.

Part 1 compares the empirical (P_F, P_D) with the exact desynchronized point W for the
worked examples of the paper and for a random eight-node chain with three desynchronized
nodes, and reports the deviation in units of the Monte Carlo standard error.
Part 2 simulates what the final LD observes, (Z_t, u_{n-1,t}) under the prior P0, and
checks the moment estimator of the marginal p' of Section V-D against its predicted
standard deviation.
Usage: python experiments/exp11_montecarlo.py [--quick]
"""
import sys
import numpy as np
from common import *
from tandem_sim.desync import desync_point, sync_point

QUICK = "--quick" in sys.argv
T_OP = 400_000 if QUICK else 4_000_000        # epochs per hypothesis, Part 1
REPS, T_EST = (100, 20_000) if QUICK else (400, 20_000)


def simulate(models, sA, sB, p, S, T, hyp, rng, marg=None):
    """Simulate T epochs under hypothesis hyp (0 or 1; or an array of 0/1 per epoch).
    S: 1-based indices of the nodes that draw independently (marginal marg[k], default p).
    Returns (u_n, u_{n-1}, Z) as boolean arrays; Z True means strategy A."""
    n = len(models)
    hyp = np.broadcast_to(np.asarray(hyp), (T,))
    Z = rng.random(T) < p
    u = prev = None
    for k in range(1, n + 1):
        m = models[k - 1]
        cdf0, cdf1 = np.cumsum(m.p0), np.cumsum(m.p1)
        x = rng.random(T)
        y = np.where(hyp == 1, np.searchsorted(cdf1, x), np.searchsorted(cdf0, x))
        y = np.minimum(y, m.K - 1)
        rank = np.empty(m.K, dtype=int)
        rank[m.order] = np.arange(m.K)              # position of each observation in ascending-LR order
        r = rank[y]
        useA = (rng.random(T) < (p if marg is None else marg.get(k, p))) if k in S else Z
        if k == 1:
            level = np.where(useA, sA.j1, sB.j1)
        else:
            (a0, a1), (b0, b1) = sA.pairs[k - 2], sB.pairs[k - 2]
            level = np.where(useA, np.where(u, a1, a0), np.where(u, b1, b0))
        prev, u = u, r >= level                      # region of level j: rank >= j
    return u, prev, Z


rng = np.random.default_rng(11)
out = {"epochs_per_hypothesis": T_OP, "operating_points": [], "estimator": {}}

# ------------------------------------------------------------------ Part 1
d5 = load_json("exp5_correction.json")


def three_node(alpha):
    t = [t for t in d5["three_node"] if abs(t["alpha"] - alpha) < 1e-9][0]
    sA = Strategy(t["sA"]["j1"], tuple(tuple(x) for x in t["sA"]["pairs"]))
    sB = Strategy(t["sB"]["j1"], tuple(tuple(x) for x in t["sB"]["pairs"]))
    return sA, sB, t["p"]


sA3, sB3, p3 = three_node(0.078)          # N_2 is the switching node
sA3b, sB3b, p3b = three_node(0.1675)      # N_1 and N_2 switch
rng8 = np.random.default_rng(3)
models8 = family_models("dirichlet3", 8, rng8, identical=False)
sA8, sB8 = random_pair(models8, rng8)
cases = [
    ("two-node running model, alpha = 0.075, N1 desynchronized", [RUNNING] * 2, Strategy(1, ((3, 1),)), Strategy(2, ((2, 1),)), 0.5, [1]),
    ("two-node violating model, alpha = 0.21125, N1 desynchronized", [VIOLATING] * 2, Strategy(2, ((2, 0),)), Strategy(1, ((2, 1),)), 0.5, [1]),
    ("three-node running model, alpha = 0.078, N2 desynchronized", [RUNNING] * 3, sA3, sB3, p3, [2]),
    ("three-node running model, alpha = 0.078, synchronized", [RUNNING] * 3, sA3, sB3, p3, []),
    ("three-node running model, alpha = 0.1675, N1 desynchronized", [RUNNING] * 3, sA3b, sB3b, p3b, [1]),
    ("random eight-node chain (Dirichlet K = 3), nodes 2, 5, 7 desynchronized", models8, sA8, sB8, 0.4, [2, 5, 7]),
]
zmax = 0.0
print("case | exact (P_F, P_D) | simulated (P_F, P_D) | deviation / standard error")
for name, models, sA, sB, p, S in cases:
    W = desync_point(models, sA, sB, p, S)[0] if S else sync_point(models, sA, sB, p)
    est, z = [], []
    for hyp in (0, 1):
        u, _, _ = simulate(models, sA, sB, p, S, T_OP, hyp, rng)
        e = float(u.mean())
        se = float(np.sqrt(max(W[hyp] * (1 - W[hyp]), 1e-12) / T_OP))
        est.append(e); z.append((e - W[hyp]) / se)
    zmax = max(zmax, max(abs(v) for v in z))
    out["operating_points"].append({"case": name, "p": p, "S": S, "exact": list(W), "simulated": est, "z": z})
    print(f"{name} | ({W[0]:.5f}, {W[1]:.5f}) | ({est[0]:.5f}, {est[1]:.5f}) | ({z[0]:+.2f}, {z[1]:+.2f})")
out["max_abs_z"] = zmax
print(f"largest deviation: {zmax:.2f} standard errors over {2 * len(cases)} estimates")

# ------------------------------------------------------------------ Part 2
# Two-node running model, alpha = 0.075, prior P0 = 0.9, N1 desynchronized with p' = p = 0.5.
P0, p = 0.9, 0.5
models, sA, sB = [RUNNING] * 2, Strategy(1, ((3, 1),)), Strategy(2, ((2, 1),))
fA, dA = RUNNING.region_probs(sA.j1); fB, dB = RUNNING.region_probs(sB.j1)
mA, mB = P0 * fA + (1 - P0) * dA, P0 * fB + (1 - P0) * dB        # rate of u_1 = 1 when N1 uses rule A / B
T = REPS * T_EST
hyp = (rng.random(T) >= P0).astype(int)
_, u1, Z = simulate(models, sA, sB, p, [1], T, hyp, rng)
u1 = u1.reshape(REPS, T_EST); Z = Z.reshape(REPS, T_EST)
rA = (u1 & Z).sum(1) / Z.sum(1); rB = (u1 & ~Z).sum(1) / (~Z).sum(1)
p_hat = (rA - mB) / (mA - mB)                                    # moment estimator from the Z = A epochs
TA = float(Z.sum(1).mean())
r_des = p * mA + (1 - p) * mB
out["estimator"] = {"P0": P0, "p_true": p, "repetitions": REPS, "epochs_per_repetition": T_EST, "mean_T_A": TA,
                    "sync_rates": [mA, mB], "desync_rate": r_des, "mean_rate_A": float(rA.mean()), "mean_rate_B": float(rB.mean()),
                    "p_hat_mean": float(p_hat.mean()), "p_hat_std": float(p_hat.std(ddof=1)),
                    "std_formula_sync_rate": float(np.sqrt(mA * (1 - mA) / TA) / abs(mA - mB)),
                    "std_formula_observed_rate": float(np.sqrt(r_des * (1 - r_des) / TA) / abs(mA - mB))}
e = out["estimator"]
print(f"\nDFC monitoring, two-node running model, P0 = {P0}: synchronized rates ({mA:.3f}, {mB:.3f}); "
      f"simulated rates given Z = A / B: {e['mean_rate_A']:.4f} / {e['mean_rate_B']:.4f} (predicted {r_des:.3f} for both)")
print(f"moment estimator of p' over {REPS} repetitions of {T_EST} epochs (T_A = {TA:.0f}): mean {e['p_hat_mean']:.4f}, "
      f"std {e['p_hat_std']:.4f}; formula {e['std_formula_sync_rate']:.4f} (synchronized rate), {e['std_formula_observed_rate']:.4f} (observed rate)")
save_json("exp11_montecarlo.json", out)
