"""Experiment 4 - where the desynchronized node sits (reviewer 4).

8-node chains; a single node k = 1..7 desynchronizes.  Records |W_F - alpha| versus k, the
injected size Delta_k, the downstream product prod_{j>k} max(rho_j^A, rho_j^B), eq. (14)
(checked against the exact computation), on (a) random strategy pairs and (b) NP-optimal
designs on Dirichlet(1) K=3 models (one model per node).
Usage: python experiments/exp4_position.py [--quick]
"""
import sys, time
import numpy as np
from common import *

QUICK = "--quick" in sys.argv
rng = np.random.default_rng(4)
n = 8
N_RANDOM = 60 if QUICK else 300
N_DESIGN = 6 if QUICK else 30
rows = []

def record(ens, cfg, models, A, B, p, alpha=None):
    D = [k for k in range(1, n) if (A.j1 != B.j1 if k == 1 else A.pairs[k - 2] != B.pairs[k - 2])]
    for k in range(1, n):
        if ens == "design" and k not in D:
            continue                      # a node that does not switch cannot desynchronize
        out = prop6_terms(models, A, B, p, [k])
        eq14 = (prop5_two_node(models, A, B, p)["dPF"] if (k == 1 and n == 2)
                else (None if k == 1 else prop5_midchain(models, A, B, p, k, 0)))
        if k == 1:   # eq. (14) with the dummy-input convention: p(1-p)(H^A - H^B)(f_B - f_A)
            fA, fB = models[0].region_probs(A.j1)[0], models[0].region_probs(B.j1)[0]
            HA = float(np.prod([models[j].attenuation(*A.pairs[j - 1])[0] for j in range(1, n)]))
            HB = float(np.prod([models[j].attenuation(*B.pairs[j - 1])[0] for j in range(1, n)]))
            eq14 = p * (1 - p) * (HA - HB) * (fB - fA)
        Hmax = float(np.prod([max(models[j].attenuation(*A.pairs[j - 1])[0],
                                  models[j].attenuation(*B.pairs[j - 1])[0]) for j in range(k, n)]))
        rows.append({"ensemble": ens, "cfg": cfg, "alpha": alpha, "k": k, "p": p,
                     "shift_F": out["shift"][0], "shift_D": out["shift"][1], "eq14_F": eq14,
                     "bound_F": out["bound_tight"][0], "Hmax_downstream": Hmax})

t0 = time.time()
for cfg in range(N_RANDOM):
    models = family_models("dirichlet3", n, rng, identical=False)
    A, B = random_pair(models, rng)
    record("random", cfg, models, A, B, float(rng.uniform(0.2, 0.8)))
for cfg in range(N_DESIGN):
    models = family_models("dirichlet3", n, rng, identical=False)
    for alpha in (0.05, 0.1, 0.2):
        des = np_design(models, alpha, restarts=8, seed=cfg, lambdas=np.exp(np.linspace(-5, 6, 25)))
        if des["classes"]["terminal_only"] or des["deterministic_at_alpha"]:
            continue
        record("design", cfg, models, des["sA"], des["sB"], des["p"], alpha)
print(f"done in {time.time()-t0:.1f}s")

eq14_err = max(abs(r["shift_F"] - r["eq14_F"]) for r in rows)
print("max |exact - eq.(14)| over all rows:", eq14_err)
summary = {"eq14_max_abs_error": eq14_err}
for ens in ("random", "design"):
    for k in range(1, n):
        sel = [r for r in rows if r["ensemble"] == ens and r["k"] == k]
        if sel:
            summary[f"{ens}k={k}"] = {"count": len(sel),
                                      "abs_shift_F": summarize([abs(r["shift_F"]) for r in sel]),
                                      "bound_F": summarize([r["bound_F"] for r in sel]),
                                      "Hmax_downstream": summarize([r["Hmax_downstream"] for r in sel]),
                                      "frac_zero_shift": float(np.mean([abs(r["shift_F"]) < 1e-12 for r in sel]))}
save_json("exp4_position_rows.json", rows)
save_json("exp4_position_summary.json", summary)
print("\nensemble  k  count  median|dPF|  p90|dPF|  median bound  median prod(rho) downstream  frac zero")
for key, v in summary.items():
    if key == "eq14_max_abs_error":
        continue
    print(f"{key:12s} {v['count']:5d}  {v['abs_shift_F']['median']:.5f}   {v['abs_shift_F']['p90']:.5f}   "
          f"{v['bound_F']['median']:.5f}     {v['Hmax_downstream']['median']:.4f}       {v['frac_zero_shift']:.2f}")
