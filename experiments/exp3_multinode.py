"""Experiment 3 - several desynchronized nodes (new Proposition 6).

For 8-node chains, |S| = 1..4 nodes draw independently.  The desynchronized point W is
computed exactly; we record the shift W - C, the superposition bound of Prop. 6, and the
ratio.  Two configuration ensembles:
  (a) random strategy pairs (A, B) on Dirichlet(1) models (one model per node), p ~ U(0.2,0.8)
  (b) NP-optimal dependent-randomization designs (Lagrangian-PBPO hull) at alpha in
      {0.05, 0.1, 0.2} on models drawn in the same way: 300 models (over disjoint blocks of
      30 models the medians differ by factors of 2.5 to 5.5; RESULTS.md, Section 5).
The ratio of the shift to the bound is taken over the cases with a positive bound (if no
node of S switches between A and B, shift and bound are both zero).
Usage: python experiments/exp3_multinode.py [--quick]
"""
import sys, time
import numpy as np
from common import *

QUICK = "--quick" in sys.argv
rng = np.random.default_rng(3)
n = 8
N_RANDOM = 60 if QUICK else 300
N_DESIGN = 6 if QUICK else 300
rows = []

t0 = time.time()
# (a) random pairs
for cfg in range(N_RANDOM):
    models = family_models("dirichlet3", n, rng, identical=False)
    A, B = random_pair(models, rng)
    p = float(rng.uniform(0.2, 0.8))
    for size in (1, 2, 3, 4):
        S = sorted(rng.choice(np.arange(1, n), size=size, replace=False).tolist())  # exclude the DFC
        out = prop6_terms(models, A, B, p, S)
        rows.append({"ensemble": "random", "cfg": cfg, "size": size, "S": S, "p": p,
                     "shift_F": out["shift"][0], "shift_D": out["shift"][1],
                     "norm": float(np.hypot(*out["shift"])),
                     "bound_F": out["bound_tight"][0], "bound_D": out["bound_tight"][1],
                     "bound_simple_F": out["bound_simple"][0]})
print(f"random ensemble done in {time.time()-t0:.1f}s")

# (b) NP-optimal designs
t0 = time.time()
for cfg in range(N_DESIGN):
    models = family_models("dirichlet3", n, rng, identical=False)
    for alpha in (0.05, 0.1, 0.2):
        des = np_design(models, alpha, restarts=8, seed=cfg, lambdas=np.exp(np.linspace(-5, 6, 25)))
        if des["classes"]["terminal_only"] or des["deterministic_at_alpha"]:
            continue                                   # no synchronization needed: nothing to lose
        A, B, p = des["sA"], des["sB"], des["p"]
        # nodes whose rule actually switches between A and B (desynchronizing any other
        # node has no effect at all); the DFC is excluded
        D = [k for k in range(1, n) if (A.j1 != B.j1 if k == 1 else A.pairs[k - 2] != B.pairs[k - 2])]
        rows.append({"ensemble": "design_meta", "cfg": cfg, "alpha": alpha, "n_switching_nodes": len(D),
                     "switching_nodes": D, "p": p})
        for size in (1, 2, 3, 4):
            if size > len(D):
                break
            S = sorted(rng.choice(np.array(D), size=size, replace=False).tolist())
            out = prop6_terms(models, A, B, p, S)
            rows.append({"ensemble": "design", "cfg": cfg, "alpha": alpha, "size": size, "S": S, "p": p,
                         "shift_F": out["shift"][0], "shift_D": out["shift"][1],
                         "norm": float(np.hypot(*out["shift"])),
                         "bound_F": out["bound_tight"][0], "bound_D": out["bound_tight"][1],
                         "bound_simple_F": out["bound_simple"][0]})
print(f"design ensemble done in {time.time()-t0:.1f}s")

# summary
summary = {"design_switching_nodes": summarize([r["n_switching_nodes"] for r in rows if r["ensemble"] == "design_meta"])}
print("number of switching nodes (excluding the DFC) in optimal designs:", summary["design_switching_nodes"])
for ens in ("random", "design"):
    for size in (1, 2, 3, 4):
        sel = [r for r in rows if r["ensemble"] == ens and r["size"] == size]
        if not sel:
            continue
        absF = np.array([abs(r["shift_F"]) for r in sel])
        bF = np.array([r["bound_F"] for r in sel])
        summary[f"{ens}|S|={size}"] = {
            "count": len(sel),
            "abs_shift_F": summarize(absF), "norm": summarize([r["norm"] for r in sel]),
            "bound_F": summarize(bF),
            "ratio_shift_over_bound": summarize(absF / np.maximum(bF, 1e-15)),
            "positive_bound": int(np.sum(bF > 1e-12)),
            "ratio_shift_over_positive_bound": summarize((absF / np.maximum(bF, 1e-15))[bF > 1e-12]),
            "violations_of_bound": int(np.sum(absF > bF + 1e-12)),
            "max_abs_shift_F": float(absF.max()),
        }
for ens in ("random", "design"):                       # pooled over |S|
    sel = [r for r in rows if r["ensemble"] == ens]
    absF = np.array([abs(r["shift_F"]) for r in sel]); bF = np.array([r["bound_F"] for r in sel])
    summary[f"{ens}|pooled"] = {"cases": len(sel), "positive_bound": int(np.sum(bF > 1e-12)),
                                "ratio_shift_over_positive_bound": summarize((absF / np.maximum(bF, 1e-15))[bF > 1e-12]),
                                "max_abs_shift_F": float(absF.max())}
    print(f"{ens}: {len(sel)} cases, {summary[f'{ens}|pooled']['positive_bound']} with a positive bound; "
          f"median shift / bound over them {summary[f'{ens}|pooled']['ratio_shift_over_positive_bound']['median']:.3f}; largest shift {absF.max():.3f}")
save_json("exp3_multinode_rows.json", rows)
save_json("exp3_multinode_summary.json", summary)
print("\n|S|  ensemble  count  median|dPF|   p90|dPF|   max|dPF|   median bound   median ratio  violations")
for k, v in summary.items():
    if "count" not in v:
        continue
    print(f"{k:16s} {v['count']:5d}  {v['abs_shift_F']['median']:.5f}   {v['abs_shift_F']['p90']:.5f}   "
          f"{v['max_abs_shift_F']:.5f}   {v['bound_F']['median']:.5f}      {v['ratio_shift_over_bound']['median']:.3f}       {v['violations_of_bound']}")
