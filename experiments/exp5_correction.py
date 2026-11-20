"""Experiment 5 - the correction at the final LD and its benchmarks (Section V), and the
direction of the false-alarm shift in designs (Section IV-E).

Designs use the realization of the two supporting vertices with the fewest switching nodes
(ties: smallest worst-case shift), see common.np_design.  A failure is the loss of
synchronization of one switching node k < n.

5a  Two-node worked examples: the running model at alpha = 0.075 (P_F falls, the budget is
    re-spent) and the constraint-violating model of Sec. IV-E at alpha = 0.21125 (P_F
    rises, the constraint is restored).
5b  Three-node running model: every hull edge whose design is susceptible, every switching
    node.  alpha = 0.078 (N_2 switches, the rule of the final LD switches: the adjustment
    restores the constraint) and alpha = 0.1675 (N_1 and N_2 switch, the rule of the final
    LD does not: the adjustment has nothing to act on) are the examples of the paper.
5c  Statistics over random Dirichlet(1) K = 3 models (one model per node), alpha uniform on
    [0.02, 0.3], n = 2, 3, 4; 20000 cases for each n.  The unit is a case = (susceptible
    design, failing node); a design with two failing nodes contributes two cases.
    For each case: direction of the P_F shift; whether the rule of the final LD switches;
    whether the adjustment can meet the constraint; its P_D against the DFC-layer optimum
    and the downstream optimum (nodes k+1..n redesigned); and whether the corrected P_D
    stays above the deterministic optimum G, which needs no common draw.
Usage: python experiments/exp5_correction.py [--quick]
"""
import sys
import numpy as np
from common import *
from tandem_sim.desync import downstream_benchmark

QUICK = "--quick" in sys.argv
TOL = 1e-9
N_CASES = 60 if QUICK else 20000    # cases per chain length: standard error of a percentage about 0.5 point
                                    # (designs as clusters); with 300 cases it is 3 to 5 points


def failure(models, A, B, p, alpha, k):
    """Everything about the loss of synchronization of node k in the design (A, B, p)."""
    C = sync_point(models, A, B, p)
    W, _ = desync_point(models, A, B, p, [k])
    corr = terminal_adjust(models, A, B, p, alpha, S=(k,))
    dfc = dfc_layer_benchmark(models, A, B, p, alpha, S=(k,))["PD"]
    down = downstream_benchmark(models, A, B, p, alpha, k)["PD"]
    return {"k": k, "W": W, "shift": (W[0] - C[0], W[1] - C[1]), "dfc_switches": corr["dfc_switches"],
            "feasible": corr["feasible"], "q": corr.get("q"), "point": corr.get("point"),
            "L": corr["L"], "U": corr["U"], "dfc_layer_PD": dfc, "downstream_PD": down}


def worked(models, alpha, label, verbose=True):
    des = np_design(models, alpha, compute_G=True)
    A, B, p = des["sA"], des["sB"], des["p"]
    out = {"label": label, "alpha": alpha, "G": des["G"], "A": des["A"], "B": des["B"], "p": p, "C": des["C"],
           "classes": des["classes"], "sA": A, "sB": B, "switching": des["switching"], "failures": {}}
    if verbose:
        print(f"\n=== {label}: alpha = {alpha}")
        print(f"  deterministic optimum G = {des['G']}")
        print(f"  A = {des['A']}  B = {des['B']}  p = {p:.4f}  C = {des['C']}")
        print(f"  strategies: A = {A}, B = {B}; switching nodes (1..n-1): {des['switching']}; "
              f"rule of the final LD switches: {tuple(A.pairs[-1]) != tuple(B.pairs[-1])}")
    for k in des["switching"]:
        f = failure(models, A, B, p, alpha, k)
        out["failures"][str(k)] = f
        if verbose:
            print(f"  N{k} desynchronized: W = ({f['W'][0]:.5f}, {f['W'][1]:.5f}), P_F - alpha = {f['W'][0]-alpha:+.5f}")
            if len(models) == 2:
                print(f"    Prop. 5: {prop5_two_node(models, A, B, p)['dPF']:+.5f}")
            print(f"    L = ({f['L'][0]:.5f}, {f['L'][1]:.5f}), U = ({f['U'][0]:.5f}, {f['U'][1]:.5f})")
            if f["feasible"]:
                print(f"    adjustment: q* = {f['q']:.4f} -> ({f['point'][0]:.5f}, {f['point'][1]:.5f})")
            else:
                print("    adjustment cannot meet the constraint")
            print(f"    DFC-layer optimum P_D = {f['dfc_layer_PD']:.5f}; downstream optimum P_D = {f['downstream_PD']:.5f}")
    return out


out = {}
out["running_0.075"] = worked([RUNNING, RUNNING], 0.075, "running model, two nodes (P_F falls)")
out["violating_0.21125"] = worked([VIOLATING, VIOLATING], 0.21125, "constraint-violating model, two nodes (P_F rises)")

# 5b: three-node running model, every susceptible hull edge (mid-edge alpha)
print("\n=== 5b: three-node running model, hull edges with a susceptible design")
models3 = [RUNNING] * 3
arr, codes, tables = enumerate_nested_arrays(models3)
hull = hull_of_array(arr)
out["three_node"] = []
for (x1, y1), (x2, y2) in zip(hull[:-1], hull[1:]):
    alpha = round(0.5 * (x1 + x2), 10)
    des = np_design(models3, alpha)
    if des["deterministic_at_alpha"] or not des["switching"]:
        continue
    if worst_single_shift(models3, des["sA"], des["sB"], des["p"]) <= TOL:
        continue
    out["three_node"].append(worked(models3, alpha, "running model, three nodes"))

# 5c: statistics over random models
print("\n=== 5c: random Dirichlet(1) K = 3 models, one failing node, cases = (design, failing node)")
rng = np.random.default_rng(5)
stats = {}
for n in (2, 3, 4):
    rec = []; designs = 0; susceptible = 0; n_sw = []
    while len(rec) < N_CASES and designs < 1000000:
        models = family_models("dirichlet3", n, rng, identical=False)
        alpha = float(rng.uniform(0.02, 0.3))
        des = np_design(models, alpha, compute_G=True)
        if des["deterministic_at_alpha"]:
            continue
        designs += 1
        A, B, p = des["sA"], des["sB"], des["p"]
        cases = []
        for k in des["switching"]:
            f = failure(models, A, B, p, alpha, k)
            if np.hypot(*f["shift"]) > TOL:
                cases.append(f)
        if not cases:
            continue
        susceptible += 1; n_sw.append(len(cases))
        for f in cases:
            rec.append({"alpha": alpha, "n": n, "k": f["k"], "C_D": des["C"][1], "G_D": des["G"][1], "W": f["W"], "overshoot": f["W"][0] - alpha,
                        "dfc_switches": f["dfc_switches"], "feasible": f["feasible"],
                        "q_PD": f["point"][1] if f["feasible"] else None,
                        "q_PF": f["point"][0] if f["feasible"] else None,
                        "dfc_PD": f["dfc_layer_PD"], "down_PD": f["downstream_PD"]})
    rise = [r for r in rec if r["overshoot"] > 1e-12]
    fall = [r for r in rec if r["overshoot"] < -1e-12]
    sw = [r for r in rec if r["dfc_switches"]]
    nosw = [r for r in rec if not r["dfc_switches"]]
    # the adjustment 'acts' when the rule of the final LD switches; then it either meets the constraint or not
    sw_feas = [r for r in sw if r["feasible"]]
    gap_dfc = np.array([r["dfc_PD"] - r["q_PD"] for r in sw_feas])
    gap_down = np.array([r["down_PD"] - r["q_PD"] for r in sw_feas])
    frac = lambda sel, tot: (len(sel) / len(tot)) if tot else None
    stats[n] = {"designs": designs, "susceptible_designs": susceptible, "cases": len(rec),
                "frac_designs_susceptible": susceptible / designs,
                "mean_failing_nodes_per_susceptible_design": float(np.mean(n_sw)),
                "frac_PF_rises": frac(rise, rec), "frac_PF_falls": frac(fall, rec),
                "frac_dfc_switches": frac(sw, rec),
                "frac_feasible_given_dfc_switches": frac(sw_feas, sw),
                "frac_feasible_given_rise_and_dfc_switches": frac([r for r in sw_feas if r["overshoot"] > 1e-12], [r for r in sw if r["overshoot"] > 1e-12]),
                "frac_constraint_violated_no_switch": frac([r for r in nosw if r["overshoot"] > 1e-12], nosw),
                "frac_constraint_restorable_all_rises": frac([r for r in rise if r["dfc_switches"] and r["feasible"]], rise),
                "frac_q_equals_dfc": float(np.mean(gap_dfc <= 1e-9)) if len(gap_dfc) else None,
                "frac_q_equals_down": float(np.mean(gap_down <= 1e-9)) if len(gap_down) else None,
                "gap_dfc_minus_q": summarize(gap_dfc), "gap_down_minus_q": summarize(gap_down),
                "min_gap_dfc": float(gap_dfc.min()) if len(gap_dfc) else None,
                "abs_overshoot": summarize(np.abs([r["overshoot"] for r in rec])),
                "loss_uncorrected_PD": summarize(np.array([r["C_D"] - r["W"][1] for r in rec])),
                "residual_loss_after_q": summarize(np.array([r["C_D"] - r["q_PD"] for r in sw_feas])),
                # the deterministic optimum G needs no common draw: how often does a correction at the
                # final LD end below it?  (W itself is compared only when it meets the constraint)
                "frac_q_below_G": frac([r for r in sw_feas if r["q_PD"] < r["G_D"] - 1e-9], sw_feas),
                "frac_dfc_below_G": frac([r for r in rec if r["dfc_PD"] < r["G_D"] - 1e-9], rec),
                "frac_down_below_G": frac([r for r in rec if r["down_PD"] < r["G_D"] - 1e-9], rec),
                "randomization_gain": summarize(np.array([r["C_D"] - r["G_D"] for r in rec])),
                "dfc_minus_G": summarize(np.array([r["dfc_PD"] - r["G_D"] for r in rec])),
                "by_node": {str(k): {"cases": sum(r["k"] == k for r in rec),
                                     "frac_PF_rises": frac([r for r in rise if r["k"] == k], [r for r in rec if r["k"] == k])}
                            for k in range(1, n)}}
    s = stats[n]
    print(f"  n={n}: {s['designs']} designs, {s['susceptible_designs']} susceptible ({s['frac_designs_susceptible']:.2f}); {s['cases']} cases; "
          f"P_F rises in {s['frac_PF_rises']:.2f}, falls in {s['frac_PF_falls']:.2f}; median |P_F shift| {s['abs_overshoot']['median']:.4f}")
    print(f"        rule of the final LD switches in {s['frac_dfc_switches']:.2f} of the cases; the adjustment then meets the constraint in "
          f"{s['frac_feasible_given_dfc_switches']:.2f}; all rises restorable: {s['frac_constraint_restorable_all_rises']}")
    print(f"        among those: equals the DFC-layer optimum in {s['frac_q_equals_dfc']}, the downstream optimum in {s['frac_q_equals_down']}; "
          f"gap to DFC-layer median {s['gap_dfc_minus_q']['median']:.2e} max {s['gap_dfc_minus_q']['max']:.2e} (min {s['min_gap_dfc']:.1e}); "
          f"to downstream median {s['gap_down_minus_q']['median']:.2e} max {s['gap_down_minus_q']['max']:.2e}")
    print(f"        cases by failing node: {s['by_node']}")
    print(f"        corrected P_D below the deterministic optimum G: adjustment {s['frac_q_below_G']:.2f} (of the cases it can restore), "
          f"DFC-layer optimum {s['frac_dfc_below_G']:.2f}, downstream optimum {s['frac_down_below_G']:.2f}; "
          f"median gain of randomization C_D - G_D {s['randomization_gain']['median']:.4f}")
out["stats"] = stats
save_json("exp5_correction.json", out)
