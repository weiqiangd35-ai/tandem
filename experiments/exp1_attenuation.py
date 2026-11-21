"""Experiment 1 - attenuation factors across chain lengths, observation models and
threshold configurations (Theorem 1; reviewers 1, 2 and the AE).

For every configuration we record the per-node factors (rho_F, rho_D) of nodes 2..n, the
end-to-end product prod rho_F, and (as a numerical check of the linear transfer) the
largest difference, over the stages, between the exact perturbation produced by a wrong
rule at N_1 and the product transfer of Theorem 1.  A random configuration is checked
against one other rule of N_1 drawn at random (its perturbations are stored); an optimal
or adversarial one against every other non-trivial level of N_1 (continuous families:
false-alarm rates 0.1, 0.5 and 0.9), without any random draw.

Families
  discrete  : running (K=3), dirichlet3 / dirichlet5 / dirichlet10 (one model per node),
              qgauss1 / qgauss2 (8-level quantized Gaussian, d' = 1, 2)
  continuous: Gaussian d' in {0.5, 1, 2, 3}, Swerling-I power ROC with SNR in {1, 3, 10}
              (beta = 1/(1+SNR); identical to an exponential shift-in-rate model with
              beta = l1/l0), logistic mu = 2
Threshold modes
  random     : every node uniform over its realizable non-trivial region pairs
               (continuous: node 1 false-alarm rate f ~ U(0, 1); node k >= 2 false-alarm rate
               on message 0, a_0 ~ U(0, 1), which fixes eta_k and, through eq. (3), a_1; no
               node is trivial, as in the discrete families)
  optimal    : the two supporting strategies of the NP-optimal design at alpha in
               {0.05, 0.1, 0.2}, in the realization with the fewest switching nodes
               (exhaustive for n <= 5 and K <= 5, Lagrangian PBPO with aligned irrelevant
               nodes for n = 10); continuous families: Lagrangian PBPO at lambda in
               {0.5, 2, 8} with the closed-form best responses of Section S3, iterated to a
               fixed point from the design of a grid search and from five random starting
               points (the six fixed points are compared; the row keeps the best one)
  adversarial: greedy per-node choice maximizing (relay-like) or minimizing rho_F
Sample sizes: 300 random configurations per family and chain length; 1000 models per cell
for the adversaries; 300 models per cell for the optimal designs of the Dirichlet families
(60 for n = 10).  The models of a cell come from three generators: the first 30 (optimal
designs) or 10 (adversaries) from the main one, which also draws the random configurations;
the other models of the optimal designs and the adversaries' models up to the hundredth from
a second one; the adversaries' last 900 models from a third.  The subsamples of 10, 30 and
100 models are therefore nested in the full sample: over the first 100 models the
adversaries' medians differ from those over 1000 by up to 0.05, e.g. 0.84 against 0.82 for
Dirichlet K=3 at n = 2 (experiments/sampling_error.py; RESULTS.md, Section 1).

The greedy adversaries choose among the pairs that the random configurations draw from:
the pairs that ignore the message (R(0) = R(1)) and the pure relay (R(0) empty, R(1) the
whole alphabet, rho_F = rho_D = 1) are excluded whenever another pair is realizable.
Option --relay replays the random streams of the discrete families without recomputing
the optimal designs, checks that the random and adversarial rows so obtained equal those
of results/exp1_rows.json, and adds, on the models and first-node rules of the relay-like
adversary, the same greedy choice over all realizable pairs, the pure relay included.  It
writes results/exp1_relay.json (about 2 minutes).
Usage: python experiments/exp1_attenuation.py [--quick] [--relay]
"""
import sys, time
import numpy as np
from scipy.stats import binom
from common import *
from tandem_sim.continuous import (GaussianShift, PowerROC, LogisticShift, strategy_from_etas,
                                   evaluate_chain_cont, pbpo_cont, pbpo_cont_exact, chain_factors_cont, ContStrategy)
from tandem_sim.discrete import chain_factors, realizable_pairs

QUICK = "--quick" in sys.argv
RELAY = "--relay" in sys.argv
rng = np.random.default_rng(1)
NS = (2, 3, 5, 10, 20)
N_CFG = 40 if QUICK else 300          # random configurations per (family, n)
N_OPT = 5 if QUICK else 300           # models per (family, n) for the optimal designs of the Dirichlet families
N_ADV = 3 if QUICK else 1000          # models per (family, n) for the greedy adversaries
N_OPT0, N_ADV0 = (5, 3) if QUICK else (30, 10)   # models of a cell drawn from the main generator
N_ADV1 = 3 if QUICK else 100          # adversaries: models up to this index come from the first two generators
rng2 = np.random.default_rng(101)     # generator of the additional models
rng3 = np.random.default_rng(202)     # generator of the adversaries' models beyond the first 100
DISCRETE = ["running", "dirichlet3", "dirichlet5", "dirichlet10", "qgauss1", "qgauss2"]
CONTINUOUS = {"gauss0.5": GaussianShift(0.5), "gauss1": GaussianShift(1.0), "gauss2": GaussianShift(2.0),
              "gauss3": GaussianShift(3.0), "swerling_snr1": PowerROC(snr=1.0), "swerling_snr3": PowerROC(snr=3.0),
              "swerling_snr10": PowerROC(snr=10.0), "logistic2": LogisticShift(2.0)}
rows = []
relay_rows = []                       # --relay: the greedy adversary with the pure relay allowed


def random_cont_strategy(models, rng):
    """Random configuration of a continuous chain with no trivial node: node 1 at a false-alarm
    rate f ~ U(0, 1); node k >= 2 with a_0 ~ U(0, 1) (false-alarm rate of its region for message
    0), the threshold eta_k that realizes it at the node's upstream point, and a_1 from eq. (3).
    This is the continuous counterpart of 'uniform over the realizable region pairs'."""
    f1 = float(rng.uniform(0.0, 1.0))
    f, d = models[0].local_point(f1)
    pairs = []
    for m in models[1:]:
        a0 = float(rng.uniform(0.0, 1.0))
        if abs(f - d) < 1e-15:                 # constant or uninformative message: nothing to fuse
            a1 = a0
        elif d >= 1.0:                         # message 0 excludes H1: R(0) is empty, R(1) is free
            a0, a1 = 0.0, a0
        elif f <= 0.0:                         # message 1 excludes H0: R(1) is the whole space
            a1 = 1.0
        else:
            tau1 = float(m.rprime(a0)) * (f * (1 - d)) / (d * (1 - f))   # tau(1)/tau(0) = [f/d] / [(1-f)/(1-d)]
            a1 = max(a0, float(m.astar(tau1)))
        pairs.append((a0, a1))
        f, d = m.node_output(a0, a1, f, d)
    return ContStrategy(f1, tuple(pairs))


def rec(family, n, mode, factors, extra=None):
    rF = np.array([f[0] for f in factors]); rD = np.array([f[1] for f in factors])
    row = {"family": family, "n": n, "mode": mode, "rho_F": rF.tolist(), "rho_D": rD.tolist(),
           "prod_rho_F": float(np.prod(rF)), "prod_rho_D": float(np.prod(rD)),
           "max_abs_rho": float(max(np.abs(rF).max(), np.abs(rD).max()))}
    if extra:
        row.update(extra)
    rows.append(row)


def transfer_err(models, A, fac):
    """Largest difference, over the other non-trivial levels of N_1 and over the stages, between the
    exact perturbation of the operating points and the product transfer (discrete families)."""
    alts = [j for j in models[0].levels if 0 < j < models[0].K and j != A.j1]
    if not alts:
        return {}
    pA = evaluate_chain(models, A); cp = np.cumprod([[1, 1]] + [[f[0], f[1]] for f in fac], axis=0); err = 0.0
    for j in alts:
        pB = evaluate_chain(models, Strategy(int(j), A.pairs))
        d = np.array([(b[0] - a[0], b[1] - a[1]) for a, b in zip(pA, pB)])
        err = max(err, float(np.abs(d - d[0] * cp).max()))
    return {"transfer_check_max_err": err}


def transfer_err_cont(models, A, fac):
    """The same check for the continuous families, with N_1 moved to the false-alarm rates 0.1, 0.5, 0.9."""
    pA = evaluate_chain_cont(models, A); cp = np.cumprod([[1, 1]] + [[f[0], f[1]] for f in fac], axis=0); err = 0.0
    for a1 in (0.1, 0.5, 0.9):
        pB = evaluate_chain_cont(models, ContStrategy(a1, A.pairs))
        d = np.array([(b[0] - a[0], b[1] - a[1]) for a, b in zip(pA, pB)])
        err = max(err, float(np.abs(d - d[0] * cp).max()))
    return {"transfer_check_max_err": err}


t0 = time.time()
# ------------------------------------------------------------- discrete
for family in DISCRETE:
    for n in NS:
        for cfg in range(N_CFG):
            models = family_models(family, n, rng, identical=(family in ("running", "qgauss1", "qgauss2")))
            A = random_strategy(models, rng)
            fac = chain_factors(models, A)
            # wrong rule at N_1: exact per-stage perturbation equals the product transfer
            alt = [j for j in models[0].levels if 0 < j < models[0].K and j != A.j1]
            extra = {}
            if alt:
                B = Strategy(int(rng.choice(alt)), A.pairs)
                pA, pB = evaluate_chain(models, A), evaluate_chain(models, B)
                d = np.array([(b[0] - a[0], b[1] - a[1]) for a, b in zip(pA, pB)])
                pred = d[0] * np.cumprod([[1, 1]] + [[f[0], f[1]] for f in fac], axis=0)
                extra = {"transfer_check_max_err": float(np.abs(d - pred).max()),
                         "injected": d[0].tolist(), "output": d[-1].tolist()}
            rec(family, n, "random", fac, extra)
        # adversarial (greedy relay-like / least message-sensitive)
        for cfg in range(N_ADV):
            g = rng if cfg < N_ADV0 else (rng2 if cfg < N_ADV1 else rng3)
            models = family_models(family, n, g, identical=(family in ("running", "qgauss1", "qgauss2")))
            for mode, pick in (("adversarial_max", max), ("adversarial_min", min)):
                j = int(g.choice([j for j in models[0].levels if 0 < j < models[0].K]))
                f, d = models[0].region_probs(j); pairs = []
                for m in models[1:]:
                    cands = realizable_pairs(m, f, d)
                    nt = [(a, b) for (a, b) in cands if a != b and not (a == m.K and b == 0)] or cands
                    pr = pick(nt, key=lambda ab: m.attenuation(*ab)[0])
                    pairs.append(pr); f, d = m.node_output(pr[0], pr[1], f, d)
                S_adv = Strategy(j, tuple(pairs)); fac = chain_factors(models, S_adv)
                rec(family, n, mode, fac, transfer_err(models, S_adv, fac))
                if RELAY and mode == "adversarial_max":
                    f, d = models[0].region_probs(j); pairs = []
                    for m in models[1:]:
                        pr = max(realizable_pairs(m, f, d), key=lambda ab: m.attenuation(*ab)[0])
                        pairs.append(pr); f, d = m.node_output(pr[0], pr[1], f, d)
                    fac = chain_factors(models, Strategy(j, tuple(pairs)))
                    relay_rows.append({"family": family, "n": n, "prod_rho_F": float(np.prod([x[0] for x in fac])),
                                       "prod_rho_F_restricted": rows[-1]["prod_rho_F"],
                                       "relays": int(sum(a == m.K and b == 0 for (a, b), m in zip(pairs, models[1:])))})
        # optimal designs: exhaustive hull for n <= 5 and K <= 5; Lagrangian PBPO (few
        # multipliers, 5 restarts) for n = 10 on the K = 3 families only
        K_fam = {"running": 3, "dirichlet3": 3, "dirichlet5": 5, "dirichlet10": 10, "qgauss1": 8, "qgauss2": 8}[family]
        if (n in (2, 3, 5) and K_fam <= 5) or (n == 10 and K_fam == 3):
            n_opt0 = N_OPT0 if n <= 5 else max(2, N_OPT0 // 5)
            n_opt = n_opt0 if family == "running" else (N_OPT if n <= 5 else max(2, N_OPT // 5))   # the running model is fixed
            for cfg in range(n_opt):
                g = rng if cfg < n_opt0 else rng2
                models = family_models(family, n, g, identical=(family in ("running", "qgauss1", "qgauss2")))
                if RELAY:
                    continue                                   # the models are drawn, the designs are not recomputed
                if n <= 5:
                    arr, codes, tables = enumerate_nested_arrays(models)
                    hull = hull_of_array(arr)
                else:
                    hull, items = hull_by_pbpo(models, np.exp(np.linspace(-3, 4, 8)), restarts=5, seed=cfg)
                for alpha in (0.05, 0.1, 0.2):
                    try:
                        sp = support_pair(hull, alpha)
                    except ValueError:
                        continue
                    if sp.deterministic:
                        pair = ((decode_strategy(realizations_of(arr, codes, sp.A)[0], tables),) if n <= 5
                                else (strategies_at(items, sp.A)[0],))
                    elif n <= 5:
                        pair = min_switch_realizations(arr, codes, tables, sp.A, sp.B,
                                                       score=lambda a, b: worst_single_shift(models, a, b, sp.p))[:2]
                    else:
                        pair = align_pair(models, strategies_at(items, sp.A)[0], strategies_at(items, sp.B)[0])
                    for s in pair:
                        fac = chain_factors(models, s)
                        rec(family, n, "optimal", fac, {"alpha": alpha, **transfer_err(models, s, fac)})
        print(f"  {family:12s} n={n:2d} done ({time.time()-t0:.0f}s)", flush=True)

if RELAY:
    stored = [r for r in load_json("exp1_rows.json") if r["mode"] in ("random", "adversarial_max", "adversarial_min") and r["family"] in DISCRETE]
    same = len(stored) == len(rows) and all(a["family"] == b["family"] and a["n"] == b["n"] and a["mode"] == b["mode"] and a["rho_F"] == b["rho_F"]
                                            for a, b in zip(stored, rows))
    print(f"replayed rows (random, adversarial) equal to results/exp1_rows.json: {same} ({len(rows)} rows)")
    assert same

    def med(x):                                                # median and distribution-free 95% interval (order statistics)
        xs = np.sort(np.asarray(x, float)); N = len(xs)
        l = max(k for k in range(1, N // 2 + 1) if binom.cdf(N - k, N, 0.5) - binom.cdf(k - 1, N, 0.5) >= 0.95)
        return {"median": float(np.median(xs)), "interval": [float(xs[l - 1]), float(xs[N - l])]}

    out = {"replay_equals_stored_rows": bool(same), "rows": relay_rows}
    print(f"\nmedian end-to-end product of the greedy relay-like adversary, {N_ADV} models per cell")
    print("family        n   pure relay excluded      pure relay allowed       nodes that relay")
    for family in DISCRETE:
        for n in NS:
            sel = [r for r in relay_rows if r["family"] == family and r["n"] == n]
            a, b = med([r["prod_rho_F_restricted"] for r in sel]), med([r["prod_rho_F"] for r in sel])
            frac = float(np.mean([r["relays"] / (n - 1) for r in sel]))
            out[f"{family}|n={n}"] = {"models": len(sel), "excluded": a, "allowed": b, "frac_relay_nodes": frac,
                                      "chains_with_a_relay": int(sum(r["relays"] > 0 for r in sel))}
            print(f"{family:13s} {n:3d}   {a['median']:.3f} [{a['interval'][0]:.3f}, {a['interval'][1]:.3f}]   "
                  f"{b['median']:.3f} [{b['interval'][0]:.3f}, {b['interval'][1]:.3f}]   {frac:.2f}")
    save_json("exp1_relay.json", out)
    sys.exit(0)

# ----------------------------------------------------------- continuous
for family, m in CONTINUOUS.items():
    for n in NS:
        models = [m] * n
        for cfg in range(N_CFG):
            A = random_cont_strategy(models, rng)
            fac = chain_factors_cont(models, A)
            a1b = float(rng.uniform(0.0, 1.0))
            B = ContStrategy(a1b, A.pairs)
            pA, pB = evaluate_chain_cont(models, A), evaluate_chain_cont(models, B)
            d = np.array([(b[0] - a[0], b[1] - a[1]) for a, b in zip(pA, pB)])
            pred = d[0] * np.cumprod([[1, 1]] + [[f[0], f[1]] for f in fac], axis=0)
            rec(family, n, "random", fac, {"transfer_check_max_err": float(np.abs(d - pred).max()),
                                            "injected": d[0].tolist(), "output": d[-1].tolist()})
        if n in (2, 3, 5) and not QUICK:
            for lam in (0.5, 2.0, 8.0):
                rs = np.random.default_rng(7)
                g, gpt, _, _ = pbpo_cont(models, lam=lam, rng=rs, a_grid=np.linspace(0.002, 0.998, 120),
                                         eta_grid=np.exp(np.linspace(-6, 6, 97)), max_sweeps=12)
                # The grid search only gives a starting point; the design is the fixed point of the exact best responses.
                # Five more starting points: false-alarm rates drawn uniformly, sorted within a node.  (Started from
                # random configurations of the 'random' mode instead, the iteration stops in three logistic designs at
                # lambda = 8, above the largest likelihood ratio e^2 of that model, at the chain that never decides 1.)
                starts = [g] + [ContStrategy(float(rs.random()), tuple(tuple(sorted(rs.random(2).tolist())) for _ in range(n - 1)))
                                for _ in range(5)]
                runs = [pbpo_cont_exact(models, lam, init) for init in starts]
                obj = [q[1][1] - lam * q[1][0] for q in runs]
                s, pt, sw, conv = runs[int(np.argmax(obj))]
                par = lambda q: np.array([q[0].a1] + [x for pr in q[0].pairs for x in pr])
                same = sum(abs(o - max(obj)) < 1e-10 and np.abs(par(q) - par((s,))).max() < 1e-8 for q, o in zip(runs, obj))
                fac = chain_factors_cont(models, s)
                rec(family, n, "optimal", fac, {"lam": lam, "point": pt, "sweeps": sw, "converged": conv, "starts": len(runs),
                                                "starts_at_this_point": int(same), "objective": float(max(obj)),
                                                "gain_over_grid_search": float(max(obj) - (gpt[1] - lam * gpt[0])),
                                                **transfer_err_cont(models, s, fac)})
        print(f"  {family:14s} n={n:2d} done ({time.time()-t0:.0f}s)", flush=True)

save_json("exp1_rows.json", rows)

# ---------------------------------------------------------------- summary
summary = {"max_abs_rho_over_all_rows": max(r["max_abs_rho"] for r in rows),
           "rows_with_abs_rho_above_1": sum(r["max_abs_rho"] > 1 + 1e-12 for r in rows),
           "transfer_check_max_err": max(r.get("transfer_check_max_err", 0) for r in rows),
           "rows_with_transfer_check": sum("transfer_check_max_err" in r for r in rows), "rows": len(rows)}
print("\nmax |rho| over all rows:", summary["max_abs_rho_over_all_rows"], " rows with |rho|>1:", summary["rows_with_abs_rho_above_1"])
print("max |exact per-stage perturbation - product transfer|:", summary["transfer_check_max_err"],
      f"({summary['rows_with_transfer_check']} of {summary['rows']} rows checked)")
co = [r for r in rows if r["mode"] == "optimal" and r["family"] in CONTINUOUS]
if co:
    summary["continuous_optimal"] = {"designs": len(co), "converged": sum(r["converged"] for r in co),
                                     "starts": co[0]["starts"],
                                     "designs_with_all_starts_at_one_point": sum(r["starts_at_this_point"] == r["starts"] for r in co),
                                     "fewest_starts_at_the_design": min(r["starts_at_this_point"] for r in co),
                                     "largest_number_of_sweeps": max(r["sweeps"] for r in co),
                                     "gain_over_grid_search": {"min": min(r["gain_over_grid_search"] for r in co),
                                                               "max": max(r["gain_over_grid_search"] for r in co)}}
    c = summary["continuous_optimal"]
    print(f"continuous families, optimal designs: {c['designs']} designs, {c['converged']} converged (at most {c['largest_number_of_sweeps']} sweeps); "
          f"the {c['starts']} starting points give one fixed point in {c['designs_with_all_starts_at_one_point']} of them; "
          f"P_D - lambda P_F exceeds that of the grid search by {c['gain_over_grid_search']['min']:.1e} to {c['gain_over_grid_search']['max']:.1e}")
print("\nfamily        n   mode            count  rho_F median  p10    p90   frac rho_F>0.9   prod rho_F median  p90")
for family in DISCRETE + list(CONTINUOUS):
    for n in NS:
        for mode in ("random", "optimal", "adversarial_max", "adversarial_min"):
            sel = [r for r in rows if r["family"] == family and r["n"] == n and r["mode"] == mode]
            if not sel:
                continue
            rF = np.concatenate([r["rho_F"] for r in sel])
            prod = np.array([r["prod_rho_F"] for r in sel])
            key = f"{family}|n={n}|{mode}"
            summary[key] = {"count": len(sel), "rho_F": summarize(rF), "frac_rho_F_gt_0.9": float(np.mean(rF > 0.9)),
                            "prod_rho_F": summarize(prod)}
            s = summary[key]
            print(f"{family:13s} {n:3d}  {mode:15s} {len(sel):5d}  {s['rho_F']['median']:.3f}        {s['rho_F']['p10']:.3f}  {s['rho_F']['p90']:.3f}   "
                  f"{s['frac_rho_F_gt_0.9']:.2f}            {s['prod_rho_F']['median']:.2e}         {s['prod_rho_F']['p90']:.2e}")
save_json("exp1_summary.json", summary)
