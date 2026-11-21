"""Experiment 2 - tandem versus parallel under single-node synchronization loss, with a
paired protocol (Section IV-D of the revision).

Protocol.  For every sampled observation model (one Dirichlet(1) model shared by all
nodes), chain length n and budget alpha:
  * tandem  : dependent-randomization design = supporting pair (A, B) of the upper hull
              of all deterministic chain points (exhaustive over all threshold strategies),
              mixing probability p;
  * parallel: same observation model, same alpha; LD thresholds and the fusion rule
              enumerated exhaustively (fusion = every likelihood-ratio test on the local
              decisions, 'cv'; or k-out-of-n only, 'kofn'); supporting pair and p from the
              upper hull in the same way.
  * realization: a hull vertex usually has several realizations (a node whose message is
    ignored downstream, or an LD ignored by the fusion rule, may use any rule; identical
    sensors can exchange roles).  Both architectures use the pair of realizations of A and
    B with the fewest switching nodes other than the fusion node (exhaustive search), ties
    broken by the smallest worst-case shift over single-node failures.
    --first-realization uses the first enumerated realization instead (for comparison).
  * one node draws independently with the same marginal p.  Tandem: every node whose
    rule switches between A and B (excluding the DFC), parallel: every LD whose rule
    switches.  The reported shift is the maximum over those nodes (worst single-node
    failure) and the mean.  Designs in which no upstream node / LD switches need no
    synchronization and are excluded (their count is reported).
  * shift = |W - C| (Euclidean) and |W_F - alpha|.
Statistics (all on susceptible designs, i.e. shift > TOL = 1e-9; a switching node whose
desynchronization leaves the operating point unchanged is immune, Corollary 1, and is
counted in the immunity fractions instead): paired ratios tandem/parallel; median with
bootstrap 95 % CI (2000 resamples of whole models, since the designs of one model at the
three values of alpha are not independent); Wilcoxon signed-rank on log ratios (pairs taken
as independent; the same test with one value per model is stored next to it); Mann-Whitney
U on the two shift samples; P(ratio < 1).  Position-resolved: the tandem shift at node k is
paired with the mean shift over the susceptible LDs of the parallel design of the same model
and alpha.
The sample is 1000 models; the position-resolved median ratios are also given for each of
the five disjoint blocks of 200 models, because a sample of 200 models leaves the medians
at some positions uncertain by a factor of about two (key block_median_ratios).
Usage: python experiments/exp2_tandem_vs_parallel.py [--quick] [--summary-only] [--seed=K] [--family=F]
       (--quick runs 30 models with n = 3, 4 and writes files with the suffix _quick;
        --summary-only recomputes the statistics from results/exp2_rows.json;
        --seed=K draws an independent sample of models and writes exp2_rows_seedK.json and
        exp2_summary_seedK.json; the paper uses the default seed, 2;
        --family=qgauss3:LO:HI replaces the Dirichlet(1) models by three-level quantizations of a
        Gaussian shift-in-mean observation, N(0,1) versus N(d',1), with d' uniform on [LO, HI] and
        the two cut points uniform on [-1, d'+1]; the files carry the suffix _qgauss3_LO_HI)
"""
import sys, time
import numpy as np
from scipy import stats as st
from common import *
from tandem_sim.parallel import parallel_design, parallel_desync, min_switch_parallel
from tandem_sim.desync import worst_single_shift

QUICK = "--quick" in sys.argv
SUMMARY_ONLY = "--summary-only" in sys.argv
FIRST = "--first-realization" in sys.argv
SEED = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--seed=")), 2)
FAMILY = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--family=")), "dirichlet3")
TAG = ("_first" if FIRST else "") + (f"_seed{SEED}" if SEED != 2 else "") + ("" if FAMILY == "dirichlet3" else "_" + FAMILY.replace(":", "_")) + ("_quick" if QUICK else "")
ROWS_FILE = f"exp2_rows{TAG}.json"
SUMMARY_FILE = f"exp2_summary{TAG}.json"
TOL = 1e-9          # a shift below TOL is numerically zero: the design is immune to that failure
rng = np.random.default_rng(SEED)
N_MODELS = 30 if QUICK else 1000
BLOCK = 200         # the position statistics are also reported per block of BLOCK consecutive models
NS = (3, 4) if QUICK else (3, 4, 5, 6)
ALPHAS = (0.05, 0.1, 0.2)
rows = []


def draw_model():
    """One observation model, shared by all nodes of a chain and by all LDs of the parallel system."""
    if FAMILY == "dirichlet3":
        return family_models("dirichlet3", 1, rng)[0]
    name, lo, hi = FAMILY.split(":")
    if name != "qgauss3":
        raise ValueError(FAMILY)
    from tandem_sim.discrete import quantized_gaussian_cuts
    d = rng.uniform(float(lo), float(hi))
    return quantized_gaussian_cuts(d, rng.uniform(-1.0, d + 1.0, size=2))


capped = {"tandem": 0, "parallel": 0}
ties = {"tandem": [], "parallel": []}
t0 = time.time()
if SUMMARY_ONLY:
    rows = load_json(ROWS_FILE)
    NS = tuple(sorted(set(r["n"] for r in rows)))
for mi in (range(0) if SUMMARY_ONLY else range(N_MODELS)):
    model = draw_model()
    for n in NS:
        models = [model] * n
        # enumerate once per (model, n): tandem points and parallel points per fusion class
        t_arr, t_codes, t_tables = enumerate_nested_arrays(models)
        t_hull = hull_of_array(t_arr)
        t_items = enumerate_system_points(models) if FIRST else None
        from tandem_sim.parallel import enumerate_parallel
        p_items = {fus: enumerate_parallel(models, fus) for fus in ("cv", "kofn")}
        p_hull = {fus: upper_hull([pt for pt, _ in p_items[fus]]) for fus in p_items}
        for alpha in ALPHAS:
            sp_t = support_pair(t_hull, alpha)
            if FIRST:
                sA_t, sB_t = strategies_at(t_items, sp_t.A)[0], strategies_at(t_items, sp_t.B)[0]
            elif sp_t.deterministic:
                sA_t = sB_t = decode_strategy(realizations_of(t_arr, t_codes, sp_t.A)[0], t_tables)
            else:
                info_t = {}
                sA_t, sB_t, _ = min_switch_realizations(t_arr, t_codes, t_tables, sp_t.A, sp_t.B, info=info_t,
                                                        score=lambda a, b: worst_single_shift(models, a, b, sp_t.p))
                capped["tandem"] += int(info_t["capped"]); ties["tandem"].append(info_t["ties"])
            td = {"sA": sA_t, "sB": sB_t, "p": sp_t.p, "A": sp_t.A, "B": sp_t.B, "C": sp_t.C,
                  "deterministic_at_alpha": sp_t.deterministic, "classes": classify_pair(sA_t, sB_t)}
            row = {"model": mi, "n": n, "alpha": alpha,
                   "tandem_C_D": td["C"][1],
                   "tandem_deterministic": td["deterministic_at_alpha"],
                   "tandem_terminal_only": td["classes"]["terminal_only"],
                   "tandem_n_switching_all": len(switching_nodes(sA_t, sB_t, include_last=True))}
            A, B, p = td["sA"], td["sB"], td["p"]
            switching = switching_nodes(A, B)
            if switching and not td["deterministic_at_alpha"]:
                shifts = []
                wrong = []
                ptsA = evaluate_chain(models, A)
                for k in switching:
                    W, _ = desync_point(models, A, B, p, [k])
                    shifts.append((float(np.hypot(W[0] - td["C"][0], W[1] - td["C"][1])), abs(W[0] - alpha), k))
                    # deterministic 'wrong rule' perturbation: chain A with node k using B's rule
                    assign = ["A"] * n; assign[k - 1] = "B"
                    from tandem_sim.desync import evaluate_assignment
                    Ap = evaluate_assignment(models, A, B, assign)[-1]
                    wrong.append(float(np.hypot(Ap[0] - ptsA[-1][0], Ap[1] - ptsA[-1][1])))
                row["tandem_shift_max"] = max(s[0] for s in shifts)
                row["tandem_shift_mean"] = float(np.mean([s[0] for s in shifts]))
                row["tandem_dPF_max"] = max(s[1] for s in shifts)
                row["tandem_n_switching"] = len(switching)
                row["tandem_shift_N1"] = next((s[0] for s in shifts if s[2] == 1), None)
                row["tandem_shift_by_k"] = {str(s[2]): s[0] for s in shifts}
                row["tandem_dPF_by_k"] = {str(s[2]): s[1] for s in shifts}
                row["tandem_wrongrule_max"] = max(wrong)
                row["tandem_wrongrule_N1"] = wrong[0] if switching[0] == 1 else None
            for fus in ("cv", "kofn"):
                sp = support_pair(p_hull[fus], alpha)
                at = lambda point: [s for pt, s in p_items[fus] if abs(pt[0] - point[0]) <= 1e-9 and abs(pt[1] - point[1]) <= 1e-9]
                if FIRST or sp.deterministic:
                    sA, sB = at(sp.A)[0], at(sp.B)[0]
                    sw = [i for i in range(n) if sA.levels[i] != sB.levels[i]]
                else:
                    info_p = {}
                    sA, sB, sw = min_switch_parallel(models, p_items[fus], sp.A, sp.B, p=sp.p, info=info_p)
                    capped["parallel"] += int(info_p["capped"]); ties["parallel"].append(info_p["ties"])
                row[f"parallel_{fus}_C_D"] = sp.C[1]
                row[f"parallel_{fus}_deterministic"] = sp.deterministic
                if sw and not sp.deterministic:
                    from tandem_sim.parallel import system_point
                    shifts = [parallel_desync(models, sA, sB, sp.p, i) for i in sw]
                    row[f"parallel_{fus}_shift_max"] = max(s["norm"] for s in shifts)
                    row[f"parallel_{fus}_shift_mean"] = float(np.mean([s["norm"] for s in shifts]))
                    row[f"parallel_{fus}_dPF_max"] = max(abs(s["W"][0] - alpha) for s in shifts)
                    row[f"parallel_{fus}_n_switching"] = len(sw)
                    row[f"parallel_{fus}_shift_by_i"] = {str(i): sh["norm"] for i, sh in zip(sw, shifts)}
                    row[f"parallel_{fus}_dPF_by_i"] = {str(i): abs(sh["W"][0] - alpha) for i, sh in zip(sw, shifts)}
                    Apt = system_point(models, sA.levels, sA.fusion)
                    wrong = []
                    for i in sw:
                        lv = list(sA.levels); lv[i] = sB.levels[i]
                        Ap = system_point(models, lv, sA.fusion)
                        wrong.append(float(np.hypot(Ap[0] - Apt[0], Ap[1] - Apt[1])))
                    row[f"parallel_{fus}_wrongrule_max"] = max(wrong)
            rows.append(row)
    if (mi + 1) % 10 == 0:
        print(f"  {mi+1}/{N_MODELS} models, {time.time()-t0:.0f}s")
if not SUMMARY_ONLY:
    save_json(ROWS_FILE, rows)
    if not FIRST:
        print("tied realization pairs scored per design: tandem median %.0f max %d, parallel median %.0f max %d; cap reached: tandem %d, parallel %d"
              % (np.median(ties["tandem"]), max(ties["tandem"]), np.median(ties["parallel"]), max(ties["parallel"]), capped["tandem"], capped["parallel"]))


def paired_tests(t, q, tol=1e-9):
    """One-sided Wilcoxon signed-rank tests on the log ratios with ties at ratio 1 (|log ratio| < tol,
    i.e. identical shifts up to floating point) set to exact zeros, which the test discards
    (zero_method='wilcox'); returns (p_less, p_greater, fraction of ties)."""
    d = np.log(np.asarray(t) / np.asarray(q))
    d = np.where(np.abs(d) < tol, 0.0, d)
    ties = float(np.mean(d == 0))
    if np.all(d == 0):
        return 1.0, 1.0, ties
    return (float(st.wilcoxon(d, alternative="less", zero_method="wilcox").pvalue),
            float(st.wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue), ties)


def boot_median(x, groups, B=2000, seed=0):
    """Median of x with a 95 % bootstrap interval that resamples whole groups (models): the
    designs of one model at the three values of alpha are not independent.  A resample draws
    the models with replacement; its median is the weighted median of x with the number of
    times each model was drawn as weight (same convention as np.median)."""
    r = np.random.default_rng(seed)
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    xs = x[order]
    _, g = np.unique(np.asarray(groups)[order], return_inverse=True)
    G = int(g.max()) + 1
    meds = np.empty(B)
    for b in range(B):
        w = r.multinomial(G, np.full(G, 1.0 / G))[g]       # weight of each sorted value
        c = np.cumsum(w)
        T = int(c[-1])
        lo = xs[np.searchsorted(c, (T - 1) // 2, side="right")]
        hi = xs[np.searchsorted(c, T // 2, side="right")]
        meds[b] = 0.5 * (lo + hi)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def by_model_tests(t, q, groups, tol=1e-9):
    """The same one-sided signed-rank tests with one value per model (the mean log ratio of its pairs)."""
    d = np.log(np.asarray(t) / np.asarray(q))
    d = np.where(np.abs(d) < tol, 0.0, d)
    _, g = np.unique(np.asarray(groups), return_inverse=True)
    m = np.bincount(g, weights=d) / np.bincount(g)
    if np.all(m == 0):
        return 1.0, 1.0, int(len(m))
    return (float(st.wilcoxon(m, alternative="less", zero_method="wilcox").pvalue),
            float(st.wilcoxon(m, alternative="greater", zero_method="wilcox").pvalue), int(len(m)))


summary = {"tol": TOL}
print("\nimmunity fractions (no single-node desynchronization moves the operating point)")
for n in NS:
    sel = [r for r in rows if r["n"] == n]
    ti = float(np.mean([("tandem_shift_max" not in r) or r["tandem_shift_max"] <= TOL for r in sel]))
    summary[f"immune|n={n}"] = {"designs": len(sel), "tandem_immune": ti,
                                "tandem_no_switching_node": float(np.mean([("tandem_shift_max" not in r) for r in sel])),
                                "median_sync_PD_tandem": float(np.median([r["tandem_C_D"] for r in sel]))}
    for fus in ("cv", "kofn"):
        summary[f"immune|n={n}"][f"parallel_{fus}_immune"] = float(np.mean([(f"parallel_{fus}_shift_max" not in r) or r[f"parallel_{fus}_shift_max"] <= TOL for r in sel]))
        summary[f"immune|n={n}"][f"median_sync_PD_parallel_{fus}"] = float(np.median([r[f"parallel_{fus}_C_D"] for r in sel]))
    print(f"  n={n}: tandem {ti:.2f}  parallel(cv) {summary[f'immune|n={n}']['parallel_cv_immune']:.2f}  "
          f"sync P_D {summary[f'immune|n={n}']['median_sync_PD_tandem']:.3f} / {summary[f'immune|n={n}']['median_sync_PD_parallel_cv']:.3f}")

print("\nposition-resolved shift |W - C| over susceptible designs (shift > TOL), likelihood-ratio fusion")
print("n  k   N(tandem)  median  p90   | paired N  median ratio [95% CI]  P(ratio<1)  Wilcoxon(less) p  Wilcoxon(greater) p")
for n in NS:
    sel = [r for r in rows if r["n"] == n]
    pooled = [x for r in sel if "parallel_cv_shift_by_i" in r for x in r["parallel_cv_shift_by_i"].values() if x > TOL]
    summary[f"position|n={n}|parallel"] = {"N": len(pooled), "median": float(np.median(pooled)), "p90": float(np.percentile(pooled, 90))}
    for k in range(1, n):
        v = [r["tandem_shift_by_k"][str(k)] for r in sel if "tandem_shift_by_k" in r and str(k) in r["tandem_shift_by_k"]]
        v = [x for x in v if x > TOL]
        pairs, pair_model = [], []
        for r in sel:
            if "tandem_shift_by_k" not in r or str(k) not in r["tandem_shift_by_k"] or r["tandem_shift_by_k"][str(k)] <= TOL:
                continue
            if "parallel_cv_shift_by_i" not in r:
                continue
            pv = [x for x in r["parallel_cv_shift_by_i"].values() if x > TOL]
            if pv:
                pairs.append((r["tandem_shift_by_k"][str(k)], float(np.mean(pv))))
                pair_model.append(r["model"])
        ent = {"N": len(v), "median": float(np.median(v)) if v else None, "p90": float(np.percentile(v, 90)) if v else None, "pairs": len(pairs)}
        if len(pairs) >= 5:
            t = np.array([a for a, _ in pairs]); q = np.array([b for _, b in pairs]); ratio = t / q
            med, lo, hi = boot_median(ratio, pair_model)
            p_less, p_greater, ties = paired_tests(t, q)
            m_less, m_greater, n_models = by_model_tests(t, q, pair_model)
            ent.update({"median_ratio": med, "ci": [lo, hi], "frac_ratio_lt_1": float(np.mean(ratio < 1 - 1e-9)),
                        "frac_ties": ties, "wilcoxon_less_p": p_less, "wilcoxon_greater_p": p_greater,
                        "models": n_models, "by_model_wilcoxon_less_p": m_less, "by_model_wilcoxon_greater_p": m_greater,
                        "geometric_mean_ratio": float(np.exp(np.mean(np.log(ratio)))),
                        "frac_ratio_below_half": float(np.mean(ratio < 0.5)), "frac_ratio_above_two": float(np.mean(ratio > 2.0)),
                        "mannwhitney_p": float(st.mannwhitneyu(t, q, alternative="less").pvalue),
                        "median_parallel_meanLD": float(np.median(q))})
            # the same median on disjoint blocks of BLOCK models: the spread shows what a sample of that size gives
            blk = np.array(pair_model) // BLOCK
            ent["block_median_ratios"] = [float(np.median(ratio[blk == b])) if np.sum(blk == b) >= 5 else None
                                          for b in range(int(blk.max()) + 1)]
            ent["block_pairs"] = [int(np.sum(blk == b)) for b in range(int(blk.max()) + 1)]
        summary[f"position|n={n}|k={k}"] = ent
        if "median_ratio" in ent:
            print(f"{n}  {k}   {ent['N']:5d}   {ent['median']:.4f} {ent['p90']:.4f} | {ent['pairs']:5d}   {ent['median_ratio']:.2f} [{ent['ci'][0]:.2f}, {ent['ci'][1]:.2f}]   "
                  f"{ent['frac_ratio_lt_1']:.2f}     {ent['wilcoxon_less_p']:.1e}     {ent['wilcoxon_greater_p']:.1e}")
    print(f"{n}  parallel LDs pooled: N={len(pooled)} median {np.median(pooled):.4f} p90 {np.percentile(pooled, 90):.4f}")

print("\nmetric: max over switching nodes of |W - C| (p-mixture desynchronization), susceptible designs on both sides")
print("fusion  n  alpha  pairs  median ratio [95% CI]   P(ratio<1)   Wilcoxon p   Mann-Whitney p   "
      "median tandem shift   median parallel shift   sync P_D tandem / parallel")
VARIANTS = {"max": ("tandem_shift_max", "parallel_{fus}_shift_max"),
            "wrongrule_max": ("tandem_wrongrule_max", "parallel_{fus}_wrongrule_max"),
            "wrongrule_N1": ("tandem_wrongrule_N1", "parallel_{fus}_wrongrule_max")}
for variant, (tkey, pkey_t) in VARIANTS.items():
  if variant != "max":
    print(f"\nmetric variant: {variant}  (tandem key {tkey}, parallel key {pkey_t})")
  for fus in ("cv", "kofn"):
    pkey = pkey_t.format(fus=fus)
    for n in NS:
        for alpha in ALPHAS + ("all",):
            sel = [r for r in rows if r["n"] == n and (alpha == "all" or r["alpha"] == alpha)
                   and r.get(tkey) is not None and r.get(pkey) is not None and r[pkey] > TOL and r[tkey] > TOL]
            if len(sel) < 5:
                continue
            tsh = np.array([r[tkey] for r in sel])
            psh = np.array([r[pkey] for r in sel])
            ratio = tsh / psh
            med, lo, hi = boot_median(ratio, [r["model"] for r in sel])
            wil, wil_g, ties = paired_tests(tsh, psh)
            mw = st.mannwhitneyu(tsh, psh, alternative="less").pvalue
            key = f"{variant}|{fus}|n={n}|alpha={alpha}"
            summary[key] = {"pairs": len(sel), "median_ratio": med, "ci": [lo, hi], "frac_ratio_lt_1": float(np.mean(ratio < 1 - 1e-9)),
                            "frac_ties": ties, "wilcoxon_p": float(wil), "wilcoxon_greater_p": float(wil_g), "mannwhitney_p": float(mw),
                            "median_tandem_shift": float(np.median(tsh)), "median_parallel_shift": float(np.median(psh)),
                            "median_sync_PD_tandem": float(np.median([r["tandem_C_D"] for r in sel])),
                            "median_sync_PD_parallel": float(np.median([r[f"parallel_{fus}_C_D"] for r in sel]))}
            s = summary[key]
            print(f"{fus:5s} {n:3d} {str(alpha):5s} {s['pairs']:5d}   {med:.3f} [{lo:.3f}, {hi:.3f}]     {s['frac_ratio_lt_1']:.2f}       "
                  f"{wil:.1e}      {mw:.1e}        {s['median_tandem_shift']:.4f}              {s['median_parallel_shift']:.4f}            "
                  f"{s['median_sync_PD_tandem']:.3f} / {s['median_sync_PD_parallel']:.3f}")
save_json(SUMMARY_FILE, summary)
