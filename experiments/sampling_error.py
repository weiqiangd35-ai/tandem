"""Sampling error of the statistics over random models (RESULTS.md, paragraphs "Sampling error"),
and the few counts quoted in the paper or in RESULTS.md that the experiment scripts do not
print themselves (lines marked "text").

Part A uses the stored sample only: parts of it (the first models of a cell, disjoint blocks)
and distribution-free intervals of medians.  Part B compares the stored summaries with those
of a second sample of the same size, obtained with every seed increased by 1000:

    mkdir -p results/second_seed
    python experiments/reseed.py exp1_attenuation.py 1000 results/second_seed > results/second_seed/exp1_attenuation_log.txt
    python experiments/reseed.py exp1_attenuation.py 1000 results/second_seed --relay > results/second_seed/exp1_relay_log.txt
    (the same, without --relay, for exp3_multinode.py, exp4_position.py, exp5_correction.py, exp8_sign_ensemble.py,
     exp9_pbpo_stats.py and, after exp5, whose worked examples they read, exp11_montecarlo.py and exp13_delayed_link.py)
    python experiments/exp2_tandem_vs_parallel.py --seed=1002 > results/exp2_seed1002_log.txt
    python experiments/exp2_tandem_vs_parallel.py --first-realization --seed=1002 > results/exp2_first_seed1002_log.txt

The package contains the logs and summaries of these runs, not their rows.  The numbers of the
paper are those of the first sample.
Usage: python experiments/sampling_error.py        (a few seconds; writes results/sampling_error.json)
"""
import os
from itertools import product
import numpy as np
from scipy.stats import binom, norm
from common import *
from tandem_sim.design import upper_hull, hull_value

SECOND = os.path.join(RESULTS, "second_seed")
out = {}


def second(name):
    path = os.path.join(SECOND, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def median_interval(x, level=0.95):
    """Distribution-free interval of a median from order statistics (as in supp_tables.py)."""
    xs = np.sort(np.asarray(x, float)); N = len(xs)
    l = max(k for k in range(1, N // 2 + 1) if binom.cdf(N - k, N, 0.5) - binom.cdf(k - 1, N, 0.5) >= level)
    return float(xs[l - 1]), float(xs[N - l])


def factor(a, b):
    return max(a / b, b / a) if min(a, b) > 0 else float("inf")


def expected_inside(level):
    """Probability that a second, independent median falls inside the interval of the first (normal approximation)."""
    return float(2 * norm.cdf(norm.ppf(0.5 + level / 2) / np.sqrt(2)) - 1)


def fm(x):
    return f"{x:.2f}" if x >= 0.0995 else (f"{x:.3f}" if x >= 0.00995 else f"{x:.1e}")


NS = (2, 5, 10, 20)
DISC = ["running", "dirichlet3", "dirichlet5", "dirichlet10", "qgauss1", "qgauss2"]
QUANT = ("qgauss1", "qgauss2")

# =========================================================================== exp1
print("=== exp1: attenuation statistics (Table SII) ===")
rows1 = load_json("exp1_rows.json")
FAMS = list(dict.fromkeys(r["family"] for r in rows1))
adv = {}
for r in rows1:
    if r["mode"] == "adversarial_max":
        adv.setdefault((r["family"], r["n"]), []).append(r["prod_rho_F"])
adv = {k: np.array(v) for k, v in adv.items()}
o = out.setdefault("exp1", {})
modes = {m: sum(r["mode"].startswith(m) for r in rows1) for m in ("random", "optimal", "adversarial")}
o["configurations"] = {"all": len(rows1), **modes, "stages": sum(len(r["rho_F"]) for r in rows1),
                       "largest_abs_rho": max(r["max_abs_rho"] for r in rows1),
                       "with_transfer_check": sum("transfer_check_max_err" in r for r in rows1),
                       "transfer_check_max_err": max(r["transfer_check_max_err"] for r in rows1 if "transfer_check_max_err" in r)}
c = o["configurations"]
print(f"text. {c['all']} configurations: {c['random']} random, {c['optimal']} NP-optimal, {c['adversarial']} adversarial; {c['stages']} stages; "
      f"largest |rho| - 1 = {c['largest_abs_rho'] - 1:.1e}; product transfer checked in {c['with_transfer_check']} of them, largest error {c['transfer_check_max_err']:.1e}")
Aq = np.array([r["rho_F"] for r in rows1 if r["family"] == "qgauss2" and r["n"] == 20 and r["mode"] == "adversarial_max"])
two_bins, one_bin = 1 - 2 * norm.cdf(-3), 1 - norm.cdf(-3)
near = (np.abs(Aq - two_bins) < 1e-9) | (np.abs(Aq - one_bin) < 1e-9)
first_node = 2 + next(j for j in range(Aq.shape[1]) if near[:, j:].all())
o["qgauss2_n20_near_relay"] = {"chains": int(Aq.shape[0]), "rho_two_bins": float(two_bins), "rho_one_bin": float(one_bin), "every_chain_from_node": int(first_node),
                               "chains_at_the_node_before": int(near[:, first_node - 3].sum())}
print(f"text. quantized Gaussian d' = 2, relay-like adversary at n = 20 ({Aq.shape[0]} chains): rho_F is {two_bins:.4f} (the relay except on the two "
      f"extreme bins) or {one_bin:.4f} (one bin) in every chain from node {first_node} on; at node {first_node - 1} in {o['qgauss2_n20_near_relay']['chains_at_the_node_before']} chains")
print("A. relay-like adversary, median end-to-end product over the first 10, the first 100 and all 1000 models of a cell")
for fam in ("dirichlet3", "qgauss1"):
    for n in NS:
        a = adv[(fam, n)]
        o[f"adversary|{fam}|n={n}"] = [float(np.median(a[:10])), float(np.median(a[:100])), float(np.median(a))]
        print(f"   {fam:11s} n = {n:2d}: {np.median(a[:10]):.3f}  {np.median(a[:100]):.3f}  {np.median(a):.3f}")
o["adversary_first100_vs_all_max_diff"] = max(abs(np.median(a[:100]) - np.median(a)) for a in adv.values())
print(f"   largest difference between the first 100 and all 1000 models, over the {len(adv)} cells: "
      f"{o['adversary_first100_vs_all_max_diff']:.3f}")
if os.path.exists(os.path.join(RESULTS, "exp1_relay.json")):
    rel = {}
    for r in load_json("exp1_relay.json")["rows"]:
        rel.setdefault((r["family"], r["n"]), []).append(r["prod_rho_F"])
    rel = {k: np.array(v) for k, v in rel.items()}
    o["relay_first100_vs_all_max_diff"] = max(abs(np.median(a[:100]) - np.median(a)) for a in rel.values())
    a = rel[("dirichlet3", 20)]
    o["relay|dirichlet3|n=20"] = [float(np.median(a[:100])), float(np.median(a))]
    print(f"   pure relay admitted: largest difference {o['relay_first100_vs_all_max_diff']:.3f}; "
          f"dirichlet3 at n = 20: {np.median(a[:100]):.3f} on the first 100 models, {np.median(a):.3f} on 1000")
print("A. quantized Gaussian families, random configurations at n = 5: the median factor sits at a jump of the distribution")
for fam in QUANT:
    st = np.concatenate([r["rho_F"] for r in rows1 if r["family"] == fam and r["n"] == 5 and r["mode"] == "random"])
    vals = np.unique(np.round(st, 9)); med = float(np.median(st))
    lo = float(vals[vals < med - 1e-9].max()); hi = float(vals[vals > lo + 1e-9].min())
    k = int(np.sum(st <= lo + 1e-9))
    o[f"jump|{fam}"] = {"values": len(vals), "median": med, "below": lo, "above": hi, "stages_at_or_below": k, "stages": len(st)}
    print(f"   {fam}: rho_F takes {len(vals)} values; {k} of the {len(st)} stages ({100 * k / len(st):.2f} %) have rho_F <= {lo:.3f}, "
          f"the next value is {hi:.3f}; median {med:.3f}")

s1, t1, iv = load_json("exp1_summary.json"), second("exp1_summary.json"), load_json("tab_s2_intervals.json")
if t1:
    print("B. second sample (seeds 1001, 1101, 1202)")
    o["theorem_1"] = {k: [s1[k], t1[k]] for k in ("rows", "rows_with_transfer_check", "rows_with_abs_rho_above_1", "max_abs_rho_over_all_rows", "transfer_check_max_err")}
    print(f"   configurations {s1['rows']} / {t1['rows']}; with |rho| above 1: {s1['rows_with_abs_rho_above_1']} / {t1['rows_with_abs_rho_above_1']}; "
          f"largest |rho| - 1: {s1['max_abs_rho_over_all_rows'] - 1:.1e} / {t1['max_abs_rho_over_all_rows'] - 1:.1e}; "
          f"largest error of the product transfer: {s1['transfer_check_max_err']:.1e} / {t1['transfer_check_max_err']:.1e} "
          f"(checked in {s1['rows_with_transfer_check']} / {t1['rows_with_transfer_check']} configurations)")
    med = lambda s, fam, n, mode="random": s[f"{fam}|n={n}|{mode}"]["prod_rho_F"]["median"]
    inside = 0; cells = 0; fac = {n: {"other": 1.0, "quantized": 1.0} for n in NS}
    for fam in FAMS:
        for n in NS:
            a, b = med(s1, fam, n), med(t1, fam, n)
            lo, hi = iv["random"][f"{fam}|n={n}"]["interval"]
            cells += 1; inside += lo <= b <= hi
            g = "quantized" if fam in QUANT else "other"
            fac[n][g] = max(fac[n][g], factor(a, b))
    lev = iv["random"]["dirichlet3|n=5"]["level"]
    o["random_products"] = {"cells": cells, "second_inside_interval_of_first": inside, "interval_level": lev,
                            "expected_share_inside": expected_inside(lev), "largest_factor": {f"n={n}": fac[n] for n in NS}}
    print(f"   median product of the random configurations: the second median lies inside the 95 % interval of the first in "
          f"{inside} of the {cells} cells ({100 * inside / cells:.0f} %; expected for two independent estimates and an interval of level "
          f"{100 * lev:.1f} %: {100 * expected_inside(lev):.0f} %)")
    print("   largest factor between the two medians at n = 2, 5, 10, 20: "
          + ", ".join(f"{fac[n]['other']:.2f}" for n in NS) + "; quantized Gaussian families: "
          + ", ".join(f"{fac[n]['quantized']:.2f}" for n in NS))
    print(f"   Dirichlet K=3 at n = 5: {med(s1, 'dirichlet3', 5):.4f} and {med(t1, 'dirichlet3', 5):.4f}")
    d = {"median": 0.0, "p10": 0.0, "p90": 0.0}; dfrac = 0.0; worst = None
    print("   per-stage factor of the random configurations at n = 5 (first / second sample): median, 10th and 90th percentiles, fraction above 0.9")
    for fam in FAMS:
        a, b = s1[f"{fam}|n=5|random"], t1[f"{fam}|n=5|random"]
        o[f"random_factor_n5|{fam}"] = {q: [a["rho_F"][q], b["rho_F"][q]] for q in d}
        o[f"random_factor_n5|{fam}"]["frac_above_0.9"] = [a["frac_rho_F_gt_0.9"], b["frac_rho_F_gt_0.9"]]
        print(f"      {fam:14s} " + "   ".join(f"{a['rho_F'][q]:.3f} / {b['rho_F'][q]:.3f}" for q in d)
              + f"   {a['frac_rho_F_gt_0.9']:.3f} / {b['frac_rho_F_gt_0.9']:.3f}")
        if fam not in QUANT:
            if abs(a["rho_F"]["median"] - b["rho_F"]["median"]) > d["median"]:
                worst = fam
            for q in d:
                d[q] = max(d[q], abs(a["rho_F"][q] - b["rho_F"][q]))
            dfrac = max(dfrac, abs(a["frac_rho_F_gt_0.9"] - b["frac_rho_F_gt_0.9"]))
    o["random_factors_n5_max_change"] = {**d, "frac_above_0.9": dfrac, "family_of_the_largest_change_of_the_median": worst}
    print(f"      twelve families other than the quantized Gaussian ones: largest change of the median {d['median']:.3f} ({worst}), "
          f"of the 10th percentile {d['p10']:.3f}, of the 90th {d['p90']:.3f}, of the fraction above 0.9 {dfrac:.3f}")
    print("   optimal designs, median factor at n = 5 (first / second sample):")
    dopt = 0.0
    for fam in FAMS:
        key = f"{fam}|n=5|optimal"
        if key in s1 and key in t1:
            a, b = s1[key]["rho_F"]["median"], t1[key]["rho_F"]["median"]
            o[f"optimal_factor_n5|{fam}"] = [a, b, s1[key]["count"], t1[key]["count"]]
            fixed = fam == "running" or fam not in DISC      # one fixed model: nothing is sampled
            dopt = dopt if fixed else max(dopt, abs(a - b))
            print(f"      {fam:14s} {a:.3f} / {b:.3f}" + ("   (fixed model: the same designs)" if fixed else ""))
    o["optimal_factor_n5_max_change"] = dopt
    dprod = [factor(s1[f"{fam}|n=5|optimal"]["prod_rho_F"]["median"], t1[f"{fam}|n=5|optimal"]["prod_rho_F"]["median"])
             for fam in ("dirichlet3", "dirichlet5")]
    o["optimal_product_n5"] = {fam: [s1[f"{fam}|n=5|optimal"]["prod_rho_F"]["median"], t1[f"{fam}|n=5|optimal"]["prod_rho_F"]["median"]]
                               for fam in ("dirichlet3", "dirichlet5")}
    print(f"      largest change over the sampled families: {dopt:.3f}; median product of the optimal designs, Dirichlet K=3 at n = 5: "
          f"{o['optimal_product_n5']['dirichlet3'][0]:.3f} and {o['optimal_product_n5']['dirichlet3'][1]:.3f}")
    print("   relay-like adversary, median product at n = 2, 5, 10, 20 (first sample with its 95 % interval / second sample):")
    DIR = [f for f in DISC if f.startswith("dirichlet")]         # the other discrete families use one fixed model at every node
    inside = cells = 0; dadv = 0.0
    for fam in DISC:
        line = []
        for n in NS:
            e = iv["adversary"][f"{fam}|n={n}"]; b = med(t1, fam, n, "adversarial_max")
            if fam in DIR:
                cells += 1; inside += e["interval"][0] <= b <= e["interval"][1]
            dadv = max(dadv, abs(e["median"] - b))
            o[f"adversary_second|{fam}|n={n}"] = [e["median"], e["interval"][0], e["interval"][1], b]
            line.append(f"{fm(e['median'])} [{fm(e['interval'][0])}, {fm(e['interval'][1])}] / {fm(b)}")
        print(f"      {fam:12s} " + ";  ".join(line))
    lev = iv["adversary"]["dirichlet3|n=5"]["level"]
    o["adversary_products"] = {"dirichlet_cells": cells, "second_inside_interval_of_first": inside, "largest_change": dadv,
                               "expected_share_inside": expected_inside(lev)}
    print(f"      Dirichlet families: the second median lies inside the interval of the first in {inside} of the {cells} cells "
          f"(expected: {100 * expected_inside(lev):.0f} %); largest change over all cells {dadv:.3f}")
    r1, r2 = load_json("exp1_relay.json"), second("exp1_relay.json")
    if r2:
        print("   pure relay admitted, median product at n = 2, 5, 10, 20 (first sample with its 95 % interval / second sample):")
        inside = cells = 0; drel = 0.0
        for fam in DISC:
            line = []
            for n in NS:
                e, b = r1[f"{fam}|n={n}"]["allowed"], r2[f"{fam}|n={n}"]["allowed"]["median"]
                if fam in DIR:
                    cells += 1; inside += e["interval"][0] <= b <= e["interval"][1]
                drel = max(drel, abs(e["median"] - b))
                o[f"relay_second|{fam}|n={n}"] = [e["median"], e["interval"][0], e["interval"][1], b]
                line.append(f"{fm(e['median'])} [{fm(e['interval'][0])}, {fm(e['interval'][1])}] / {fm(b)}")
            print(f"      {fam:12s} " + ";  ".join(line))
        o["relay_products"] = {"dirichlet_cells": cells, "second_inside_interval_of_first": inside, "largest_change": drel,
                               "share_of_relaying_nodes_dirichlet3_n20": [r1["dirichlet3|n=20"]["frac_relay_nodes"], r2["dirichlet3|n=20"]["frac_relay_nodes"]]}
        print(f"      Dirichlet families: the second median lies inside the interval of the first in {inside} of the {cells} cells; "
              f"largest change over all cells {drel:.3f}; share of the nodes that relay, Dirichlet K=3 at n = 20: "
              f"{r1['dirichlet3|n=20']['frac_relay_nodes']:.2f} / {r2['dirichlet3|n=20']['frac_relay_nodes']:.2f}")

# =========================================================================== exp2
print("\n=== exp2: tandem versus parallel (Section IV-D, Fig. 2, Tables SIII-SIV) ===")
s2 = load_json("exp2_summary.json")
t2 = load_json("exp2_summary_seed1002.json") if os.path.exists(os.path.join(RESULTS, "exp2_summary_seed1002.json")) else None
o = out.setdefault("exp2", {})
rows2 = load_json("exp2_rows.json")
single = [float(np.mean([r.get("parallel_cv_n_switching") == 1 for r in rows2 if r["n"] == n])) for n in (3, 4, 5, 6)]
none = [float(np.mean([r.get("parallel_cv_n_switching") is None for r in rows2 if r["n"] == n])) for n in (3, 4, 5, 6)]
ppm = [s2[f"position|n={n}|k={k}"]["pairs"] / s2[f"position|n={n}|k={k}"]["models"] for n in (3, 4, 5, 6) for k in range(1, n)]
o["parallel_designs_switching_one_LD"] = single; o["parallel_designs_switching_no_LD"] = none; o["pairs_per_model"] = [min(ppm), max(ppm)]
print("text. parallel designs whose two supporting strategies differ in the threshold of a single LD, n = 3..6: "
      + ", ".join(f"{100 * x:.1f}" for x in single) + " % (of no LD, the fusion rule alone: " + ", ".join(f"{100 * x:.1f}" for x in none)
      + f" %);  pairs per model at a position: {min(ppm):.2f} to {max(ppm):.2f}")
print("A. median ratio at node k: the five blocks of 200 models" + (" (B: first sample with its bootstrap interval / second sample, --seed=1002)" if t2 else ""))
inside = inside2 = cells = 0
side = lambda m, ci: (ci[1] < 1) if m < 1 else (ci[0] > 1)        # the interval lies on the same side of one as the median
for n in (3, 4, 5, 6):
    for k in range(1, n):
        e = s2[f"position|n={n}|k={k}"]
        blocks = e["block_median_ratios"]
        txt = f"   n = {n}, k = {k}: blocks " + ", ".join(f"{b:.2f}" for b in blocks)
        o[f"n={n}|k={k}"] = {"blocks": blocks, "first": [e["median_ratio"]] + list(e["ci"]), "first_pairs": e["pairs"],
                             "first_wilcoxon": [e["wilcoxon_less_p"], e["wilcoxon_greater_p"]], "first_on_one_side": bool(side(e["median_ratio"], e["ci"]))}
        if t2:
            f = t2[f"position|n={n}|k={k}"]; b = f["median_ratio"]
            cells += 1; inside += e["ci"][0] <= b <= e["ci"][1]; inside2 += f["ci"][0] <= e["median_ratio"] <= f["ci"][1]
            o[f"n={n}|k={k}"].update({"second": [b] + list(f["ci"]), "second_pairs": f["pairs"], "second_wilcoxon": [f["wilcoxon_less_p"], f["wilcoxon_greater_p"]],
                                      "second_on_one_side": bool(side(b, f["ci"])), "second_blocks": f["block_median_ratios"]})
            txt += (f";   {e['median_ratio']:.2f} [{e['ci'][0]:.2f}, {e['ci'][1]:.2f}] / {b:.2f} [{f['ci'][0]:.2f}, {f['ci'][1]:.2f}]"
                    f"   pairs {e['pairs']} / {f['pairs']}   signed-rank p for tandem smaller {e['wilcoxon_less_p']:.1e} / {f['wilcoxon_less_p']:.1e}")
        print(txt)
if t2:
    o["positions"] = {"cells": cells, "second_inside_interval_of_first": inside, "first_inside_interval_of_second": inside2}
    print(f"B. the second median lies inside the interval of the first at {inside} of the {cells} positions, the first inside the interval of the second at {inside2}")
    later = [(n, k) for n in (4, 5, 6) for k in range(2, n)]
    o["positions"]["later_nodes_on_one_side"] = [sum(o[f"n={n}|k={k}"][s] for n, k in later) for s in ("first_on_one_side", "second_on_one_side")]
    print(f"   second and later nodes of chains of four to six nodes ({len(later)} positions): interval on its side of one at "
          f"{o['positions']['later_nodes_on_one_side'][0]} / {o['positions']['later_nodes_on_one_side'][1]} positions")
    for lab, key in (("share of the pairs with a ratio below 0.5", "frac_ratio_below_half"), ("share with a ratio above 2", "frac_ratio_above_two"),
                     ("geometric mean of the ratio", "geometric_mean_ratio")):
        a = [s2[f"position|n={n}|k=1"][key] for n in (3, 4, 5, 6)]; b = [t2[f"position|n={n}|k=1"][key] for n in (3, 4, 5, 6)]
        o[f"N1|{key}"] = [a, b]
        print(f"   first node, {lab}, n = 3..6: " + ", ".join(f"{x:.2f}" for x in a) + " / " + ", ".join(f"{x:.2f}" for x in b))
    for lab, key in (("median synchronized P_D, tandem", "median_sync_PD_tandem"), ("median synchronized P_D, parallel", "median_sync_PD_parallel_cv")):
        a = [s2[f"immune|n={n}"][key] for n in (3, 4, 5, 6)]; b = [t2[f"immune|n={n}"][key] for n in (3, 4, 5, 6)]
        o[key] = [a, b]
        print(f"   {lab}, n = 3..6: " + ", ".join(f"{x:.2f}" for x in a) + " / " + ", ".join(f"{x:.2f}" for x in b))
    a = [s2[f"position|n={n}|parallel"]["median"] for n in (3, 4, 5, 6)]; b = [t2[f"position|n={n}|parallel"]["median"] for n in (3, 4, 5, 6)]
    o["median_parallel_shift"] = [a, b]
    print("   median parallel single-LD shift, n = 3..6: " + ", ".join(f"{x:.4f}" for x in a) + " / " + ", ".join(f"{x:.4f}" for x in b))
    for n in (3, 4, 5, 6):
        a = [s2[f"position|n={n}|k={k}"]["median"] for k in range(1, n)]; b = [t2[f"position|n={n}|k={k}"]["median"] for k in range(1, n)]
        c = [s2[f"position|n={n}|k={k}"]["N"] for k in range(1, n)]; d = [t2[f"position|n={n}|k={k}"]["N"] for k in range(1, n)]
        o[f"median_tandem_shift|n={n}"] = [a, b]; o[f"susceptible_cases|n={n}"] = [c, d]
        print(f"   n = {n}, median tandem shift at k = 1..{n - 1}: " + ", ".join(f"{x:.4f}" for x in a) + " / " + ", ".join(f"{x:.4f}" for x in b)
              + ";  susceptible cases: " + ", ".join(map(str, c)) + " / " + ", ".join(map(str, d)))
    for lab, key in (("immune designs, tandem", "tandem_immune"), ("immune designs, parallel", "parallel_cv_immune")):
        a = [s2[f"immune|n={n}"][key] for n in (3, 4, 5, 6)]; b = [t2[f"immune|n={n}"][key] for n in (3, 4, 5, 6)]
        o[key] = [a, b]
        print(f"   {lab}, n = 3..6: " + ", ".join(f"{x:.2f}" for x in a) + " / " + ", ".join(f"{x:.2f}" for x in b))
    for lab, key in (("worst switching node, median ratio", "max|cv|n={n}|alpha=all"),
                     ("wrong rule at N_1 against the worst parallel LD, median ratio", "wrongrule_N1|cv|n={n}|alpha=all")):
        a = [s2[key.format(n=n)]["median_ratio"] for n in (3, 4, 5, 6)]; b = [t2[key.format(n=n)]["median_ratio"] for n in (3, 4, 5, 6)]
        o[key.split("|")[0]] = [a, b]
        print(f"   {lab}, n = 3..6: " + ", ".join(f"{x:.2f}" for x in a) + " / " + ", ".join(f"{x:.2f}" for x in b))

f1 = load_json("exp2_summary_first.json")
f2 = load_json("exp2_summary_first_seed1002.json") if os.path.exists(os.path.join(RESULTS, "exp2_summary_first_seed1002.json")) else None
if f2:
    print("B. first realization enumerated (--first-realization), first / second sample (--seed=1002)")
    q = o.setdefault("first_realization", {})
    for lab, get, f in (("designs in which N_1 is a susceptible switching node, n = 3..6", lambda s, n: s[f"position|n={n}|k=1"]["N"], "{:d}"),
                        ("immune designs, tandem", lambda s, n: s[f"immune|n={n}"]["tandem_immune"], "{:.2f}"),
                        ("immune designs, parallel", lambda s, n: s[f"immune|n={n}"]["parallel_cv_immune"], "{:.2f}"),
                        ("median ratio at N_1", lambda s, n: s[f"position|n={n}|k=1"]["median_ratio"], "{:.2f}"),
                        ("median ratio at N_2 (n = 4, 5, 6)", lambda s, n: s[f"position|n={n}|k=2"]["median_ratio"] if n >= 4 else None, "{:.2f}"),
                        ("median ratio next to the fusion node (n = 4, 5, 6)", lambda s, n: s[f"position|n={n}|k={n - 1}"]["median_ratio"] if n >= 4 else None, "{:.2f}")):
        a = [get(f1, n) for n in (3, 4, 5, 6)]; b = [get(f2, n) for n in (3, 4, 5, 6)]
        a = [x for x in a if x is not None]; b = [x for x in b if x is not None]
        q[lab.split(" (")[0].split(",")[0] if lab.startswith("designs") else lab.split(" (")[0]] = [a, b]
        print(f"   {lab}: " + ", ".join(f.format(x) for x in a) + " / " + ", ".join(f.format(x) for x in b))
    ins = [f1[f"position|n={n}|k={k}"]["ci"][0] <= f2[f"position|n={n}|k={k}"]["median_ratio"] <= f1[f"position|n={n}|k={k}"]["ci"][1]
           for n in (3, 4, 5, 6) for k in range(1, n)]
    q["positions"] = {"cells": len(ins), "second_inside_interval_of_first": int(sum(ins))}
    print(f"   the second median lies inside the interval of the first at {sum(ins)} of the {len(ins)} positions")

# =========================================================================== exp3
print("\n=== exp3: several desynchronized nodes (Section IV-F, Fig. S2) ===")
rows3 = load_json("exp3_multinode_rows.json"); s3, t3 = load_json("exp3_multinode_summary.json"), second("exp3_multinode_summary.json")
o = out.setdefault("exp3", {})
des = [r for r in rows3 if r["ensemble"] == "design"]
n_models = max(r["cfg"] for r in des) + 1


def design_stats(sel):
    med = [float(np.median([abs(r["shift_F"]) for r in sel if r["size"] == s])) if any(r["size"] == s for r in sel) else None
           for s in (1, 2, 3, 4)]
    pos = [abs(r["shift_F"]) / r["bound_F"] for r in sel if r["bound_F"] > 0]
    return {"designs": sum(r["size"] == 1 for r in sel), "cases": len(sel), "cases_S4": sum(r["size"] == 4 for r in sel),
            "median_abs_shift": med, "pooled_ratio": float(np.median(pos))}


full = design_stats(des)
assert all(abs(a - s3[f"design|S|={s}"]["abs_shift_F"]["median"]) < 1e-15 for s, a in zip((1, 2, 3, 4), full["median_abs_shift"]))
assert abs(full["pooled_ratio"] - s3["design|pooled"]["ratio_shift_over_positive_bound"]["median"]) < 1e-15
blocks = [design_stats([r for r in des if b <= r["cfg"] < b + 30]) for b in range(0, n_models, 30)]
o["first_30_models"] = blocks[0]; o["all_models"] = full
o["blocks_of_30"] = {"median_abs_shift_min": [min(b["median_abs_shift"][i] for b in blocks) for i in range(4)],
                     "median_abs_shift_max": [max(b["median_abs_shift"][i] for b in blocks) for i in range(4)],
                     "pooled_ratio": [min(b["pooled_ratio"] for b in blocks), max(b["pooled_ratio"] for b in blocks)],
                     "cases_S4": [min(b["cases_S4"] for b in blocks), max(b["cases_S4"] for b in blocks)]}
o["blocks_of_30"]["factor"] = [hi / lo for lo, hi in zip(o["blocks_of_30"]["median_abs_shift_min"], o["blocks_of_30"]["median_abs_shift_max"])]
b0 = blocks[0]
print(f"A. optimal designs, first 30 of the {n_models} models: {b0['designs']} designs, {b0['cases']} cases, {b0['cases_S4']} with |S| = 4; "
      "median |W_F - alpha| for |S| = 1..4: " + ", ".join(f"{x:.1e}" for x in b0["median_abs_shift"])
      + f"; pooled shift / bound {b0['pooled_ratio']:.3f}")
print(f"   over the {len(blocks)} disjoint blocks of 30 models: "
      + "; ".join(f"|S| = {i + 1}: {o['blocks_of_30']['median_abs_shift_min'][i]:.1e} to {o['blocks_of_30']['median_abs_shift_max'][i]:.1e}" for i in range(4))
      + f"; pooled shift / bound {o['blocks_of_30']['pooled_ratio'][0]:.3f} to {o['blocks_of_30']['pooled_ratio'][1]:.3f}"
      + f"; cases with |S| = 4: {o['blocks_of_30']['cases_S4'][0]} to {o['blocks_of_30']['cases_S4'][1]}"
      + "; largest / smallest block median: " + ", ".join(f"{x:.1f}" for x in o["blocks_of_30"]["factor"]))
print("   all models: " + ", ".join(f"{x:.1e}" for x in full["median_abs_shift"]) + f"; pooled shift / bound {full['pooled_ratio']:.3f}")
if t3:
    print("B. second sample (seed 1003): first / second")
    for ens in ("random", "design"):
        for s in (1, 2, 3, 4):
            a, b = s3[f"{ens}|S|={s}"], t3[f"{ens}|S|={s}"]
            o[f"{ens}|S={s}"] = {"count": [a["count"], b["count"]],
                                 "median": [a["abs_shift_F"]["median"], b["abs_shift_F"]["median"]],
                                 "p90": [a["abs_shift_F"]["p90"], b["abs_shift_F"]["p90"]],
                                 "max": [a["max_abs_shift_F"], b["max_abs_shift_F"]],
                                 "ratio": [a["ratio_shift_over_positive_bound"]["median"], b["ratio_shift_over_positive_bound"]["median"]]}
            print(f"   {ens:6s} |S| = {s}: cases {a['count']} / {b['count']}; median |W_F - alpha| {a['abs_shift_F']['median']:.1e} / {b['abs_shift_F']['median']:.1e}; "
                  f"p90 {a['abs_shift_F']['p90']:.3f} / {b['abs_shift_F']['p90']:.3f}; max {a['max_abs_shift_F']:.3f} / {b['max_abs_shift_F']:.3f}; "
                  f"median shift / bound {a['ratio_shift_over_positive_bound']['median']:.3f} / {b['ratio_shift_over_positive_bound']['median']:.3f}")
        a, b = s3[f"{ens}|pooled"], t3[f"{ens}|pooled"]
        o[f"{ens}|pooled"] = {"ratio": [a["ratio_shift_over_positive_bound"]["median"], b["ratio_shift_over_positive_bound"]["median"]],
                              "max": [a["max_abs_shift_F"], b["max_abs_shift_F"]], "cases": [a["cases"], b["cases"]]}
        print(f"   {ens:6s} pooled: median shift / bound {a['ratio_shift_over_positive_bound']['median']:.3f} / {b['ratio_shift_over_positive_bound']['median']:.3f}; "
              f"largest shift {a['max_abs_shift_F']:.3f} / {b['max_abs_shift_F']:.3f}")
    a, b = s3["design_switching_nodes"], t3["design_switching_nodes"]
    o["design_switching_nodes"] = {"designs": [a["n"], b["n"]], "mean": [a["mean"], b["mean"]]}
    print(f"   designs with a switching node among N_1..N_7: {a['n']} / {b['n']}; switching nodes per design {a['mean']:.2f} / {b['mean']:.2f}")

# =========================================================================== exp4
print("\n=== exp4: position of the desynchronized node (Section IV-F) ===")
rows4 = load_json("exp4_position_rows.json"); s4, t4 = load_json("exp4_position_summary.json"), second("exp4_position_summary.json")
o = out.setdefault("exp4", {})
for k in (1, 7):
    x = [abs(r["shift_F"]) for r in rows4 if r["ensemble"] == "random" and r["k"] == k]
    lo, hi = median_interval(x)
    o[f"k={k}"] = {"median": float(np.median(x)), "interval": [lo, hi], "zero": int(sum(v < 1e-12 for v in x)), "pairs": len(x)}
    print(f"A. random pairs, k = {k}: median |W_F - alpha| {np.median(x):.1e}, 95 % interval [{lo:.1e}, {hi:.1e}]; "
          f"{o[f'k={k}']['zero']} of the {len(x)} shifts are zero (below 1e-12)")
if t4:
    for k in (1, 7):
        a, b = s4[f"randomk={k}"], t4[f"randomk={k}"]
        o[f"k={k}"]["second"] = {"median": b["abs_shift_F"]["median"], "zero_frac": b["frac_zero_shift"], "product": b["Hmax_downstream"]["median"]}
        o[f"k={k}"]["product"] = a["Hmax_downstream"]["median"]
        print(f"B. second sample (seed 1004), k = {k}: median |W_F - alpha| {a['abs_shift_F']['median']:.1e} / {b['abs_shift_F']['median']:.1e}; "
              f"zero shifts {a['frac_zero_shift']:.3f} / {b['frac_zero_shift']:.3f}; "
              f"median downstream product {a['Hmax_downstream']['median']:.3f} / {b['Hmax_downstream']['median']:.3f}")
    o["eq14_max_abs_error"] = [s4["eq14_max_abs_error"], t4["eq14_max_abs_error"]]
    print(f"   largest |exact - eq. (14)|: {s4['eq14_max_abs_error']:.1e} / {t4['eq14_max_abs_error']:.1e}")

# =========================================================================== exp5
print("\n=== exp5: single-node failures of susceptible designs (Table SVI) ===")
s5 = load_json("exp5_correction.json")["stats"]; t5 = second("exp5_correction.json")
o = out.setdefault("exp5", {})
for n in ("2", "3", "4"):
    e = s5[n]
    o[f"n={n}"] = {"cases": e["cases"], "cases_per_susceptible_design": e["cases"] / e["susceptible_designs"],
                   "largest_binomial_standard_error": 0.5 / np.sqrt(e["cases"])}
    print(f"A. n = {n}: {e['cases']} cases from {e['susceptible_designs']} susceptible designs ({e['cases'] / e['susceptible_designs']:.2f} per design); "
          f"binomial standard error of a share of all cases: at most {100 * 0.5 / np.sqrt(e['cases']):.2f} points")
if t5:
    t5 = t5["stats"]
    print("B. second sample (seed 1005): first / second, n = 2, 3, 4")
    KEYS = (("cases", "cases", "{:d}"), ("susceptible designs", "susceptible_designs", "{:d}"), ("randomized designs", "designs", "{:d}"),
            ("P_F rises", "frac_PF_rises", "{:.3f}"), ("median |W_F - alpha|", ("abs_overshoot", "median"), "{:.4f}"),
            ("largest |W_F - alpha|", ("abs_overshoot", "max"), "{:.3f}"),
            ("rule of the DFC switches", "frac_dfc_switches", "{:.3f}"),
            ("adjustment meets the constraint", "frac_feasible_given_dfc_switches", "{:.3f}"),
            ("  when P_F rises", "frac_feasible_given_rise_and_dfc_switches", "{:.3f}"),
            ("it equals the DFC-layer optimum", "frac_q_equals_dfc", "{:.3f}"), ("it equals the downstream optimum", "frac_q_equals_down", "{:.3f}"),
            ("median gap to the DFC-layer optimum", ("gap_dfc_minus_q", "median"), "{:.1e}"),
            ("median gap to the downstream optimum", ("gap_down_minus_q", "median"), "{:.1e}"),
            ("largest gap", ("gap_dfc_minus_q", "max"), "{:.2f}"),
            ("DFC rule does not switch: P_F > alpha", "frac_constraint_violated_no_switch", "{:.3f}"),
            ("below G: adjustment", "frac_q_below_G", "{:.3f}"), ("below G: DFC-layer optimum", "frac_dfc_below_G", "{:.3f}"),
            ("below G: downstream optimum", "frac_down_below_G", "{:.3f}"),
            ("median C_D - G_D", ("randomization_gain", "median"), "{:.4f}"), ("median C_D - S_D", ("common_draw_gain", "median"), "{:.4f}"),
            ("90 % quantile of C_D - S_D", ("common_draw_gain", "p90"), "{:.4f}"), ("largest C_D - S_D", ("common_draw_gain", "max"), "{:.3f}"),
            ("below S: adjustment", "frac_q_below_S", "{:.3f}"), ("below S: DFC-layer optimum", "frac_dfc_below_S", "{:.3f}"),
            ("below S: downstream optimum", "frac_down_below_S", "{:.3f}"))
    get = lambda e, key: (e.get(key) if isinstance(key, str) else e.get(key[0], {}).get(key[1]))
    dmax = 0.0
    for lab, key, f in KEYS:
        cells = []
        for n in ("2", "3", "4"):
            a, b = get(s5[n], key), get(t5[n], key)
            if a is None or b is None:
                cells.append("-"); continue
            cells.append(f"{f.format(a)} / {f.format(b)}")
            o[f"n={n}"][lab.strip()] = [a, b]
            if isinstance(key, str) and key.startswith("frac_"):
                dmax = max(dmax, abs(a - b))
        print(f"   {lab:40s} " + ";  ".join(cells))
    o["largest_change_of_a_share"] = dmax
    print(f"   largest change of a share: {100 * dmax:.1f} points")
    for n in ("2", "3", "4"):
        a, b = ([e[n]["by_node"][k]["cases"] for k in sorted(e[n]["by_node"])] for e in (s5, t5))
        o[f"n={n}"]["cases_by_failing_node"] = [a, b]
        print(f"   n = {n}, cases by failing node (N_1" + (f" to N_{int(n) - 1}" if n != "2" else "") + "): " + ", ".join(map(str, a)) + " / " + ", ".join(map(str, b)))

# =========================================================================== exp7
print("\n=== exp7: nominal piecewise-linear model of Proposition 3 (knots 0, 0.1, 0.2, 1) ===")
LINES = [(6.0, 0.0), (1.0, 0.5), (3 / 8, 5 / 8)]; KN = [0.0, 0.1, 0.2, 1.0]
r_pl = lambda a: np.min([m * np.asarray(a, float) + b for m, b in LINES], axis=0)


def pl_envelope(al):
    """Corollary S1: the best strategy with two of (f, a0, a1) at knots and the third solved from P_F = alpha."""
    al = np.asarray(al, float); one = np.ones_like(al); best = np.full(len(al), -np.inf)

    def update(f, a0, a1, ok):
        d = r_pl(f)
        np.maximum(best, np.where(ok, (1 - d) * r_pl(a0) + d * r_pl(a1), -np.inf), out=best)

    for f, a0 in product(KN, KN):
        if f > 0:
            a1 = (al - (1 - f) * a0) / f
            update(f * one, a0 * one, np.clip(a1, 0, 1), (a1 >= -1e-15) & (a1 <= 1 + 1e-15))
    for f, a1 in product(KN, KN):
        if f < 1:
            a0 = (al - f * a1) / (1 - f)
            update(f * one, np.clip(a0, 0, 1), a1 * one, (a0 >= -1e-15) & (a0 <= 1 + 1e-15))
    for a0, a1 in product(KN, KN):
        if a1 != a0:
            f = (al - a0) / (a1 - a0)
            update(np.clip(f, 0, 1), a0 * one, a1 * one, (f >= -1e-15) & (f <= 1 + 1e-15))
    return best


pts = [((1 - f) * a0 + f * a1, float((1 - r_pl(f)) * r_pl(a0) + r_pl(f) * r_pl(a1))) for f, a0, a1 in product(KN, repeat=3)]
hull = upper_hull(pts)
o = out.setdefault("exp7", {}); o["nominal_pl_hull_edges"] = []
for u, v in zip(hull[:-1], hull[1:]):
    xs = np.linspace(u[0], v[0], 2001)
    g = float((np.array([hull_value(hull, float(x)) for x in xs]) - pl_envelope(xs)).max())
    o["nominal_pl_hull_edges"].append([u[0], v[0], g])
print("text. largest gap between the hull and the deterministic envelope on each hull edge (2001 values of alpha per edge):")
print("   " + ";  ".join(f"[{a:.4g}, {b:.4g}]: {g:.1e}" for a, b, g in o["nominal_pl_hull_edges"]))
k = o["knot_draws"] = load_json("exp7_prop3_gap.json")["knot_perturbation"]["models"]
print(f"A. a share p of the {k} perturbed models has the binomial standard error sqrt(p (1 - p) / {k}): "
      f"{100 * np.sqrt(0.7 * 0.3 / k):.1f} points at 70 %, {100 * np.sqrt(0.24 * 0.76 / k):.1f} points at 24 %")
o["knot_draws"] = k

# =========================================================================== exp8
print("\n=== exp8: direction of the shift on two-node chains (Section IV-E) ===")
s8, t8 = load_json("exp8_sign_ensemble.json"), second("exp8_sign_ensemble.json")
o = out.setdefault("exp8", {})
if t8:
    print("B. second sample (seed 1000): first / second")
    for lab, get in (("hull edges", lambda e: e["designs"]["edges"]), ("chains with a susceptible edge", lambda e: e["models_with_a_susceptible_edge"]),
                     ("P_F rises", lambda e: e["designs"]["counts"]["rise"]), ("P_F falls", lambda e: e["designs"]["counts"]["fall"]),
                     ("pairs of realizations", lambda e: e["all_single_threshold_realization_pairs"]["realization_pairs"]),
                     ("  unchanged", lambda e: e["all_single_threshold_realization_pairs"]["fractions"]["unchanged"]),
                     ("  rise", lambda e: e["all_single_threshold_realization_pairs"]["fractions"]["rise"]),
                     ("  fall", lambda e: e["all_single_threshold_realization_pairs"]["fractions"]["fall"]),
                     ("with the trivial edges: unchanged", lambda e: e["all_pairs_including_trivial_edges"]["fractions"]["unchanged"]),
                     ("  rise", lambda e: e["all_pairs_including_trivial_edges"]["fractions"]["rise"]),
                     ("  fall", lambda e: e["all_pairs_including_trivial_edges"]["fractions"]["fall"]),
                     ("susceptible designs with nested regions and a feasible adjustment", lambda e: e["proposition_7"]["adjustment feasible"]),
                     ("susceptible pairs of realizations", lambda e: e["proposition_7"]["susceptible realization pairs"]),
                     ("  feasible", lambda e: e["proposition_7"]["realization pairs feasible"])):
        a, b = get(s8), get(t8)
        o[lab.strip() if lab.strip() not in o else lab.strip() + " (trivial edges included)"] = [a, b]
        print(f"   {lab:66s} " + (f"{a} / {b}" if isinstance(a, int) else f"{a:.3f} / {b:.3f}"))

# =========================================================================== exp9
print("\n=== exp9: PBPO (Table SI) ===")
rows9 = [r for r in load_json("exp9_pbpo_rows.json") if r["mode"] == "eta"]
o = out.setdefault("exp9", {})
chains = max(r["chain"] for r in rows9) + 1
mid, al = [], []
for b in range(0, chains, 20):
    sel = [r for r in rows9 if b <= r["chain"] < b + 20]
    mid.append(float(np.mean([r["best20_hit"] for r in sel if not r["end"]]))); al.append(float(np.mean([r["best20_hit"] for r in sel])))
o["blocks_of_20_chains"] = {"best20_mid": mid, "best20_all": al}
nv = {}
for r in rows9:
    nv[r["chain"]] = max(nv.get(r["chain"], 0), r["vertex"] + 1)
miss = [r for r in rows9 if not r["best20_hit"]]
o["missed"] = {"vertices": len(rows9), "missed": len(miss), "end_or_within_two": sum(min(r["vertex"], nv[r["chain"]] - 1 - r["vertex"]) <= 2 for r in miss)}
print(f"text. best of 20 initializations: {len(miss)} of the {len(rows9)} vertices are missed, {o['missed']['end_or_within_two']} of them the first or last "
      "vertex of a hull or within two positions of it")
o["lambda_max"] = max(r["lam"] for r in rows9)
print(f"A. best of 20 initializations over the {len(mid)} disjoint blocks of 20 chains: mid-hull vertices "
      + ", ".join(f"{100 * x:.1f}" for x in mid) + " %; all vertices " + ", ".join(f"{100 * x:.1f}" for x in al) + " %")
first = [r["lam"] for r in rows9 if r["vertex"] == 0]
o["lambda_first_vertex"] = {"hulls": len(first), "median": float(np.median(first)), "above_1e3": int(sum(l >= 1e3 for l in first)), "max": max(first)}
print(f"   supporting lambda at the first vertex of the {len(first)} hulls: median {np.median(first):.0f}, above 1e3 in "
      f"{o['lambda_first_vertex']['above_1e3']}, largest {max(first):.1e}")
s9, t9 = load_json("exp9_pbpo_summary.json"), second("exp9_pbpo_summary.json")
if t9:
    print("B. second sample (seed 1009): first / second")
    for lab, grp, key, f in (("hull vertices", "eta|all", "vertices", "{:d}"), ("first or last vertices", "eta|end", "vertices", "{:d}"),
                             ("single initialization, mid-hull: median hit rate", "eta|mid", "hit_median", "{:.3f}"),
                             ("  25th percentile", "eta|mid", "hit_p25", "{:.3f}"), ("  75th percentile", "eta|mid", "hit_p75", "{:.3f}"),
                             ("single initialization, end vertices: median", "eta|end", "hit_median", "{:.3f}"),
                             ("  90th percentile", "eta|end", "hit_p90", "{:.3f}"),
                             ("best of 20, mid-hull vertices", "eta|mid", "best20_frac", "{:.3f}"), ("best of 20, all vertices", "eta|all", "best20_frac", "{:.3f}"),
                             ("mean sweeps of a single run", "eta|all", "sweeps_mean", "{:.2f}"),
                             ("largest number of sweeps, best of 20", "eta|all", "best20_sweeps_max", "{:d}"),
                             ("largest number of sweeps, single runs", "eta|all", "sweeps_max", "{:d}"),
                             ("fixed point at every vertex", "eta|all", "fixed_point_frac", "{:.3f}"),
                             ("regions kept: median hit rate", "regions|all", "hit_median", "{:.3f}"),
                             ("regions kept: best of 20, mid-hull", "regions|mid", "best20_frac", "{:.3f}"),
                             ("regions kept: best of 20, all vertices", "regions|all", "best20_frac", "{:.3f}"),
                             ("regions kept: mean sweeps", "regions|all", "sweeps_mean", "{:.2f}")):
        a, b = s9[grp][key], t9[grp][key]
        o[f"{grp}|{key}"] = [a, b]
        print(f"   {lab:50s} {f.format(a)} / {f.format(b)}")

# =========================================================================== exp11, exp13
print("\n=== exp11: Monte Carlo check; exp13: delayed message ===")
m1, m2 = load_json("exp11_montecarlo.json"), second("exp11_montecarlo.json")
d1, d2 = load_json("exp13_delayed_link.json"), second("exp13_delayed_link.json")
o = out.setdefault("exp11", {})
if m2:
    z = [np.array([c["z"] for c in m["operating_points"]]).ravel() for m in (m1, m2)]
    worst = max(m2["operating_points"], key=lambda c: max(abs(v) for v in c["z"]))
    o.update({"estimates": [len(z[0]), len(z[1])], "max_abs_z": [m1["max_abs_z"], m2["max_abs_z"]],
              "rms_z_both_runs": float(np.sqrt(np.mean(np.concatenate(z) ** 2))), "worst_case_second_run": worst["case"],
              "p_hat_mean": [m1["estimator"]["p_hat_mean"], m2["estimator"]["p_hat_mean"]],
              "p_hat_std": [m1["estimator"]["p_hat_std"], m2["estimator"]["p_hat_std"]], "std_predicted": m1["estimator"]["std_formula_observed_rate"],
              "prob_max_of_12_below_first": float((2 * norm.cdf(m1["max_abs_z"]) - 1) ** 12),
              "prob_max_of_12_above_second": float(1 - (2 * norm.cdf(m2["max_abs_z"]) - 1) ** 12)})
    print(f"B. second run (seeds 1011, 1003): largest deviation over the {len(z[0])} estimates {m1['max_abs_z']:.2f} / {m2['max_abs_z']:.2f} standard errors "
          f"(second run: {worst['case']}); root mean square of the {len(z[0]) + len(z[1])} deviations of the two runs {o['rms_z_both_runs']:.2f}")
    print(f"   for twelve independent standard normal deviations the largest is below {m1['max_abs_z']:.2f} with probability {o['prob_max_of_12_below_first']:.2f} "
          f"and above {m2['max_abs_z']:.2f} with probability {o['prob_max_of_12_above_second']:.2f}")
    print(f"   estimator of p': mean {m1['estimator']['p_hat_mean']:.4f} / {m2['estimator']['p_hat_mean']:.4f}, standard deviation "
          f"{m1['estimator']['p_hat_std']:.4f} / {m2['estimator']['p_hat_std']:.4f} (predicted {o['std_predicted']:.4f})")
o = out.setdefault("exp13", {})
if d2:
    o.update({"cases": [d1["cases"], d2["cases"]], "max_abs_error": [d1["max_abs_error"], d2["max_abs_error"]]})
    print(f"B. second sample (seed 1013): cases {d1['cases']} / {d2['cases']}; largest |exact - block form of eq. (13)| {d1['max_abs_error']:.1e} / {d2['max_abs_error']:.1e}")

save_json("sampling_error.json", out)
