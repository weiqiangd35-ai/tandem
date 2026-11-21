"""Tables of the supplementary material, generated from the saved experiment summaries.

Writes (to the directory given as the first argument, default results/):
  tab_s2.tex  attenuation statistics (Table SII), from exp1_summary.json; the last column is the
              median product of the relay-like adversary at n = 5 and n = 20
  tab_s2_intervals.json  distribution-free 95% confidence intervals of the median products of
              Table SII (random configurations and greedy adversary), from exp1_rows.json; the
              caption of the table quotes how far the intervals extend from the medians
  tab_s3.tex  paired tandem-versus-parallel comparison (Table SIII), from exp2_summary.json;
              the column after the confidence interval is the range of the median ratio over
              the five disjoint blocks of 200 models
  tab_s4.tex  immunity fractions, number of susceptible cases and median shift by position (Table SIV),
              from exp2_summary.json
  tab_s8.tex  position-resolved comparison for the two families of quantized Gaussian sensors (Table SVIII),
              from exp2_summary_qgauss3_0.3_1.5.json and exp2_summary_qgauss3_1.5_3.0.json
Usage: python experiments/supp_tables.py [outdir]
"""
import json, os, sys
import numpy as np
from scipy.stats import binom
from common import load_json

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results")


def sci(x):
    if x is None:
        return "--"
    if x == 0:
        return "$0$"
    m, e = f"{x:.1e}".split("e")          # round first, so that 9.99e-5 prints as 1.0e-4
    return f"${m}\\times10^{{{int(e)}}}$"


def two(x):
    """Two decimals, three below 0.1."""
    return f"{x:.2f}" if x >= 0.1 else f"{x:.3f}"


# ------------------------------------------------------------------ Table S-III (exp1)
s1 = load_json("exp1_summary.json")
FAMS = [("running", "running model ($K=3$)"), ("dirichlet3", "Dirichlet $K=3$"), ("dirichlet5", "Dirichlet $K=5$"),
        ("dirichlet10", "Dirichlet $K=10$"), ("qgauss1", "quantized Gaussian $d'=1$"), ("qgauss2", "quantized Gaussian $d'=2$"),
        ("gauss0.5", "Gaussian $d'=0.5$"), ("gauss1", "Gaussian $d'=1$"), ("gauss2", "Gaussian $d'=2$"), ("gauss3", "Gaussian $d'=3$"),
        ("swerling_snr1", "Swerling I, SNR $1$"), ("swerling_snr3", "Swerling I, SNR $3$"), ("swerling_snr10", "Swerling I, SNR $10$"),
        ("logistic2", "logistic $\\mu=2$")]
lines = []
for fam, lab in FAMS:
    r5 = s1.get(f"{fam}|n=5|random")
    if r5 is None:
        continue
    rho = r5["rho_F"]
    prods = [s1.get(f"{fam}|n={n}|random", {}).get("prod_rho_F", {}).get("median") for n in (2, 5, 10, 20)]
    opt = s1.get(f"{fam}|n=5|optimal", {}).get("rho_F", {}).get("median")
    adv = [s1.get(f"{fam}|n={n}|adversarial_max", {}).get("prod_rho_F", {}).get("median") for n in (5, 20)]
    lines.append(f"{lab} & {rho['median']:.2f} ({rho['p10']:.2f}--{rho['p90']:.2f}) & {r5['frac_rho_F_gt_0.9']:.2f} & "
                 + " & ".join(sci(p) for p in prods)
                 + f" & {'--' if opt is None else f'{opt:.2f}'} & {'--' if adv[0] is None else ' / '.join(two(x) for x in adv)} \\\\")
open(os.path.join(OUT, "tab_s2.tex"), "w").write("\n".join(lines) + "\n")


# Sampling error of the median products of Table SII.  A cell is a median over independent
# configurations (300 random ones, or 1000 models for an adversary), so the order statistics
# x_(l) and x_(N+1-l) bound the population median with probability P(l <= B <= N-l),
# B ~ Bin(N, 1/2), whatever the distribution; l is the largest rank with a level of at least 95%.
def median_interval(x, level=0.95):
    xs = np.sort(np.asarray(x, float)); N = len(xs)
    l = max(k for k in range(1, N // 2 + 1) if binom.cdf(N - k, N, 0.5) - binom.cdf(k - 1, N, 0.5) >= level)
    med, lo, hi = float(np.median(xs)), float(xs[l - 1]), float(xs[N - l])
    return {"N": N, "ranks": [l, N + 1 - l], "level": float(binom.cdf(N - l, N, 0.5) - binom.cdf(l - 1, N, 0.5)),
            "median": med, "interval": [lo, hi], "factor": (max(hi / med, med / lo) if lo > 0 else None)}


prod = {}
for r in load_json("exp1_rows.json"):
    if r["mode"] in ("random", "adversarial_max"):
        prod.setdefault((r["mode"], r["family"], r["n"]), []).append(r["prod_rho_F"])
QUANT = ("qgauss1", "qgauss2")
iv = {"random": {}, "adversary": {}, "random_largest_factor": {}}
for fam, _ in FAMS:
    for n in (2, 5, 10, 20):
        if ("random", fam, n) in prod:
            iv["random"][f"{fam}|n={n}"] = median_interval(prod[("random", fam, n)])
        if ("adversarial_max", fam, n) in prod:
            iv["adversary"][f"{fam}|n={n}"] = median_interval(prod[("adversarial_max", fam, n)])
for n in (2, 5, 10, 20):
    fac = {fam: iv["random"][f"{fam}|n={n}"]["factor"] for fam, _ in FAMS if f"{fam}|n={n}" in iv["random"]}
    iv["random_largest_factor"][f"n={n}"] = {"quantized": max(v for f, v in fac.items() if f in QUANT),
                                             "other": max(v for f, v in fac.items() if f not in QUANT)}
with open(os.path.join(OUT, "tab_s2_intervals.json"), "w") as fh:
    json.dump(iv, fh, indent=1)
print("95% interval of a median product of random configurations: largest distance from the median, as a factor")
for n in (2, 5, 10, 20):
    e = iv["random_largest_factor"][f"n={n}"]
    print(f"  n = {n:2d}: {e['other']:.2f} (quantized Gaussian families: {e['quantized']:.2f})")

# ------------------------------------------------------------------ Tables S-IV and S-V (exp2)
s2 = load_json("exp2_summary.json")
NS = sorted(int(k.split("|")[1].split("=")[1]) for k in s2 if k.startswith("immune|"))


def ratio_cell(e):
    return f"{e['median_ratio']:.2f} [{e['ci'][0]:.2f}, {e['ci'][1]:.2f}]"


lines = []
first = True
for n in NS:
    for k in range(1, n):
        e = s2.get(f"position|n={n}|k={k}")
        if not e or "median_ratio" not in e:
            continue
        lab = "node $k$ vs.\\ mean susceptible parallel LD" if first else ""
        first = False
        bm = [x for x in e.get("block_median_ratios", []) if x is not None]
        blocks = f"{min(bm):.2f}--{max(bm):.2f}" if len(bm) >= 2 else "--"      # range over the blocks of 200 models
        lines.append(f"{lab} & {n} & {k} & {e['pairs']} & {ratio_cell(e)} & {blocks} & {e['frac_ratio_lt_1']:.2f} & {e['frac_ties']:.2f} & "
                     f"{e['wilcoxon_less_p']:.1e} & {e['wilcoxon_greater_p']:.1e} & {e['mannwhitney_p']:.1e} \\\\")
lines.append("\\midrule")
for variant, lab in (("max", "worst switching node"),
                     ("wrongrule_N1", "$N_1$ wrong rule vs.\\ worst parallel LD")):
    first = True
    for n in NS:
        e = s2.get(f"{variant}|cv|n={n}|alpha=all")
        if not e:
            continue
        lines.append(f"{lab if first else ''} & {n} & -- & {e['pairs']} & {ratio_cell(e)} & -- & {e['frac_ratio_lt_1']:.2f} & {e['frac_ties']:.2f} & "
                     f"{e['wilcoxon_p']:.1e} & {e['wilcoxon_greater_p']:.1e} & {e['mannwhitney_p']:.1e} \\\\")
        first = False
open(os.path.join(OUT, "tab_s3.tex"), "w").write("\n".join(lines) + "\n")

lines = []
for n in NS:
    im = s2[f"immune|n={n}"]
    med = " / ".join(f"{s2[f'position|n={n}|k={k}']['median']:.4f}" for k in range(1, n))
    cnt = " / ".join(str(s2[f'position|n={n}|k={k}']['N']) for k in range(1, n))
    par = s2[f"position|n={n}|parallel"]
    lines.append(f"{n} & {im['tandem_immune']:.2f} / {im['parallel_cv_immune']:.2f} & {cnt} & {med} & {par['median']:.4f} & "
                 f"{im['median_sync_PD_tandem']:.3f} / {im['median_sync_PD_parallel_cv']:.3f} \\\\")
open(os.path.join(OUT, "tab_s4.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ Table S-VIII (exp2 with --family=qgauss3:LO:HI)
fams = [load_json(f"exp2_summary_qgauss3_{t}.json") for t in ("0.3_1.5", "1.5_3.0")]
lines = []
for n in (3, 4, 5, 6):
    for k in range(1, n):
        cells = []
        for s8 in fams:
            e = s8[f"position|n={n}|k={k}"]
            cells.append(f"{e['pairs']} & {e['median_ratio']:.2f} [{e['ci'][0]:.2f}, {e['ci'][1]:.2f}] & {e['frac_ratio_lt_1']:.2f}")
        lines.append(("\\midrule\n" if (k == 1 and n > 3) else "") + f"{n} & {k} & " + " & ".join(cells) + " \\\\")
open(os.path.join(OUT, "tab_s8.tex"), "w").write("\n".join(lines) + "\n")
print("tables written to", OUT)

# ------------------------------------------------------------------ Table S-VI (exp5): correction statistics
s5 = load_json("exp5_correction.json")["stats"]


def pct(x):
    return "--" if x is None else f"{100 * x:.0f}\\%"


def num(x):
    return "--" if x is None else (f"{x:.3f}" if x >= 1e-2 else sci(x))


def thou(x):
    return f"{x:,}".replace(",", "\\,")


NS5 = [k for k in ("2", "3", "4") if k in s5]
def dec(x):
    return "--" if x is None else f"{x:.4f}"


rows5 = [
    ("randomized designs", lambda s: thou(s["designs"])),
    ("\\quad susceptible", lambda s: thou(s["susceptible_designs"])),
    # what randomization gains over the two designs that need no common draw: the deterministic optimum G and
    # the best design S in which a single node randomizes (C_D - S_D is the part that needs the common draw)
    ("$C_D-G_D$: median", lambda s: f"{s['randomization_gain']['median']:.3f}"),
    ("$C_D-S_D$: median", lambda s: f"{s['common_draw_gain']['median']:.4f}"),
    ("\\quad 90\\% quantile", lambda s: f"{s['common_draw_gain']['p90']:.4f}"),
    ("\\quad largest", lambda s: f"{s['common_draw_gain']['max']:.3f}"),
    ("$P_F$ rises / falls", lambda s: f"{pct(s['frac_PF_rises'])} / {pct(s['frac_PF_falls'])}"),
    ("$|W_{P_F}-\\alpha|$: median / max", lambda s: f"{s['abs_overshoot']['median']:.3f} / {s['abs_overshoot']['max']:.2f}"),
    ("rule of the DFC switches", lambda s: pct(s["frac_dfc_switches"])),
    ("\\quad adjustment meets the constraint", lambda s: pct(s["frac_feasible_given_dfc_switches"])),
    ("\\quad \\quad when $P_F$ rises", lambda s: pct(s["frac_feasible_given_rise_and_dfc_switches"])),
    ("\\quad \\quad equals DFC-layer / downstream optimum", lambda s: f"{pct(s['frac_q_equals_dfc'])} / {pct(s['frac_q_equals_down'])}"),
    ("\\quad \\quad median gap to the DFC-layer optimum", lambda s: dec(s["gap_dfc_minus_q"]["median"])),
    ("\\quad \\quad median gap to the downstream optimum", lambda s: dec(s["gap_down_minus_q"]["median"])),
    ("\\quad \\quad largest gap", lambda s: f"{max(s['gap_dfc_minus_q']['max'], s['gap_down_minus_q']['max']):.2f}"),
    # corrected P_D below the deterministic optimum G: the adjustment among the cases in which it meets the
    # constraint (hence in the indented block), the two optimal corrections among all cases (last rows);
    # the same comparison with the best design S in which a single node randomizes
    ("\\quad \\quad below $G$ / below $S$", lambda s: f"{pct(s['frac_q_below_G'])} / {pct(s['frac_q_below_S'])}"),
    ("rule of the DFC fixed: $P_F>\\alpha$", lambda s: pct(s["frac_constraint_violated_no_switch"])),
    ("all cases: DFC-layer below $G$ / below $S$", lambda s: f"{pct(s['frac_dfc_below_G'])} / {pct(s['frac_dfc_below_S'])}"),
    ("all cases: downstream below $G$ / below $S$", lambda s: f"{pct(s['frac_down_below_G'])} / {pct(s['frac_down_below_S'])}"),
]
lines = [lab + " & " + " & ".join(f(s5[k]) for k in NS5) + " \\\\" for lab, f in rows5]
open(os.path.join(OUT, "tab_s6.tex"), "w").write("\n".join(lines) + "\n")
print("correction statistics written (tab_s6.tex)")
