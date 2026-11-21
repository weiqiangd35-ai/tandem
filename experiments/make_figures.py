"""Quick-look figures from the JSON results (for inspection; the figures of the paper are
drawn by paper_figs.py).  Usage: python experiments/make_figures.py"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 8})


def load(name):
    with open(os.path.join(R, name)) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- exp1
try:
    rows = load("exp1_rows.json")
    fams = ["running", "dirichlet3", "dirichlet5", "dirichlet10", "qgauss1", "gauss1", "gauss2", "swerling_snr3", "logistic2"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    # (a) rho_F distributions, n = 5, random vs optimal
    data, labels = [], []
    for fam in fams:
        for mode in ("random", "optimal"):
            sel = [r for r in rows if r["family"] == fam and r["n"] == 5 and r["mode"] == mode]
            if sel:
                data.append(np.concatenate([r["rho_F"] for r in sel])); labels.append(f"{fam}\n{mode}")
    axes[0].boxplot(data, showfliers=False)
    axes[0].set_xticks(range(1, len(labels) + 1)); axes[0].set_xticklabels(labels, rotation=90, fontsize=6)
    axes[0].set_ylabel(r"$\rho_F$ per stage (n = 5)")
    # (b) end-to-end product vs n (median, IQR) for random configurations
    for fam in ["dirichlet3", "dirichlet10", "gauss1", "swerling_snr3"]:
        ns, med, lo, hi = [], [], [], []
        for n in (2, 3, 5, 10, 20):
            sel = [r["prod_rho_F"] for r in rows if r["family"] == fam and r["n"] == n and r["mode"] == "random"]
            if sel:
                ns.append(n); med.append(np.median(sel)); lo.append(np.percentile(sel, 25)); hi.append(np.percentile(sel, 75))
        axes[1].errorbar(ns, med, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)], marker="o", capsize=2, label=fam)
    axes[1].set_yscale("log"); axes[1].set_xlabel("chain length n"); axes[1].set_ylabel(r"$\prod_{k \geq 2} \rho_{F,k}$ (median, IQR)")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_exp1_attenuation.png"), dpi=200)
    print("wrote fig_exp1_attenuation.png")
except FileNotFoundError:
    print("exp1 results missing")

# ---------------------------------------------------------------- exp2
try:
    s = load("exp2_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for variant, lab in (("max", "worst switching node, |W-C|"), ("N1", "N_1 only, |W-C|"), ("wrongrule_N1", "N_1 wrong rule (deterministic)")):
        ns, med, lo, hi = [], [], [], []
        for n in (3, 4, 5, 6):
            k = f"{variant}|cv|n={n}|alpha=all"
            if k in s:
                ns.append(n); med.append(s[k]["median_ratio"]); lo.append(s[k]["ci"][0]); hi.append(min(s[k]["ci"][1], 10))
        axes[0].errorbar(ns, med, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)], marker="o", capsize=2, label=lab)
    axes[0].axhline(1, color="k", lw=0.8, ls="--"); axes[0].set_yscale("log")
    axes[0].set_xlabel("n"); axes[0].set_ylabel("median tandem/parallel shift ratio [95% CI]"); axes[0].legend(fontsize=6)
    for key, lab in (("median_sync_PD_tandem", "tandem"), ("median_sync_PD_parallel", "parallel")):
        axes[1].plot([3, 4, 5, 6], [s[f"max|cv|n={n}|alpha=all"][key] for n in (3, 4, 5, 6)], marker="o", label=lab)
    axes[1].set_xlabel("n"); axes[1].set_ylabel("synchronized $P_D$ (median)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_exp2_tandem_vs_parallel.png"), dpi=200)
    print("wrote fig_exp2_tandem_vs_parallel.png")
except FileNotFoundError:
    print("exp2 results missing")

# ------------------------------------------------------------- exp3 / exp4
try:
    r3 = load("exp3_multinode_rows.json"); r4 = load("exp4_position_rows.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    sel = [r for r in r3 if r["ensemble"] == "random"]
    axes[0].scatter([r["bound_F"] for r in sel], [abs(r["shift_F"]) for r in sel], s=4, alpha=0.4, c=[r["size"] for r in sel], cmap="viridis")
    lim = [1e-6, 1]
    axes[0].plot(lim, lim, "k--", lw=0.8); axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel("Prop. 6 bound on $|W_F-\\alpha|$"); axes[0].set_ylabel("exact $|W_F-\\alpha|$ (color: |S| = 1..4)")
    for ens, lab in (("random", "random pairs"), ("design", "optimal designs")):
        ks, med, p90 = [], [], []
        for k in range(1, 8):
            v = [abs(r["shift_F"]) for r in r4 if r["ensemble"] == ens and r["k"] == k]
            if v:
                ks.append(k); med.append(np.median(v)); p90.append(np.percentile(v, 90))
        axes[1].plot(ks, med, marker="o", label=f"{lab}: median"); axes[1].plot(ks, p90, marker="x", ls=":", label=f"{lab}: p90")
    axes[1].set_yscale("log"); axes[1].set_xlabel("desynchronized node k (n = 8)"); axes[1].set_ylabel("$|W_F-\\alpha|$"); axes[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig_exp3_exp4.png"), dpi=200)
    print("wrote fig_exp3_exp4.png")
except FileNotFoundError:
    print("exp3/exp4 results missing")
