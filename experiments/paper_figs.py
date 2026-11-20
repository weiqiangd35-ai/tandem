"""Figures of the manuscript and of the supplementary material, generated from the saved
results (run the experiments first).  Usage: python experiments/paper_figs.py [outdir]
Writes fig_position.pdf (Fig. 2 of the paper) and, for the supplement, fig_gap_eps.pdf (Fig. S1),
fig_multinode.pdf (Fig. S2), fig_dr_two_panel.pdf (Fig. S3), fig_three_node.pdf (Fig. S4) and
fig_attenuation_stats.pdf (Fig. S5).
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SIM)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
from tandem_sim import *
from tandem_sim.design import distinct_points, strategies_at
R = os.path.join(SIM, "results")

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.1, "lines.markersize": 3.5,
                     "figure.dpi": 100, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.3})
W1 = 3.45   # one column (inches)


def load(name):
    with open(os.path.join(R, name)) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- Fig. S3: dependent randomization, two panels
# Same content and style as Figs. 2-3 of the submitted version (one panel per alpha), side by side.
RUN = DiscreteModel([0.8, 0.1, 0.1], [0.3, 0.1, 0.6])
items = enumerate_system_points([RUN, RUN])
pts = distinct_points(items)
hull = upper_hull(pts)
fig, axes = plt.subplots(1, 2, figsize=(W1, 1.55))
for ax, alpha, xlim, ylim in ((axes[0], 0.17, (-0.01, 0.42), (0.3, 1.0)), (axes[1], 0.075, (-0.004, 0.165), (0.3, 0.9))):
    sp = support_pair(hull, alpha)
    G = deterministic_optimum(pts, alpha)
    hx, hy = zip(*hull)
    ax.plot(hx, hy, "-", color="red", lw=1.3, label="randomized ROC (hull)", zorder=2)
    ax.plot(*zip(*pts), "o", color="steelblue", ms=3.2, alpha=0.8, label="deterministic points", zorder=3)
    ax.plot([sp.A[0], sp.B[0]], [sp.A[1], sp.B[1]], "o", color="green", ms=5, label="$A$, $B$ (supporting)", zorder=6)
    ax.plot(G[0], G[1], "s", color="orange", mec="k", mew=0.5, ms=5, label="$G$ (deterministic optimum)", zorder=5)
    ax.plot(sp.C[0], sp.C[1], "o", color="red", mec="k", mew=0.5, ms=5.5, label="$C$ (randomized)", zorder=7)
    ax.axvline(alpha, color="gray", lw=0.8, ls=":")
    ax.annotate("", xy=(alpha, G[1]), xytext=(alpha, sp.C[1]), arrowprops=dict(arrowstyle="->", color="purple", lw=0.9))
    ax.annotate(f"$\\Delta P_D=+{sp.C[1]-G[1]:.3f}$", (alpha, 0.5 * (G[1] + sp.C[1])), xytext=(4, 0), textcoords="offset points",
                color="purple", fontsize=6.5, va="center")
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xlabel("$P_F$"); ax.set_title(f"$\\alpha={alpha}$")
    ax.annotate("B", sp.B, xytext=(4, -7), textcoords="offset points", fontsize=7, weight="bold")
    ax.annotate("C", sp.C, xytext=(-8, 3), textcoords="offset points", fontsize=7, weight="bold", color="red")
    if np.hypot(sp.A[0] - G[0], sp.A[1] - G[1]) < 1e-9:
        ax.annotate("A = G", sp.A, xytext=(4, -8), textcoords="offset points", fontsize=7, weight="bold")
    else:
        ax.annotate("A", sp.A, xytext=(-9, 2), textcoords="offset points", fontsize=7, weight="bold")
        ax.annotate("G", G, xytext=(3, -8), textcoords="offset points", fontsize=7, weight="bold", color="darkorange")
axes[0].set_ylabel("$P_D$")
# symbol key is given in the caption (no room for a legend in the two-panel layout)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_dr_two_panel.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------------------------------------------------------------- Fig. S4: three-node example (Figs. 4-5 of the submitted version)
models3 = [RUN] * 3
sA3 = Strategy(1, ((2, 0), (2, 0)))
sB3 = Strategy(2, ((2, 0), (2, 0)))
sync3 = evaluate_chain(models3, sA3)
from tandem_sim.desync import evaluate_assignment
des3 = evaluate_assignment(models3, sA3, sB3, ("B", "A", "A"))
mags3 = [float(np.hypot(a[0] - b[0], a[1] - b[1])) for a, b in zip(sync3, des3)]
fig, axes = plt.subplots(1, 2, figsize=(W1, 1.38), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
ax.plot(*zip(*sync3), "o-", color="steelblue", ms=5, label="synchronized")
ax.plot(*zip(*des3), "s--", color="crimson", ms=5, label="desynchronized")
for a, b, k in zip(sync3, des3, (1, 2, 3)):
    ax.plot([a[0], b[0]], [a[1], b[1]], ":", color="gray", lw=0.8)
    ax.annotate(f"$N_{k}$", a, xytext=(4, -9), textcoords="offset points", fontsize=6.5, color="steelblue")
    ax.annotate(f"$N_{k}$", b, xytext=((2, 4) if k == 3 else (-12, 2)), textcoords="offset points", fontsize=6.5, color="crimson")
ax.set_xlabel("$P_F$"); ax.set_ylabel("$P_D$"); ax.set_xlim(0.04, 0.44); ax.set_ylim(0.56, 1.0)
ax.text(0.265, 0.695, "synchronized", color="steelblue", fontsize=6)
ax.text(0.05, 0.962, "desynchronized", color="crimson", fontsize=6)
ax = axes[1]
bars = ax.bar(["$N_1$\n(injected)", f"$N_2$\n$\\times${mags3[1] / mags3[0]:.2f}", f"$N_3$ (system)\n$\\times${mags3[2] / mags3[1]:.2f}"],
              mags3, color=["#d62728", "#ff9896", "#ffbb91"], edgecolor="k", lw=0.5, width=0.55)
for b_, m in zip(bars, mags3):
    ax.annotate(f"{m:.4f}", (b_.get_x() + b_.get_width() / 2, m), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6)
ax.set_ylabel("$|\\Delta|$ at node"); ax.set_ylim(0, 0.175); ax.tick_params(axis="x", labelsize=6.5)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_three_node.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Fig. S5: attenuation statistics (exp1)
# Abbreviations (explained in the caption): Dir.3 / Dir.10 = Dirichlet K = 3 / 10; qGauss = quantized
# Gaussian d' = 1; Gauss1 / Gauss3 = Gaussian d' = 1 / 3; Swer10 = Swerling I, SNR 10; opt. = NP-optimal
# designs on Dirichlet K = 3 models.
rows = load("exp1_rows.json")
fig, axes = plt.subplots(1, 2, figsize=(W1, 1.55))
fams = [("dirichlet3", "Dir.3"), ("dirichlet10", "Dir.10"), ("qgauss1", "qGauss"), ("gauss1", "Gauss1"),
        ("gauss3", "Gauss3"), ("swerling_snr10", "Swer10")]
data, labels, colors = [], [], []
for fam, lab in fams:
    sel = [r for r in rows if r["family"] == fam and r["n"] == 5 and r["mode"] == "random"]
    data.append(np.concatenate([r["rho_F"] for r in sel])); labels.append(lab); colors.append("#5b84b1")
opt = [r for r in rows if r["family"] == "dirichlet3" and r["n"] == 5 and r["mode"] == "optimal"]
data.append(np.concatenate([r["rho_F"] for r in opt])); labels.append("opt."); colors.append("#c0392b")
bp = axes[0].boxplot(data, showfliers=False, whis=(10, 90), widths=0.6, patch_artist=True, medianprops={"color": "k"})
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.5)
axes[0].set_xticks(range(1, len(labels) + 1)); axes[0].set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right", rotation_mode="anchor")
axes[0].set_ylabel(r"$\rho_{F,k}$ per stage, $n=5$"); axes[0].set_ylim(0, 1.02)
for fam, lab, c in [("dirichlet3", "Dir.3", "#5b84b1"), ("dirichlet10", "Dir.10", "#1e8449"),
                    ("gauss1", "Gauss1", "#e67e22"), ("swerling_snr10", "Swer10", "#8e44ad")]:
    ns, med, lo, hi = [], [], [], []
    for n in (2, 3, 5, 10, 20):
        sel = [r["prod_rho_F"] for r in rows if r["family"] == fam and r["n"] == n and r["mode"] == "random"]
        sel = [v for v in sel if v > 0]
        ns.append(n); med.append(np.median(sel)); lo.append(np.percentile(sel, 25)); hi.append(np.percentile(sel, 75))
    axes[1].errorbar(ns, med, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)], marker="o", capsize=1.5, color=c, label=lab)
ns, med = [], []
for n in (2, 3, 5, 10):
    sel = [r["prod_rho_F"] for r in rows if r["family"] == "dirichlet3" and r["n"] == n and r["mode"] == "optimal"]
    ns.append(n); med.append(np.median(sel))
axes[1].plot(ns, med, "s--", color="#c0392b", label="Dir.3 opt.")
axes[1].set_yscale("log"); axes[1].set_xlabel("chain length $n$"); axes[1].set_ylabel(r"$\prod_{k\geq 2}\rho_{F,k}$")
axes[1].set_xticks([2, 5, 10, 20]); axes[1].set_ylim(1e-12, 3)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=0.9, handletextpad=0.5)
fig.subplots_adjust(left=0.13, right=0.985, bottom=0.36, top=0.97, wspace=0.55)
fig.savefig(os.path.join(OUT, "fig_attenuation_stats.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------------------------------------------------------------- Fig. 2: position-resolved comparison (exp2)
rows2 = load("exp2_rows.json")
TOL = 1e-9          # susceptible = shift above TOL (immune cases are reported separately)
fig, ax = plt.subplots(figsize=(W1, 1.55))
cols = {4: "#5b84b1", 5: "#1e8449", 6: "#c0392b"}
mark = {4: "o", 5: "s", 6: "^"}     # one marker shape per chain length, so that the curves also read in black and white
XP = 5.9            # abscissa of the parallel reference markers
for n in (4, 5, 6):
    sel = [r for r in rows2 if r["n"] == n and "tandem_shift_by_k" in r]
    ks, med, p90, cnt = [], [], [], []
    for k in range(1, n):
        v = [r["tandem_shift_by_k"][str(k)] for r in sel if str(k) in r["tandem_shift_by_k"] and r["tandem_shift_by_k"][str(k)] > TOL]
        ks.append(k); med.append(np.median(v)); p90.append(np.percentile(v, 90)); cnt.append(len(v))
    ax.plot(ks, med, "-", marker=mark[n], color=cols[n], ms=4.5)
    ax.plot(ks, p90, ":", marker=mark[n], color=cols[n], lw=0.9, ms=3.5, mfc="white", mew=0.8)
    pv = [x for r in rows2 if r["n"] == n and "parallel_cv_shift_by_i" in r for x in r["parallel_cv_shift_by_i"].values() if x > TOL]
    ax.plot([XP + 0.17 * (n - 5)], [np.median(pv)], mark[n], mfc="white", mec=cols[n], mew=1.2, ms=5.5)   # side by side
    ax.annotate(f"$n={n}$", (ks[-1], med[-1]), xytext=(4, 3 if n == 4 else -2), textcoords="offset points", color=cols[n], fontsize=7, va="center")
from matplotlib.lines import Line2D
style = [Line2D([0], [0], color="k", marker="o", ms=4.5, ls="-", label="tandem median"),
         Line2D([0], [0], color="k", marker="o", ms=3.5, mfc="white", mew=0.8, ls=":", lw=0.9, label="tandem 90th pct."),
         Line2D([0], [0], color="k", marker="o", ms=5.5, mfc="white", mec="k", mew=1.2, ls="none", label="parallel LD median (P)")]
fig.legend(handles=style, loc="lower center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=0.9)
ax.set_yscale("log"); ax.set_xlabel("position $k$ of the desynchronized node"); ax.set_ylabel(r"$\|W-C\|$")
ax.set_xticks([1, 2, 3, 4, 5, XP]); ax.set_xticklabels(["1", "2", "3", "4", "5", "P"]); ax.set_xlim(0.6, 6.3); ax.set_ylim(2e-3, 0.2)
ax.axvline(5.45, color="gray", lw=0.6, ls="-")
fig.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(OUT, "fig_position.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------------------------------------------------------------- Fig. S2: several desynchronized nodes (exp3)
r3 = load("exp3_multinode_rows.json")
fig, axes = plt.subplots(1, 2, figsize=(W1, 1.45))
sel = [r for r in r3 if r["ensemble"] == "random"]
sc = axes[0].scatter([r["bound_F"] for r in sel], [abs(r["shift_F"]) for r in sel], s=3, alpha=0.35,
                     c=[r["size"] for r in sel], cmap="viridis", vmin=1, vmax=4)
lim = [1e-6, 1]
axes[0].plot(lim, lim, "k--", lw=0.8); axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlim(lim); axes[0].set_ylim(lim)
axes[0].set_xlabel("bound of Prop. 6"); axes[0].set_ylabel(r"exact $|W_{P_F}-\alpha|$")
cb = fig.colorbar(sc, ax=axes[0], ticks=[1, 2, 3, 4], pad=0.02); cb.set_label("$|S|$", fontsize=7); cb.ax.tick_params(labelsize=6)
for ens, lab, c in (("random", "random pairs", "#5b84b1"), ("design", "PBPO designs", "#c0392b")):
    sizes, med, p90 = [], [], []
    for s in (1, 2, 3, 4):
        v = [abs(r["shift_F"]) for r in r3 if r["ensemble"] == ens and r["size"] == s]
        if v:
            sizes.append(s); med.append(np.median(v)); p90.append(np.percentile(v, 90))
    axes[1].plot(sizes, med, "o-", color=c, label=lab); axes[1].plot(sizes, p90, "x:", color=c)
# solid with circles: median; dotted with crosses: 90th percentile (stated in the caption)
axes[1].set_yscale("log"); axes[1].set_xlabel("$|S|$"); axes[1].set_ylabel(r"$|W_{P_F}-\alpha|$")
axes[1].set_xticks([1, 2, 3, 4]); axes[1].set_xlim(0.8, 4.2); axes[1].set_ylim(3e-5, 0.2)
axes[1].legend(fontsize=5.5, loc="lower right", handlelength=1.5, labelspacing=0.2, borderpad=0.3)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_multinode.pdf")); plt.close(fig)

# ---------------------------------------------------------------- supplement: gap vs eps (exp7)
e7 = load("exp7_prop3_gap.json")
fig, ax = plt.subplots(figsize=(W1, 1.8))
eps = [r["eps"] for r in e7["gap_vs_eps"]]; g = [r["max_gap"] for r in e7["gap_vs_eps"]]
ax.semilogx(eps, g, "o-", color="#5b84b1", label="maximal concavity gap")
ax.axvline(1 / 768, color="gray", ls="--", lw=0.8); ax.annotate(r"$\varepsilon=1/768$", (1 / 768, 0.0062), xytext=(3, 0), textcoords="offset points", fontsize=6)
ax.axhline(1 / 160, color="gray", ls=":", lw=0.8); ax.annotate(r"$1/160$", (1.2e-4, 1 / 160), xytext=(0, 2), textcoords="offset points", fontsize=6)
ax.set_xlabel(r"smoothing parameter $\varepsilon$"); ax.set_ylabel("gap"); ax.set_ylim(0, 0.0078)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_gap_eps.pdf")); plt.close(fig)
print("figures written")
