"""Experiment 9 - convergence statistics of PBPO (Section II-D, Table SI).

Random four-node chains (Dirichlet(1) K = 3, one model per node).  Exhaustive enumeration
gives the vertices of the upper hull of the deterministic set; every non-trivial vertex is
the maximizer of P_D - lambda P_F for the supporting lambda used here (midpoint of the
slopes of the two adjacent edges; for the two end vertices, the adjacent slope times 2 or
1/2).  For each vertex: the fraction of 30 single random initializations of the
Lagrangian PBPO that reach it (hit rate), reported separately for mid-hull and end
vertices; whether the best of 20 initializations (the first with eta_k = lambda at every
node) reaches it; sweeps to termination; and whether the procedure, started at a strategy
that realizes the vertex, stays there (the exhaustive optimum is a fixed point).  Both variants: the other nodes keep their
thresholds (mode 'eta', the paper's) or their region pairs (mode 'regions').
Usage: python experiments/exp9_pbpo_stats.py [--quick]
"""
import sys, time
import numpy as np
from common import *
from tandem_sim.design import pbpo, pbpo_best, init_from_lambda, _objective

QUICK = "--quick" in sys.argv


def reached(pt, v, tol=1e-9):
    """The run ends at the vertex v.  Operating points are compared, not objective values:
    the hull vertices are stored rounded to 1e-12, and at the low-P_F end of a hull the
    supporting lambda can exceed 1e3, so that an objective computed from the rounded
    vertex can differ from the exact one by more than 1e-9 (a comparison of objective
    values would count such runs as misses)."""
    return abs(pt[0] - v[0]) <= tol and abs(pt[1] - v[1]) <= tol


N_CHAINS = 4 if QUICK else 100         # blocks of 20 chains differ by up to 5 points (RESULTS.md, Section 6)
N_SINGLE = 10 if QUICK else 30
rng = np.random.default_rng(9)
rows = []
t0 = time.time()
for ci in range(N_CHAINS):
    models = family_models("dirichlet3", 4, rng, identical=False)
    items = enumerate_system_points(models)
    hull = upper_hull(distinct_points(items))
    verts = [v for v in hull if v not in ((0.0, 0.0), (1.0, 1.0))]
    slopes = [(hull[i + 1][1] - hull[i][1]) / (hull[i + 1][0] - hull[i][0]) for i in range(len(hull) - 1)]
    for vi, v in enumerate(verts):
        i = hull.index(v)
        s_left, s_right = slopes[i - 1], slopes[i]        # slope of the edge into / out of v
        end = vi == 0 or vi == len(verts) - 1
        lam = 0.5 * (s_left + s_right)
        target = _objective(v, None, lam)
        # sanity: v maximizes P_D - lam P_F over the hull vertices
        assert all(_objective(w, None, lam) <= target + 1e-9 for w in hull), "supporting lambda"
        for mode in ("eta", "regions"):
            # the vertex is a fixed point: started at a strategy that realizes it, the procedure stays there
            s_fp, pt_fp, sw_fp, _ = pbpo(models, lam=lam, init=strategies_at(items, v)[0], mode=mode)
            fixed = bool(reached(pt_fp, v))
            hits, sweeps = [], []
            for r in range(N_SINGLE):
                s, pt, sw, _ = pbpo(models, lam=lam, rng=np.random.default_rng(1000 * ci + 50 * vi + r), mode=mode)
                hits.append(reached(pt, v)); sweeps.append(sw)
            s20, pt20, st = pbpo_best(models, lam=lam, restarts=20, seed=ci, mode=mode)
            rows.append({"chain": ci, "vertex": vi, "end": end, "lam": lam, "mode": mode,
                         "hit_rate": float(np.mean(hits)), "sweeps_mean": float(np.mean(sweeps)), "sweeps_max": int(max(sweeps)),
                         "best20_hit": bool(reached(pt20, v)),
                         "fixed_point": fixed, "fixed_point_sweeps": int(sw_fp),
                         "best20_sweeps_max": int(max(st["sweeps"]))})
    print(f"  chain {ci+1}/{N_CHAINS}: {len(verts)} vertices, {time.time()-t0:.0f}s", flush=True)
save_json("exp9_pbpo_rows.json", rows)

summary = {}
for mode in ("eta", "regions"):
    for grp, sel in (("mid", [r for r in rows if r["mode"] == mode and not r["end"]]),
                     ("end", [r for r in rows if r["mode"] == mode and r["end"]]),
                     ("all", [r for r in rows if r["mode"] == mode])):
        h = np.array([r["hit_rate"] for r in sel])
        summary[f"{mode}|{grp}"] = {"vertices": len(sel), "hit_median": float(np.median(h)), "hit_p10": float(np.percentile(h, 10)),
                                     "hit_p25": float(np.percentile(h, 25)), "hit_p75": float(np.percentile(h, 75)),
                                     "hit_p90": float(np.percentile(h, 90)), "hit_min": float(h.min()), "hit_max": float(h.max()),
                                     "best20_frac": float(np.mean([r["best20_hit"] for r in sel])),
                                     "fixed_point_frac": float(np.mean([r["fixed_point"] for r in sel])),
                                     "fixed_point_sweeps_max": int(max(r["fixed_point_sweeps"] for r in sel)),
                                     "sweeps_mean": float(np.mean([r["sweeps_mean"] for r in sel])),
                                     "sweeps_max": int(max(r["sweeps_max"] for r in sel)),
                                     "best20_sweeps_max": int(max(r["best20_sweeps_max"] for r in sel))}
        s = summary[f"{mode}|{grp}"]
        print(f"{mode:8s} {grp:4s} vertices={s['vertices']:4d} single-init hit: median {s['hit_median']:.2f} "
              f"p10-p90 [{s['hit_p10']:.2f}, {s['hit_p90']:.2f}] min-max [{s['hit_min']:.2f}, {s['hit_max']:.2f}]  "
              f"best-of-20 reaches {s['best20_frac']:.2f}  sweeps mean {s['sweeps_mean']:.1f} max {s['sweeps_max']}  "
              f"fixed point {s['fixed_point_frac']:.3f}")
save_json("exp9_pbpo_summary.json", summary)
