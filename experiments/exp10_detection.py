"""Experiment 10 - what the final LD can see (Section V-D).

The DFC observes the common draw Z_t and the arriving message u_{n-1,t}.  Under
synchronization the conditional rates r_S = P(u_{n-1} = 1 | Z = S) are known at design time
(hypothesis mixture with prior P0).  If node k draws independently with marginal p', the
rate becomes p' m_S^{(k,A)} + (1 - p') m_S^{(k,B)}, where m_S^{(k,X)} is the rate of the chain
that follows S everywhere except at node k, which uses X.  The script prints, for the worked
examples of Section V-C, the synchronized and desynchronized conditional rates, the shift
delta_S, the per-epoch standard deviation sigma_S, the number of Z = S epochs for a 6-sigma
separation (36 sigma^2 / delta^2) and the standard deviation of the moment estimator of p'
times sqrt(T_S), sqrt(r (1 - r)) / |m_S^{(k,A)} - m_S^{(k,B)}| with r the desynchronized rate.
Usage: python experiments/exp10_detection.py
"""
import json
import numpy as np
from common import *
from tandem_sim.desync import evaluate_assignment


def rate(models, sA, sB, assign, P0):
    f, d = evaluate_assignment(models, sA, sB, assign)[-2]      # output of node n-1
    return P0 * f + (1 - P0) * d


def analyze(models, sA, sB, p, k, pprime, P0):
    n = len(models)
    out = {}
    for S in ("A", "B"):
        base = [S] * n
        mS = rate(models, sA, sB, base, P0)
        aX = list(base); aX[k - 1] = "A"; mA = rate(models, sA, sB, aX, P0)
        bX = list(base); bX[k - 1] = "B"; mB = rate(models, sA, sB, bX, P0)
        r = pprime * mA + (1 - pprime) * mB
        delta = r - mS
        sigma = float(np.sqrt(mS * (1 - mS)))
        pS = p if S == "A" else 1 - p
        out[S] = {"sync_rate": mS, "desync_rate": r, "delta": delta, "sigma": sigma,
                  "T_S_6sigma": (36 * sigma ** 2 / delta ** 2) if abs(delta) > 1e-12 else None,
                  "T_total_6sigma": (36 * sigma ** 2 / delta ** 2 / pS) if abs(delta) > 1e-12 else None,
                  # moment estimator p' = (r_hat - m^B) / (m^A - m^B): its standard deviation uses the
                  # rate actually observed, i.e. the desynchronized rate r
                  "pprime_std_times_sqrtT_S": (float(np.sqrt(r * (1 - r))) / abs(mA - mB)) if abs(mA - mB) > 1e-12 else None}
    return out


results = {}
sA, sB = Strategy(1, ((3, 1),)), Strategy(2, ((2, 1),))              # running model, alpha = 0.075
for P0 in (0.9, 0.99):
    results[f"two_node_alpha0.075_P0={P0}"] = analyze([RUNNING] * 2, sA, sB, 0.5, 1, 0.5, P0)
d = load_json("exp5_correction.json")


def three_node(alpha):
    t = [t for t in d["three_node"] if abs(t["alpha"] - alpha) < 1e-9][0]
    sA = Strategy(t["sA"]["j1"], tuple(tuple(x) for x in t["sA"]["pairs"]))
    sB = Strategy(t["sB"]["j1"], tuple(tuple(x) for x in t["sB"]["pairs"]))
    return sA, sB, t["p"], t["switching"]


# alpha = 0.078: N_2 is the only switching node; alpha = 0.1675: N_1 and N_2 both switch,
# and their failures move the two conditional rates with opposite sign patterns
for alpha in (0.078, 0.1675):
    sA3, sB3, p3, sw = three_node(alpha)
    for k in sw:
        results[f"three_node_alpha{alpha}_k={k}_P0=0.9"] = analyze([RUNNING] * 3, sA3, sB3, p3, k, p3, 0.9)
for key, out in results.items():
    print(key)
    for S, v in out.items():
        print(f"  Z={S}: sync {v['sync_rate']:.4f} desync {v['desync_rate']:.4f} delta {v['delta']:+.4f} sigma {v['sigma']:.3f} "
              f"T_S(6 sigma) {v['T_S_6sigma'] and round(v['T_S_6sigma'])}  std(p') x sqrt(T_S) {v['pprime_std_times_sqrtT_S'] and round(v['pprime_std_times_sqrtT_S'], 2)}")
save_json("exp10_detection.json", results)
