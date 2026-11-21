"""Run every experiment script in sequence, logging to results/<name>_log.txt."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
scripts = sys.argv[1:] or ["exp1_attenuation.py", "exp3_multinode.py", "exp4_position.py",
                           "exp5_correction.py", "exp2_tandem_vs_parallel.py", "exp6_prop2_parametric.py",
                           "exp7_prop3_gap.py", "knot_summary.py", "exp8_sign_ensemble.py", "exp9_pbpo_stats.py",
                           "exp10_detection.py", "exp11_montecarlo.py", "exp12_certificate.py",
                           "exp13_delayed_link.py", "make_figures.py"]
for s in scripts:
    t0 = time.time()
    with open(os.path.join(RES, s.replace(".py", "") + "_log.txt"), "w") as log:
        rc = subprocess.call([sys.executable, "-u", os.path.join(HERE, s)], stdout=log, stderr=subprocess.STDOUT)
    print(f"{s}: exit {rc} in {time.time()-t0:.0f}s", flush=True)
