"""Run an experiment script on another sample of models.

Every random generator that the script creates at top level with a literal seed,
`name = np.random.default_rng(N)`, gets the seed N + OFFSET instead, and the files that the
script saves go to OUTDIR instead of results/.  Nothing else changes: the script is executed
as it is, with these seeds (OFFSET = 0 reproduces the stored results).  The second sample of
RESULTS.md was obtained with OFFSET = 1000:

    mkdir -p results/second_seed
    python experiments/reseed.py exp1_attenuation.py 1000 results/second_seed > results/second_seed/exp1_attenuation_log.txt
    python experiments/reseed.py exp1_attenuation.py 1000 results/second_seed --relay > results/second_seed/exp1_relay_log.txt
    (the same, without --relay, for exp3_multinode.py, exp4_position.py, exp5_correction.py,
     exp8_sign_ensemble.py, exp9_pbpo_stats.py and, after exp5, whose worked examples they read from
     OUTDIR, exp11_montecarlo.py and exp13_delayed_link.py; about 55 minutes in all)

exp2_tandem_vs_parallel.py has its own option for this (--seed=1002, alone and with --first-realization).
experiments/sampling_error.py compares the two samples.
Usage: python experiments/reseed.py SCRIPT OFFSET OUTDIR [options of the script]
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
script, offset, outdir = os.path.basename(sys.argv[1]), int(sys.argv[2]), os.path.abspath(sys.argv[3])
sys.path.insert(0, HERE)
import common

os.makedirs(outdir, exist_ok=True)
common.RESULTS = outdir
path = os.path.join(HERE, script)
with open(path) as fh:
    src = fh.read()
seeds = []


def shift(m):
    seeds.append((m.group(1), int(m.group(2)), int(m.group(2)) + offset))
    return f"{m.group(1)} = np.random.default_rng({seeds[-1][2]})"


src = re.sub(r"^(\w+) = np\.random\.default_rng\((\d+)\)", shift, src, flags=re.M)
if not seeds:
    sys.exit(f"{script}: no top-level generator with a literal seed")
print(f"{script} with " + ", ".join(f"{name}: seed {new} (instead of {old})" for name, old, new in seeds)
      + f"; output in {sys.argv[3]}", flush=True)
sys.argv = [path] + sys.argv[4:]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
