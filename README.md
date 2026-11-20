# tandem_sim

Numerical companion to

> W. Dong and J. Zhou, "Dependent Randomization in Tandem Binary Decision Fusion:
> Non-Amplification of Synchronization Loss," IEEE TAES, manuscript TAES-2026-3140
> (major revision, November 2026).

Repository: https://github.com/weiqiangd35-ai/tandem

Everything in the paper's numerical sections can be recomputed from this package; the
regression tests in `tests/` check the numbers of the worked examples (the twelve achievable
points of the running model, the designs at alpha = 0.17 and 0.075, the three-node
attenuation example, the sign law of Proposition 5, the desynchronized point
W = (0.0725, 0.560), the correction q = 0.471, the constraint-violating example of
Sec. IV-E, the three-node examples at alpha = 0.078 and 0.1675, the parallel 2-out-of-4
shift 0.0428 of the submitted version), the identities (13), (14), Proposition 6 and the
delayed-message identity on random chains, Proposition 7 on random two-node chains (and in
exact arithmetic on models with likelihood-ratio ties), the realization rule of the
designs, the fixed point of the closed-form best responses on a continuous chain, and the
exact rational values of the certificate of Proposition 3.

All desynchronized operating points are computed **exactly** (probability-weighted sums
over the 2^(|S|+1) draw combinations).  Randomness enters only through the sampling of
observation models and threshold configurations, and every script fixes its seed.

## Layout

| path | content |
| --- | --- |
| `tandem_sim/discrete.py` | discrete models, threshold regions (levels), region-pair enumeration for the single-threshold class, forward recursion (4)-(5), attenuation factors (10)-(11) |
| `tandem_sim/design.py` | exhaustive enumeration, upper concave hull, support pair (A, B, p) and its realization (fewest switching nodes, `min_switch_realizations`; `align_pair` for PBPO designs), Proposition 4 classification, PBPO (Algorithm S1 of the supplement) with restarts, Lagrangian hull for long chains |
| `tandem_sim/desync.py` | exact desynchronized point for any set S of nodes, eq. (13)/(14), Proposition 6 (superposition identity and bound), message delayed on a link (block form of eq. (13), Section IV-F), terminal-node correction with the coupled mechanism (Proposition 7), DFC-layer and downstream benchmarks (`downstream_benchmark` redesigns the nodes after the desynchronized one) |
| `tandem_sim/continuous.py` | Gaussian, power-ROC (Swerling I / exponential), logistic local models; regions as false-alarm parameters; Lagrangian PBPO on grids (`pbpo_cont`) and with the closed-form best responses, iterated to a fixed point (`pbpo_cont_exact`) |
| `tandem_sim/exact.py` | exact rational check of Proposition 7 on two-node chains with integer-weight models (frequent likelihood-ratio ties), independent of the floating-point modules |
| `tandem_sim/parallel.py` | parallel architecture: k-out-of-n and Chair-Varshney fusion, exhaustive achievable set, support pair and its realization (`min_switch_parallel`), single-LD desynchronization |
| `experiments/exp1_attenuation.py` | Theorem 1 (Theorem 2 of the submitted version) statistics: rho distributions, end-to-end products, n = 2..20, discrete and continuous families, random / optimal / adversarial thresholds (300 random configurations per family and chain length; optimal designs of 300 models per Dirichlet family and chain length, and of the continuous families at three multipliers (fixed points of the closed-form best responses, reached from a grid search and from five random starting points); 1000 models per greedy adversary, which chooses among the pairs that use both the message and the observation; `--relay` adds the same greedy choice with the pure relay allowed, `results/exp1_relay.json`); in every configuration the exact effect of a wrong rule at N_1 is compared with the product transfer of the theorem |
| `experiments/exp2_tandem_vs_parallel.py` | paired tandem-versus-parallel protocol with bootstrap CIs that resample whole models (the pairs of one model at the three alphas are dependent), pair-level Wilcoxon and Mann-Whitney tests (the summary also stores the signed-rank test on one value per model); position-resolved, worst-node and wrong-rule metrics; immunity fractions; 1000 models, with the position medians also given for the five blocks of 200 models (`--summary-only` recomputes the statistics from the saved rows; `--first-realization` repeats the experiment with the first realization enumerated, Sec. S4; `--seed=K` uses another sample of models; `--family=qgauss3:LO:HI` replaces the Dirichlet models by three-level quantizations of N(0,1) versus N(d',1) with d' uniform on [LO, HI], Table SVIII; `--quick` writes files with the suffix `_quick`) |
| `experiments/exp13_delayed_link.py` | message delayed on the link into node k: exact mixture over the two draws against the block form of eq. (13); shift versus the position of the link; the link into the last node is the loss of the common draw at the final LD itself (Section IV-F) |
| `experiments/knot_summary.py` | statistics of the knot perturbations quoted in Sections III-B and S3, from the per-draw values stored by exp7 (`results/exp7_knot_summary.json`); re-derives the classification at epsilon = 0 in exact rational arithmetic |
| `experiments/exp12_certificate.py` | exact rational enumeration behind Proposition 3 (two-knot candidates, Corollary S1): 91/160 at alpha = 3/40 and 29/50 at alpha = 2/25, with a brute-force cross-check |
| `experiments/exp11_montecarlo.py` | independent epoch-by-epoch Monte Carlo simulation of the desynchronized chain, compared with the exact operating points; Monte Carlo check of the estimator of p' (Section V-D) |
| `experiments/exp10_detection.py` | conditional message rates at the DFC under synchronization and under a desynchronized node: detection and estimation of p' (Section V-D) |
| `experiments/exp9_pbpo_stats.py` | PBPO convergence statistics on 100 random four-node chains (single-initialization hit rates, best of 20, sweeps, fixed-point check at every hull vertex; Table SI) |
| `experiments/exp7_prop3_gap.py` | example of Proposition 3: deterministic envelope and concavity gap versus the smoothing parameter, validation at epsilon = 0, perturbation of the knots (420 draws: exact enumeration at epsilon = 0 by Corollary S1, smoothed model at epsilon = 0.004, and at 0.002 and 0.001 where the piecewise-linear envelope is concave; its gap is taken against the concave hull over the whole interval of alpha) (Section S3, Fig. S1); `--procs N` distributes the draws over N processes, `--checks` reports the sensitivity of the largest smoothed gaps to the grids (`results/exp7_checks.json`) |
| `experiments/exp6_prop2_parametric.py` | criterion of Proposition 2 for Gaussian, Swerling I and logistic local ROCs: number of maximizers of Phi_lambda on a uniform and a logit-spaced grid of f, and concavity gap of the envelope with local refinement (Section III-B, Section S3) |
| `experiments/exp8_sign_ensemble.py` | every hull edge of random two-node chains: immune / rise / fall under the design realization (Section IV-E), checks of Proposition 7 (non-trivial N_1 rules, nested DFC regions, feasibility; exact-arithmetic check on models with ties), and the counts over all realization pairs |
| `experiments/supp_tables.py` | writes the LaTeX rows of Tables SII to SIV, SVI and SVIII of the supplementary material from the saved summaries, and the confidence intervals of the median products of Table SII (`results/tab_s2_intervals.json`) |
| `experiments/reseed.py` | runs an experiment script with every generator seed increased by an offset and with another output folder; used for the second sample of RESULTS.md (offset 1000, `results/second_seed/`) |
| `experiments/sampling_error.py` | sampling error of the statistics over random models: parts of the stored sample (first models of a cell, disjoint blocks), distribution-free intervals of medians, and the comparison with the second sample (`results/sampling_error.json`) |
| `experiments/paper_figs.py` | draws Fig. 2 of the paper and Figs. S1-S5 of the supplement from the saved results |
| `experiments/exp3_multinode.py` | several desynchronized nodes: shift versus the Proposition 6 bound, for 300 random strategy pairs and for the PBPO designs of 300 eight-node models |
| `experiments/exp4_position.py` | position of the desynchronized node, eq. (14) check |
| `experiments/exp5_correction.py` | worked correction examples (Table SV), every susceptible hull edge of the three-node running model, statistics over 20000 single-node failures of random designs for each of n = 2, 3, 4 (Table SVI): direction of the shift, feasibility of the adjustment, DFC-layer, downstream and deterministic optima |
| `results/` | JSON rows and summaries written by the scripts, plus the logs of the runs used for the revision; `second_seed/` and the four files `exp2_*seed1002*` are the summaries and logs of the second sample (its rows are not included); the four files `exp2_*qgauss3*` are the summaries and logs of the two families of quantized Gaussian sensors (rows not included) |

## Running

```
pip install numpy scipy pytest
python -m pytest tests -q                      # ~6 s
python experiments/exp5_correction.py --quick  # seconds; the full run (20000 cases for each n) takes ~12 min
python experiments/exp12_certificate.py        # seconds; exact fractions
python experiments/exp13_delayed_link.py       # seconds
python experiments/exp3_multinode.py --quick   # ~1 min; the full run (PBPO designs of 300 eight-node models) takes ~30 min
python experiments/exp4_position.py --quick
python experiments/exp1_attenuation.py --quick # ~20 s quick; full run ~10 min
python experiments/exp1_attenuation.py --relay # ~2 min, after the full run: greedy adversary with the pure relay allowed
python experiments/exp2_tandem_vs_parallel.py  # full run ~20 min (1000 models, n = 3..6); --summary-only for the statistics alone
python experiments/exp2_tandem_vs_parallel.py --family=qgauss3:0.3:1.5   # ~25 min: quantized Gaussian sensors, d' in [0.3, 1.5]; likewise --family=qgauss3:1.5:3.0 (Table SVIII)
python experiments/exp6_prop2_parametric.py    # ~2 min
python experiments/exp7_prop3_gap.py           # ~2 h on one core, nearly all of it for the 420 knot perturbations; --procs 2 takes ~70 min
python experiments/exp7_prop3_gap.py --checks  # ~10 min, after the full run: grid sensitivity of the largest smoothed gaps
python experiments/knot_summary.py             # seconds, after exp7
python experiments/exp8_sign_ensemble.py       # ~30 s
python experiments/exp9_pbpo_stats.py          # ~5 min (100 chains); --quick for 4 chains
python experiments/exp10_detection.py          # after exp5 (reads its three-node examples); so do exp11 and exp13
python experiments/supp_tables.py              # Tables SII to SIV, SVI and SVIII; confidence intervals of the products of Table SII
python experiments/paper_figs.py <outdir>      # Fig. 2 of the paper and Figs. S1-S5 of the supplement
python experiments/reseed.py exp4_position.py 1000 results/second_seed   # a script on a second sample (every seed + 1000); all of them: ~55 min; exp2 --seed=1002: ~20 min, and as much with --first-realization
python experiments/sampling_error.py           # seconds: parts of the stored sample, intervals, comparison with the second sample
```

## Conventions that matter

* **Realization of a supporting pair.**  A hull vertex is in general realized by several
  strategies (a node whose message is ignored downstream may use any rule, a region for a
  message that cannot occur is free, identical sensors can give the same point with
  different assignments).  The realization decides which nodes switch between A and B.
  All designs use: (i) the fewest switching nodes among N_1..N_{n-1} (parallel: the fewest
  LDs whose threshold differs); (ii) the smallest worst-case shift |W - C| over single-node
  failures; (iii) a final LD (parallel: DFC) whose rule does not switch, then enumeration order.
  `common.np_design(..., realization="first")` and `exp2 --first-realization` reproduce the
  first-enumerated convention for comparison.

* A node's region pair is indexed by *levels* (j0, j1): level j is the set of the K - j
  observations with the largest likelihood ratios.  A single node threshold eta induces
  the pair through tau(1) = eta f/d and tau(0) = eta (1-f)/(1-d) at the *designed* upstream
  point (f, d); `realizable_pairs` enumerates every pair a single eta can produce.  This
  is the strategy family of the paper.  `all_nested_pairs` is the larger class used only
  in benchmarks.
* A strategy is stored in region form (`Strategy(j1, pairs)`); under synchronization
  loss every node keeps the regions it was designed with.
* The terminal-node correction for n >= 3 uses the *coupled* mechanism: the DFC changes
  its marginal from p to q while staying as correlated with the common draw as possible
  (monotone coupling), so that q = p reproduces the uncorrected point W.  An independent
  DFC draw would itself desynchronize the DFC from the intermediate nodes.  For n = 2 the
  two mechanisms coincide, eqs. (17)-(20) of the paper.  `terminal_adjust` tests the
  constraint with a tolerance of 1e-9 in P_F: the constraint is often met with equality at
  an end point, and alpha itself comes from hull points computed in floating point.
* The parallel system's "shift by 0.0428" of the submitted manuscript is the deterministic
  wrong-rule perturbation (all LDs at (0.2, 0.7), 2-out-of-4, one LD switches to
  (0.1, 0.6)); `exp2` reports that metric alongside the p-mixture shift |W - C|.
