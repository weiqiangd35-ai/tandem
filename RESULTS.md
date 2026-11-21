# Results of the runs used for the revision (2026-11-15)

The numbers below come from the scripts in `experiments/`.  Those of the paper use the seeds
fixed in the scripts: the logs are in `results/*_log.txt` and the rows and summaries in
`results/*.json`.  Designs use the realization rule described in the README (fewest
switching nodes, then smallest worst-case shift).

Sampling error.  Every script that uses a random generator, except exp7 (the knot
perturbations, with `knot_summary.py`), was run a second time: exp1 to exp5, exp8, exp9,
exp11 and exp13.  The seed of every generator that a script creates at top level was
increased by 1000 (`experiments/reseed.py`; exp2 has the option `--seed=1002`, which was
also used with `--first-realization`); these generators draw the models, the configurations
and the simulated epochs, while the initializations of the PBPO and the bootstrap keep their
own seeds.  For the experiments that sample models this is a second sample of the same size.
The package contains the logs and summary files of these runs, in `results/second_seed/`
and, for exp2, in the four files `results/exp2_*seed1002*`; their row files are not
included.  `experiments/sampling_error.py` compares the two samples, computes statistics on
parts of the first one (the first models of a cell, disjoint blocks) and distribution-free
intervals of medians, and prints the few counts of the text that the experiment scripts do
not print themselves; its output is `results/sampling_error_log.txt`.  The paragraphs
"Sampling error" below quote it.  The numbers of the paper are those of the first sample.
The second sample reproduces what does not depend on the sample (no factor above one and the
product transfer to 1e-15 in exp1, eq. (14) to 3e-16 in exp4 and its block form to 3e-16 in
exp13, Proposition 7 on every susceptible two-node design of exp8 and exp5, every vertex a
fixed point of the PBPO in exp9), and the statistics change by amounts consistent with their
intervals: shares by at most 1.4 points (exp5); medians of factors and ratios in the second
decimal (exp1; exp2, where each of the 14 position medians lies inside the interval of the
other sample); medians of quantities that span several decades by factors of up to 1.6
(shifts in exp3) and 3 (products of random configurations at n = 20 in exp1; 6 in the
quantized Gaussian families).  Two outcomes depend on the run: the signed-rank test at the
first node of exp2, which at the 1 % level favors the chain for n <= 5 in the first sample
and for n <= 4 in the second (Section 2; the paper draws no conclusion from this test), and
the largest deviation of the Monte Carlo check of exp11, 1.2 standard errors in the first
run and 2.8 in the second (Section 6).  exp7 was not repeated: its 420 draws are classified
exactly at eps = 0, a share of the 415 models has a binomial standard error of about 2
points, and the largest gaps quoted are maxima over these draws (Section 6).

## 0. Regression tests

`python -m pytest tests -q`: 34 tests pass.  Reproduced exactly: the 12 achievable
two-node points; alpha = 0.17 (G = (0.12, 0.67), A = (0.11, 0.66), B = (0.19, 0.84),
p = 0.25, C = (0.17, 0.795), terminal-node regime); alpha = 0.075 (A = (0.04, 0.49),
B = (0.11, 0.66), p = 0.5, C = (0.075, 0.575), upstream-involving); the three-node chain
(0.20, 0.70) -> (0.28, 0.88) -> (0.352, 0.952) and its desynchronized version with
deviations 0.1414 -> 0.0985 -> 0.0826, rho_F = 0.9, rho_D = 0.4; eq. (13) = -0.0025 and
W = (0.0725, 0.560); q = 0.471 -> (0.075, 0.5662); the constraint-violating model
(rho_F^A = 0.95, rho_F^B = 0.50, W_F - alpha = +0.056); the three-node designs at
alpha = 0.078 and 0.1675; the best designs with a single randomizing node in the four worked
examples (P_D = 0.56875, 0.3199, 0.6543, 0.8275; Section 4); Proposition 7 on random two-node chains, in floating point and
in exact arithmetic on models with likelihood-ratio ties; and the parallel
"0.0428" of the submitted version (the deterministic wrong-rule shift: all four LDs at
(0.2, 0.7), 2-out-of-4, one LD switched to (0.1, 0.6)).

## 1. Attenuation statistics (exp1; Section IV-C, Fig. S5, Table SII)

92 808 configurations (21 000 random, 11 808 from designs, 60 000 adversarial; 597 192 stages),
n in {2, 3, 5, 10, 20}, six discrete and eight continuous families.  Largest |rho| = 1 (to
floating point).  In every configuration the per-stage perturbation produced by a wrong rule
at N_1 equals the product transfer of Theorem 1, to 8.3e-16 at worst.  A random configuration
is checked against one other rule of N_1 drawn at random; an optimal or adversarial one
against every other non-trivial level of N_1 (optimal designs of the continuous families:
against the false-alarm rates 0.1, 0.5 and 0.9).  Of the 11 808 configurations from designs,
468 come from PBPO designs (396 at n = 10, 72 for the continuous families) and the others
from exhaustive designs (n <= 5).

| family, n = 5 | rho_F median (p10, p90), random | rho_F median, optimal | prod rho_F median, random | adversary, n = 5 / n = 20 |
| --- | --- | --- | --- | --- |
| running K=3 | 0.20 (0.10, 0.90) | 0.90 | 1.0e-2 | 0.66 / 0.14 |
| Dirichlet K=3 | 0.43 (0.08, 0.88) | 0.88 | 1.4e-2 | 0.45 / 0.036 |
| Dirichlet K=5 | 0.41 (0.06, 0.89) | 0.79 | 1.1e-2 | 0.54 / 0.17 |
| Dirichlet K=10 | 0.39 (0.04, 0.82) | - | 6.5e-3 | 0.53 / 0.31 |
| quantized Gaussian d'=1 | 0.48 (0.00, 0.84) | - | 3.3e-4 | 0.52 / 0.30 |
| quantized Gaussian d'=2 | 0.34 (0.00, 0.84) | - | 6.0e-4 | 0.62 / 0.57 |
| Gaussian d'=1 | 0.48 (0.10, 0.81) | 0.58 | 2.6e-2 | - |
| Gaussian d'=3 | 0.48 (0.09, 0.87) | 0.37 | 1.9e-2 | - |
| Swerling I, SNR 1 / 3 / 10 | 0.46 / 0.47 / 0.48 | 0.98 / 0.99 / 1.00 | 1.5e-2 / 2.0e-2 / 2.2e-2 | - |
| logistic mu=2 | 0.49 (0.10, 0.83) | 0.48 | 2.0e-2 | - |

Random configurations of the continuous families draw uniform false-alarm rates (N_1: f;
later nodes: a_0, with a_1 from eq. (3)), so that no node is trivial.  End-to-end products
decay geometrically with n for random configurations and, on Dirichlet K=3 models, for
the greedy relay-like adversary (0.82, 0.45, 0.19, 0.036 at n = 2, 5, 10, 20); with larger
alphabets the adversary's product falls more slowly (last column and the table below).
Optimal designs
attenuate less than random ones (Dirichlet K=3, n = 5: median factor 0.88 against 0.43,
median product 0.12 against 0.014), and for the Swerling I model hardly at all: its
likelihood ratio is bounded below by 1/(1+SNR), so the optimal node declares presence
whenever its upstream neighbor does (a_1 = 1) and rho_F = 1 - a_0 is close to one.

Optimal designs of the continuous families.  For lambda = 0.5, 2, 8 and n = 2, 3, 5 (72
designs) the design is a fixed point of the closed-form best responses of Section S3
(`pbpo_cont_exact`): with the regions of the other nodes fixed, the best regions of node k
are those of eq. (3) with eta_k = lambda A_F / A_D, where A_F and A_D are the products of
the factors of the nodes after it.  The iteration is started from the design of a grid
search (`pbpo_cont`; 120 false-alarm rates for N_1, 97 thresholds for each later node) and
from five random points: all six runs end at the same fixed point in every design, after at
most 102 sweeps, and P_D - lambda P_F exceeds that of the grid design by 5e-6 to 6e-3.

Sample.  The random configurations are 300 per family and chain length.  The optimal designs
of the Dirichlet families use 300 models per chain length (60 for n = 10) and each greedy
adversary 1000 models.  The models of a cell come from three generators: the first 30
(optimal designs; 6 of the 60 for n = 10) and 10 (adversaries) from the main one, which also
draws the random configurations; the other models of the optimal designs and the
adversaries' models up to the hundredth from a second one; the adversaries' last 900 models
from a third.  The adversaries' first 10 and first 100 models are thus parts of the sample
of 1000: for Dirichlet K=3 the median products at n = 2, 5, 10, 20 are 0.71, 0.39, 0.28,
0.031 on the first 10 models, 0.84, 0.45, 0.20, 0.035 on the first 100 and 0.82, 0.45, 0.19,
0.036 on all of them.  Over the 30 cells the medians of the first 100 models differ from
those of the 1000 by up to 0.05 (0.06 with the pure relay admitted, where Dirichlet K=3 at n = 20
gives 0.60 on the first 100 models and 0.55 on 1000).

The greedy adversaries.  They choose among the pairs that the random configurations draw
from: a pair that ignores the message (rho = 0) and the pure relay (R(0) empty, R(1) the
whole alphabet; rho_F = rho_D = 1) are excluded whenever another pair is realizable, so the
"relay-like" adversary is the most relay-like chain whose nodes still use their observations.
`exp1_attenuation.py --relay` replays the same models and first-node rules with the pure
relay admitted wherever a single threshold realizes it (`results/exp1_relay.json`; the replay
reproduces the 69 000 random and adversarial rows of the discrete families exactly).  Median
end-to-end products at n = 2, 3, 5, 10, 20, 1000 models per cell:

| family | pure relay excluded (Table SII, Section IV-C) | pure relay admitted | share of the nodes that relay |
| --- | --- | --- | --- |
| running K=3 | 0.90, 0.81, 0.66, 0.39, 0.14 | 0.90 for every n | 0, 0.50, 0.75, 0.89, 0.95 |
| Dirichlet K=3 | 0.82, 0.66, 0.45, 0.19, 0.036 | 0.93, 0.79, 0.70, 0.60, 0.55 | 0.34, 0.41, 0.53, 0.67, 0.80 |
| Dirichlet K=5 | 0.84, 0.72, 0.54, 0.34, 0.17 | 0.85, 0.75, 0.62, 0.50, 0.47 | 0.18, 0.24, 0.36, 0.52, 0.69 |
| Dirichlet K=10 | 0.78, 0.66, 0.53, 0.40, 0.31 | 0.78, 0.66, 0.53, 0.43, 0.37 | 0.05, 0.07, 0.12, 0.28, 0.48 |
| quantized Gaussian d'=1 | 0.82, 0.67, 0.52, 0.36, 0.30 | the same | 0 |
| quantized Gaussian d'=2 | 0.82, 0.67, 0.62, 0.59, 0.57 | the same | 0 |

In the running model every node from N_3 on can relay, and the product stays at the factor
0.9 of N_2 for every n.  For Dirichlet K=3 the distribution-free 95 % intervals of the
medians with the relay admitted are [0.90, 0.94], [0.68, 0.72], [0.58, 0.62], [0.53, 0.57] at
n = 2, 5, 10, 20.  The larger the alphabet, the closer the restricted adversary comes to a
relay, and the more slowly its product falls: at n = 20 it is 0.036, 0.17 and 0.31 for
Dirichlet K = 3, 5 and 10.  No relay is realizable in the quantized Gaussian chains (eight
levels), but for d' = 2 the restricted adversary is close to one: a rule that departs from
the relay only on the two extreme bins has rho_F = 1 - 2 Q(3) = 0.9973 (0.9987 with one
bin), every chain uses one of these two rules from the eleventh node on, and the product
only falls from 0.62 at n = 5 to 0.57 at n = 20.  The statement of Section IV-C that the
product of the adversary decays geometrically is made for Dirichlet K=3; Theorem 1 excludes
growth of a perturbation, and a chain of relays shows that it need not decay.

Sampling error: intervals.  `supp_tables.py` computes the distribution-free 95 % confidence
interval of every median product of Table SII (`results/tab_s2_intervals.json`): order
statistics 133 and 168 of the 300 random configurations of a cell, 469 and 532 of the 1000
models of an adversary.  The products of the random configurations span several decades, and
their sampling error grows with n.  For the twelve families other than the quantized
Gaussian ones the interval lies within a factor of 1.2 of the median at n = 2, 1.5 at n = 5,
1.8 at n = 10 and 2.4 at n = 20; for the two quantized Gaussian families, within a factor of
3.0, 2.3, 3.9 and 5.6.  The caption of Table SII quotes 1.2, 2.4 and 5.6, with "about",
since the width of such an interval itself varies from one sample of 300 to another; the
entries of the table are the point estimates.  For the adversary on Dirichlet K=3 models the
intervals are 0.82 [0.81, 0.83], 0.45 [0.44, 0.46], 0.19 [0.18, 0.20] and 0.036 [0.034,
0.038] at n = 2, 5, 10, 20.

Sampling error: second sample (seeds 1001, 1101 and 1202 for the three generators).

Median products of the random configurations: the two medians of a cell differ by factors of
up to 1.15, 1.63, 1.86 and 2.97 at n = 2, 5, 10, 20 (Dirichlet K=3 at n = 5: 0.0144 and
0.0135) and, in the quantized Gaussian families, of 1.00, 1.26, 1.79 and 5.97; the second
median lies inside the interval of the first in 46 of the 56 cells (82 %; for two
independent estimates and intervals of this level, 95.7 %, the expected share is 85 %).

Per-stage factors of the random configurations at n = 5 (first columns of Table SII): in the
twelve families other than the quantized Gaussian ones the median changes by at most 0.04
(Gaussian d' = 3: 0.48 and 0.52; Dirichlet K=3: 0.43 and 0.44), the 10th and the 90th
percentile by at most 0.02 and the fraction above 0.9 by at most 0.02.  In the quantized
Gaussian families rho_F takes 19 values and the median sits at a jump of the distribution:
for d' = 1, 597 of the 1200 stages have rho_F <= 0.341 and the next value is 0.477, so that
the median is 0.48 in the first sample and 0.34 in the second (90th percentile: 0.84 and
0.95); for d' = 2, 586 stages are at or below 0.159, the next value is 0.341, and the median
is 0.34 in both.

Optimal designs: the median factor at n = 5 is 0.88 and 0.85 for Dirichlet K=3 and 0.79 and
0.79 for Dirichlet K=5 (median product for K=3: 0.12 and 0.13); the designs of the other
families use one fixed model and do not change.

Relay-like adversary: for Dirichlet K=3 at n = 2, 5, 10, 20 the second sample gives 0.83,
0.45, 0.19, 0.033 (0.85, 0.53, 0.35, 0.16 for K=5 and 0.78, 0.53, 0.40, 0.32 for K=10); in
the three Dirichlet families its median lies inside the interval of the first in 10 of the
12 cells, and the largest change is 0.018.  The running model and the quantized Gaussian
models are fixed, and their values are the same in both samples.  With the pure relay
admitted the second sample gives 0.93, 0.71, 0.61, 0.55 for Dirichlet K=3 (first sample:
0.93, 0.70, 0.60, 0.55), inside the interval of the first in 11 of the 12 Dirichlet cells;
80 % and 80 % of the nodes of a twenty-node chain relay.

## 2. Tandem versus parallel, paired protocol (exp2; Section IV-D, Fig. 2, Tables SIII-SIV and SVIII)

1000 Dirichlet(1) K = 3 models (one model shared by all nodes), n = 3..6, alpha in
{0.05, 0.1, 0.2}, i.e. 3000 designs for each n; both architectures designed exhaustively at
the same alpha and realized by the same rule; one node draws independently with the same
marginal p; exact W - C.  A case is susceptible when the shift exceeds 1e-9 and immune
otherwise.

| | n = 3 | n = 4 | n = 5 | n = 6 |
| --- | --- | --- | --- | --- |
| immune designs, tandem / parallel | 0.70 / 0.51 | 0.55 / 0.45 | 0.42 / 0.44 | 0.34 / 0.42 |
| susceptible cases at k = 1..n-1 | 267 / 863 | 217 / 888 / 1156 | 172 / 929 / 1250 / 1075 | 133 / 831 / 1122 / 1137 / 1004 |
| median tandem shift at k = 1..n-1 | 0.0150 / 0.0224 | 0.0121 / 0.0125 / 0.0219 | 0.0131 / 0.0071 / 0.0145 / 0.0204 | 0.0197 / 0.0037 / 0.0092 / 0.0129 / 0.0200 |
| median parallel single-LD shift | 0.0174 | 0.0122 | 0.0094 | 0.0075 |
| median sync. P_D, tandem / parallel | 0.60 / 0.61 | 0.66 / 0.70 | 0.69 / 0.77 | 0.71 / 0.83 |

Median ratio of the tandem shift at node k to the mean shift over the susceptible LDs of
the parallel design of the same model and alpha, with its bootstrap 95 % confidence
interval (2000 resamples of whole models), the number of pairs, the one-sided Wilcoxon p
for "tandem smaller" (<) or "parallel smaller" (>) with the pairs taken as independent, and
the range of the median over the five disjoint blocks of 200 models:

| k | n = 3 | n = 4 | n = 5 | n = 6 |
| --- | --- | --- | --- | --- |
| 1 | 0.97 [0.83, 1.05]; 215 pairs; < 2e-03; blocks 0.67-1.04 | 0.85 [0.67, 0.99]; 184 pairs; < 1e-04; blocks 0.59-1.00 | 0.98 [0.78, 1.03]; 135 pairs; < 6e-03; blocks 0.53-1.06 | 1.02 [0.85, 1.12]; 103 pairs; n.s.; blocks 0.72-1.23 |
| 2 | 0.97 [0.85, 1.08]; 476 pairs; n.s.; blocks 0.83-1.04 | 0.79 [0.68, 0.94]; 558 pairs; < 2e-05; blocks 0.67-1.02 | 0.63 [0.52, 0.73]; 569 pairs; < 1e-15; blocks 0.54-0.78 | 0.42 [0.35, 0.50]; 538 pairs; < 4e-33; blocks 0.31-0.60 |
| 3 |  | 1.40 [1.15, 1.69]; 677 pairs; > 5e-06; blocks 1.09-1.84 | 1.30 [1.08, 1.54]; 740 pairs; > 7e-03; blocks 0.97-1.51 | 0.87 [0.74, 0.99]; 682 pairs; < 2e-03; blocks 0.61-1.14 |
| 4 |  |  | 2.18 [1.77, 2.63]; 651 pairs; > 1e-23; blocks 1.87-2.64 | 1.41 [1.14, 1.62]; 656 pairs; > 1e-04; blocks 0.76-2.07 |
| 5 |  |  |  | 2.62 [2.19, 2.92]; 575 pairs; > 2e-22; blocks 1.91-3.57 |

Worst switching node: median ratio 1.08, 1.28, 1.65, 1.56 (1.07, 1.30, 1.66, 1.57 with the
parallel system restricted to k-out-of-n rules).  Deterministic wrong rule at N_1 against
the worst parallel LD: 0.78, 0.61, 0.69, 0.76 (332, 268, 202, 147 pairs).  Adjacent hull
vertices of the parallel system differ in the threshold of a single LD in 77 to 82 % of
the designs.

Reading: the ratio rises along the chain.  In chains of four to six nodes a failure at the
second node shifts the operating point less than the failure of one parallel detector (0.79,
0.63, 0.42), and a failure at the node next to the final LD more (1.40, 2.18, 2.62); in
between the ratio crosses one (n = 5: 1.30 at N_3; n = 6: 0.87 at N_3, 1.41 at N_4).  The
first node seldom has to switch (it is susceptible in 133 to 267 of the 3000 designs); there
the median ratio is 0.85, 0.98, 1.02 for n = 4, 5, 6 and the two architectures are
comparable: the block medians range from 0.53 to 1.23.  The signed-rank test favors the
chain for n <= 5 in this sample, because the ratio has a heavy lower tail at N_1 (below 0.5
in 24 to 34 % of the pairs, above 2 in 13 to 17 %; geometric mean 0.64 to 0.74), to which
the test responds and the median does not; its outcome is different in the second sample
(see "Sampling error" below).  The shifts are larger in both architectures at this node (for
n = 4, 5, 6 the median parallel shift is 0.021 to 0.029 in the paired designs, against 0.008
to 0.012 over all designs), which is why N_1 lies above the parallel median in Fig. 2.  For
n = 3 the median ratio is 0.97 at both nodes.

Sampling error.  The block ranges of the table show what a sample of 200 models gives at one
position: the five block medians are 2.07, 1.48, 0.76, 1.07, 1.70 at N_4 for n = 6 and 1.67,
1.84, 1.32, 1.09, 1.18 at the node next to the final LD for n = 4.  The second sample of
1000 models (`--seed=1002`) gives the following median ratios (first sample / second sample,
each with its bootstrap interval):

| k | n = 3 | n = 4 | n = 5 | n = 6 |
| --- | --- | --- | --- | --- |
| 1 | 0.97 [0.83, 1.05] / 0.90 [0.75, 1.02] | 0.85 [0.67, 0.99] / 0.87 [0.69, 1.01] | 0.98 [0.78, 1.03] / 0.98 [0.86, 1.17] | 1.02 [0.85, 1.12] / 1.01 [0.73, 1.20] |
| 2 | 0.97 [0.85, 1.08] / 1.01 [0.87, 1.29] | 0.79 [0.68, 0.94] / 0.84 [0.77, 0.93] | 0.63 [0.52, 0.73] / 0.57 [0.47, 0.66] | 0.42 [0.35, 0.50] / 0.41 [0.33, 0.48] |
| 3 |  | 1.40 [1.15, 1.69] / 1.35 [1.14, 1.60] | 1.30 [1.08, 1.54] / 1.28 [1.10, 1.47] | 0.87 [0.74, 0.99] / 0.90 [0.80, 1.00] |
| 4 |  |  | 2.18 [1.77, 2.63] / 2.05 [1.82, 2.44] | 1.41 [1.14, 1.62] / 1.57 [1.32, 1.84] |
| 5 |  |  |  | 2.62 [2.19, 2.92] / 2.83 [2.37, 3.45] |

Each median lies inside the interval of the other sample at all 14 positions.  In both
samples the nine medians at the second and later nodes of chains of four to six nodes have
their interval on their side of one: below one at N_2 (0.79, 0.63, 0.42 and 0.84, 0.57, 0.41
for n = 4, 5, 6) and at N_3 for n = 6 (0.87 and 0.90, the weakest of them: the intervals end
at 0.990 and 0.995), above one at the other positions (next to the final LD: 1.40, 2.18,
2.62 and 1.35, 2.05, 2.83).  At the first node the medians are 0.85, 0.98, 1.02 and 0.87,
0.98, 1.01 for n = 4, 5, 6, and every interval contains one except that of n = 4 in the
first sample.  The signed-rank test at this node (pairs taken as independent, alternative
"tandem smaller") gives p = 0.002, 0.0001, 0.006, 0.14 for n = 3..6 in the first sample and
0.010, 0.0007, 0.19, 0.034 in the second: at the 1 % level it favors the chain for n <= 5 in
the first sample and for n <= 4 in the second.  The lower tail is as heavy in the second
sample (below 0.5 in 27 to 32 % of the pairs, above 2 in 11 to 18 %; geometric mean 0.69 to
0.85).  The outcome of this test depends on the sample; the reading that the two
architectures are comparable at the first node does not, and the paper states only the
latter.

Other statistics of the table, n = 3..6, first and second sample: immune designs, tandem
0.70, 0.55, 0.42, 0.34 and 0.68, 0.52, 0.41, 0.33, parallel 0.51, 0.45, 0.44, 0.42 and 0.50,
0.45, 0.44, 0.42; median synchronized P_D, tandem 0.60, 0.66, 0.69, 0.71 and 0.62, 0.68,
0.71, 0.73, parallel 0.61, 0.70, 0.77, 0.83 and 0.63, 0.72, 0.80, 0.85; median parallel
single-LD shift 0.0174, 0.0122, 0.0094, 0.0075 and 0.0185, 0.0130, 0.0101, 0.0082; worst
switching node 1.08, 1.28, 1.65, 1.56 and 1.11, 1.27, 1.64, 1.66; wrong rule at N_1 against
the worst parallel LD 0.78, 0.61, 0.69, 0.76 and 0.79, 0.69, 0.73, 0.67.  The median tandem
shift at N_1 for n = 6, over 133 and 138 cases, is 0.0197 and 0.0111; at the other positions
the median shifts of the two samples differ by at most 0.0011.

Dependence within a model.  A model contributes up to three pairs to a position (one for
each alpha; 1.24 to 1.41 on average, depending on the position), and they are dependent.
The confidence intervals therefore resample whole models.  The nine medians quoted in the
paper for the second and later nodes of chains of four to six nodes each have their interval
on their side of one.  The Wilcoxon and Mann-Whitney p-values in the table and in Table SIII
take the pairs as independent.  The summary also stores the signed-rank test on one value
per model (the mean log ratio of its pairs, `by_model_wilcoxon_*`): it gives the same
direction at every position, with p <= 0.03 wherever the pair-level p is at most 0.01 (n = 5,
N_3: 0.027 against 0.007; n = 6, N_4: 0.006 against 0.0001), and 0.002, 1e-5, 0.006, 0.03 at
N_1 for n = 3..6.

Dependence on the realization (`--first-realization`, files `exp2_summary_first.json` and
`exp2_first_log.txt`; the rows are not shipped and are regenerated by the script in about 25
minutes): with the first single-threshold strategy enumerated, N_1 is a susceptible
switching node in 851, 571, 430, 335 designs, the immune fractions are 0.53, 0.45, 0.37,
0.30 (tandem) and 0.44, 0.42, 0.42, 0.41 (parallel), and the median ratios are 0.53, 0.50,
0.43, 0.55 at N_1 (n = 3..6), 0.76, 0.66, 0.47 at N_2 and 1.26, 1.82, 2.21 next to the
final LD (n = 4, 5, 6).  Second sample (`--first-realization --seed=1002`): N_1 is a
susceptible switching node in 805, 553, 418, 289 designs, the immune fractions are 0.52,
0.44, 0.36, 0.30 (tandem) and 0.44, 0.42, 0.42, 0.41 (parallel), and the median ratios are
0.56, 0.53, 0.60, 0.56 at N_1, 0.78, 0.62, 0.46 at N_2 and 1.26, 1.86, 2.26 next to the
final LD; the second median lies inside the interval of the first at 13 of the 14
positions.

Dependence on the observation model (`--family=qgauss3:0.3:1.5` and `--family=qgauss3:1.5:3.0`,
files `exp2_summary_qgauss3_*.json` and `exp2_qgauss3_*_log.txt`; the rows are not shipped and are
regenerated by the script in about 25 minutes for each family; Section IV-D, Table SVIII): the same
protocol on three-level quantizations of a Gaussian shift-in-mean observation, N(0,1) versus N(d',1),
with the two cut points uniform on [-1, d' + 1] and d' uniform on [0.3, 1.5] (weak sensors) or on
[1.5, 3] (strong sensors); 1000 models in each family.  Ratio of the tandem shift at node k to the
mean shift of the susceptible LDs of the parallel design (pairs, median [95 % CI], fraction below one):

| n | k | weak: pairs | median ratio [95 % CI] | P(ratio < 1) | strong: pairs | median ratio [95 % CI] | P(ratio < 1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 1 | 457 | 0.53 [0.47, 0.63] | 0.67 | 695 | 0.34 [0.33, 0.37] | 0.86 |
| 3 | 2 | 512 | 1.60 [1.16, 1.93] | 0.42 | 584 | 0.37 [0.33, 0.49] | 0.67 |
| 4 | 1 | 383 | 0.50 [0.42, 0.65] | 0.64 | 502 | 0.23 [0.21, 0.28] | 0.85 |
| 4 | 2 | 461 | 1.40 [1.18, 1.64] | 0.43 | 482 | 0.62 [0.43, 0.81] | 0.59 |
| 4 | 3 | 520 | 1.28 [1.02, 1.62] | 0.45 | 643 | 0.43 [0.34, 0.61] | 0.62 |
| 5 | 1 | 257 | 0.65 [0.49, 0.78] | 0.63 | 379 | 0.18 [0.16, 0.22] | 0.83 |
| 5 | 2 | 338 | 1.59 [1.28, 1.99] | 0.38 | 363 | 0.88 [0.54, 1.19] | 0.53 |
| 5 | 3 | 463 | 2.60 [1.99, 3.19] | 0.33 | 559 | 0.72 [0.50, 0.92] | 0.56 |
| 5 | 4 | 434 | 3.75 [2.87, 5.38] | 0.24 | 376 | 1.35 [0.84, 2.58] | 0.45 |
| 6 | 1 | 168 | 0.47 [0.31, 0.61] | 0.68 | 281 | 0.16 [0.14, 0.22] | 0.80 |
| 6 | 2 | 244 | 1.53 [1.25, 2.08] | 0.40 | 333 | 0.91 [0.48, 1.42] | 0.50 |
| 6 | 3 | 393 | 2.28 [1.72, 2.74] | 0.35 | 447 | 0.91 [0.55, 1.28] | 0.51 |
| 6 | 4 | 424 | 3.14 [2.50, 5.12] | 0.28 | 341 | 1.15 [0.72, 1.47] | 0.48 |
| 6 | 5 | 412 | 6.10 [4.71, 8.65] | 0.25 | 317 | 2.04 [1.25, 2.82] | 0.40 |

With weak sensors the median ratio is below one at N_1 and above one at every later node, each
interval on its side of one.  With strong sensors it is below one at N_1, at N_2 for n = 3, 4 and
at N_3 for n = 4, 5, above one next to the final LD for n = 6, and its interval contains one at the
other positions.  Median shift of one parallel LD (n = 3..6): 0.0087, 0.0056, 0.0037, 0.0026 (weak) and
0.0369, 0.0176, 0.0091, 0.0056 (strong); the median tandem shift lies between 0.0049 and
0.0211 at every position in both families.  Worst switching node: 1.24, 1.14, 2.57, 2.81 (weak) and
0.40, 0.40, 0.69, 1.07 (strong).  Immune designs, tandem / parallel: 0.70, 0.60, 0.54, 0.49 / 0.54, 0.47, 0.45, 0.42 (weak) and
0.66, 0.59, 0.56, 0.54 / 0.45, 0.43, 0.39, 0.37 (strong).  Median synchronized P_D, tandem / parallel: 0.46, 0.50, 0.54, 0.56 /
0.46, 0.53, 0.60, 0.65 (weak) and 0.920, 0.956, 0.972, 0.981 / 0.928, 0.971, 0.988, 0.995 (strong).

## 3. Direction of the shift and Proposition 7 on two-node chains (exp8; Section IV-E)

5 000 Dirichlet(1) K = 3 models shared by both nodes; every hull edge except the two
adjacent to the trivial strategies (26 825 edges).  Every chain has five edges on which only
the final LD switches (immune) and at most one on which N_1 switches (1 825 chains); on that
edge P_F rises in 928 chains and falls in 897.  In each of the 1 825 susceptible designs
both N_1 rules are non-trivial, the DFC regions are nested as in Proposition 7 and the
adjustment restores P_F = alpha; over all 26 825 edges it is also feasible for each of the 7 412
pairs of realizations of the two vertices with distinct N_1 rules and a nonzero shift.  Over
all pairs of single-threshold realizations (46 825): unchanged 84 %, rise 11 %, fall 5 %;
with the trivial edges included: 57 / 23 / 20 %.

Sampling error.  In the second sample (5 000 other models, 26 839 edges) 1 839 chains have
an edge on which N_1 switches; P_F rises on it in 912 chains and falls in 927 (928 and 897
in the first sample).  Every one of these designs, and every one of the 7 446 susceptible
pairs of realizations, passes the checks of Proposition 7.  Over all pairs of
single-threshold realizations: unchanged 84 %, rise 11 %, fall 5 %; with the trivial edges
included: 57 / 23 / 20 %.  The exact-arithmetic check below has its own generator and is the
same in both runs.

Exact arithmetic (`tandem_sim/exact.py`, run by exp8): 20 000 two-node chains whose models
have integer weights (two to four levels, totals 6 to 12), so that equal likelihood ratios
and observations exactly at a threshold are frequent; 124 229 hull edges.  In all 68 668
pairs of realizations with distinct non-trivial N_1 rules the DFC regions are nested as in
Proposition 7 and alpha lies between L_F and U_F (q* in [0, 1] restores P_F = alpha exactly
in the 44 833 pairs with L_F != U_F; in the other 23 835 the two DFC rules coincide and
L_F = U_F = alpha).  With a trivial N_1 rule the region for the message that never occurs
is free, and alpha is outside [L_F, U_F] in 7 596 of 277 549 such pairs (2.7 %): the
hypothesis of the proposition is needed, and the design rule never produces such a pair.

## 4. Correction (exp5; Sections III-D, IV-E and V-C, Tables SV-SVI)

Worked examples (running model unless stated):

| | synchronized C | desynchronized W | adjustment | DFC-layer / downstream optimum | deterministic optimum G | single randomizing node S (node) |
| --- | --- | --- | --- | --- | --- | --- |
| two nodes, alpha = 0.075, N_1 | (0.075, 0.575) | (0.0725, 0.560) | q* = 0.4706 -> (0.075, 0.5662) | 0.5662 / 0.5662 | (0.04, 0.49) | (0.075, 0.56875), final LD |
| violating model, alpha = 0.21125, N_1 | (0.21125, 0.32375) | (0.2675, 0.3719) | q* = 0.0833 -> (0.21125, 0.3172) | 0.3173 / 0.3173 | (0.0975, 0.19) | (0.21125, 0.31986), N_1 |
| three nodes, alpha = 0.078, N_2 | (0.078, 0.678) | (0.10775, 0.6465) | q* = 0.6715 -> (0.078, 0.5760) | 0.5824 / 0.5824 | (0.072, 0.637) | (0.078, 0.65432), final LD |
| three nodes, alpha = 0.1675, N_1 or N_2 | (0.1675, 0.830) | (0.16525, 0.824) | none (same DFC rule) | 0.8253 / 0.8265 (N_1), 0.8251 / 0.8251 (N_2) | (0.136, 0.796) | (0.1675, 0.82750), N_2 |

Single-node failures of susceptible designs on random Dirichlet(1) K = 3 models (one model
per node, alpha uniform on [0.02, 0.3]; 20 000 cases for each n; a case is a susceptible
design with one of its switching nodes):

| | n = 2 | n = 3 | n = 4 |
| --- | --- | --- | --- |
| susceptible / randomized designs | 20000 / 114206 | 13824 / 41972 | 10994 / 26546 |
| P_F rises / falls | 0.80 / 0.20 | 0.75 / 0.25 | 0.72 / 0.28 |
| abs(W_F - alpha): median / max | 0.013 / 0.18 | 0.008 / 0.20 | 0.006 / 0.19 |
| rule of the DFC switches | 1.00 | 0.60 | 0.40 |
| adjustment then meets the constraint (when P_F rises) | 1.00 (1.00) | 0.93 (0.91) | 0.92 (0.89) |
| it then equals the DFC-layer / downstream optimum | 0.36 / 0.36 | 0.42 / 0.34 | 0.40 / 0.28 |
| median gap to the DFC-layer / downstream optimum (largest gap) | 1.0e-03 / 1.0e-03 (0.37) | 3.3e-04 / 1.3e-03 (0.33) | 1.9e-04 / 1.3e-03 (0.35) |
| DFC rule does not switch: P_F > alpha | - | 0.73 | 0.71 |
| adjusted point below G (share of the cases in which the adjustment meets the constraint) | 0.19 | 0.35 | 0.41 |
| DFC-layer / downstream optimum below G (share of all cases) | 0.08 / 0.08 | 0.31 / 0.23 | 0.44 / 0.32 |
| median gain of randomization over G, C_D - G_D | 0.061 | 0.030 | 0.017 |
| part that needs the common draw, C_D - S_D: median / 90 % quantile / largest | 0.0021 / 0.0106 / 0.056 | 0.0017 / 0.0104 / 0.055 | 0.0013 / 0.0095 / 0.058 |
| C_D - S_D as a share of C_D - G_D: median over the susceptible designs | 0.04 | 0.07 | 0.09 |
| C_D - S_D above 0.01 (share of the susceptible designs) | 0.11 | 0.11 | 0.09 |
| S attained with the final LD as the randomizing node / of which an upstream node attains the same value (share of the susceptible designs) | 0.64 / 0.23 | 0.42 / 0.06 | 0.32 / 0.01 |
| S attained only with an upstream node as the randomizing node (share of the susceptible designs) | 0.36 | 0.58 | 0.68 |
| adjusted point below S (share of the cases in which the adjustment meets the constraint) | 1.00 | 0.97 | 0.92 |
| DFC-layer / downstream optimum below S (share of all cases) | 1.00 / 1.00 | 0.98 / 0.95 | 0.96 / 0.92 |

For n = 2 the adjustment always restores P_F = alpha (Proposition 7).  For n >= 3 it acts
only when the DFC's own regions differ under A and B; otherwise the DFC has to change its
region pair, still a single-node action.  In 31 % and 44 % of the failures of three- and
four-node designs even the best correction at the final LD ends below the deterministic
optimum G, which needs no common draw.

S is the best design in which a single node randomizes with a private coin and every other
node keeps one rule (`tandem_sim.design.single_node_optimum`).  With the rules of the other
nodes fixed, the system point is affine in the operating points of the randomizing node, so
these designs reach the convex hull of the system points of the strategies that differ at
that node alone; the search is exhaustive over the node, the rules of the other nodes and the
two rules that are mixed, within the nested threshold strategies.  Like G, S needs no common
draw; it is a lower bound on what independent randomization at several nodes attains
(Papastavrou and Athans, 1995), so C_D - S_D bounds from above what the common draw adds.
Its median is 0.0021, 0.0017, 0.0013 for n = 2, 3, 4, against a median gain of
randomization over G of 0.061, 0.030, 0.017; it exceeds 0.01 in about one susceptible design in
ten.  After a failure the best correction at the final LD ends below S in 100 %, 98 %, 96 %
of the cases.  In the two-node running model at alpha = 3/40, C_D - S_D = 23/40 - 91/160 = 1/160,
the gap that the certificate of Proposition 3 gives at epsilon = 0 (exp12): a node on a segment
of a piecewise-linear ROC is a node of the discrete model that randomizes between two adjacent
rules.  Several nodes can attain the value of S; the table counts the designs in which the
final LD does (alone or with an upstream node) and those in which only an upstream node does.
exp5 was run again on 2026-11-20 with this reference added; every statistic stored before is
reproduced exactly, in both samples.

Sampling error.  With 20 000 cases a share of the cases has a binomial standard error of at
most 0.35 point; a design contributes 1.45 cases on average for n = 3 and 1.82 for n = 4, so
the cases are not all independent.  Second sample (seed 1005):

| first / second sample | n = 2 | n = 3 | n = 4 |
| --- | --- | --- | --- |
| susceptible / randomized designs | 20000 / 114206 and 20000 / 110623 | 13824 / 41972 and 13824 / 41479 | 10994 / 26546 and 11038 / 26508 |
| P_F rises | 0.795 / 0.794 | 0.754 / 0.759 | 0.717 / 0.725 |
| abs(W_F - alpha): median | 0.0133 / 0.0133 | 0.0084 / 0.0088 | 0.0061 / 0.0062 |
| rule of the DFC switches | 1.000 / 1.000 | 0.601 / 0.597 | 0.398 / 0.412 |
| adjustment then meets the constraint | 1.000 / 1.000 | 0.933 / 0.930 | 0.917 / 0.907 |
| it then equals the DFC-layer optimum | 0.357 / 0.355 | 0.422 / 0.420 | 0.405 / 0.403 |
| it then equals the downstream optimum | 0.357 / 0.355 | 0.340 / 0.339 | 0.283 / 0.290 |
| DFC rule does not switch: P_F > alpha | - | 0.732 / 0.740 | 0.714 / 0.720 |
| adjusted point below G (share of the cases in which the adjustment meets the constraint) | 0.185 / 0.184 | 0.354 / 0.350 | 0.406 / 0.403 |
| DFC-layer optimum below G (share of all cases) | 0.085 / 0.085 | 0.305 / 0.306 | 0.438 / 0.443 |
| downstream optimum below G (share of all cases) | 0.085 / 0.085 | 0.231 / 0.227 | 0.317 / 0.317 |
| median C_D - G_D | 0.0607 / 0.0612 | 0.0301 / 0.0308 | 0.0170 / 0.0166 |
| C_D - S_D: median | 0.0021 / 0.0021 | 0.0017 / 0.0017 | 0.0013 / 0.0013 |
| C_D - S_D: 90 % quantile | 0.0106 / 0.0105 | 0.0104 / 0.0104 | 0.0095 / 0.0093 |
| C_D - S_D: largest | 0.056 / 0.058 | 0.055 / 0.073 | 0.058 / 0.067 |
| adjusted point below S (share of the cases in which the adjustment meets the constraint) | 1.000 / 1.000 | 0.970 / 0.970 | 0.918 / 0.918 |
| DFC-layer optimum below S (share of all cases) | 1.000 / 1.000 | 0.978 / 0.978 | 0.956 / 0.954 |
| downstream optimum below S (share of all cases) | 1.000 / 1.000 | 0.946 / 0.950 | 0.915 / 0.915 |

The largest change of a share is 1.4 points.  For n = 3 the two samples reach their 20 000
cases with the same number of susceptible designs, 13 824; the samples differ (8 319 and 8 390
of the cases have N_1 as the failing node).  The second sample has 20 001 cases for n = 4
(the last design contributes more than one).

## 5. Several desynchronized nodes (exp3) and position (exp4); Section IV-F, Fig. S2

Eight-node chains, Dirichlet(1) K = 3 (one model per node).  300 random strategy pairs and
PBPO designs (300 models x 3 alphas, Lagrangian-PBPO hull; 831 designs with at least
one switching node among N_1..N_7, 1.9 on average; 1543 cases).  The ratio of the shift to
the bound (16) is taken over the cases with a positive bound.

| \|S\| | random pairs: median / p90 / max of abs(W_F - alpha) | median shift / bound (cases with a positive bound, of 300) | PBPO designs: cases, median / p90 / max | median shift / bound |
| --- | --- | --- | --- | --- |
| 1 | 3.6e-04 / 0.023 / 0.128 | 0.39 (226) | 831, 2.1e-04 / 0.015 / 0.105 | 0.017 |
| 2 | 2.3e-03 / 0.042 / 0.220 | 0.26 (279) | 437, 4.4e-03 / 0.042 / 0.164 | 0.062 |
| 3 | 4.5e-03 / 0.049 / 0.196 | 0.24 (291) | 196, 1.3e-02 / 0.058 / 0.164 | 0.086 |
| 4 | 9.9e-03 / 0.064 / 0.183 | 0.21 (298) | 79, 2.0e-02 / 0.069 / 0.107 | 0.093 |

Pooled over |S|, the exact shift is 0.25 of the bound in the median for random pairs
(1094 of 1200 cases have a positive bound) and 0.048 for PBPO designs; the largest
shift is 0.22 (random pairs) and 0.16 (PBPO designs).

Sampling error.  Optimal designs, parts of the sample: over the ten disjoint blocks of 30
models the medians of abs(W_F - alpha) range from 1.0e-4 to 5.5e-4, 2.3e-3 to 8.9e-3, 6.6e-3
to 1.6e-2 and 9.0e-3 to 4.3e-2 for |S| = 1..4 (factors of 5.5, 3.9, 2.5, 4.8; a block has 6
to 11 cases with four desynchronized nodes), and the pooled ratio to the bound from 0.03 to
0.07.  The first block has 87 designs and 155 cases.  Second sample (seed 1003; 300 other
pairs and 300 other models, 833 designs with a switching node among N_1..N_7, 1.9 on
average, 1569 cases):

| \|S\| | random pairs: median / p90 / max of abs(W_F - alpha) | median shift / bound (cases with a positive bound, of 300) | PBPO designs: cases, median / p90 / max | median shift / bound |
| --- | --- | --- | --- | --- |
| 1 | 2.4e-04 / 0.018 / 0.110 | 0.35 (209) | 833, 3.3e-04 / 0.017 / 0.146 | 0.018 |
| 2 | 3.3e-03 / 0.039 / 0.169 | 0.30 (281) | 453, 5.1e-03 / 0.044 / 0.125 | 0.064 |
| 3 | 7.0e-03 / 0.045 / 0.151 | 0.22 (296) | 209, 1.6e-02 / 0.057 / 0.103 | 0.081 |
| 4 | 9.1e-03 / 0.051 / 0.157 | 0.19 (298) | 74, 2.1e-02 / 0.058 / 0.103 | 0.092 |

Pooled over |S|, the shift is 0.25 of the bound in the median for random pairs and 0.052 for
PBPO designs (0.25 and 0.048 in the first sample); the largest shift is 0.17 and 0.15
(0.22 and 0.16).

Position (exp4): eq. (14) agrees with the exact shift to 3e-16 over 2 251 rows (2 100 from
random pairs, 151 from PBPO designs); for random pairs the median abs(W_F - alpha) rises
from 6e-7 at k = 1 to 7e-3 at k = 7 while the median downstream product rises from 0.013 to
0.64.  The median at k = 1 lies in a distribution that spans many decades and is not a
stable statistic: 146 of the 300 shifts are zero, so that the median lies at the edge of
this mass, and its distribution-free 95 % interval reaches from a numerical zero (6e-17) to
7e-6 (at k = 7: [5e-3, 1e-2]).  The second sample (seed 1004; 2 239 rows, 139 from optimal
designs; agreement with eq. (14) to 3e-16) gives 8e-7 at k = 1, with 143 zero shifts, 8e-3
at k = 7, and 0.012 to 0.62 for the product.  The paper says "below 1e-6" and "about 0.01 to
about 0.65".

## 6. Other experiments

* **Proposition 2, parametric ROCs (exp6).**  Gaussian d' in {0.5, 1, 1.5, 2, 3, 4} and
  the pairs (1,2), (2,1), (0.5,3), (3,0.5); Swerling I SNR 1, 3, 10 and (1,10), (10,1);
  logistic mu in {1, 2, 3} and (1,3).  Phi_lambda has a single maximizer for every lambda
  on a uniform (999 values) and a logit-spaced (801 values) grid of f; the concavity gap of
  the envelope, computed with local refinement of f, is at most 6e-17.
* **Proposition 3 (exp7, exp12).**  Exact rational certificate: envelope 91/160 at
  alpha = 3/40 (hull chord 23/40, gap 1/160) and 29/50 at alpha = 2/25 (gap 1/140).  Maximal
  gap versus the smoothing parameter: 0.0071 (piecewise-linear limit), 0.0067 at 0.004,
  0.0056 at 0.01, 0.0029 at 0.02, 0.0007 at 0.03, 0.0004 at 0.05.
  Perturbation of the two interior knots by up to 5 % (420 draws, 415 of which keep the
  concave-hull structure; `python experiments/knot_summary.py` prints the statistics below
  from the per-draw values).  At eps = 0 every draw is treated exactly: by Corollary S1,
  whose lemmas use only that the local ROC is concave and piecewise linear, the envelope is
  attained at a strategy with two of (f, a0, a1) at knots, and its concave hull has its
  vertices among the 64 strategies with all three at knots.  The gap is at least 1e-3 in 291
  draws (70 %; maximum 0.0084), exactly zero in 101 (24 %) and positive but smaller in 23.
  `knot_summary.py` repeats the classification in exact rational arithmetic (knots as the
  exact binary fractions of the draws; chord minus the best Corollary-S1 candidate at the
  quarter points of every hull edge): zero on every edge for the same 101 draws, positive on
  some edge for the other 314.
  The envelope is concave exactly when every hull edge joins two strategies that differ in
  one coordinate by one knot, so that one threshold moving along one linear segment traces
  the edge; this happens when the middle segment is flat (nominal slope 1; 0.42 to 1.79 over
  the draws): in all 81 draws with a middle slope below 0.728 and in none of the 298 above
  0.809 (rounded inward: all 79 below 0.72, none of the 298 above 0.81).
  Smoothed models.  The gap is the concave hull of the envelope minus the envelope, with
  the hull taken over the whole interval of alpha (grid: step 1e-3 up to 0.2, 5e-3 up to 0.4,
  0.05 beyond).  At eps = 0.004 it is at least 1e-3 in 296 draws (71 %; maximum 0.0080); 7
  draws fall below 1e-3 and 12 rise above it relative to eps = 0.  In the 101 draws whose
  piecewise-linear envelope is concave the gap is 0.8e-4 to 3.2e-4 at eps = 0.004 (median 1.6e-4),
  and it is an effect of the smoothing.  Over eps = 0.001, 0.002, 0.004 it is at most
  0.103 eps (medians 0.043, 0.041, 0.039 eps); only the two draws with the flattest middle
  segments exceed 0.081 eps.  In 98 draws it is roughly proportional to eps (it doubles from
  0.001 to 0.002 and from 0.002 to 0.004 to within an eighth: ratios 1.83 to 2.21).  The other
  three have the three flattest middle segments (slopes 0.415 to 0.426, within 0.01 of the
  slope of the last segment, so that the second knot almost disappears), and there the gap is
  not monotone in eps.  The largest gap of these draws lies at alpha between 0.07 and 0.11.
  The smoothed nominal model has gaps of the same kind away from the gap of its certificate:
  for alpha outside [0.04, 0.11], where its piecewise-linear envelope coincides with its hull,
  the largest gap is 0.158, 0.130, 0.089 eps at eps = 0.001, 0.002, 0.004 (near alpha = 0.03, on
  the hull edge between the vertices at 0.02 and 0.04; alpha refined to steps of 1e-4 around the
  maximum, without which the first value is 0.147).  For the nominal model this ratio grows as
  eps decreases, and the medians of the concave draws grow slightly; the bound of 0.11 eps is a
  statement about the three values of eps computed, not a limit for smaller eps, and the main
  text says "of order 0.1 eps".  A bound for every eps <= 1/20 follows from the argument of
  the certificate (Section S3).  The softmin h_eps satisfies min - eps ln 3 <= h_eps <= min
  at every a, the end points included, so that |r_eps - r_PL| <= eps ln 3 / (1 - eps ln 3)
  <= 1.2 eps whatever the knots.  (The certificate of the nominal model also uses that at
  a = 0 and a = 1 the two inactive lines lie at least 1/2 above the active one; in the
  perturbed models the middle line can pass close to the last one at a = 1, and the
  constant 1.2 does not need that margin.)  The envelope and its concave hull then move by
  at most 2.4 eps each, and a smoothed model whose piecewise-linear envelope is concave has
  a gap of at most 4.8 eps; the computed gaps are more than forty times smaller.  The
  example therefore tolerates most perturbations of this size but needs a sufficiently
  steep middle segment.
  Grid checks (`python experiments/exp7_prop3_gap.py --checks`, `results/exp7_checks.json`), on
  the nine largest ratios gap / eps of the concave draws: refining alpha to steps of 1e-4 around
  the maximum changes them by at most 0.5 % (largest ratio 0.1037 instead of 0.1035); for the
  three largest, a uniform alpha grid of step 1e-3 over [0, 1], and 1601 or 3201 uniform values
  of f instead of 801, leave them unchanged (relative differences below 1e-9: the maximizing f
  lies among the values clustered around the knots).
  Two choices of the computation matter for these small gaps.  (i) The grid of f: a
  perturbed knot is not a point of the uniform grid of 801 values, and at eps = 0 the
  maximizing f is a knot unless a0 and a1 both are (Corollary S1).  The grid therefore has
  400 additional values within 12 transition widths eps / (difference of slopes) of each
  knot.  (ii) The range of alpha: the concave hull is taken over the whole interval [0, 1].
  The hull over a window of alpha lies on or below it, so a window that does not contain
  the whole non-concave region of a smoothed model loses a tangent point of the hull and
  returns a gap that is too small.  The values at eps = 0 are exact and depend on neither
  choice.  Fig. S1 uses the window [0.01, 0.30], which contains both tangent points of the
  nominal gap: on the grid over [0, 1] the maxima at eps = 0.001, 0.002, 0.004 are 0.00703,
  0.00691, 0.00661, against 0.00705, 0.00690, 0.00669 on [0.01, 0.30] with step 5e-4.
  In `results/exp7_prop3_gap_log.txt` the values printed after "paper:" are those of the
  submitted text (0.0066, 0.0055 and 0.0007; maxima of Phi_lambda at f ~ 0.102 and 0.197 with
  a dip of 0.0048).
* **PBPO (exp9, Table SI).**  100 random four-node chains, 2 255 hull vertices (200 of them
  the first or last vertex of a hull): a single random initialization reaches a mid-hull
  vertex in a median 37 % of the runs (IQR 17-57 %); the best of 20 initializations reaches
  95 % of the mid-hull vertices and 90 % of all, in at most 7 sweeps.  Of the 234 vertices
  missed, 190 are the first or last vertex of a hull or within two positions of it.  Keeping
  the downstream regions instead of the thresholds: 3 % and 55 %.  Started at a strategy that
  realizes the vertex, the procedure stays there at every vertex (fixed point).
  Test for "the run ends at the vertex": operating points are compared.  A comparison of
  objective values to 1e-9 would not be reliable, because the vertices are stored rounded
  to 1e-12 and the supporting lambda is large at the low-P_F end of a hull (above 1e3 at
  the first vertex of 27 of the 100 hulls, 2e5 at most), so that the objective computed
  from a stored vertex can be off by more than 1e-9.
  Sampling error.  Over the five disjoint blocks of 20 chains the best-of-20 figures range
  from 94 % to 97 % (mid-hull vertices) and from 88 % to 93 % (all vertices).  The second
  sample (seed 1009; 100 other chains, 2 182 vertices) gives a median single-run hit rate of
  37 % at mid-hull vertices (IQR 17-60 %), 94 % of the mid-hull vertices and 90 % of all for
  the best of 20 initializations, in at most 7 sweeps, and 3 % and 57 % with the downstream
  regions kept; every vertex is a fixed point there as well.
* **Monitoring at the DFC (exp10, Section V-D).**  Two-node running model, alpha = 0.075,
  P0 = 0.9: rates (0.25, 0.15) -> (0.20, 0.20); 2 700 / 1 836 epochs for a 6-sigma
  separation; std of the estimate of p' = 4.0/sqrt(T_S).  Three-node model, alpha = 0.078
  (N_2 switches): (0.255, 0.060) -> (0.1575, 0.1575), 719 / 214 epochs.  alpha = 0.1675
  (N_1 and N_2 switch): (0.085, 0.165); N_1 desynchronized shifts (-0.0125, +0.005), N_2
  (+0.045, -0.0525): opposite sign patterns identify the node.
* **Monte Carlo (exp11).**  4e6 epochs per hypothesis; twelve simulated rates within 1.2
  standard errors of the exact ones, e.g. (0.10780, 0.64631) for the exact
  W = (0.10775, 0.6465) of the three-node model.  Estimator of p': mean 0.4999, standard
  deviation 0.0395 (predicted 0.0400).  Second run (seeds 1011 and 1003): the largest of the
  twelve deviations is 2.8 standard errors (three-node running model, alpha = 0.078, N_2
  desynchronized); the estimator of p' has mean 0.4992 and standard deviation 0.0390.  The
  root mean square of the 24 deviations of the two runs is 0.98; for twelve independent
  standard normal deviations the largest is below 1.2 with probability 0.04 and above 2.8
  with probability 0.05.
* **Delayed message (exp13, Section IV-F).**  Block form of eq. (13) against the exact
  mixture over the two draws: agreement to 3e-16 over 3 200 cases; to 3e-16 over 3 200 other
  cases in the second sample (seed 1013).  The cases include the link into the last node,
  which is the loss of the common draw at the final LD itself (the test of the delayed
  message compares the two computations).
