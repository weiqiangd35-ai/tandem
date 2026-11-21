"""tandem_sim: numerical companion to

    Dong & Zhou, "Dependent Randomization in Tandem Binary Decision Fusion:
    Non-Amplification of Synchronization Loss" (IEEE TAES, manuscript
    TAES-2026-3140, revision; submitted under the subtitle "Inherent Robustness to
    Synchronization Loss").

Modules
-------
discrete    discrete-observation models, threshold regions, forward recursion
design      exhaustive enumeration, upper concave hull, support pair, PBPO, and the
            best design in which a single node randomizes (no common draw)
desync      synchronization loss: exact operating point, Prop. 5, Prop. 6 bound,
            terminal-node correction and its benchmarks
continuous  Gaussian / exponential / Swerling-I local models (regions as
            threshold intervals, attenuation factors in closed form)
parallel    parallel architecture (k-out-of-n and Chair-Varshney fusion) for the
            tandem-versus-parallel comparison
exact       exact rational check of Proposition 7 on two-node chains whose models have
            likelihood-ratio ties (imported on demand: tandem_sim.exact)
"""

from .discrete import DiscreteModel, Strategy, evaluate_chain, evaluate_from, realizable_pairs, all_nested_pairs
from .design import (enumerate_system_points, enumerate_from, upper_hull, support_pair,
                     deterministic_optimum, classify_pair, pbpo, pbpo_best, hull_by_pbpo,
                     enumerate_nested_arrays, min_switch_realizations, align_pair, switching_nodes,
                     single_node_optimum)
from .desync import (desync_point, sync_point, prop5_two_node, prop5_midchain,
                     prop6_terms, terminal_adjust, dfc_layer_benchmark,
                     full_downstream_benchmark, downstream_benchmark, effective_point,
                     worst_single_shift,
                     delayed_link_point, delayed_link_identity)

__all__ = [
    "DiscreteModel", "Strategy", "evaluate_chain", "evaluate_from", "realizable_pairs",
    "all_nested_pairs", "enumerate_system_points", "enumerate_from", "upper_hull",
    "support_pair", "deterministic_optimum", "classify_pair", "pbpo", "pbpo_best",
    "hull_by_pbpo", "desync_point", "sync_point", "prop5_two_node", "prop5_midchain",
    "prop6_terms", "terminal_adjust", "dfc_layer_benchmark", "full_downstream_benchmark",
    "effective_point", "delayed_link_point", "delayed_link_identity", "downstream_benchmark",
    "enumerate_nested_arrays", "min_switch_realizations", "align_pair", "switching_nodes",
    "worst_single_shift", "single_node_optimum",
]
