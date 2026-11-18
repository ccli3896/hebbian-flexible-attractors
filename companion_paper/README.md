# Companion paper: figure → code

Li & Erlich (2026). *Prospective configuration learns by drift and diffusion along line attractors in closed-loop
environments.* PDF: `Li_Erlich_2026_prospective_configuration.pdf` (this folder).

Needs only the `ssb` package (see the [top-level README](../README.md)); no data files. Simulations run in JAX on
CPU. Run the scripts from this folder; they cache to `cache/` and write figures to `figs/` here.

**Run order.** `python n1_run.py att` and `python n1_run.py beh` once (about 10 minutes in total), then
`python n1_analysis.py`, which draws the analysis figures and prints every number quoted in the paper
(saved output: `results.txt`).

| Paper item | Code |
|---|---|
| Simulations (behavior sweep, attractor runs) | `n1_sim.py`, `n1_run.py` |
| Figs. 2-3, 6, 9a, b and 10 | `n1_analysis.py` |
| Fig. 4a (line attractor) | `make_fig4_line_attractor.py` |
| Fig. 4b (two branches) | `make_fig5_branches.py` |
| Fig. 5 (rewarded and unrewarded deviations) | `make_fig6_gate.py` |
| Fig. 7 (diffusion after learning) | `make_fig_diffusion_check.py` |
| Fig. 8 (drift terms) | `make_fig_drift_terms.py` |
| Fig. 9c (choice densities) | `make_fig10_density.py` (after `n1_analysis.py`) |
| §5 drift and diffusion derivation | `n1_derive.py` (saved output: `derive_results.txt`) |
| §7 large network, Fig. 11 | `large_network_line.py` (writes `big.npz`), then `make_fig13_large_network.py` |
| §7 large network, mechanism tests | `large_network_tests.py`, then `large_network_tests_analysis.py` |
| Noise sweep (eta = 0.0015, 0.006) read by `n1_analysis.py` | `n1_noise_run.py` |
| PC learning rule (`update_weights`) | `../ssb/models/rules.py` |
| Activity update, network init | `../ssb/models/ssb.py` |
| Deterministic bandit (`bandit_det`) | `../ssb/envs/bandits.py` |

Script names follow an earlier figure numbering; this table gives the final figures.
