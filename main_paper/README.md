# Main paper: figure → code

*Adaptive behaviors emerge from Hebbian-like plasticity through flexible attractors.* Install and data: see the
[top-level README](../README.md). Notebooks expect `data/` at the repository root. File names give the figure and panels each
notebook produces.

| Paper item | Code |
|---|---|
| Fig. 1b, c, e, f, g; plasticity-off control (Supp. Methods S1) | `fig1/fig1_signal_seeking.ipynb` |
| Fig. 1j | `fig1/fig1j_attractors.ipynb` |
| Fig. 2b left | `fig2/fig2b_mouse_data.ipynb` |
| Fig. 2b right | Traced by hand from Gong et al. (2026), Fig. 4B, in Illustrator; no code |
| Fig. 2c, d; Supp. Fig. 4 | `fig2/fig2cd_sfig4_reward_magnitude.ipynb` |
| Fig. 2e-i; Supp. Methods "Reward-magnitude mechanism" | `fig2/fig2e-i_reward_magnitude_mechanism.ipynb` |
| Fig. 3b-d; Supp. Fig. 5 | `fig3/fig3b-d_sfig5_probability_matching.ipynb` (+ `plots.py`) |
| Fig. 3e-g | `fig3/fig3e-g_attractor_length.ipynb` |
| Fig. 3h-k; matching-pennies rows of Supp. Table 2 | `fig3/fig3h-k_matching_pennies.ipynb`; cells chosen with `fig3/select_mp_cells.py` |
| Fig. 4b-h; Supp. Fig. 6 | `fig4/fig4b-h_sfig6_explore_exploit.ipynb` |
| Fig. 4i-k | `fig4/fig4i-k_changing_environments.ipynb` |
| Fig. 5a-c; Supp. Fig. 7; Supp. Table 1 | `fig5/fig5a-c_sfig7_forgetting.ipynb` |
| Fig. 5e, f; Supp. Fig. 8 | `fig5/fig5e-f_sfig8_memory_retention.ipynb` |
| Supp. Fig. 1a, c (examples) | `supp/sfig1/sfig1_behavior_grid.ipynb` |
| Supp. Fig. 1c (scores) | `supp/sfig1/sfig1c_scores.py` |
| Supp. Figs. 2, 3 | `supp/sfig2_3/sfig2-3_parameter_sweeps.ipynb` |
| Supp. Table 2 | Operating points: `../ssb/analysis/best_params.py`; other rows are set in the notebooks above |
| Parameter sweeps behind Supp. Figs. 1-4 and Fig. 3h-k | `../sweeps/` and `../ssb/scripts/` (see the top-level README) |

Schematics and cartoons (e.g. Figs. 1a, 1d, 1h, 2a, 3a, 3h, 4a, 5d) were drawn in Illustrator. The example traces
in Supp. Fig. 1b (LPL target, OJA example) were made with earlier code that is not included; the slow-feature
scores they illustrate are reproduced by `sfig2-3_parameter_sweeps.ipynb`.
