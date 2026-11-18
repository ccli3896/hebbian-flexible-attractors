# Main paper: figure → code

*Adaptive behaviors emerge from Hebbian-like plasticity through flexible attractors.* Install and data: see the
[top-level README](../README.md). Notebooks expect `data/` at the repository root. Notebook names follow an
earlier figure order; this table gives the final panels.

| Paper item | Code |
|---|---|
| Fig. 1b, c, e, f, g; plasticity-off control (Supp. Methods S1) | `fig1/F1_demo.ipynb` |
| Fig. 1j | `fig1/f6i_phase.ipynb` |
| Fig. 2b left | `fig2/F2_animals.ipynb` |
| Fig. 2b right | Traced by hand from Gong et al. (2026), Fig. 4B, in Illustrator; no code |
| Fig. 2c, d; Supp. Fig. 4 | `fig2/F2_magrate.ipynb` |
| Fig. 2e-i; Supp. Methods "Reward-magnitude mechanism" | `fig2/f6ii_mag.ipynb` |
| Fig. 3b-d; Supp. Fig. 5 | `fig3/f4_probmatching.ipynb` (+ `plots.py`) |
| Fig. 3e-g | `fig3/f6iv_pm.ipynb` |
| Fig. 3h-k; matching-pennies rows of Supp. Table 2 | `fig3/f4_matchingpennies.ipynb`; cells chosen with `fig3/select_mp_cells.py` |
| Fig. 4b-h; Supp. Fig. 6 | `fig4/f5_exploreexploit.ipynb` |
| Fig. 4i-k | `fig4/f6vi_blocks.ipynb` |
| Fig. 5a-c; Supp. Fig. 7; Supp. Table 1 | `fig5/f3_forgetting.ipynb` |
| Fig. 5e, f; Supp. Fig. 8 | `fig5/f3_continual.ipynb` |
| Supp. Fig. 1a, c (examples) | `supp/sfig1/S1_behavior_grid.ipynb` |
| Supp. Fig. 1c (scores) | `supp/sfig1/sf1c_scores.py` |
| Supp. Figs. 2, 3 | `supp/sfig2_3/FGrid_S2.ipynb` |
| Supp. Table 2 | Operating points: `../ssb/analysis/best_params.py`; other rows are set in the notebooks above |
| Parameter sweeps behind Supp. Figs. 1-4 and Fig. 3h-k | `../sweeps/` and `../ssb/scripts/` (see the top-level README) |

Schematics and cartoons (e.g. Figs. 1a, 1d, 1h, 2a, 3a, 3h, 4a, 5d) were drawn in Illustrator. The example traces
in Supp. Fig. 1b (LPL target, OJA example) were made with earlier code that is not included; the slow-feature
scores they illustrate are reproduced by `FGrid_S2.ipynb`.
