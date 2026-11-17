# Code for "Adaptive behaviors emerge from Hebbian-like plasticity through flexible attractors"

Li, Brenner, Ye, Du Pont, Kudarauskas, Aloor, Duan, Chen and Erlich (2026). Preprint: TODO (bioRxiv DOI).

This repository holds the code needed to regenerate the figures, tables and numbers in the paper
and its Supplementary Information, and in the companion paper (Li & Erlich, "Prospective
configuration learns by drift and diffusion along line attractors in closed-loop environments").
Exploratory analyses, alternative figure versions and statistics that do not appear in the papers
have been removed.

## Install

```bash
pip install -e .            # installs the `ssb` package (JAX, NumPy, SciPy, pandas, Matplotlib, powerlaw)
pip install adjustText jupyter pyarrow
```

Figures were drawn in Matplotlib (Arial) and assembled in Adobe Illustrator.

## Layout

| Folder | Contents |
|---|---|
| `ssb/` | Model (`models/`), environments (`envs/`), simulator (`scripts/run.py`), analysis helpers |
| `ssb/scripts/run_sweep.py` | Parameter-sweep driver (deterministic bandit, bandits, feature and slow-feature selectivity) |
| `ssb/scripts/run_mp_steady.py` | Matching-pennies sweep (seven alternating phases; Fig. 3h-k, Supp. Figs. 2-3) |
| `ssb/scripts/run_mag_sweep.py` | Reward-magnitude sweep over all reward-seeking cells (Supp. Fig. 4) |
| `ssb/analysis/aggregate*.py`, `merge_supp_stages.py`, `recompute_relative_criterion.py` | Turn sweep shards into the arrays in `data/` |
| `ssb/scripts/compute_targets.py` | Beron fitting targets (Fig. 3d, Supp. Fig. 5) |
| `sweeps/slurm/` | SLURM launchers (see "Rebuilding the sweep data") |
| `sweeps/fit_runlength_dists.py` | Run-length fits for every deterministic-bandit cell (heavy-tail column, Supp. Figs. 2-3) |
| `figures/` | One notebook (or script) per figure, below. Folder numbers follow an earlier figure order; use the table below. |
| `data/` | Not included; see "Data" |

## Figure → code

| Paper item | Code |
|---|---|
| Fig. 1b, c, e, f, g; plasticity-off control (Supp. Methods S1) | `figures/fig1/F1_demo.ipynb` |
| Fig. 1j | `figures/fig6/f6i_phase.ipynb` |
| Fig. 2b left | `figures/fig2/F2_animals.ipynb` |
| Fig. 2b right | Traced by hand from Gong et al. (2026), Fig. 4B, in Illustrator; no code |
| Fig. 2c, d; Supp. Fig. 4 | `figures/fig2/F2_magrate.ipynb` |
| Fig. 2e-i; Supp. Methods "Reward-magnitude mechanism" | `figures/fig6/f6ii_mag.ipynb` |
| Fig. 3b-d; Supp. Fig. 5 | `figures/fig4/f4_probmatching.ipynb` (+ `plots.py`) |
| Fig. 3e-g | `figures/fig6/f6iv_pm.ipynb` |
| Fig. 3h-k; matching-pennies rows of Supp. Table 2 | `figures/fig4/f4_matchingpennies.ipynb`; cells chosen with `figures/fig4/select_mp_cells.py` |
| Fig. 4b-h; Supp. Fig. 6 | `figures/fig5/f5_exploreexploit.ipynb` |
| Fig. 4i-k | `figures/fig6/f6vi_blocks.ipynb` |
| Fig. 5a-c; Supp. Fig. 7; Supp. Table 1 | `figures/fig3/f3_forgetting.ipynb` |
| Fig. 5e, f; Supp. Fig. 8 | `figures/fig3/f3_continual.ipynb` |
| Supp. Fig. 1a, c (examples) | `figures/supp/sfig1/S1_behavior_grid.ipynb` |
| Supp. Fig. 1c (scores) | `figures/supp/sfig1/sf1c_scores.py` |
| Supp. Figs. 2, 3 | `figures/supp/sfig2_3/FGrid_S2.ipynb` |
| Supp. Table 2 | Operating points: `ssb/analysis/best_params.py`; other rows are set in the notebooks above |
| Companion paper (all figures and tables except Fig. 12) | `figures/supp_note1/`: run `n1_run.py att`, `n1_run.py beh` and `n1_noise_run.py` once (they cache to `cache/`), then `n1_analysis.py` (writes `figs/` and prints every quoted number) |
| Companion paper, large network (Fig. 12) | `figures/supp_note1/large_network_line.py` |

Schematics and cartoons (e.g. Figs. 1a, 1d, 1h, 2a, 3a, 3h, 4a, 5d) were drawn in Illustrator.

The example traces in Supp. Fig. 1b (LPL target, OJA example) were made with earlier code that is not
included; the slow-feature scores they illustrate are reproduced by `FGrid_S2.ipynb`.

## Rebuilding the sweep data (optional; SLURM)

1. `sweeps/slurm/launch_array.sh` runs `ssb/scripts/run_sweep.py` over the grid (envs `det`, `bandit_bait`,
   `bandit_unbait`, `selectivity`, `slowfeature`; sizes `small`, `big`). Its `matchingpennies` env is an
   earlier three-phase protocol that the paper no longer uses (step 5 replaces it).
2. `python -m ssb.analysis.aggregate --results-dir <sweep results> --out-dir data/sweeps --det-raw-dir <det raw>`
   joins the per-slice files (move the `det_*.npz` outputs to `data/det/`).
3. `python sweeps/det_reward_summary.py` writes `data/det/det_*.npy`; `python sweeps/fit_runlength_dists.py --size small|big`
   writes `data/sweeps/dist_fits/`.
4. Reward-magnitude sweep (Supp. Fig. 4): `python -m ssb.analysis.cell_select` (cell manifest),
   `sweeps/slurm/launch_supp.sh` (`mag` mode), then `ssb.analysis.aggregate_supp`, `merge_supp_stages`
   and `recompute_relative_criterion` to produce `data/magrate/`.
5. Matching-pennies sweep (Fig. 3h-k, Supp. Figs. 2-3): `sweeps/slurm/launch_mp_steady.sh`; then
   `sweeps/slurm/check_mp_steady.sh` reports missing slices and, with `--submit`, resubmits only those;
   finally `python -m ssb.analysis.aggregate_mp_steady --indir sweep_mp_steady/results --outdir data/sweeps`.
   Details in `sweeps/README_mp_steady.md`.

The SLURM scripts carry Sainsbury Wellcome Centre partition and memory settings; adjust them for other clusters.

## Hardware

Figures were produced with JAX on an Apple M3 Max, using the Metal (GPU) backend except in notebooks that
set `jax_platform_name='cpu'`; sweeps ran on CPU nodes of the SWC cluster. Runs on other hardware or
backends can differ in the last digits of reported numbers.

## Data

Place the data in `data/` next to `ssb/` (the notebooks use relative paths). Simulation data: TODO (Zenodo DOI).
Expected contents:

| Path | Used by | Source |
|---|---|---|
| `data/det/det_{rule}_{small,big}.npz/.npy` | operating points (`best_params`), Fig. 4, Supp. Figs. 2-3 | `run_sweep.py` (env `det`) + `aggregate.py` |
| `data/sweeps/{selectivity,slowfeature,bandit_bait,bandit_unbait}_{rule}_{size}.npy` | Supp. Figs. 1-3 | `run_sweep.py` + `aggregate.py` |
| `data/sweeps/matchingpennies_steady_{rule}_{small,big}.npz` | Fig. 3h-k, Supp. Figs. 2-3 | `run_mp_steady.py` + `aggregate_mp_steady.py` |
| `data/sweeps/dist_fits/{rule}_{size}_fits.npz` | Supp. Figs. 2-3 (heavy tail) | `sweeps/fit_runlength_dists.py` |
| `data/magrate/mag_{rule}_{size}.npz` | Supp. Fig. 4 | `run_mag_sweep.py` + `aggregate_supp.py` + `merge_supp_stages.py` + `recompute_relative_criterion.py` |
| `data/animals_pm/bariEtAl_allBehData_20190208.mat` | Fig. 3b, Supp. Fig. 5 | Bari et al. (2019), obtained from the authors |
| `data/animals_pm/Beron_Bandit data.csv` | Fig. 3c, d, Supp. Fig. 5 | Beron et al. (2022), Harvard Dataverse, doi:10.7910/DVN/7E0NM5 |
| `data/animals_mp/matchingpennies_data_MP_CF.parquet` | Fig. 3i | Aloor et al. (2026); TODO: sharing statement |
| `data/animals_Gong/gong2026_nav_latencies.npz` | Fig. 2b left | extracted from figshare, doi:10.25378/janelia.32145868 |

Large intermediate results (forgetting runs, HMM fits, reward-magnitude curves, matching-pennies re-runs) are
cached by the notebooks themselves (`cache/` folders) and regenerated if missing.

Sweep grid: alpha, beta in logspace(-5, 0, 15); eta in logspace(-6, -1, 6); tau in
logspace(-3, -1, 4) for BCM, LC, COV and LPL; 20 seeds per cell.

Rule names in code: `HR` = RPE, `HZ` = LPL.

## License

TODO (e.g. MIT).
