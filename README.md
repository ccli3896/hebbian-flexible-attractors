# hebbian-flexible-attractors

Code for two papers:

| Paper | Folder |
|---|---|
| **Main paper:** Li, Brenner, Ye, Du Pont, Kudarauskas, Aloor, Duan, Chen & Erlich (2026). *Adaptive behaviors emerge from Hebbian-like plasticity through flexible attractors.* Preprint: TODO (bioRxiv DOI) | [`main_paper/`](main_paper/) |
| **Companion paper:** Li & Erlich (2026). *Prospective configuration learns by drift and diffusion along line attractors in closed-loop environments.* PDF in [`companion_paper/`](companion_paper/) | [`companion_paper/`](companion_paper/) |

Each folder has its own README with a figure → code table. Both papers use the shared model package `ssb/`.

## Layout

| Folder | Contents |
|---|---|
| `ssb/` | Shared model: plasticity rules (`models/rules.py`), network dynamics (`models/ssb.py`), environments (`envs/`), simulator (`scripts/run.py`), sweep drivers (`scripts/`) and analysis helpers (`analysis/`) |
| `sweeps/` | Parameter sweeps for the main paper: SLURM launchers (`slurm/`) and post-processing scripts; see "Rebuilding the sweep data" |
| `main_paper/` | One folder per main-paper figure (`fig1/` … `fig5/`) and `supp/` for the Supplementary Information |
| `companion_paper/` | Scripts, printed results and PDF of the companion paper |
| `data/` | Not included; see "Data" |

## Install

```bash
pip install -e .            # installs the `ssb` package (JAX, NumPy, SciPy, pandas, Matplotlib, powerlaw)
pip install adjustText jupyter pyarrow
```

Figures were drawn in Matplotlib (Arial) and assembled in Adobe Illustrator.

## Data

The main-paper notebooks expect a `data/` folder next to `ssb/` (relative paths). Simulation data: Zenodo, [doi:10.5281/zenodo.24342269](https://doi.org/10.5281/zenodo.24342269) (unpack into `data/`; see `readme.txt` there).
The companion paper needs no data: its scripts simulate everything and cache results beside themselves.

| Path | Used by | Source |
|---|---|---|
| `data/det/det_{rule}_{small,big}.npz/.npy` | operating points (`ssb/analysis/best_params.py`), Fig. 4, Supp. Figs. 2-3 | `run_sweep.py` (env `det`) + `aggregate.py` |
| `data/sweeps/{selectivity,slowfeature,bandit_bait,bandit_unbait}_{rule}_{size}.npy` | Supp. Figs. 1-3 | `run_sweep.py` + `aggregate.py` |
| `data/sweeps/matchingpennies_steady_{rule}_{small,big}.npz` | Fig. 3h-k, Supp. Figs. 2-3 | `run_mp_steady.py` + `aggregate_mp_steady.py` |
| `data/sweeps/dist_fits/{rule}_{size}_fits.npz` | Supp. Figs. 2-3 (heavy tail) | `sweeps/fit_runlength_dists.py` |
| `data/magrate/mag_{rule}_{size}.npz` | Supp. Fig. 4 | `run_mag_sweep.py` + `aggregate_supp.py` + `merge_supp_stages.py` + `recompute_relative_criterion.py` |
| `data/animals_pm/bariEtAl_allBehData_20190208.mat` | Fig. 3b, Supp. Fig. 5 | Bari et al. (2019), obtained from the authors |
| `data/animals_pm/Beron_Bandit data.csv` | Fig. 3c, d, Supp. Fig. 5 | Beron et al. (2022), Harvard Dataverse, doi:10.7910/DVN/7E0NM5 |
| `data/animals_mp/matchingpennies_datatable.parquet` (full 47-mouse table, `data/behaviour/mouse/` in the GIN repo) | Fig. 3i, 3j | Aloor et al. (2026), G-Node GIN: https://gin.g-node.org/DELab/matching-pennies-widefield-2026 |
| `data/animals_Gong/gong2026_nav_latencies.npz` | Fig. 2b left | extracted from figshare, doi:10.25378/janelia.32145868 |

Large intermediate results (forgetting runs, HMM fits, reward-magnitude curves, matching-pennies re-runs) are
cached by the notebooks themselves (`cache/` folders) and regenerated if missing.

Sweep grid: alpha, beta in logspace(-5, 0, 15); eta in logspace(-6, -1, 6); tau in logspace(-3, -1, 4) for
BCM, LC, COV and LPL; 20 seeds per cell. Rule names in code: `HR` = RPE, `HZ` = LPL.

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

Figures were produced with JAX on an Apple M3 Max, using the Metal (GPU) backend except in notebooks that set
`jax_platform_name='cpu'`; sweeps ran on CPU nodes of the SWC cluster. Runs on other hardware or backends can
differ in the last digits of reported numbers.

## License

© 2026 Chenguang Li & Jeffrey C Erlich. All Rights Reserved (see [LICENSE](LICENSE)). The code is provided solely for the purpose of peer review; we intend to release it under an open-source license on publication.
