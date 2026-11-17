"""Pick each rule's best deterministic-bandit hyperparameters from the raw sweep.

For a given ``(rule, size)``, ``best_params`` returns the ``(eta, alpha, beta,
tau)`` cell with the highest median-over-time performance across the last
20,000 timesteps of the det sweep's raw per-seed trajectories
(``data/det/det_{rule}_{size}.npz`` -- unclipped, consistent with
``ssb.scripts.run.run_quick``), restricted to eta >= 1e-5 (see "selection" in
``data/readme.txt``). Performance at each timestep is the MEAN across seeds.

Single source of truth for the operating point used across the main-figure
notebooks (F1_demo, F2_SFig_magrate, f3_continual) instead of each one
re-deriving or hand-tuning its own numbers.

Distinct from ``ssb.analysis.cell_select``, which selects a whole *region* of
reward-seeking cells (for supplementary parameter sweeps) from the aggregated
``.npy`` summary; this module picks the single best cell from the raw
``.npz`` per-seed trajectories.

Degenerate cells. For the (rule, size) pairs in ``EXCLUDE_DEGENERATE`` the
unrestricted argmax lands on alpha = 1 (activity step 2*alpha = 2, so the
Euler update x <- 2*target - x oscillates instead of relaxing) with beta at
the grid floor (1e-5, plasticity effectively off); with beta = 0 those
networks score the same, so their reward seeking is not produced by the rule.
For these pairs the alpha = 1 column and the beta = 1e-5 row are excluded
before taking the argmax. Selected cells (Nov 2026):
  BCM/big: eta=1e-3, tau=0.1, alpha=0.0848, beta=1      (score 0.80)
  LC/big : eta=1e-4, tau=0.1, alpha=0.0072, beta=0.0848 (score 0.75)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ssb.analysis.cell_select import ALPHAS, BETAS, ETAS, TAUS

# data/det/, resolved relative to this file so callers get the right path
# regardless of a notebook's own working directory or depth in the repo.
DET_FOLDER = Path(__file__).resolve().parents[2] / 'data' / 'det'

# (rule, size) pairs whose unrestricted best cell is degenerate (see module docstring).
EXCLUDE_DEGENERATE = {('BCM', 'big'), ('LC', 'big')}


def best_params(rule: str, size: str, det_folder: str | Path = DET_FOLDER,
                exclude_degenerate: bool = True):
    """Return (eta, alpha, beta, tau) for rule/size's best det-sweep cell.

    With ``exclude_degenerate`` (default), pairs in ``EXCLUDE_DEGENERATE`` skip
    alpha = 1 and beta = 1e-5; pass False to get the original unrestricted cell."""
    x = np.load(Path(det_folder) / f'det_{rule}_{size}.npz')['x']
    mean_seeds = x.mean(axis=-3)              # mean across seeds, per timestep
    last20k = mean_seeds[..., -20000:, :, :]
    score = np.median(last20k, axis=-3)       # median over time within the last 20k steps
    score[:1] = -1                            # exclude eta_idx 0 (eta < 1e-5)
    if exclude_degenerate and (rule, size) in EXCLUDE_DEGENERATE:
        score[..., -1, :] = -1                # alpha = 1 (last alpha index)
        score[..., :, 0] = -1                 # beta = 1e-5 (first beta index)

    idx = np.unravel_index(np.argmax(score), score.shape)
    eta = ETAS[idx[0]]
    if score.ndim == 3:
        alpha, beta, tau = ALPHAS[idx[1]], BETAS[idx[2]], 0.
    else:
        alpha, beta, tau = ALPHAS[idx[2]], BETAS[idx[3]], TAUS[idx[1]]
    return eta, alpha, beta, tau
