"""Fit run-length distributions for every cell of the deterministic-bandit sweep.

Writes data/sweeps/dist_fits/{rule}_{size}_fits.npz (favored model, power-law exponent,
reward), used for the heavy-tail column of Supplementary Figs. 2-3. Extracted from the
fitting half of the original internal script (det_sweep_processing.py); the PDF booklet
it also drew is not part of the paper and is omitted.

Usage:  python sweeps/fit_runlength_dists.py --size small --det-dir data/det --out data/sweeps/dist_fits
"""
import argparse, os
import numpy as np
from ssb.analysis import powerlaw_fit as plf

RULES = ['PC', 'HR', 'COV', 'OJA', 'BCM', 'LC', 'HZ']
N_ALPHA = N_BETA = 15
FAVOR_KEY = {'power_law': 1, 'inconclusive': 0, 'exponential': -1}


def has_tau(arr):
    return arr.ndim == 6


def get_seq(arr, eta_idx, tau_idx, alpha, beta):
    """Return (T, seeds) action sequence for one hyperparameter cell."""
    if has_tau(arr):       # (eta, tau, T, seeds, alpha, beta)
        return arr[eta_idx, tau_idx, :, :, alpha, beta]
    return arr[eta_idx, :, :, alpha, beta]   # (eta, T, seeds, alpha, beta)


def get_lengths(seq_across_seeds):
    """Concatenate run lengths across seeds. No decimation."""
    n_seeds = seq_across_seeds.shape[1]
    out = []
    for s in range(n_seeds):
        out.append(plf.action_lengths(seq_across_seeds[:, s]))
    return np.concatenate(out) if out else np.array([], dtype=int)


def _fit_one_cell(args):
    """Worker for parallel fitting. args = (lengths, fixed_xmin)."""
    lengths, fixed_xmin = args
    if lengths.size < 30 or np.unique(lengths).size < 2:
        return (np.nan, np.nan)
    try:
        favored_str, alpha = plf.fit_favored_alpha(
            lengths, discrete=True, xmin=fixed_xmin)
        return (FAVOR_KEY[favored_str], alpha)
    except Exception:
        return (np.nan, np.nan)


def fit_all_cells(arr, min_runs=10, fixed_xmin=1, n_workers=12):
    """Fit every (eta, tau, alpha, beta) cell.

    Speed measures:
      - fast path: uses plf.fit_favored_alpha, which skips the expensive
        power_law.KS() computation (~2x vs fit_and_compare). The booklet
        only plots favored + exponent, so KS isn't needed per cell.
      - prefilter (min_runs): skip dead/degenerate cells.
      - fixed_xmin: skip the xmin KS-search.
      - n_workers > 1: parallelize across cells with multiprocessing
        (fits are independent). Near-linear speedup in cores; the dominant
        cost is the per-cell powerlaw fit, so this is the big lever on a
        multi-core machine. n_workers=1 runs serially (default).

    Returns (favored, exponent), each (n_eta, n_tau, N_ALPHA, N_BETA).
    """
    n_eta = arr.shape[0]
    n_tau = arr.shape[1] if has_tau(arr) else 1

    favored = np.full((n_eta, n_tau, N_ALPHA, N_BETA), np.nan)
    exponent = np.full((n_eta, n_tau, N_ALPHA, N_BETA), np.nan)

    # Build the full list of (cell-index, lengths) up front.
    cells = []
    payloads = []
    for e in range(n_eta):
        for t in range(n_tau):
            for a in range(N_ALPHA):
                for w in range(N_BETA):
                    lengths = get_lengths(get_seq(arr, e, t, a, w))
                    cells.append((e, t, a, w))
                    payloads.append((lengths, fixed_xmin))

    if n_workers and n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            results = pool.map(_fit_one_cell, payloads, chunksize=16)
    else:
        results = [_fit_one_cell(p) for p in payloads]

    n_fit = 0
    for (e, t, a, w), (fav, alpha) in zip(cells, results):
        favored[e, t, a, w] = fav
        exponent[e, t, a, w] = alpha
        if not np.isnan(fav):
            n_fit += 1
    print(f"    fit {n_fit}/{len(cells)} cells "
          f"({len(cells) - n_fit} skipped/failed)")
    return favored, exponent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', choices=['small', 'big'], required=True)
    ap.add_argument('--det-dir', default='data/det')
    ap.add_argument('--out', default='data/sweeps/dist_fits')
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for rule in RULES:
        arr = np.load(os.path.join(a.det_dir, f'det_{rule}_{a.size}.npz'))['x']
        favored, exponent = fit_all_cells(arr, n_workers=a.workers)
        reward = np.load(os.path.join(a.det_dir, f'det_{rule}_{a.size}.npy'))
        if reward.ndim == 3: reward = reward[:, None]
        np.savez(os.path.join(a.out, f'{rule}_{a.size}_fits.npz'), favored=favored, exponent=exponent, reward=reward)
        print(rule, a.size, 'saved')


if __name__ == '__main__':
    main()
