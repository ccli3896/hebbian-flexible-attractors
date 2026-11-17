"""Stitch run_mp_steady.py slices into one tensor per (rule, arch) and print a summary.

Usage:
    python -m ssb.analysis.aggregate_mp_steady --indir sweep_mp_steady/results --outdir data/sweeps

Writes  <outdir>/matchingpennies_steady_<rule>_<arch>.npz  with
    scores : (6 eta, n_tau, 15 alpha, 15 beta, n_phases, 4 metrics, n_seeds)   NaN where a slice is missing
    plus phases, metrics and the full grids.
The summary uses only phases >= 2 (i.e. after the first computer -> coin-flip cycle), so it reflects
settled behaviour: per cell, the median over seeds and over the later computer / coin-flip phases.
"""
import argparse, glob, re
from collections import defaultdict
from pathlib import Path
import numpy as np


def aggregate(indir, outdir):
    files = sorted(glob.glob(str(Path(indir) / 'mp_steady_*.npz')))
    groups = defaultdict(list)
    for f in files:
        m = re.match(r'mp_steady_([A-Z]+)_([A-Za-z0-9_]+?)_eta\d', Path(f).name)
        if m: groups[(m.group(1), m.group(2))].append(f)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    for (rule, arch), fs in sorted(groups.items()):
        z0 = np.load(fs[0])
        ne, nt = len(z0['etas_full']), len(z0['taus_full'])
        na, nb = len(z0['alphas_full']), len(z0['betas_full'])
        P, M, S = z0['scores'].shape[4:]
        full = np.full((ne, nt, na, nb, P, M, S), np.nan, np.float32)
        filled = np.zeros((ne, nt, na, nb), bool)
        for f in fs:
            z = np.load(f)
            ie = slice(None) if int(z['eta_idx']) < 0 else slice(int(z['eta_idx']), int(z['eta_idx']) + 1)
            it = slice(None) if int(z['tau_idx']) < 0 else slice(int(z['tau_idx']), int(z['tau_idx']) + 1)
            ia = slice(None) if int(z['alpha_idx']) < 0 else slice(int(z['alpha_idx']), int(z['alpha_idx']) + 1)
            ib = slice(None) if int(z['beta_idx']) < 0 else slice(int(z['beta_idx']), int(z['beta_idx']) + 1)
            full[ie, it, ia, ib] = z['scores']; filled[ie, it, ia, ib] = True
        out = outdir / f'matchingpennies_steady_{rule}_{arch}.npz'
        np.savez_compressed(out, scores=full, phases=z0['phases'], metrics=z0['metrics'], etas=z0['etas_full'],
                            taus=z0['taus_full'], alphas=z0['alphas_full'], betas=z0['betas_full'],
                            timesteps=z0['timesteps'], n_cycles=z0['n_cycles'], seed=z0['seed'])
        print(f'{rule}/{arch}: {len(fs)} slices, {filled.mean():.0%} of cells filled -> {out}')
        summarize(full, z0, rule, arch)


def summarize(full, z, rule, arch, top=5):
    phases = list(z['phases']); later = range(2, len(phases))
    comp = [p for p in later if phases[p] == 'computer']; coin = [p for p in later if phases[p] == 'coinflip']
    if not comp or not coin:
        return
    med = lambda ps, k: np.nanmedian(full[..., ps, k, :].reshape(full.shape[:4] + (-1,)), axis=-1)
    Hc, Hr = med(comp, 0), med(coin, 0)
    Bc, Br = med(comp, 1), med(coin, 1)
    Wc = med(comp, 3)
    gap = Hc - Hr
    order = np.argsort(np.nan_to_num(gap, nan=-9).ravel())[::-1][:top]
    print(f'  {rule}/{arch}: top cells by settled entropy gap (computer - coin flip), phases >= 2')
    for flat in order:
        i = np.unravel_index(flat, gap.shape)
        print(f'    eta={z["etas_full"][i[0]]:.0e} tau={z["taus_full"][i[1]]:.3g} alpha={z["alphas_full"][i[2]]:.3g} '
              f'beta={z["betas_full"][i[3]]:.3g} | H comp {Hc[i]:.2f} coin {Hr[i]:.2f} | bias comp {Bc[i]:.2f} '
              f'coin {Br[i]:.2f} | win comp {Wc[i]:.2f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--indir', required=True)
    p.add_argument('--outdir', default='data/sweeps')
    a = p.parse_args()
    aggregate(a.indir, a.outdir)
