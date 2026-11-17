"""Noise sweep for Supp. Note 1 Table 2 (eta = 0.0015 and 0.006; eta = 0.003 is the attractor run).
Caches to cache/ beside this file."""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[2]))   # this folder; repo root (contains ssb/)
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np, time
from n1_sim import simulate
for eta in (0.0015, 0.006):
    for r in (0.2, 0.5, 1.0):
        t=time.time(); o=simulate(np.arange(200), r, 300_000, eta=eta, stride=10)
        np.savez(f'{HERE}/cache/noise_s{r}_e{eta}.npz', **o)
        print(f'noise eta={eta} s={r}: {time.time()-t:.0f}s finalP={o["A"][-5000:].mean():.3f}', flush=True)
