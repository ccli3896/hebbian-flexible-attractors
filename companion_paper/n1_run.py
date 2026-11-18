"""Behaviour sweep (`python n1_run.py beh`) and attractor runs (`python n1_run.py att`) for companion paper.
Caches to cache/ beside this file."""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))   # this folder; repo root (contains ssb/)
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np, time, sys
os.makedirs(HERE / 'cache', exist_ok=True)
from n1_sim import simulate, ALPHA, BETA, ETA
R=[0.2,0.3,0.5,0.7,1.0]
which=sys.argv[1]
if which=='att':
    # attractor runs: 200 seeds (0..199), 300k trials, every 10th trial
    for r in R:
        t=time.time(); o=simulate(np.arange(200), r, 300_000, stride=10)
        np.savez(f'{HERE}/cache/att_s{r}.npz', **{k:v for k,v in o.items()})
        print(f'att s={r}: {time.time()-t:.0f}s amax={o["amax"]:.2f} finalP={o["A"][-5000:].mean():.3f}', flush=True)
elif which=='beh':
    seeds=np.concatenate([235+9173*k+np.arange(80) for k in range(10)])
    for r in R:
        t=time.time(); o=simulate(seeds, r, 300_000, window=100)
        np.savez(f'{HERE}/cache/beh_s{r}.npz', B=o['B'].reshape(o['B'].shape[0],10,80).transpose(1,0,2), amax=o['amax'])
        print(f'beh s={r}: {time.time()-t:.0f}s amax={o["amax"]:.2f} final={o["B"][-300:].mean():.3f}', flush=True)
