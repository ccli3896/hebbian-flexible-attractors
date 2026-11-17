"""Summarize the deterministic-bandit sweep: data/det/det_{rule}_{size}.npy.

Per cell, the fraction of stored steps on the rewarded arm (mean over time and
seeds) of the raw action array in data/det/det_{rule}_{size}.npz. Streams the
array one leading slice at a time so it fits in memory.

Usage:  python sweeps/det_reward_summary.py --det-dir data/det
"""
import argparse, os, zipfile
import numpy as np
from numpy.lib import format as fmt

RULES = ['PC', 'HR', 'COV', 'OJA', 'BCM', 'LC', 'HZ']


def summarize(path):
    with zipfile.ZipFile(path) as z, z.open('x.npy') as f:
        version = fmt.read_magic(f)
        shape, _, dtype = fmt._read_array_header(f, version)
        lead = shape[:-4]                 # (eta,) or (eta, tau); then (T, seeds, alpha, beta)
        per = int(np.prod(shape[len(lead):])) * dtype.itemsize
        out = np.zeros(lead + shape[-2:])
        for idx in np.ndindex(*lead):
            a = np.frombuffer(f.read(per), dtype=dtype).reshape(shape[len(lead):])
            out[idx] = a.mean(axis=(0, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det-dir', default='data/det')
    a = ap.parse_args()
    for size in ('small', 'big'):
        for rule in RULES:
            p = os.path.join(a.det_dir, f'det_{rule}_{size}.npz')
            if not os.path.exists(p):
                continue
            np.save(os.path.join(a.det_dir, f'det_{rule}_{size}.npy'), summarize(p))
            print(rule, size, 'done')


if __name__ == '__main__':
    main()
