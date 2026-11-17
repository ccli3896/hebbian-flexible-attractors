"""Stitch supplementary-sweep shards, and drive the two-stage censoring rerun.

Two jobs:

1. **Aggregate.** Concatenate the per-shard ``.npz`` files for each
   (kind, rule, arch, stage) back into single arrays, in cell order. Missing
   shards are reported rather than silently skipped -- a gap means a job never
   landed, and quietly returning a short array would corrupt every downstream
   figure.

2. **Emit a stage-2 manifest.** The magnitude sweep runs stage 1 at a short
   timestep budget. Any (cell, reward) whose learning curve never reached
   criterion is *censored*: its time-to-criterion is a lower bound, not a
   measurement. Fitting rate sensitivity against censored points is exactly the
   threshold artifact the Fig. 2d analysis is designed to exclude, so those
   cells are re-run at the long budget. ``--emit-rerun`` writes a manifest
   containing only the affected cells, which ``launch_supp.sh`` consumes with
   ``STAGE=s2 TIMESTEPS=100000``.

   Why this is worth the extra stage: the cached ``magrate`` runs show 20k
   truncating BCM and LC at the lowest reward by a factor of ~4 (12,700 vs
   52,700 steps), while five of seven rules are unaffected. Running everything
   at the long budget would cost ~5x for a correction that only a small
   minority of cells need.

Merging stages: ``load_stage`` takes a ``prefer`` argument so a stage-2 result
overrides the stage-1 result for the same cell, giving one final array with the
long-budget measurement wherever it was needed.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from ssb.analysis.cell_select import (ALPHAS, BETAS, ETAS, TAUS, RULES,
                                      RULES_WITH_TAU, MANIFEST_FIELDS)

# mag_PC_small_s1_c0.npz  /  cl_HZ_big_s2_c125.npz
SHARD_RE = re.compile(
    r'^(?P<kind>mag|cl|forget)_(?P<rule>[A-Z]+)_(?P<arch>[A-Za-z0-9_]+?)'
    r'_(?P<stage>s\d+)_c(?P<start>\d+)\.npz$')

# Arrays whose leading axis is the cell axis, so shards concatenate along axis 0.
CELL_AXIS_KEYS = {
    'mag': ['curves', 't_crit', 'censored', 'asymptote', 'attractor',
            'sensitivity', 'cell_idx', 'cell_ids'],
    'cl': ['acc', 'memory', 'end_w', 'd0_end', 'd1_end', 'cell_idx', 'cell_ids'],
    'forget': ['v_learned', 'p_learned', 'v_curve', 'p_curve',
               'retention_noinput', 'retention_constant', 'reversal_index',
               'cell_idx', 'cell_ids'],
}


def find_shards(results_dir):
    """Group shard paths by (kind, rule, arch, stage), sorted by cell start."""
    groups = defaultdict(list)
    for p in sorted(Path(results_dir).glob('*.npz')):
        m = SHARD_RE.match(p.name)
        if not m:
            continue
        key = (m['kind'], m['rule'], m['arch'], m['stage'])
        groups[key].append((int(m['start']), p))
    for key in groups:
        groups[key].sort()
    return groups


def load_stage(results_dir, kind, rule, arch, stage, expected_cells=None):
    """Concatenate one group's shards along the cell axis.

    Raises if the shards do not tile the cell axis contiguously from 0 -- a gap
    means a failed job, and the caller must re-run it (RESUME=1) rather than
    build a figure from a silently short array.
    """
    shards = find_shards(results_dir).get((kind, rule, arch, stage), [])
    if not shards:
        raise FileNotFoundError(f'no shards for {kind}/{rule}/{arch}/{stage} in {results_dir}')

    out, cursor = {}, 0
    for start, path in shards:
        if start != cursor:
            raise ValueError(
                f'{kind}/{rule}/{arch}/{stage}: shard gap — expected cell '
                f'{cursor}, found {start} ({path.name}). Re-run the missing '
                f'shard with RESUME=1 before aggregating.')
        d = np.load(path, allow_pickle=True)
        for k in CELL_AXIS_KEYS[kind]:
            if k in d:
                out.setdefault(k, []).append(d[k])
        cursor = start + len(d['cell_idx'])

    merged = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    if expected_cells is not None and cursor != expected_cells:
        raise ValueError(f'{kind}/{rule}/{arch}/{stage}: got {cursor} cells, '
                         f'expected {expected_cells}')
    # Scalar metadata is identical across shards; take it from the first.
    d0 = np.load(shards[0][1], allow_pickle=True)
    for k in ('rewards', 'metric_keys', 'timesteps', 'n_seeds', 'threshold',
              'bin_width', 'num_revs', 'reward',
              'conditions', 'ts_learn', 'ts_probe', 'n_checkpoints'):
        if k in d0:
            merged[k] = d0[k]
    return merged


def _still_rising(curves, frac=0.1, eps=0.005):
    """Per (cell, reward): is the learning curve still climbing at the end?

    Compares the mean of the final ``frac`` of bins against the mean of the
    ``frac`` immediately before it. A curve still going up ran out of budget; a
    flat one has converged, wherever it converged to.
    """
    n = curves.shape[-1]
    k = max(1, int(round(frac * n)))
    med = np.median(curves, axis=2)              # (cells, rewards, bins)
    late = med[..., -k:].mean(axis=-1)
    prev = (med[..., -2 * k:-k].mean(axis=-1) if n >= 2 * k
            else med[..., :k].mean(axis=-1))
    return (late - prev) > eps


def censored_cells(merged, time_limited_only=True, asym_margin=0.05):
    """Cell indices worth re-running at the long budget.

    A cell is censored when its learning curve never reached criterion within
    the budget. That happens for two different reasons, and only one of them is
    fixed by running longer:

    * **time-limited** -- the curve was still climbing when the budget ran out,
      or had already got close to criterion. More steps would get there.
    * **capability-limited** -- the curve flattened well below criterion. This
      parameter setting does not reach that level of performance, and running it
      five times longer gives the same answer at five times the cost.

    Because the sweep covers every reward-seeking cell, including ones that only
    just clear the 0.55 selection threshold, the second group is large: those
    cells are legitimately excluded by the learnability mask of the Fig. 2d
    analysis, not waiting on more compute. With ``time_limited_only`` (default)
    only the first group is returned. Pass ``time_limited_only=False`` for every
    censored cell, i.e. the strict "anything that missed criterion" reading.

    The rate-sensitivity fit needs at least three usable reward levels, so one
    censored level can be the difference between a fit and no fit; a cell
    qualifies if ANY of its levels is worth re-running.
    """
    cens = merged['censored']                    # (cells, rewards)
    if not time_limited_only or 'curves' not in merged:
        return np.where(cens.any(axis=1))[0]

    thr = float(merged.get('threshold', 0.9))
    asym = merged['asymptote']                   # (cells, rewards)
    worth = cens & (_still_rising(merged['curves']) | (asym >= thr - asym_margin))
    return np.where(worth.any(axis=1))[0]


def censoring_breakdown(merged, asym_margin=0.05):
    """Return (n_cells, n_censored, n_time_limited, n_capability_limited)."""
    total = len(merged['cell_idx'])
    all_c = len(censored_cells(merged, time_limited_only=False))
    time_c = len(censored_cells(merged, time_limited_only=True,
                                asym_margin=asym_margin))
    return total, all_c, time_c, all_c - time_c


def emit_rerun_manifest(results_dir, out_path, stage='s1', kinds=('mag',),
                        rules=None, archs=('small', 'big')):
    """Write a manifest of the cells that need the long-budget rerun."""
    rules = list(rules or RULES)
    rows, summary = [], []
    for kind in kinds:
        for arch in archs:
            for rule in rules:
                try:
                    merged = load_stage(results_dir, kind, rule, arch, stage)
                except FileNotFoundError:
                    continue
                idx = censored_cells(merged)
                total, all_c, time_c, cap_c = censoring_breakdown(merged)
                summary.append((rule, arch, len(idx), total, all_c, cap_c))
                has_tau = rule in RULES_WITH_TAU
                ids = merged.get('cell_ids', np.array([]))
                for i in idx:
                    e, t, a, b = (int(x) for x in merged['cell_idx'][i])
                    rows.append({
                        'cell_id': str(ids[i]) if len(ids) > i else f'{rule}_{arch}_r{i:04d}',
                        'rule': rule, 'arch': arch, 'det_arch': arch,
                        'eta_idx': e, 'tau_idx': t if has_tau else -1,
                        'alpha_idx': a, 'beta_idx': b,
                        'eta': f'{ETAS[e]:.6g}',
                        'tau': f'{TAUS[t]:.6g}' if (has_tau and t >= 0) else '',
                        'alpha': f'{ALPHAS[a]:.6g}', 'beta': f'{BETAS[b]:.6g}',
                        'det_score': '0.000000',
                        'selection': f'rerun-from-{stage}',
                        # -1 = "not carried through this stage". The rerun
                        # manifest exists only to re-run cells; join back on
                        # cell_id against the master manifest for the flag.
                        'in_balanced': -1,
                        'source': f'{kind}_{rule}_{arch}_{stage}',
                    })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f'rerun manifest: {len(rows)} cells -> {out_path}\n')
    print(f'{"rule":5} {"arch":8} {"cells":>6} {"censored":>9} '
          f'{"time-lim":>9} {"cap-lim":>8}   rerun')
    for rule, arch, n, total, all_c, cap_c in summary:
        pct = 100.0 * n / total if total else 0.0
        print(f'  {rule:4s} {arch:8s} {total:6d} {all_c:9d} {n:9d} {cap_c:8d}'
              f'   {pct:.1f}%')
    print('\ntime-lim  = curve still rising, or already near criterion; more '
          'steps would reach it -> re-run')
    print('cap-lim   = curve flat and below criterion; more steps change '
          'nothing -> excluded, and the')
    print('            Fig. 2d learnability mask drops these from the '
          'rate-sensitivity fit anyway')
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', required=True)
    p.add_argument('--stage', default='s1')
    p.add_argument('--kinds', nargs='+', default=['mag'], choices=['mag', 'cl', 'forget'])
    p.add_argument('--rules', nargs='+', default=RULES)
    p.add_argument('--archs', nargs='+', default=['small', 'big'])
    p.add_argument('--out-dir', help='write aggregated arrays here (.npz per group)')
    p.add_argument('--emit-rerun', help='write a stage-2 manifest of censored cells')
    args = p.parse_args(argv)

    groups = find_shards(args.results_dir)
    print(f'found {sum(len(v) for v in groups.values())} shards in '
          f'{len(groups)} groups')

    if args.out_dir:
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        for kind in args.kinds:
            for arch in args.archs:
                for rule in args.rules:
                    try:
                        merged = load_stage(args.results_dir, kind, rule, arch, args.stage)
                    except FileNotFoundError:
                        continue
                    dest = out_dir / f'{kind}_{rule}_{arch}_{args.stage}.npz'
                    np.savez_compressed(dest, **merged)
                    print(f'  wrote {dest.name}  ({len(merged["cell_idx"])} cells)')

    if args.emit_rerun:
        emit_rerun_manifest(args.results_dir, args.emit_rerun, stage=args.stage,
                            kinds=[k for k in args.kinds if k == 'mag'],
                            rules=args.rules, archs=args.archs)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
