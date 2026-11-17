"""Select the reward-seeking region of the hyperparameter grid.

The supplementary sweeps (reward magnitude, continual learning) do NOT re-scan
the whole (eta, tau, alpha, beta) grid. They run only at parameter settings
where the rule can reward-seek *at all*, judged from the already-completed
deterministic-bandit sweep. Everything below 55% choice of the good arm is a
parameter setting where the rule simply doesn't learn the task, so measuring
its reward-magnitude scaling or its retention would be measuring noise.

Terminology
-----------
A **cell** is one point of the learning-rule hyperparameter grid:

    (eta, alpha, beta)          for PC, HR, OJA        -> 6 x 15 x 15 = 1350
    (eta, tau, alpha, beta)     for COV, BCM, LC, HZ   -> 6 x 4 x 15 x 15 = 5400

(Both trees now call the weight learning rate `beta`; it was `omega` before the
rename, and `batchify_hps` raises if the old key is still passed.)
The det sweep scored every cell; `SEEK_THRESHOLD` keeps the seeking ones.

Input
-----
``data/det/det_{rule}_{size}.npy``, written by ``sweeps/det_reward_summary.py`` as
the mean over (T, seeds) of the raw action array — i.e. per cell, the fraction of steps on the rewarded arm.
Shapes: ``(eta, alpha, beta)`` for no-tau rules, ``(eta, tau, alpha, beta)``
for tau rules.

Output
------
A labelled CSV manifest, one row per selected cell, carrying BOTH the absolute
grid indices (so a sweep job can reconstruct the exact hyperparameters) and the
resolved float values (so the manifest is readable without re-deriving the
grids). Every downstream artifact is keyed by ``cell_id``, so any number in the
supplement can be traced back to a specific hyperparameter setting.

Selection
---------
``all`` (default) takes every seeking cell, so each rule/arch gets exactly its
own region. Those regions are very unequal — OJA/small has 67 seeking cells,
HZ/small has 2613 — which is the point: a flat quota would mean 100% coverage
of one rule's region and under 4% of another's. Running all of them costs about
77 CPU-h, so there is no reason to subsample.

Because the regions are unequal, every row also carries ``in_balanced``, marking
the stratified equal-N subset (default 100 per rule/arch). Per-rule analyses use
all rows; cross-rule comparisons, where the largest regions would otherwise
dominate, can filter to ``in_balanced == 1``.

``stratified`` spends a fixed per-(rule, arch) budget across the seeking region
rather than clustering at the peak, and is what computes that subset:

1. The argmax cell is always included, so the sweep contains the exact setting
   the main figures use and the supplement can be cross-checked against them.
2. The remaining budget is split across (eta, tau) planes in proportion to
   ``sqrt(plane population)`` — square-root allocation so a plane holding most
   of the seeking mass doesn't swallow the budget and leave sparse planes
   unrepresented.
3. Within a plane, cells are chosen by farthest-point sampling in
   (alpha_idx, beta_idx), seeded from that plane's best-scoring cell. This
   spreads picks over the region's extent instead of its density, which is what
   makes the result a robustness check rather than a re-measurement of the peak.

Farthest-point sampling and the allocation are both deterministic, so the same
inputs always produce the same manifest (no RNG seed to record).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

# Grid definitions. These MUST match ssb/scripts/run_sweep.py, which is the
# code that actually produced the det sweep being read here. Duplicated as
# module constants (not imported) so this module stays free of the jax import.
ALPHAS = np.logspace(-5, 0, 15)
BETAS = np.logspace(-5, 0, 15)
ETAS = np.logspace(-6, -1, 6)
TAUS = np.logspace(-3, -1, 4)

RULES = ['PC', 'HR', 'COV', 'OJA', 'BCM', 'LC', 'HZ']
RULES_WITH_TAU = {'COV', 'BCM', 'LC', 'HZ'}

# Fraction-of-steps-on-good-arm below which a cell is treated as non-seeking.
# Cells at or below this are excluded from every supplementary sweep.
SEEK_THRESHOLD = 0.55

# Per-(rule, arch) cell budget for the stratified selection.
DEFAULT_BUDGET = 100

MANIFEST_FIELDS = [
    'cell_id', 'rule', 'arch', 'det_arch',
    'eta_idx', 'tau_idx', 'alpha_idx', 'beta_idx',
    'eta', 'tau', 'alpha', 'beta',
    'det_score', 'selection', 'in_balanced', 'source',
]


def det_score_path(agg_dir, rule: str, det_arch: str) -> Path:
    """Path of the aggregated det reward file for one (rule, arch)."""
    return Path(agg_dir) / f'det_{rule}_{det_arch}.npy'


def load_det_scores(agg_dir, rule: str, det_arch: str) -> np.ndarray:
    """Load det scores as a 4-D ``(eta, tau, alpha, beta)`` array.

    No-tau rules are stored as ``(eta, alpha, beta)``; they are given a
    length-1 tau axis so every downstream code path can index uniformly.
    ``tau_idx`` is then written as -1 in the manifest to mark "not applicable"
    rather than "tau index 0", which matters because the sweep drivers must not
    pass a tau to a rule that has none.
    """
    path = det_score_path(agg_dir, rule, det_arch)
    if not path.exists():
        raise FileNotFoundError(
            f"missing det scores: {path}\n"
            f"Expected the aggregated deterministic-bandit sweep. Point "
            f"--agg-dir at the directory holding det_*.npy (e.g. "
            f"data/det)."
        )
    arr = np.load(path)
    expect_tau = rule in RULES_WITH_TAU
    if expect_tau:
        if arr.ndim != 4:
            raise ValueError(
                f"{path}: rule {rule} has tau, expected 4-D "
                f"(eta, tau, alpha, beta), got shape {arr.shape}"
            )
    else:
        if arr.ndim != 3:
            raise ValueError(
                f"{path}: rule {rule} has no tau, expected 3-D "
                f"(eta, alpha, beta), got shape {arr.shape}"
            )
        arr = arr[:, None]          # -> (eta, 1, alpha, beta)
    return arr


def _farthest_point_order(coords: np.ndarray, start: int, n: int) -> list:
    """Indices of ``n`` points from ``coords`` spread as widely as possible.

    Greedy farthest-point sampling: repeatedly take the point maximising the
    minimum distance to everything already chosen. Deterministic, and ties
    break on the lower index via ``argmax``. ``coords`` is (n_points, n_dims).
    """
    n = min(n, len(coords))
    chosen = [start]
    if n == 1:
        return chosen
    # Running min-distance from each point to the chosen set.
    dist = np.linalg.norm(coords - coords[start], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(dist))
        if dist[nxt] <= 0:
            break                    # every remaining point is a duplicate
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(coords - coords[nxt], axis=1))
    return chosen


def select_cells(scores: np.ndarray, budget: int = DEFAULT_BUDGET,
                 threshold: float = SEEK_THRESHOLD,
                 selection: str = 'stratified') -> list:
    """Choose cells from a 4-D ``(eta, tau, alpha, beta)`` score array.

    Returns a list of ``(eta_idx, tau_idx, alpha_idx, beta_idx)`` tuples.

    selection:
      'stratified' -- argmax + sqrt-allocated, farthest-point spread (default)
      'top'        -- the ``budget`` highest-scoring cells
      'all'        -- every seeking cell, ignoring ``budget``
    """
    seeking = np.argwhere(scores > threshold)
    if len(seeking) == 0:
        return []

    vals = scores[tuple(seeking.T)]
    # Score-descending order. Every return path below is built from this, so
    # cell index 0 is ALWAYS the argmax cell for that (rule, arch) -- the same
    # operating point the main figures use. That invariant is what lets the
    # supplement be cross-checked against the main figures, and it has to hold
    # on the under-budget path too, where no sampling actually happens.
    by_score = np.argsort(-vals, kind='stable')

    if selection == 'all':
        return [tuple(int(i) for i in seeking[o]) for o in by_score]

    if selection == 'top':
        return [tuple(int(i) for i in seeking[o]) for o in by_score[:budget]]

    if selection != 'stratified':
        raise ValueError(f"unknown selection {selection!r}")

    if len(seeking) <= budget:
        # Fewer seeking cells than the budget: take them all, no sampling.
        return [tuple(int(i) for i in seeking[o]) for o in by_score]

    # --- stratified -------------------------------------------------------
    # 1. Always keep the global argmax so the sweep contains the main-figure
    #    operating point.
    best = seeking[int(np.argmax(vals))]
    picked = [tuple(int(i) for i in best)]

    # 2. Group the seeking cells by (eta, tau) plane.
    planes = {}
    for row, val in zip(seeking, vals):
        planes.setdefault((int(row[0]), int(row[1])), []).append((row, val))

    # sqrt allocation: proportional to sqrt(population), so sparse planes stay
    # represented and dense planes don't monopolise the budget.
    weights = {k: np.sqrt(len(v)) for k, v in planes.items()}
    total_w = sum(weights.values())
    remaining = budget - len(picked)
    alloc = {k: int(np.floor(remaining * w / total_w)) for k, w in weights.items()}

    # Hand out the floor remainder to the largest planes, largest first, so the
    # budget is spent exactly rather than under-filled by rounding.
    leftover = remaining - sum(alloc.values())
    for k in sorted(planes, key=lambda k: (-len(planes[k]), k))[:leftover]:
        alloc[k] += 1

    # 3. Farthest-point spread within each plane, seeded from its best cell.
    #    The argmax kept in step 1 is its own plane's best cell, so that plane's
    #    sampling re-proposes it first; ask that plane for one extra pick so the
    #    duplicate doesn't silently cost a slot (budget-1 instead of budget).
    best_plane = (int(best[0]), int(best[1]))
    seen = set(picked)
    for key in sorted(planes):
        want = alloc.get(key, 0)
        if key == best_plane:
            want += 1
        if want <= 0:
            continue
        rows = np.array([r for r, _ in planes[key]])
        pvals = np.array([v for _, v in planes[key]])
        coords = rows[:, 2:].astype(float)          # (alpha_idx, beta_idx)
        start = int(np.argmax(pvals))
        for j in _farthest_point_order(coords, start, want):
            cell = tuple(int(i) for i in rows[j])
            if cell not in seen:
                seen.add(cell)
                picked.append(cell)

    return picked[:budget]


def build_manifest(agg_dir, rules=None, archs=('small', 'big'),
                   budget: int = DEFAULT_BUDGET,
                   threshold: float = SEEK_THRESHOLD,
                   selection: str = 'all',
                   det_arch_map=None, balanced_budget: int = DEFAULT_BUDGET) -> list:
    """Build manifest rows for every (rule, arch) combination.

    Default ``selection='all'`` runs every reward-seeking cell, so each rule
    gets exactly its own region rather than a flat quota. Those regions differ
    enormously -- OJA/small has 67 seeking cells, HZ/small has 2613 -- so a flat
    quota would mean 100% coverage for one rule and 3.8% for another under the
    same nominal budget.

    Because the regions are so unequal, any *pooled* or cross-rule comparison
    would be dominated by the largest regions. Each row therefore also carries
    ``in_balanced``: 1 if the cell belongs to the stratified ``balanced_budget``
    subset for its (rule, arch), else 0. Per-rule analyses use every row; a
    cross-rule comparison can filter to ``in_balanced == 1`` for equal N. The
    subset is chosen by the same deterministic stratification as before, so it
    still contains that rule's argmax cell.

    ``det_arch_map`` maps a target arch onto the det-sweep arch whose seeking
    region it should inherit (used when an arch has no det sweep of its own).
    Unmapped archs read their own ``det_{rule}_{arch}.npy``. The arch actually
    read is recorded per row as ``det_arch``, so an inherited selection is
    never silently mistaken for a measured one.
    """
    rules = list(rules or RULES)
    det_arch_map = dict(det_arch_map or {})
    rows = []
    for arch in archs:
        det_arch = det_arch_map.get(arch, arch)
        for rule in rules:
            scores = load_det_scores(agg_dir, rule, det_arch)
            has_tau = rule in RULES_WITH_TAU
            cells = select_cells(scores, budget=budget, threshold=threshold,
                                 selection=selection)
            # The balanced subset is computed independently of `cells` so the
            # flag means the same thing regardless of which selection ran.
            balanced = set(select_cells(scores, budget=balanced_budget,
                                        threshold=threshold,
                                        selection='stratified'))
            for n, (e, t, a, b) in enumerate(cells):
                rows.append({
                    'cell_id': f'{rule}_{arch}_c{n:04d}',
                    'rule': rule,
                    'arch': arch,
                    'det_arch': det_arch,
                    'eta_idx': e,
                    'tau_idx': t if has_tau else -1,
                    'alpha_idx': a,
                    'beta_idx': b,
                    'eta': f'{ETAS[e]:.6g}',
                    'tau': f'{TAUS[t]:.6g}' if has_tau else '',
                    'alpha': f'{ALPHAS[a]:.6g}',
                    'beta': f'{BETAS[b]:.6g}',
                    'det_score': f'{float(scores[e, t, a, b]):.6f}',
                    'selection': selection,
                    'in_balanced': 1 if (e, t, a, b) in balanced else 0,
                    'source': det_score_path(agg_dir, rule, det_arch).name,
                })
    return rows


def write_manifest(rows, out_path) -> Path:
    """Write manifest rows to CSV, creating parent directories."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out_path


def read_manifest(path) -> list:
    """Read a manifest CSV back into typed dicts (indices as int)."""
    out = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            for k in ('eta_idx', 'tau_idx', 'alpha_idx', 'beta_idx'):
                row[k] = int(row[k])
            row['in_balanced'] = int(row.get('in_balanced', 0) or 0)
            row['det_score'] = float(row['det_score'])
            out.append(row)
    return out


def manifest_rows_for(rows, rule: str, arch: str) -> list:
    """Filter manifest rows down to one (rule, arch), preserving order."""
    return [r for r in rows if r['rule'] == rule and r['arch'] == arch]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--agg-dir', required=True,
                   help='directory holding aggregated det_{rule}_{arch}.npy')
    p.add_argument('--out', required=True, help='manifest CSV to write')
    p.add_argument('--archs', nargs='+', default=['small', 'big'])
    p.add_argument('--rules', nargs='+', default=RULES)
    p.add_argument('--budget', type=int, default=DEFAULT_BUDGET,
                   help='cells per (rule, arch) [default: %(default)s]')
    p.add_argument('--threshold', type=float, default=SEEK_THRESHOLD,
                   help='det score above which a cell counts as reward-seeking')
    p.add_argument('--selection', default='all',
                   choices=['stratified', 'top', 'all'],
                   help='all = every seeking cell (default); stratified/top = '
                        'subsample to --budget')
    p.add_argument('--balanced-budget', type=int, default=DEFAULT_BUDGET,
                   help='size of the equal-N subset flagged as in_balanced '
                        '[default: %(default)s]')
    p.add_argument('--inherit', nargs='*', default=[], metavar='ARCH=DET_ARCH',
                   help='arch whose seeking region is inherited from another '
                        'arch\'s det sweep, e.g. N50_L3=big')
    args = p.parse_args(argv)

    det_arch_map = {}
    for item in args.inherit:
        if '=' not in item:
            p.error(f'--inherit expects ARCH=DET_ARCH, got {item!r}')
        k, v = item.split('=', 1)
        det_arch_map[k] = v

    rows = build_manifest(args.agg_dir, rules=args.rules, archs=args.archs,
                          budget=args.budget, threshold=args.threshold,
                          selection=args.selection, det_arch_map=det_arch_map,
                          balanced_budget=args.balanced_budget)
    out = write_manifest(rows, args.out)

    print(f'wrote {len(rows)} cells -> {out}')
    for arch in args.archs:
        for rule in args.rules:
            sel = manifest_rows_for(rows, rule, arch)
            n = len(sel)
            sc = [float(r['det_score']) for r in sel]
            rng = f'{min(sc):.3f}-{max(sc):.3f}' if sc else 'none'
            nb = sum(int(r['in_balanced']) for r in sel)
            print(f'  {rule:4s} {arch:8s} {n:5d} cells '
                  f'({nb:4d} balanced)   det score {rng}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
