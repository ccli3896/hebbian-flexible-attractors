"""Recompute magrate t_crit/censored under a relative (fraction-of-asymptote)
criterion, reusing already-saved binned curves -- no simulation rerun needed
for cells resolvable from data already on disk.

Background
----------
``t_crit``/``censored`` in ``data/magrate/mag_{rule}_{arch}.npz`` were computed
by ``run_mag_sweep.criterion_metrics()`` against a FIXED absolute threshold
(P(optimal) >= 0.9). This recomputes them against a threshold relative to each
cell's own asymptote (``crit_frac * asymptote``, default 0.9) instead, using
only data already on disk:

    curves      (cells, rewards, seeds, bins) -- always the stage-1 (20k-step)
                run, kept for every cell regardless of whether it was later
                rerun at the long (100k-step) budget.
    asymptote   (cells, rewards) -- mean P(optimal) over the final 10% of
                bins. THRESHOLD-INDEPENDENT, and already the best-available
                estimate (the long-run one, for cells rerun at stage 2).

Because the new threshold is <= the old one (0.9 * asymptote <= 0.9, since
asymptote <= 1), nothing that already crossed the old criterion can un-cross.
And any cell whose asymptote was computed from THIS SAME curve necessarily
crosses 0.9 * asymptote somewhere in it too -- a mean's tail always contains at
least one value >= the mean. So only cells rerun at stage 2 (``from_s2``),
whose stored curve is the *shorter* stage-1 run while their asymptote comes
from the longer stage-2 run, can fail to resolve: for those the true crossing
may lie beyond the 20k-step window we have on disk.

Usage
-----
    python -m ssb.analysis.recompute_relative_criterion \\
        --magrate-dir data/magrate --out-dir data/magrate_relcrit \\
        --emit-rerun data/magrate_relcrit/rerun_relcrit.csv

If the raw stage-2 shards (``mag_{rule}_{arch}_s2_c*.npz``, the actual
100k-step curves) are still retrievable -- e.g. copied down from the cluster's
``sweep_results/supp/results`` before it's cleaned up -- pass
``--raw-s2-dir`` and cells that were otherwise unresolved get a real
measurement instead of a rerun: each is matched to its raw shard row by
``cell_idx`` (same join ``merge_supp_stages`` uses) and rescanned against its
own (longer, 100k-step) curve, which -- by the same tail-contains-the-mean
argument as the module docstring -- necessarily crosses 0.9 * asymptote
somewhere in it. Any row that still doesn't resolve after this (a genuine gap
in the retrieved shards) falls through to the emitted rerun manifest.

The emitted manifest is in the same format as
``ssb.analysis.aggregate_supp.emit_rerun_manifest``, so it drops straight into
the existing cluster workflow: copy it to the cluster checkout as
``sweep_results/supp/rerun_relcrit.csv`` and run

    MANIFEST=sweep_results/supp/rerun_relcrit.csv STAGE=s2 TIMESTEPS=100000 \\
        bash slurm/launch_supp.sh mag

then merge with ``ssb.analysis.merge_supp_stages`` as usual -- it overrides
t_crit/censored/asymptote per matched cell_idx regardless of how those were
computed, so it needs no changes for this new criterion.

Note on ``censored``/``resolved``: whenever ``crit_frac <= 1`` (the sensible
range -- a threshold above a cell's own asymptote isn't a meaningful "fraction
of asymptote" criterion) and every cell resolves, both fields are provably
constant: ``censored`` is always False and ``resolved`` always True, by the
same tail-contains-the-mean argument above. In that case they carry no
information and are dropped from the saved arrays rather than stored as dead
weight; ``t_crit`` alone is then always a real measurement. If a run leaves
any cell unresolved, or is run with ``crit_frac > 1``, both fields are kept
since they're no longer constant.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from ssb.analysis.cell_select import (ALPHAS, BETAS, ETAS, TAUS, RULES,
                                      RULES_WITH_TAU, MANIFEST_FIELDS)

CRIT_FRAC = 0.9
ARCHS = ('small', 'big')

S2_KEYS = ('curves', 't_crit', 'censored', 'asymptote', 'attractor',
          'sensitivity', 'cell_idx', 'cell_ids')


def _cell_key(row) -> tuple:
    """The (eta, tau, alpha, beta) integer tuple identifying a cell -- same
    join key ``merge_supp_stages._key`` uses."""
    return tuple(int(x) for x in row)


def load_raw_s2_shards(raw_s2_dir, rule, arch):
    """Concatenate this (rule, arch)'s raw stage-2 shards, if any are present.

    Mirrors ``aggregate_supp.load_stage``, but reads a flat directory of
    already-downloaded shards rather than a full results tree.
    """
    pat = re.compile(rf'^mag_{re.escape(rule)}_{re.escape(arch)}_s2_c(\d+)\.npz$')
    shards = []
    for p in Path(raw_s2_dir).glob(f'mag_{rule}_{arch}_s2_c*.npz'):
        m = pat.match(p.name)
        if m:
            shards.append((int(m.group(1)), p))
    if not shards:
        return None
    shards.sort()

    out, cursor = {}, 0
    for start, path in shards:
        if start != cursor:
            raise ValueError(f"{rule}/{arch}: s2 shard gap -- expected cell "
                             f"{cursor}, found {start} ({path.name})")
        d = np.load(path, allow_pickle=True)
        for k in S2_KEYS:
            out.setdefault(k, []).append(d[k])
        cursor = start + len(d['cell_idx'])
    merged = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    d0 = np.load(shards[0][1], allow_pickle=True)
    merged['bin_width'] = int(d0['bin_width'])
    return merged


def recompute_cell_criterion(curves, asymptote, bin_width, crit_frac=CRIT_FRAC):
    """Relative-criterion t_crit/censored/ever from stored curves + asymptote.

    Returns (t_crit, censored, ever), each shape (cells, rewards). ``t_crit``
    is -1 (not ``bin_width * n_bins``) where the stored curve never crosses --
    that sentinel distinguishes "we don't know" from a real measurement,
    unlike the sweep's own convention of filling in the budget for a
    genuinely-converged-but-below-threshold cell.
    """
    med = np.median(curves, axis=2)                 # (cells, rewards, bins)
    thresh = crit_frac * asymptote[..., None]        # (cells, rewards, 1)
    above = med >= thresh
    ever = above.any(axis=-1)
    first_bin = np.argmax(above, axis=-1)
    t_crit = np.where(ever, first_bin * bin_width, -1).astype(np.int32)
    censored = ~ever
    return t_crit, censored, ever


def process_group(magrate_dir, rule, arch, crit_frac=CRIT_FRAC, raw_s2_dir=None):
    """Recompute one (rule, arch) group. Returns (arrays, unresolved_cells)."""
    d = np.load(Path(magrate_dir) / f"mag_{rule}_{arch}.npz", allow_pickle=True)
    arrays = {k: d[k] for k in d.files}

    curves = arrays["curves"]
    asymptote = arrays["asymptote"]
    bin_width = int(arrays["bin_width"])
    n_cells = curves.shape[0]
    from_s2 = arrays.get("from_s2", np.zeros(n_cells, dtype=bool))

    t_crit_rel, censored_rel, ever = recompute_cell_criterion(
        curves, asymptote, bin_width, crit_frac)

    # Unresolved iff the stored (stage-1) curve never crosses the new
    # threshold AND the cell's asymptote came from a stage-2 curve we don't
    # have. Non-from_s2 cells always resolve (see module docstring).
    unresolved = (~ever) & from_s2[:, None]

    # If the raw stage-2 shards were retrieved, resolve as many of those
    # as possible from the actual long-budget curve instead of leaving them
    # as -1/unresolved.
    n_recovered = 0
    if raw_s2_dir is not None and unresolved.any():
        s2 = load_raw_s2_shards(raw_s2_dir, rule, arch)
        if s2 is not None:
            row_of = {_cell_key(row): j for j, row in enumerate(s2["cell_idx"])}
            unresolved_rows = np.where(unresolved.any(axis=1))[0]
            for i in unresolved_rows:
                key = _cell_key(arrays["cell_idx"][i])
                j = row_of.get(key)
                if j is None:
                    continue  # genuine gap in the retrieved shards
                t_crit_j, censored_j, ever_j = recompute_cell_criterion(
                    s2["curves"][j:j + 1], asymptote[i:i + 1],
                    s2["bin_width"], crit_frac)
                t_crit_rel[i] = t_crit_j[0]
                censored_rel[i] = censored_j[0]
                unresolved[i] = ~ever_j[0]
                n_recovered += int(ever_j[0].sum())

    resolved = ~unresolved

    # Prefer an existing legacy copy so re-running against a directory that
    # was already processed (e.g. because the true source data.magrate/ was
    # lost after the first pass) doesn't clobber the true absolute-0.9
    # lineage with an already-relative t_crit/censored.
    old_t_crit = arrays.get("t_crit_legacy_abs0p9", arrays.get("t_crit"))
    old_censored = arrays.get("censored_legacy_abs0p9", arrays.get("censored"))
    arrays["t_crit_legacy_abs0p9"] = old_t_crit
    arrays["censored_legacy_abs0p9"] = old_censored
    arrays["t_crit"] = t_crit_rel
    arrays["censored"] = censored_rel
    arrays["resolved"] = resolved
    arrays["crit_frac"] = np.float32(crit_frac)

    unresolved_cells = np.where(unresolved.any(axis=1))[0]
    return arrays, unresolved_cells, n_recovered


def strip_constant_censoring(out_dir, rules, archs):
    """Drop censored/resolved from each group's saved npz.

    Call only when the run confirmed crit_frac <= 1 and zero cells are
    unresolved -- in which case both fields are provably constant
    (censored=False, resolved=True everywhere) and carry no information.
    """
    for rule in rules:
        for arch in archs:
            path = Path(out_dir) / f"mag_{rule}_{arch}.npz"
            if not path.exists():
                continue
            d = np.load(path, allow_pickle=True)
            arrays = {k: d[k] for k in d.files if k not in ("censored", "resolved")}
            np.savez(path, **arrays)


def write_rerun_manifest(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out_path


def manifest_rows_for_cells(rule, arch, cell_idx_arr, cell_ids, cell_rows):
    """Build MANIFEST_FIELDS rows for the given cell row indices."""
    has_tau = rule in RULES_WITH_TAU
    rows = []
    for i in cell_rows:
        e, t, a, b = (int(x) for x in cell_idx_arr[i])
        rows.append({
            "cell_id": str(cell_ids[i]) if len(cell_ids) > i else f"{rule}_{arch}_r{i:04d}",
            "rule": rule, "arch": arch, "det_arch": arch,
            "eta_idx": e, "tau_idx": t if has_tau else -1,
            "alpha_idx": a, "beta_idx": b,
            "eta": f"{ETAS[e]:.6g}",
            "tau": f"{TAUS[t]:.6g}" if (has_tau and t >= 0) else "",
            "alpha": f"{ALPHAS[a]:.6g}", "beta": f"{BETAS[b]:.6g}",
            "det_score": "0.000000",
            "selection": "rerun-relcrit-s2",
            "in_balanced": -1,
            "source": f"mag_{rule}_{arch}_relcrit",
        })
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--magrate-dir", default="data/magrate")
    p.add_argument("--out-dir", default="data/magrate_relcrit")
    p.add_argument("--crit-frac", type=float, default=CRIT_FRAC)
    p.add_argument("--rules", nargs="+", default=list(RULES))
    p.add_argument("--archs", nargs="+", default=list(ARCHS))
    p.add_argument("--raw-s2-dir",
                   help="directory of downloaded raw stage-2 shards "
                        "(mag_{rule}_{arch}_s2_c*.npz) to resolve cells from")
    p.add_argument("--emit-rerun", help="write a stage-2 rerun manifest of unresolved cells")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rerun_rows = []
    print(f"{'rule':4} {'arch':6} {'cells':>6} {'unresolved':>10} {'recovered':>10}")
    total_cells = total_unresolved = total_recovered = 0
    for rule in args.rules:
        for arch in args.archs:
            path = Path(args.magrate_dir) / f"mag_{rule}_{arch}.npz"
            if not path.exists():
                continue
            arrays, unresolved_cells, n_recovered = process_group(
                args.magrate_dir, rule, arch, args.crit_frac, args.raw_s2_dir)
            n_cells = arrays["cell_idx"].shape[0]
            dest = out_dir / f"mag_{rule}_{arch}.npz"
            np.savez(dest, **arrays)
            total_cells += n_cells
            total_unresolved += len(unresolved_cells)
            total_recovered += n_recovered
            print(f"{rule:4} {arch:6} {n_cells:6d} {len(unresolved_cells):10d} {n_recovered:10d}   -> {dest.name}")

            if args.emit_rerun:
                rerun_rows.extend(manifest_rows_for_cells(
                    rule, arch, arrays["cell_idx"], arrays.get("cell_ids", np.array([])),
                    unresolved_cells))

    pct = 100 * total_unresolved / max(total_cells, 1)
    print(f"\nTOTAL: {total_cells} cells, {total_unresolved} unresolved ({pct:.1f}%), "
         f"{total_recovered} (cell, reward) entries recovered from raw s2 shards")
    print(f"recomputed arrays written -> {out_dir}")

    if args.crit_frac <= 1.0 and total_unresolved == 0:
        strip_constant_censoring(out_dir, args.rules, args.archs)
        print("\ncrit_frac <= 1 and every cell resolved: censored/resolved are "
             "provably constant (censored=False, resolved=True) for every "
             "entry -- dropped from the saved arrays (t_crit alone is always "
             "a real measurement now).")

    if args.emit_rerun:
        dest = write_rerun_manifest(rerun_rows, args.emit_rerun)
        print(f"\nrerun manifest: {len(rerun_rows)} cells -> {dest}")
        print("Copy this to the cluster checkout as "
             "sweep_results/supp/rerun_relcrit.csv, then run:\n"
             "  MANIFEST=sweep_results/supp/rerun_relcrit.csv STAGE=s2 "
             "TIMESTEPS=100000 bash slurm/launch_supp.sh mag\n"
             "then merge with ssb.analysis.merge_supp_stages as usual.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
