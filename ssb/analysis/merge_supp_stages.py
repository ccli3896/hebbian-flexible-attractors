"""Fold stage-2 (long-budget) magnitude reruns over stage-1 aggregates.

The magnitude sweep runs stage 1 at a short budget (20k steps). A cell that never
reached criterion is *censored*: its time-to-criterion is a lower bound, not a
measurement, and must not enter the rate-sensitivity fit. The stage-2 rerun
(``STAGE=s2 TIMESTEPS=100000``) re-measures the *time-limited* subset at the long
budget. This module folds those long-budget measurements back over the stage-1
arrays, cell for cell, and writes one final array per (rule, arch).

Why a separate step: ``aggregate_supp.load_stage`` loads a single stage, and its
docstring's ``prefer`` merge was never wired into the CLI, so ``--stage s1``
silently ignores s2. This closes that gap.

**How cells are matched.** The stage-2 job runs from ``rerun.csv``, which
re-indexes cells 0..N, so the s2 shards do *not* share the s1 cell axis. Each cell
is instead matched on its (eta, tau, alpha, beta) grid index -- the ``cell_idx``
tuple -- which both stages carry unchanged. Every s2 cell must find exactly one s1
cell or the merge aborts (a mismatch means the rerun manifest is stale).

**What is overridden.** Only per-(cell, reward) arrays whose shape is
budget-independent: ``t_crit``, ``censored``, ``asymptote``, ``attractor``,
``sensitivity``. The binned learning ``curves`` have more bins at 100k than at
20k, so they cannot be substituted into a rectangular array; s1 curves are kept
(they only feed the stage-1 rising check and display). The fit consumes
``t_crit`` and ``censored``, which *are* overridden -- so a cell that reached
criterion only at 100k now carries its true time, and a cell still flat at 100k
stays censored (genuinely capability-limited).

A boolean ``from_s2`` mask (length = n cells) marks which cells were replaced, and
``timesteps_s2`` records the long budget for provenance.

Usage:
    python -m ssb.analysis.merge_supp_stages \\
        --results-dir sweep_results/supp/results \\
        --out-dir     sweep_results/supp/aggregated

``cl`` has no censoring rerun, so it is passed through from s1 unchanged, giving a
complete aggregated set (mag merged + cl) in one command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ssb.analysis.aggregate_supp import CELL_AXIS_KEYS, load_stage
from ssb.analysis.cell_select import RULES

IDENTITY_KEYS = {"cell_idx", "cell_ids"}


def _key(cell_idx_row) -> tuple:
    """The (eta, tau, alpha, beta) integer tuple that identifies a cell."""
    return tuple(int(x) for x in cell_idx_row)


def merge_group(results_dir, rule, arch):
    """Merge s2 over s1 for one (mag, rule, arch). Returns (arrays, stats).

    If no s2 shards exist for the group, the s1 arrays are returned unchanged
    (with an all-False ``from_s2``).
    """
    s1 = load_stage(results_dir, "mag", rule, arch, "s1")
    n_cells = len(s1["cell_idx"])
    from_s2 = np.zeros(n_cells, dtype=bool)

    try:
        s2 = load_stage(results_dir, "mag", rule, arch, "s2")
    except FileNotFoundError:
        s1["from_s2"] = from_s2
        return s1, {"rerun": 0, "overridden": 0, "still_censored": 0,
                    "override_keys": [], "skipped_keys": []}

    # Which cell-axis metrics can be substituted (shape must match past axis 0).
    override_keys, skipped_keys = [], []
    for k in CELL_AXIS_KEYS["mag"]:
        if k in IDENTITY_KEYS or k not in s1 or k not in s2:
            continue
        if s1[k].shape[1:] == s2[k].shape[1:]:
            override_keys.append(k)
        else:
            skipped_keys.append(k)  # curves: differing bin count at 100k

    row_of = {_key(s1["cell_idx"][i]): i for i in range(n_cells)}
    if len(row_of) != n_cells:
        raise ValueError(f"{rule}/{arch}: s1 has duplicate cell_idx tuples")

    still_censored = 0
    for j in range(len(s2["cell_idx"])):
        key = _key(s2["cell_idx"][j])
        if key not in row_of:
            raise KeyError(
                f"{rule}/{arch}: stage-2 cell {key} has no stage-1 match -- "
                f"rerun.csv is stale relative to these results; re-emit it.")
        i = row_of[key]
        for k in override_keys:
            s1[k][i] = s2[k][j]
        from_s2[i] = True
        if "censored" in s2 and bool(np.all(s2["censored"][j])):
            still_censored += 1

    s1["from_s2"] = from_s2
    if "timesteps" in s2:
        s1["timesteps_s2"] = s2["timesteps"]
    return s1, {
        "rerun": len(s2["cell_idx"]),
        "overridden": int(from_s2.sum()),
        "still_censored": still_censored,
        "override_keys": override_keys,
        "skipped_keys": skipped_keys,
    }


def passthrough_group(results_dir, kind, rule, arch):
    """Aggregate a single stage (used for cl, which has no rerun)."""
    return load_stage(results_dir, kind, rule, arch, "s1")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out-dir", required=True,
                   help="one {kind}_{rule}_{arch}.npz per group written here")
    p.add_argument("--rules", nargs="+", default=list(RULES))
    p.add_argument("--archs", nargs="+", default=["small", "big"])
    p.add_argument("--kinds", nargs="+", default=["mag", "cl"],
                   choices=["mag", "cl"])
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'kind':4} {'rule':4} {'arch':6} {'cells':>6} {'rerun':>6} "
          f"{'merged':>7} {'still_cens':>10}")
    total_written = 0
    for kind in args.kinds:
        for arch in args.archs:
            for rule in args.rules:
                try:
                    if kind == "mag":
                        arrays, st = merge_group(args.results_dir, rule, arch)
                    else:
                        arrays = passthrough_group(args.results_dir, kind, rule, arch)
                        st = {"rerun": 0, "overridden": 0, "still_censored": 0}
                except FileNotFoundError:
                    continue  # this (kind, rule, arch) was not run
                dest = out_dir / f"{kind}_{rule}_{arch}.npz"
                np.savez(dest, **arrays)
                total_written += 1
                print(f"{kind:4} {rule:4} {arch:6} {len(arrays['cell_idx']):6d} "
                      f"{st['rerun']:6d} {st['overridden']:7d} "
                      f"{st['still_censored']:10d}")

    print(f"\nwrote {total_written} aggregated arrays -> {out_dir}")
    print("from_s2 marks the cells replaced by the 100k rerun; timesteps_s2 "
          "records the long budget.\ncurves are NOT merged (bin count differs "
          "at 100k) -- t_crit/censored ARE, which is what the rate fit uses.")


if __name__ == "__main__":
    main()
