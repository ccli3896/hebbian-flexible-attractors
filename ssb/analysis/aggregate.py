"""Aggregate per-slice sweep results into single tensors.

The SLURM array jobs write one .npy per (eta_idx, tau_idx) slice with
filenames encoding the indices:

    results/{env}_{rule}_{size}.npy                            # full sweep, single file (small no-tau)
    results/{env}_{rule}_{size}_tau{T}.npy                     # small tau rule, sliced over tau
    results/{env}_{rule}_{size}_eta{E}.npy                     # big no-tau, sliced over eta
    results/{env}_{rule}_{size}_eta{E}_tau{T}.npy              # big tau rule, sliced over (eta, tau)

This module reads those slice files and reconstructs the full sweep
tensor of shape (n_etas, [n_taus,] n_alphas, n_betas, [extra...]).
The "extra..." varies per env: () for bandit_bait and selectivity,
(2,) for slowfeature and bandit_unbait ([pm_index, choice_sd]),
(3,) for matchingpennies.

For 'det' env: raw action arrays are in data/DetSweep/raw/ as .npz
files with the same indexing scheme. See aggregate_det() for that case.

Usage:
    python -m ssb.analysis.aggregate \
        --results-dir sweep_results/results \
        --out-dir     sweep_results/aggregated

Or programmatically:
    from ssb.analysis.aggregate import aggregate_sweep
    full = aggregate_sweep("sweep_results/results", env="bandit_bait", rule="COV", size="big")
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

try:
    # Preferred: derive the recognised architecture tokens straight from the
    # registry so this parser never drifts from the arch names the launcher
    # actually submits. architectures.py has no heavy (jax) imports, so this
    # is cheap.
    from ssb.utils.architectures import list_architectures as _list_archs
    _ARCH_NAMES = list(_list_archs())
except Exception:  # pragma: no cover - fallback if registry moves
    _ARCH_NAMES = ["small", "big", "N2_L2", "N10_L2", "N50_L2",
                   "N200_L2", "N50_L3"]


# Default sweep grid sizes. Must match ssb/scripts/run_sweep.py.
N_ETAS = 6
N_TAUS = 4
RULES_WITH_TAU = {"COV", "BCM", "LC", "HZ"}


# Filename pattern: {env}_{rule}_{size}[_eta{E}][_tau{T}][_a{A}][_w{W}].npy
# Env names can contain underscores (e.g. bandit_bait) AND so can the arch
# token (e.g. N50_L3), so we anchor on the KNOWN set of architecture names
# from the registry. Longest-first alternation so 'N50_L3' is preferred over
# any shorter prefix. Previously this only matched (small|big), which
# silently dropped every N*_L* slice from the S3 architecture sweep.
# The optional _a{A}/_w{W} tokens are the alpha/beta grid-shard indices;
# absent when the alpha/beta grid was not sharded (the common case), so old
# eta/tau-only filenames still parse and aggregate exactly as before.
_ARCH_ALT = "|".join(
    re.escape(a) for a in sorted(_ARCH_NAMES, key=len, reverse=True)
)
_FNAME_RE = re.compile(
    r"^(?P<env>.+)_(?P<rule>[A-Za-z]+)_(?P<size>" + _ARCH_ALT + r")"
    r"(?:_eta(?P<eta>\d+))?"
    r"(?:_tau(?P<tau>\d+))?"
    r"(?:_a(?P<a>\d+))?"
    r"(?:_w(?P<w>\d+))?"
    r"\.npy$"
)


def _parse_filename(path: Path) -> dict | None:
    """Parse a results filename into its components.

    Returns dict with keys env, rule, size, eta, tau, a, w (each an int or
    None if that axis was not sliced); or None if the filename doesn't match.
    """
    m = _FNAME_RE.match(path.name)
    if m is None:
        return None
    return {
        "env": m.group("env"),
        "rule": m.group("rule"),
        "size": m.group("size"),
        "eta": int(m.group("eta")) if m.group("eta") is not None else None,
        "tau": int(m.group("tau")) if m.group("tau") is not None else None,
        "a": int(m.group("a")) if m.group("a") is not None else None,
        "w": int(m.group("w")) if m.group("w") is not None else None,
    }


def _find_slices(
    results_dir: Path,
    env: str,
    rule: str,
    size: str,
) -> list[tuple[dict, Path]]:
    """Find all slice files for one (env, rule, size) combination."""
    out = []
    for p in sorted(results_dir.iterdir()):
        if not p.is_file() or p.suffix != ".npy":
            continue
        parsed = _parse_filename(p)
        if parsed is None:
            continue
        if (parsed["env"], parsed["rule"], parsed["size"]) != (env, rule, size):
            continue
        out.append((parsed, p))
    return out


def aggregate_sweep(
    results_dir: str | Path,
    env: str,
    rule: str,
    size: str,
    n_etas: int = N_ETAS,
    n_taus: int = N_TAUS,
) -> np.ndarray:
    """Reconstruct the full sweep tensor from per-slice files.

    Returns an array of shape (n_etas, [n_taus,] n_alphas, n_betas, [extra]).
    The tau axis is present iff the rule is in RULES_WITH_TAU.

    Raises ValueError if slices are missing for the expected combinations.
    """
    results_dir = Path(results_dir)
    slices = _find_slices(results_dir, env, rule, size)
    if not slices:
        raise ValueError(
            f"no slice files found for ({env}, {rule}, {size}) in {results_dir}"
        )

    rule_uses_tau = rule in RULES_WITH_TAU
    has_tau = rule_uses_tau  # whether the per-slice file has a tau axis

    # Probe one slice to learn per-slice shape/dtype. Per-slice arrays have
    # shape (eta, [tau,] alpha, beta, *extra); any axis that was sliced down
    # by an array job has length 1 there, and the index it belongs at is in
    # the filename. The alpha/beta axes sit right after eta(/tau).
    sample_arr = np.load(slices[0][1])
    sample_shape = sample_arr.shape
    sample_dtype = sample_arr.dtype

    ax_alpha = 2 if has_tau else 1   # position of the alpha axis in a slice
    ax_beta = ax_alpha + 1
    extra_shape = sample_shape[ax_beta + 1:]  # trailing per-env dims (e.g. (2,))

    # Full sizes along each sliceable axis. eta/tau come from params. For
    # alpha/beta: if any filename carries that shard index the axis was
    # sharded (so the per-slice length is 1 and the full size is max idx + 1);
    # otherwise the slice already holds the full axis, so read it off the
    # sample. This makes an un-sharded (eta/tau-only) run reconstruct exactly
    # as before.
    any_a = any(p["a"] is not None for p, _ in slices)
    any_w = any(p["w"] is not None for p, _ in slices)
    n_alpha = (max(p["a"] for p, _ in slices if p["a"] is not None) + 1) if any_a \
        else sample_shape[ax_alpha]
    n_beta = (max(p["w"] for p, _ in slices if p["w"] is not None) + 1) if any_w \
        else sample_shape[ax_beta]

    out_shape = (n_etas,) + ((n_taus,) if has_tau else ()) + (n_alpha, n_beta) + extra_shape
    out = np.zeros(out_shape, dtype=sample_dtype)
    # Track filled (eta, tau, alpha, beta) cells for a completeness check.
    filled = np.zeros((n_etas, n_taus if has_tau else 1, n_alpha, n_beta), dtype=bool)

    def _axis_slice(idx):
        """A length-1 slice at idx if that axis was sharded, else the full axis."""
        return slice(idx, idx + 1) if idx is not None else slice(None)

    for parsed, path in slices:
        arr = np.load(path)

        e_sl = _axis_slice(parsed["eta"])
        a_sl = _axis_slice(parsed["a"])
        w_sl = _axis_slice(parsed["w"])
        t_sl = _axis_slice(parsed["tau"]) if has_tau else None

        # Basic (slice) indexing: each leading axis is either a length-1
        # window (sharded) or the whole axis (unsharded), and arr's shape
        # matches exactly, so a single assignment places the whole tile —
        # no advanced-indexing broadcasting subtleties.
        if has_tau:
            out[e_sl, t_sl, a_sl, w_sl] = arr
            filled[e_sl, t_sl, a_sl, w_sl] = True
        else:
            out[e_sl, a_sl, w_sl] = arr
            filled[e_sl, 0, a_sl, w_sl] = True

    missing = np.argwhere(~filled)
    if missing.size > 0:
        # Report (eta, tau, alpha, beta) tuples (tau dropped for non-tau rules).
        miss = missing.tolist()
        pretty = [tuple(m if has_tau else m[:1] + m[2:]) for m in miss]
        raise ValueError(
            f"missing slices for ({env}, {rule}, {size}): "
            f"(eta{',tau' if has_tau else ''},alpha,beta) = {pretty[:10]}"
            + (" ..." if len(pretty) > 10 else "")
        )

    return out


def aggregate_all(
    results_dir: str | Path,
    out_dir: str | Path,
    n_etas: int = N_ETAS,
    n_taus: int = N_TAUS,
) -> dict[str, Path]:
    """Aggregate every (env, rule, size) tuple found in the results dir.

    Writes one combined .npy per tuple into ``out_dir``. Returns a mapping
    from "{env}_{rule}_{size}" -> output path.

    Tuples with incomplete slice coverage are skipped with a warning.
    """
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover all (env, rule, size) tuples present
    tuples = set()
    for p in results_dir.iterdir():
        parsed = _parse_filename(p) if p.is_file() else None
        if parsed is not None:
            tuples.add((parsed["env"], parsed["rule"], parsed["size"]))

    written = {}
    for env, rule, size in sorted(tuples):
        try:
            agg = aggregate_sweep(results_dir, env, rule, size, n_etas, n_taus)
        except ValueError as e:
            print(f"SKIP {env}/{rule}/{size}: {e}")
            continue
        out_path = out_dir / f"{env}_{rule}_{size}.npy"
        np.save(out_path, agg)
        print(f"wrote {out_path} shape={agg.shape}")
        written[f"{env}_{rule}_{size}"] = out_path

    return written


def aggregate_det(
    raw_dir: str | Path,
    out_dir: str | Path,
    env: str = "det",
    n_etas: int = N_ETAS,
    n_taus: int = N_TAUS,
) -> dict[str, Path]:
    """Aggregate the 'det' env's raw action files efficiently into compressed booleans.

    For 'det', each slice writes a .npz with key 'x' containing actions
    shaped (T/2, seeds, n_alphas, n_betas). Filenames look like:

        eta{E}_{stem}.npz                       (non-tau)
        eta{E}_tau{T}_{stem}.npz                (tau rule)
        eta{E}[_tau{T}]_a{A}[_w{W}]_{stem}.npz  (alpha/beta grid-sharded)

    Output: one compressed .npz per (rule, size), containing key 'x' with
    shape (n_etas, [n_taus,] T/2, seeds, n_alphas, n_betas) of type bool.

    The alpha/beta axes live INSIDE the per-slice tail (positions -2/-1), so
    when a run is grid-sharded those axes are length 1 in each slice and get
    placed by the _a{A}/_w{W} filename index. Un-sharded runs (no _a/_w) keep
    the full alpha/beta axes and reconstruct exactly as before.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group files by stem (which encodes env_rule_size from the run_sweep --out arg)
    det_re = re.compile(
        r"^eta(?P<eta>\d+)(?:_tau(?P<tau>\d+))?"
        r"(?:_a(?P<a>\d+))?(?:_w(?P<w>\d+))?_(?P<stem>.+)\.npz$"
    )

    by_stem: dict[str, list] = {}
    for p in sorted(raw_dir.iterdir()):
        if not p.is_file() or p.suffix != ".npz":
            continue
        m = det_re.match(p.name)
        if m is None:
            continue
        stem = m.group("stem")
        by_stem.setdefault(stem, []).append((m, p))

    def _sl(v):
        return slice(int(v), int(v) + 1) if v is not None else slice(None)

    written = {}
    for stem, entries in by_stem.items():
        # Decide whether this stem has tau / alpha / beta slicing
        has_tau = any(m.group("tau") is not None for m, _ in entries)
        any_a = any(m.group("a") is not None for m, _ in entries)
        any_w = any(m.group("w") is not None for m, _ in entries)

        # Probe a slice for the per-slice tail (T/2, seeds, alpha, beta).
        with np.load(entries[0][1]) as sample_file:
            tail_shape = sample_file["x"].shape

        # Full alpha/beta sizes: from shard indices if sharded, else the
        # sample's own (already-full) tail axes (-2 = alpha, -1 = beta).
        n_alpha = (max(int(m.group("a")) for m, _ in entries if m.group("a") is not None) + 1) \
            if any_a else tail_shape[-2]
        n_beta = (max(int(m.group("w")) for m, _ in entries if m.group("w") is not None) + 1) \
            if any_w else tail_shape[-1]
        full_tail = tail_shape[:-2] + (n_alpha, n_beta)

        if has_tau:
            out_shape = (n_etas, n_taus) + full_tail
        else:
            out_shape = (n_etas,) + full_tail

        out = np.zeros(out_shape, dtype=bool)
        filled = np.zeros((n_etas, n_taus if has_tau else 1, n_alpha, n_beta), dtype=bool)

        for m, p in entries:
            e = int(m.group("eta"))
            with np.load(p) as f:
                arr = f["x"]

            a_sl = _sl(m.group("a"))
            w_sl = _sl(m.group("w"))
            # Full T/2 and seeds axes (slice(None)); shard only alpha/beta.
            lead_full = (slice(None),) * (arr.ndim - 2)
            if has_tau:
                t = int(m.group("tau"))
                out[(e, t) + lead_full + (a_sl, w_sl)] = arr
                filled[e, t, a_sl, w_sl] = True
            else:
                out[(e,) + lead_full + (a_sl, w_sl)] = arr
                filled[e, 0, a_sl, w_sl] = True

        if not filled.all():
            print(f"SKIP det/{stem}: missing slices {np.argwhere(~filled).tolist()}")
            continue

        # Optimization 3: Save as compressed .npz to crush disk space footprint
        out_path = out_dir / f"{stem}.npz"
        np.savez_compressed(out_path, x=out)
        print(f"wrote {out_path} shape={out.shape} (dtype=bool)")
        written[stem] = out_path

    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing per-slice .npy files (e.g. sweep_results/results)")
    parser.add_argument("--out-dir", required=True,
                        help="Where to write aggregated tensors")
    parser.add_argument("--det-raw-dir", default=None,
                        help="(Optional) data/DetSweep/raw — aggregate det-env raw files too")
    parser.add_argument("--n-etas", type=int, default=N_ETAS)
    parser.add_argument("--n-taus", type=int, default=N_TAUS)
    args = parser.parse_args()

    aggregate_all(args.results_dir, args.out_dir,
                  n_etas=args.n_etas, n_taus=args.n_taus)

    if args.det_raw_dir:
        aggregate_det(args.det_raw_dir, args.out_dir,
                      n_etas=args.n_etas, n_taus=args.n_taus)


if __name__ == "__main__":
    main()
