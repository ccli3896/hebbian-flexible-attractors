"""Reward-magnitude sweep: does reward still scale learning across parameters?

The main text establishes reward-magnitude scaling at ONE hyperparameter cell
per rule (each rule's det-sweep argmax). This sweep re-runs that measurement at
many cells, so the supplement can show the result is a property of the rule and
not of a hand-picked operating point.

Two ways to choose which cells to run:

  manifest mode  (--manifest)   the reward-seeking cells chosen by
                                ``ssb.analysis.cell_select`` -- used for the
                                small/big sweeps, where running the full grid
                                would be mostly wasted on non-seeking cells.
  grid mode      (--full-grid)  every cell of the (eta[, tau], alpha, beta)
                                grid -- used for the architecture sweep.

Both modes shard by cell (``--cell-start`` / ``--cell-count``) so one SLURM
array task handles a bounded number of cells. Sharding is a pure sub-selection
of an ordered cell list, so shards recombine into exactly the tensor an
unsharded run would have produced.

Batching
--------
Every (cell x reward x seed) combination is one lane of a single vmapped
`jax.lax.scan` over time -- the same structure as ``run_sweep.py``, which is
why cost scales with ``lanes x timesteps`` rather than with the number of
separate simulations. Lane order is

    lane = (cell_i * n_rewards + reward_j) * n_seeds + seed_k

so results reshape cleanly to ``(n_cells, n_rewards, n_seeds, ...)``.

Outputs (one .npz per shard)
----------------------------
    curves      (cells, rewards, seeds, bins) float32  binned P(optimal)
    t_crit      (cells, rewards) int32    steps to criterion (median across seeds)
    censored    (cells, rewards) bool     criterion never reached within budget
    asymptote   (cells, rewards) float32  mean P(optimal) over the final tenth
    attractor   (cells, rewards, n_metrics) float32   see analysis.attractor
    sensitivity (cells, 3) float32        d_slope_dR, intercept, r2 per cell
    cell_*      index columns identifying each cell (traceable to the manifest)

``censored`` is what drives the two-stage rerun: stage 1 runs at a short budget,
and only the (cell, reward) entries that never reached criterion are re-run at
the long budget. See ``ssb.analysis.aggregate_supp``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
from jax import vmap

from ssb.models import ssb, rules
from ssb.envs.bandits import bandit_det
from ssb.analysis import attractor as attr
from ssb.analysis.cell_select import (ALPHAS, BETAS, ETAS, TAUS, RULES_WITH_TAU,
                                      read_manifest, manifest_rows_for)
from ssb.utils.architectures import get_sizes
from ssb.utils.batch_inits import make_default_hps, init_params_batch

# Reward levels swept. Five levels spanning the range used in the main figure's
# learning-curve panel (F2), enough to fit the slope-vs-reward line.
DEFAULT_REWARDS = [0.2, 0.4, 0.6, 0.8, 1.0]

# Criterion for "learned", matching the notebooks: P(optimal) >= 0.9 measured on
# a 100-step boxcar of each seed, then the median across seeds.
CRIT_THRESHOLD = 0.9
CRIT_WINDOW = 100

CLIP = 3.0


# ---------------------------------------------------------------------------
# Cell enumeration
# ---------------------------------------------------------------------------

def grid_cells(rule: str, n_alpha=len(ALPHAS), n_beta=len(BETAS)) -> list:
    """Every cell of the full grid for one rule, in a fixed, reproducible order.

    Order is (eta, tau, alpha, beta) with tau omitted (-1) for non-tau rules.
    Used by the architecture sweep, which scans all parameter combinations
    rather than only the reward-seeking ones.
    """
    has_tau = rule in RULES_WITH_TAU
    taus = range(len(TAUS)) if has_tau else [-1]
    return [(e, t, a, b)
            for e in range(len(ETAS))
            for t in taus
            for a in range(n_alpha)
            for b in range(n_beta)]


def manifest_cells(manifest_path, rule: str, arch: str) -> tuple:
    """Cells for one (rule, arch) from a manifest, plus their cell_ids."""
    rows = manifest_rows_for(read_manifest(manifest_path), rule, arch)
    cells = [(r['eta_idx'], r['tau_idx'], r['alpha_idx'], r['beta_idx']) for r in rows]
    ids = [r['cell_id'] for r in rows]
    return cells, ids


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _build_hps_batch(rule, cells, rewards, n_seeds, arch, base_seed):
    """Per-lane hyperparameters. Lane = ((cell, reward), seed), C-order.

    Unlike ``expand_hps_batch``, cells are *paired* (alpha, beta, eta, tau)
    tuples rather than a cross product, so the arrays are built directly.
    """
    n_cells, n_rw = len(cells), len(rewards)
    n_lanes = n_cells * n_rw * n_seeds
    has_tau = rule in RULES_WITH_TAU

    base = make_default_hps(base_seed)
    sizes = jnp.array(get_sizes(arch, 1, 2))       # decision env: 1 in, 2 motor

    # Per-cell parameter values, broadcast out to lanes.
    a = np.array([ALPHAS[c[2]] for c in cells], dtype=np.float32)
    b = np.array([BETAS[c[3]] for c in cells], dtype=np.float32)
    e = np.array([ETAS[c[0]] for c in cells], dtype=np.float32)
    t = np.array([TAUS[c[1]] if has_tau and c[1] >= 0 else float(base['tau'][0])
                  for c in cells], dtype=np.float32)

    rep = n_rw * n_seeds                            # lanes per cell
    lane_alpha = np.repeat(a, rep)
    lane_beta = np.repeat(b, rep)
    lane_eta = np.repeat(e, rep)
    lane_tau = np.repeat(t, rep)
    # Reward varies within a cell, seed varies fastest.
    lane_reward = np.tile(np.repeat(np.asarray(rewards, dtype=np.float32), n_seeds), n_cells)
    lane_seed = np.tile(np.arange(n_seeds, dtype=np.int32), n_cells * n_rw) + base_seed

    hb = {
        'sizes': jnp.tile(sizes[None], (n_lanes,) + (1,) * sizes.ndim),
        'init_scale': jnp.full((n_lanes, 1), float(base['init_scale'][0])),
        'randinit': jnp.full((n_lanes, 1), float(base['randinit'][0])),
        'posinit': jnp.full((n_lanes, 1), float(base['posinit'][0])),
        'seed': jnp.asarray(lane_seed).reshape(-1, 1),
        'alpha': jnp.asarray(lane_alpha).reshape(-1, 1),
        'beta': jnp.asarray(lane_beta).reshape(-1, 1),
        'eta_a': jnp.asarray(lane_eta).reshape(-1, 1),
        'cap': jnp.full((n_lanes, 1), float(base['cap'][0])),
        'rho': jnp.full((n_lanes, 1), float(base['rho'][0])),
        'tau': jnp.asarray(lane_tau).reshape(-1, 1),
    }
    return hb, np.asarray(lane_reward)


def run_actions(rule, cells, rewards, timesteps, n_seeds, arch, base_seed=235,
                bin_width=CRIT_WINDOW):
    """Run every (cell, reward, seed) lane and return binned P(optimal).

    Returns ``(n_cells, n_rewards, n_seeds, n_bins)`` float32. Binning happens
    inside the scan-consuming loop rather than after it: the full action trace
    is ``timesteps x lanes`` booleans, which at production sizes is tens of GB,
    while the binned curve is all any downstream metric actually uses.
    """
    n_cells, n_rw = len(cells), len(rewards)
    hb, lane_reward = _build_hps_batch(rule, cells, rewards, n_seeds, arch, base_seed)
    n_lanes = len(lane_reward)

    uw, us = rules.rule2fns(rule)
    UA = vmap(ssb.update_acts); UW = vmap(uw); US = vmap(us)
    NO = vmap(ssb.act_noise);   EX = vmap(ssb.expand_stimulus)
    env = vmap(bandit_det, in_axes=(0, 0, 0, 0))

    acts, states, weights, tern, key = init_params_batch(hb)

    # bandit_det's reward vector is [reward_for_action_0, reward_for_action_1];
    # action 1 is the optimal arm, so lane reward goes in slot 1.
    rewards_b = jnp.stack([jnp.zeros(n_lanes), jnp.asarray(lane_reward)], axis=1)
    dummy = jnp.zeros((n_lanes, 1))

    @jax.jit
    def step(carry, _):
        acts, states, weights, key = carry
        rb, action, _, key = env(dummy, acts[:, -2:], rewards_b, key)
        stim = EX(rb, acts)
        acts = UA(stim, acts, states, weights, tern, hb)
        acts, key = NO(acts, key, hb)
        acts = jnp.clip(acts, 0., CLIP)
        weights = UW(stim, acts, states, weights, tern, hb)
        states = US(acts, states, hb)
        return (acts, states, weights, key), action

    # Scan in bin-sized chunks and reduce each chunk to its mean, so peak memory
    # is one bin of actions rather than the whole trace.
    n_bins = max(1, timesteps // bin_width)
    out = np.zeros((n_bins, n_lanes), dtype=np.float32)
    carry = (acts, states, weights, key)
    for i in range(n_bins):
        carry, actions = jax.lax.scan(step, carry, None, length=bin_width)
        out[i] = np.asarray(jnp.mean(actions == 1, axis=0), dtype=np.float32)

    # (bins, lanes) -> (cells, rewards, seeds, bins)
    return np.transpose(out.reshape(n_bins, n_cells, n_rw, n_seeds), (1, 2, 3, 0))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def criterion_metrics(curves, timesteps, bin_width=CRIT_WINDOW,
                      threshold=CRIT_THRESHOLD):
    """Steps-to-criterion, censoring flag and asymptote from binned curves.

    ``curves`` is (cells, rewards, seeds, bins). The criterion is applied to the
    median across seeds, matching ``get_trials_to_criterion`` in the notebooks.
    A (cell, reward) whose median never crosses is marked censored and given
    ``timesteps`` -- the value is a lower bound, not a measurement, and callers
    must not fit against censored entries (see the notebooks' `learnable` mask).
    """
    med = np.median(curves, axis=2)                     # (cells, rewards, bins)
    above = med >= threshold
    ever = above.any(axis=-1)
    first_bin = np.argmax(above, axis=-1)
    t_crit = np.where(ever, first_bin * bin_width, timesteps).astype(np.int32)
    censored = ~ever
    # Final TENTH, matching get_trials_to_criterion in the Fig. 2d notebook and
    # the Methods definition of A(s). The supplementary sweep must measure the
    # asymptote over the same window as the main figure, or the two are not
    # comparable.
    tail = max(1, med.shape[-1] // 10)
    asymptote = med[..., -tail:].mean(axis=-1).astype(np.float32)
    return t_crit, censored, asymptote


def attractor_for_cells(rule, cells, rewards, arch, n_seeds, burn, sample, seed):
    """Attractor probe at every (cell, reward), plus the per-cell slope fit."""
    n_metrics = len(attr.METRIC_KEYS)
    metrics = np.full((len(cells), len(rewards), n_metrics), np.nan, dtype=np.float32)
    sens = np.full((len(cells), 3), np.nan, dtype=np.float32)
    has_tau = rule in RULES_WITH_TAU
    for i, (e, t, a, b) in enumerate(cells):
        slopes = []
        for j, r in enumerate(rewards):
            m = attr.probe_cell(
                rule, float(ALPHAS[a]), float(BETAS[b]), float(ETAS[e]), float(r),
                arch=arch, tau=float(TAUS[t]) if (has_tau and t >= 0) else None,
                n_seeds=n_seeds, burn=burn, sample=sample, seed=seed)
            metrics[i, j] = attr.metrics_to_vector(m)
            slopes.append(m['slope'])
        fit = attr.slope_reward_sensitivity(rewards, slopes)
        sens[i] = [fit['d_slope_dR'], fit['intercept'], fit['r2']]
    return metrics, sens


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_mag_sweep(config, cells, cell_ids, out_path):
    """Run the sweep for one shard of cells and write a .npz."""
    rule = config['rule']
    arch = config['arch']
    timesteps = int(config['timesteps'])
    rewards = list(config.get('rewards', DEFAULT_REWARDS))
    n_seeds = int(config.get('n_seeds', 20))
    base_seed = int(config.get('seed', 235))

    t0 = time.time()
    print(f"mag sweep: rule={rule} arch={arch} cells={len(cells)} "
          f"rewards={rewards} seeds={n_seeds} timesteps={timesteps}", flush=True)

    curves = run_actions(rule, cells, rewards, timesteps, n_seeds, arch,
                         base_seed=base_seed)
    t_crit, censored, asymptote = criterion_metrics(curves, timesteps)
    print(f"  behaviour done ({time.time() - t0:.1f}s); "
          f"censored {int(censored.sum())}/{censored.size}", flush=True)

    if config.get('attractor', True):
        a_metrics, a_sens = attractor_for_cells(
            rule, cells, rewards, arch,
            n_seeds=int(config.get('attractor_seeds', 8)),
            burn=int(config.get('attractor_burn', 3000)),
            sample=int(config.get('attractor_sample', 1500)),
            seed=base_seed)
        print(f"  attractor done ({time.time() - t0:.1f}s)", flush=True)
    else:
        a_metrics = np.zeros((len(cells), len(rewards), 0), dtype=np.float32)
        a_sens = np.zeros((len(cells), 3), dtype=np.float32)

    cell_arr = np.array(cells, dtype=np.int32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically: a job killed at the wall must leave NO output, so the
    # launcher's RESUME pass correctly sees the slice as missing and re-runs it.
    # (savez appends '.npz' to a *path* that lacks it, which would defeat the
    # rename; writing through an open handle keeps the temp name exact.)
    tmp = out_path.with_suffix(out_path.suffix + '.tmp')
    with open(tmp, 'wb') as fh:
        np.savez_compressed(
            fh,
            curves=curves.astype(np.float32),
            t_crit=t_crit,
            censored=censored,
            asymptote=asymptote,
            attractor=a_metrics,
            sensitivity=a_sens,
            cell_idx=cell_arr,
            cell_ids=np.array(cell_ids, dtype=object) if cell_ids else np.array([]),
            rewards=np.asarray(rewards, dtype=np.float32),
            metric_keys=np.array(attr.METRIC_KEYS, dtype=object),
            rule=rule, arch=arch, timesteps=timesteps, n_seeds=n_seeds,
            bin_width=CRIT_WINDOW, threshold=CRIT_THRESHOLD,
        )
    tmp.replace(out_path)
    print(f"wrote {out_path} in {time.time() - t0:.1f}s", flush=True)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='JSON config (rule, arch, timesteps, ...)')
    p.add_argument('--out', required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--manifest', help='cell manifest CSV from ssb.analysis.cell_select')
    src.add_argument('--full-grid', action='store_true',
                     help='run every grid cell (architecture sweep)')
    p.add_argument('--cell-start', type=int, default=0)
    p.add_argument('--cell-count', type=int, default=None,
                   help='number of cells in this shard (default: all remaining)')
    args = p.parse_args(argv)

    with open(args.config) as f:
        config = json.load(f)
    rule, arch = config['rule'], config['arch']

    if args.manifest:
        cells, ids = manifest_cells(args.manifest, rule, arch)
    else:
        cells = grid_cells(rule,
                           n_alpha=int(config.get('n_alpha', len(ALPHAS))),
                           n_beta=int(config.get('n_beta', len(BETAS))))
        ids = [f'{rule}_{arch}_g{i:06d}' for i in range(len(cells))]

    lo = args.cell_start
    hi = len(cells) if args.cell_count is None else min(len(cells), lo + args.cell_count)
    if lo >= len(cells):
        raise SystemExit(f"cell-start {lo} beyond cell count {len(cells)} "
                         f"for {rule}/{arch} — check the shard arithmetic")
    cells, ids = cells[lo:hi], ids[lo:hi]
    print(f"shard: cells [{lo}, {hi}) of {hi if args.cell_count is None else 'total'}")

    run_mag_sweep(config, cells, ids, args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
