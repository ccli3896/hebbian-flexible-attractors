"""simulate_beron.py — Run SSB model on Beron reward conditions.

Two modes are provided:

1. ``simulate_beron`` (unchanged API): runs each reward condition as a
   single long, *static* session, discards the burn-in, and computes
   steady-state behavioral metrics. No block switching — each condition
   is independent. Use this for choice-vs-reward / block-variance summaries.

2. ``simulate_beron_blocks`` (new): runs ONE continuous session in which
   the reward probabilities switch at scheduled block boundaries, exactly
   like the animal task. The model state (acts / states / weights) is
   carried across switches via ``run_env``'s ``checkpoint_log`` mechanism,
   so the model genuinely re-adapts after every switch. Returns real
   per-trial choices as a Beron-format DataFrame, plus steady-state metrics.

Public API:
    simulate_beron(alpha, beta, eta_a, iti, ...)        -> metrics dict
    simulate_beron_blocks(alpha, beta, eta_a, iti, ...) -> (result_df, metrics)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import jax.numpy as jnp

from ssb.models import ssb, rules
from ssb.scripts.run import run_env
from ssb.utils.batch_inits import make_default_hps, batchify_hps
from ssb.envs.bandits import (
    bandit_unbait_iti,
    bandit_bait_iti,
    make_iti_state,
    make_iti_env_par,
)
from ssb.analysis.behavior_metrics import compute_behavior_metrics


# Beron reward conditions: (prob_high, prob_low)
# High side is always arm 1 (action=1) for simplicity;
# by symmetry, average across both sides is identical.
BERON_CONDITIONS = jnp.array([
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.1, 0.9),
]) * .3

BERON_REWARD_RATIOS = [
    p[0] / (p[0] + p[1]) for p in BERON_CONDITIONS
]


# ---------------------------------------------------------------------------
# Shared helper: build the batched hyperparameters
# ---------------------------------------------------------------------------

def _build_hps_batch(alpha, beta, eta_a, n_hidden, n_seeds, seed, tau=None):
    hps = make_default_hps(seed)
    if n_hidden > 0:
        hps['sizes'] = jnp.array([[1, 0], [n_hidden, 0], [2, 0]])
    else:
        hps['sizes'] = jnp.array([[1, 0], [2, 0]])
    hps['alpha'] = jnp.array([alpha])
    hps['beta'] = jnp.array([beta])
    hps['eta_a'] = jnp.array([eta_a])
    if tau is not None:
        hps['tau'] = jnp.array([tau])
    return batchify_hps(hps, n_seeds)


# ---------------------------------------------------------------------------
# Mode 1: static per-condition runs (original behavior, unchanged output)
# ---------------------------------------------------------------------------

def simulate_beron(
    alpha: float,
    beta: float,
    eta_a: float,
    iti: int,
    settle_time: int = 1,
    reward_scale: float = 1.0,
    n_seeds: int = 20,
    n_hidden: int = 0,
    timesteps: int = 60_000,
    rule: str = 'PC',
    baiting: bool = False,
    settle_frac: float = 0.5,
    skip: int = 25,
    seed: int = 825,
    conditions: list[tuple[float, float]] | None = None,
    reward_ratios: list[float] | None = None,
    return_actions: bool = False,
    tau: float | None = None,
) -> dict:
    """Run the SSB model on bandit conditions and return behavioral metrics.

    Each condition is one long, static run_env call; the post-settling tail
    is used for metrics. No block switching is simulated here.

    Args:
        alpha     : Activity update rate.
        beta     : Weight update rate (beta).
        eta_a     : Noise scale.
        iti       : Number of ITI null steps between bandit interactions.
        n_seeds   : Number of parallel seeds to run.
        timesteps : Total steps per condition (including ITI steps).
                    Actual bandit trials ≈ timesteps / (iti + 1).
        rule      : Weight learning rule string (e.g. 'PC').
        baiting   : If True, use bandit_bait_iti instead of bandit_unbait_iti.
        settle_frac: Fraction of timesteps discarded as burn-in.
        skip      : Extra trials skipped at start of post-settle window
                    for the _skip metrics.
        seed      : Base random seed.
        conditions: Override BERON_CONDITIONS with custom (p0, p1) tuples.
        reward_ratios: Override BERON_REWARD_RATIOS with custom values.
        return_actions: If True, return (metrics, actions_per_condition)
                    instead of just metrics. Each entry is the per-trial
                    action array (n_trials, n_seeds) for that condition.

    Returns:
        metrics dict (matching compute_behavior_metrics), or
        (metrics, actions_per_condition) if return_actions=True.
    """
    conds  = conditions    if conditions    is not None else BERON_CONDITIONS
    ratios = reward_ratios if reward_ratios is not None else BERON_REWARD_RATIOS

    env_fn = bandit_bait_iti if baiting else bandit_unbait_iti
    hps_batch = _build_hps_batch(alpha, beta, eta_a, n_hidden, n_seeds, seed, tau=tau)

    actions_per_condition = []

    for prob_high, prob_low in conds:
        env_par = make_iti_env_par(prob_low * reward_scale, prob_high * reward_scale, iti)
        env_state = make_iti_state(iti)

        log, _ = run_env(
            hps_batch,
            rule,
            env_fn,
            env_par,
            timesteps,
            settle_time=settle_time,
            update_acts=ssb.update_acts,
            env_state=env_state,
            n_motor=2,
        )

        actions = np.array(log[0])   # (T, n_seeds)

        if iti > 0:
            # ITI steps are the first `iti` steps of each (iti+1)-length cycle.
            # The real bandit trial is the LAST step of each cycle.
            actions = actions[iti::iti + 1, :]

        actions_per_condition.append(actions)

    metrics = compute_behavior_metrics(
        actions_per_condition,
        ratios,
        settle_frac=settle_frac,
        skip=skip,
    )

    if return_actions:
        return metrics, actions_per_condition
    return metrics


# ---------------------------------------------------------------------------
# Block schedule builder
# ---------------------------------------------------------------------------

def make_block_schedule(
    conds,
    n_blocks: int,
    mean_block_len: int,
    rng: np.random.Generator,
    min_block_len: int = 10,
):
    """Build a sequence of (cond_idx, block_len) for a switching session.

    Conditions are cycled through in a shuffled-but-balanced order: each
    full pass visits every condition once (in a fresh random order), so no
    condition is over- or under-represented, and consecutive blocks never
    repeat the same condition across a pass boundary unless the reshuffle
    happens to place it there. Block lengths are geometric with the given
    mean, floored at min_block_len.

    Returns:
        list of (cond_idx, block_len) tuples, length n_blocks.
    """
    n_conds = len(conds)
    order = []
    while len(order) < n_blocks:
        perm = rng.permutation(n_conds)
        # avoid an immediate repeat across the shuffle boundary
        if order and perm[0] == order[-1]:
            perm = np.roll(perm, 1)
        order.extend(perm.tolist())
    order = order[:n_blocks]

    schedule = []
    for cond_idx in order:
        block_len = max(min_block_len, int(rng.geometric(1.0 / mean_block_len)))
        schedule.append((int(cond_idx), block_len))
    return schedule


# ---------------------------------------------------------------------------
# Mode 2: continuous session with scheduled block switching
# ---------------------------------------------------------------------------

def simulate_beron_blocks(
    alpha: float,
    beta: float,
    eta_a: float,
    iti: int,
    settle_time: int = 1,
    reward_scale: float = 1.0,
    n_seeds: int = 20,
    n_hidden: int = 0,
    rule: str = 'PC',
    baiting: bool = False,
    seed: int = 825,
    conditions: list[tuple[float, float]] | None = None,
    reward_ratios: list[float] | None = None,
    n_blocks: int = 60,
    mean_block_len: int = 80,
    min_block_len: int = 10,
    rng_seed: int = 42,
    metrics_skip: int = 25,
    metrics_min_block_len: int = 50,
    tau: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run ONE continuous session with reward probabilities switching at
    scheduled block boundaries, carrying model state across switches.

    Each block is run as a separate ``run_env`` call, but the previous
    block's ``final_state`` is passed in as ``checkpoint_log`` so that the
    model's acts / states / weights persist across the switch. This makes
    the model genuinely re-adapt after every probability change, exactly
    like the animal experiences block transitions.

    EVERY seed becomes its own session (sessionid = seed index), all running
    the SAME block schedule. This is deliberate: the summary plots
    (get_block_ratios / get_block_variance) group by [sessionid, block], so
    n_seeds seeds give n_seeds independent realisations of each block. The
    cross-block spread that drives the variance plots therefore reflects real
    model variability under switching, NOT the artefact of independent
    steady-state runs that the static simulate_beron would have produced.

    Args:
        n_blocks       : Number of blocks per session.
        mean_block_len : Mean trials per block (geometric draw).
        min_block_len  : Floor on block length (trials).
        rng_seed       : Seed for the block-scheduling RNG (independent of
                         the model seed). All seeds share this one schedule.
        metrics_skip / metrics_min_block_len : transient handling for the
                         optional pooled metrics summary.

    Returns:
        result_df : pd.DataFrame with columns
                    [sessionid, block, trial, decision, reward0, reward1]
                    containing REAL per-trial model choices with true block
                    structure, one session per seed. Feed this directly to
                    plot_rolling_decisions / get_block_ratios /
                    plot_block_variance.
        metrics   : dict with {reward_ratios, choice_ratio} pooled across all
                    seeds and blocks (transient dropped). Provided for
                    convenience; the summary plots do not require it.
    """
    conds  = np.array(conditions if conditions is not None else BERON_CONDITIONS)
    ratios = reward_ratios if reward_ratios is not None else BERON_REWARD_RATIOS

    env_fn = bandit_bait_iti if baiting else bandit_unbait_iti
    hps_batch = _build_hps_batch(alpha, beta, eta_a, n_hidden, n_seeds, seed, tau=tau)

    rng = np.random.default_rng(rng_seed)
    schedule = make_block_schedule(
        conds, n_blocks, mean_block_len, rng, min_block_len=min_block_len
    )

    rows = []
    # Per-seed running trial counter, so trials are contiguous within each
    # session (sessionid). One entry per seed.
    block_trial_start = [0] * n_seeds

    # Per-condition accumulation of per-block stats across all seeds,
    # to rebuild a compute_behavior_metrics-style summary.
    # We collect raw per-block action arrays grouped by condition index.
    actions_by_cond: dict[int, list[np.ndarray]] = {}

    checkpoint = None   # final_state carried across blocks

    for block_idx, (cond_idx, block_len) in enumerate(schedule):
        prob_high, prob_low = float(conds[cond_idx][0]), float(conds[cond_idx][1])
        rr = prob_high / (prob_high + prob_low)   # reward ratio for this block

        env_par   = make_iti_env_par(prob_low * reward_scale,
                                     prob_high * reward_scale, iti)
        env_state = make_iti_state(iti)

        # timesteps for this block, including ITI null steps
        block_timesteps = block_len * (iti + 1)

        log, final_state = run_env(
            hps_batch,
            rule,
            env_fn,
            env_par,
            block_timesteps,
            settle_time=settle_time,
            update_acts=ssb.update_acts,
            env_state=env_state,
            n_motor=2,
            checkpoint_log=checkpoint,   # <-- carry model state across the switch
        )

        # carry forward for the next block
        checkpoint = final_state

        actions = np.array(log[0])   # (block_timesteps, n_seeds)
        if iti > 0:
            actions = actions[iti::iti + 1, :]   # (block_len, n_seeds)

        actions_by_cond.setdefault(cond_idx, []).append(actions)

        # Build per-trial rows for EVERY seed. Each seed is its own session
        # (sessionid = seed index), all sharing the same block schedule.
        # This is what gives the summary plots their cross-block variance:
        # get_block_variance groups by [sessionid, block], so n_seeds seeds
        # contribute n_seeds independent realisations of every block.
        n_trials_block = actions.shape[0]
        # local per-seed trial offsets so trials are contiguous within a session
        for seed_idx in range(actions.shape[1]):
            col = actions[:, seed_idx]
            for t, d in enumerate(col):
                rows.append({
                    'sessionid': int(seed_idx),
                    'block'    : block_idx,
                    'trial'    : block_trial_start[seed_idx] + t,
                    'decision' : int(d),
                    'reward0'  : float(1 - rr),
                    'reward1'  : float(rr),
                })
            block_trial_start[seed_idx] += n_trials_block

    result_df = pd.DataFrame(rows)

    # ---- Build full metrics dict from result_df ----
    # Inlined rather than imported from plot_model_behavior to avoid a
    # circular import (plot_model_behavior imports from simulate_beron).
    # Logic is identical to get_block_variance / get_block_ratios.

    # Per-block records grouped by reward ratio
    records: dict[float, list[dict]] = {}
    for (sid, blk), blk_df in result_df.groupby(['sessionid', 'block']):
        if len(blk_df) <= metrics_min_block_len:
            continue
        r0 = blk_df['reward0'].iloc[0]
        r1 = blk_df['reward1'].iloc[0]
        rr = round(float(r1 / (r0 + r1)), 6)
        d      = blk_df['decision'].values[metrics_skip:]
        if len(d) == 0:
            continue
        records.setdefault(rr, []).append({
            'choice_ratio': d.mean(),
            'switch_rate' : (d[1:] != d[:-1]).mean() if len(d) > 1 else 0.0,
        })

    reward_ratios = sorted(records.keys())

    def _agg(key):
        return [float(np.mean([r[key] for r in records[rr]])) for rr in reward_ratios]

    def _sd(key):
        return [float(np.std([r[key] for r in records[rr]])) for rr in reward_ratios]

    metrics = {
        'reward_ratios': reward_ratios,
        'choice_ratio' : _agg('choice_ratio'),
        'choice_sd'    : _sd('choice_ratio'),
        'switch_rate'  : _agg('switch_rate'),
    }

    return result_df, metrics