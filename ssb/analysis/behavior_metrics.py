"""Behavior metrics for comparing model output to animal data.

All functions accept raw action arrays from ``run_env`` and return dicts
of scalar statistics that can be directly compared to the output of the
animal analysis functions (get_block_ratios, get_block_variance).

The three metric groups are:
    1. choice_ratio   — mean P(action=1) per reward condition (post-settling)
    2. choice_sd      — cross-block SD of choice ratio per condition
    3. switch_rate    — mean trial-to-trial switch rate per condition

Each metric is computed for two windows:
    full  — all post-settling trials
    skip  — trials after the first ``skip`` in each block

Convention: actions are binary (0 or 1). reward_ratios is a 1-D array
of the unique reward ratio values used (one per condition).
"""

from __future__ import annotations

import numpy as np


def compute_behavior_metrics(
    actions_per_condition: list[np.ndarray],
    reward_ratios: list[float],
    settle_frac: float = 0.5,
    skip: int = 25,
    block_len: int | None = None,
) -> dict:
    """Compute choice ratio, cross-condition SD, and switch rate.

    Args:
        actions_per_condition:
            List of arrays, one per reward condition. Each array has
            shape (timesteps, n_seeds) — the raw action log from run_env
            for that condition (already extracted action channel).
        reward_ratios:
            Parallel list of reward ratio values (float), one per condition.
            e.g. [0.9, 0.8, 0.7, 0.3, 0.2, 0.1] for Beron 6-condition setup.
        settle_frac:
            Fraction of trials to discard as burn-in. Default 0.5 (second
            half only), matching the existing sweep scoring convention.
        skip:
            Number of additional trials to drop from the START of the
            post-settle window when computing the _skip metrics.
        block_len:
            Unused in the simple simulation (single long block per condition).
            Reserved for future block-switching simulation.

    Returns:
        dict with keys:
            'reward_ratios'   : np.ndarray, shape (n_cond,)
            'choice_ratio'    : np.ndarray, shape (n_cond,)   mean over seeds
            'choice_ratio_se' : np.ndarray, shape (n_cond,)   SE over seeds
            'choice_ratio_skip'    : np.ndarray, shape (n_cond,)
            'choice_ratio_skip_se' : np.ndarray, shape (n_cond,)
            'switch_rate'     : np.ndarray, shape (n_cond,)
            'switch_rate_skip': np.ndarray, shape (n_cond,)
            'choice_sd'       : np.ndarray, shape (n_cond,)   SD across seeds
            'choice_sd_skip'  : np.ndarray, shape (n_cond,)
    """
    n_cond = len(actions_per_condition)
    choice_ratio      = np.zeros(n_cond)
    choice_ratio_se   = np.zeros(n_cond)
    choice_ratio_skip      = np.zeros(n_cond)
    choice_ratio_skip_se   = np.zeros(n_cond)
    switch_rate       = np.zeros(n_cond)
    switch_rate_skip  = np.zeros(n_cond)
    choice_sd         = np.zeros(n_cond)
    choice_sd_skip    = np.zeros(n_cond)

    for i, (actions, rr) in enumerate(zip(actions_per_condition, reward_ratios)):
        # actions: (timesteps, n_seeds)
        T = actions.shape[0]
        settle_idx = int(T * settle_frac)

        full  = actions[settle_idx:, :]          # (T_post, n_seeds)
        skip_ = actions[settle_idx + skip:, :]   # skip further trials

        # Choice ratio per seed, then mean/SE/SD across seeds
        cr_full  = full.mean(axis=0)    # (n_seeds,)
        cr_skip  = skip_.mean(axis=0)

        choice_ratio[i]          = cr_full.mean()
        choice_ratio_se[i]       = cr_full.std() / np.sqrt(len(cr_full))
        choice_ratio_skip[i]     = cr_skip.mean()
        choice_ratio_skip_se[i]  = cr_skip.std() / np.sqrt(len(cr_skip))
        choice_sd[i]             = cr_full.std()
        choice_sd_skip[i]        = cr_skip.std()

        # Switch rate per seed: fraction of consecutive-trial switches
        switches_full = (np.diff(full, axis=0) != 0).mean(axis=0)
        switches_skip = (np.diff(skip_, axis=0) != 0).mean(axis=0)
        switch_rate[i]      = switches_full.mean()
        switch_rate_skip[i] = switches_skip.mean()

    return {
        'reward_ratios'        : np.array(reward_ratios),
        'choice_ratio'         : choice_ratio,
        'choice_ratio_se'      : choice_ratio_se,
        'choice_ratio_skip'    : choice_ratio_skip,
        'choice_ratio_skip_se' : choice_ratio_skip_se,
        'choice_sd'            : choice_sd,
        'choice_sd_skip'       : choice_sd_skip,
        'switch_rate'          : switch_rate,
        'switch_rate_skip'     : switch_rate_skip,
    }