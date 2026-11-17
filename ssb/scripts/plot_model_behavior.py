"""plot_model_behavior.py — Plot model simulation results using the same
functions as the animal analysis (plot_rolling_decisions, plot_choice_vs_reward,
plot_block_variance).

Converts raw action arrays from simulate_beron into a pseudo-DataFrame that
mirrors the Beron result format, so all existing plot functions work unchanged.

For plot_rolling_decisions, blocks are simulated by:
  - Drawing reward outcomes per trial from the condition probabilities
  - Randomly switching between conditions at a fixed mean block length

Usage (in notebook):
    from plot_model_behavior import (
        make_model_result,
        plot_rolling_decisions,
        plot_choice_vs_reward,
        plot_block_variance,
        get_block_ratios,
    )

    metrics = simulate_beron(...)
    actions_per_condition = get_actions_from_simulate(...)  # see below

    # Or use the all-in-one helper:
    result_df, actions_per_condition = run_and_make_result(
        alpha=0.04592, beta=0.1186, eta_a=0.01369,
        iti=2, reward_scale=0.9775, n_hidden=10,
        n_seeds=20, base_timesteps=5000, rule='PC',
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from ssb.scripts.simulate_beron import simulate_beron, BERON_CONDITIONS, BERON_REWARD_RATIOS

mpl.rcParams.update({
    "font.family"       : "Arial",
    "font.size"         : 7,
    "axes.linewidth"    : 0.5,
    "xtick.major.width" : 0.5,
    "ytick.major.width" : 0.5,
    "xtick.major.size"  : 2,
    "ytick.major.size"  : 2,
    "xtick.labelsize"   : 5,
    "ytick.labelsize"   : 5,
    "axes.labelsize"    : 7,
    "lines.linewidth"   : 1,
})


# ---------------------------------------------------------------------------
# Core helper: simulate_beron wrapper that also returns raw action arrays
# ---------------------------------------------------------------------------

def run_and_make_result(
    alpha        : float,
    beta        : float,
    eta_a        : float,
    iti          : int   = 0,
    settle_time  : int   = 1,
    reward_scale : float = 1.0,
    n_hidden     : int   = 0,
    n_seeds      : int   = 20,
    base_timesteps: int  = 5_000,
    rule         : str   = 'PC',
    seed         : int   = 2428,
    mean_block_len: int  = 80,
    n_blocks     : int   = 60,
    rng_seed     : int   = 42,
) -> tuple[pd.DataFrame, dict]:
    """Run simulate_beron and return (result_df, metrics).

    result_df mirrors the Beron animal DataFrame format so all existing
    plot functions work without modification.

    Args:
        mean_block_len: Mean number of trials per block in the rolling
                        decisions plot. Drawn from a geometric distribution.
        n_blocks      : Number of blocks to simulate for the rolling plot.
        rng_seed      : Seed for the block-switching RNG (separate from
                        the model seed so they can be varied independently).

    Returns:
        result_df : pd.DataFrame with columns
                    [sessionid, block, trial, decision, reward0, reward1]
        metrics   : dict from compute_behavior_metrics (choice_ratio etc.)
    """
    timesteps = base_timesteps * (iti + 1)

    metrics = simulate_beron(
        alpha        = alpha,
        beta        = beta,
        eta_a        = eta_a,
        iti          = iti,
        settle_time  = settle_time,
        reward_scale = reward_scale,
        n_hidden     = n_hidden,
        n_seeds      = n_seeds,
        timesteps    = timesteps,
        rule         = rule,
        seed         = seed,
    )

    result_df = make_model_result(
        metrics,
        mean_block_len = mean_block_len,
        n_blocks       = n_blocks,
        rng_seed       = rng_seed,
    )

    return result_df, metrics


# ---------------------------------------------------------------------------
# Build pseudo-DataFrame for plot_rolling_decisions
# ---------------------------------------------------------------------------

def make_model_result(
    metrics       : dict,
    mean_block_len: int = 80,
    n_blocks      : int = 60,
    rng_seed      : int = 42,
    n_trials      : int | None = None,
) -> pd.DataFrame:
    """Convert metrics dict into a Beron-format DataFrame.

    For each simulated block:
      - Condition (reward_ratio) is drawn uniformly at random from the
        conditions in metrics['reward_ratios'].
      - Block length is drawn from Geometric(1/mean_block_len).
      - Choices are drawn from Bernoulli(choice_ratio[condition]).
      - Rewards are drawn from Bernoulli(prob_high or prob_low) depending
        on the choice made that trial.

    This gives realistic-looking trial sequences that the existing
    plot_rolling_decisions function can render directly.

    Args:
        n_trials: If set, keep generating blocks until at least this many
                  trials have been produced (overrides n_blocks). Useful for
                  generating long sequences for good summary statistics while
                  keeping a short sequence for the rolling-decisions plot.
    """
    rng = np.random.default_rng(rng_seed)

    reward_ratios = np.array(metrics['reward_ratios'])
    choice_ratios = np.array(metrics['choice_ratio'])

    # Map reward_ratio → (prob for action=0, prob for action=1)
    # reward_ratio = p1 / (p0 + p1), so given reward_ratio r:
    #   p1 = r, p0 = 1 - r  (normalised to sum to 1 for simplicity,
    #   matching what BERON_CONDITIONS encode after reward_scale=1)
    reward_probs = {
        round(float(rr), 6): (float(1 - rr), float(rr))
        for rr in reward_ratios
    }

    rows = []
    trial_idx = 0
    block_idx = 0

    while True:
        # Stop condition: either enough blocks or enough trials
        if n_trials is None:
            if block_idx >= n_blocks:
                break
        else:
            if trial_idx >= n_trials:
                break

        # Draw condition and block length
        cond_idx    = rng.integers(0, len(reward_ratios))
        rr          = float(reward_ratios[cond_idx])
        p_choice1   = float(choice_ratios[cond_idx])
        p0, p1      = reward_probs[round(rr, 6)]
        block_len   = max(10, int(rng.geometric(1.0 / mean_block_len)))

        for _ in range(block_len):
            decision = int(rng.random() < p_choice1)
            # Reward drawn from the arm actually chosen
            reward_prob = p1 if decision == 1 else p0
            reward      = int(rng.random() < reward_prob)

            rows.append({
                'sessionid' : 0,
                'block'     : block_idx,
                'trial'     : trial_idx,
                'decision'  : decision,
                'reward0'   : 1 - rr,   # prob of reward on arm 0
                'reward1'   : rr,       # prob of reward on arm 1
                'reward'    : reward,
            })
            trial_idx += 1

        block_idx += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reuse the original plot functions verbatim (copied so this file is
# self-contained; keep in sync with analysis.py if that changes)
# ---------------------------------------------------------------------------

def plot_rolling_decisions(result, sessid=0, window=20, ax=None, show_prob_labels=False):
    """Plot rolling choice probability with block shading. Works on model result_df."""
    sess = result[result["sessionid"] == sessid].copy().reset_index(drop=True)
    if sess.empty:
        raise ValueError(f"No data for sessionid={sessid}")

    sess["rolling"] = sess["decision"].rolling(window, min_periods=1).mean()

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 1.))

    shade = ["#f0f0f0", "#ffffff"]
    # Small visual break at the end of every block's bars, so a boundary is
    # always visible even when a block happens to share its neighbor's
    # reward ratio (otherwise the black target bar reads as one unbroken
    # segment straight through the boundary).
    n_trials = len(sess)
    for i, (_, blk) in enumerate(sess.groupby("block")):
        start, end = blk.index[0], blk.index[-1]
        r0, r1 = blk["reward0"].iloc[0], blk["reward1"].iloc[0]
        ratio = r1 / (r0 + r1)
        choice_ratio = blk["decision"].mean()

        gap = min(max(1, round(0.004 * n_trials)), max(1, (end - start + 1) // 3))
        bar_end = end + 1 - gap

        ax.axvspan(start, end + 1, color=shade[i % 2], alpha=0.6, lw=0)
        ax.hlines(ratio,        start, bar_end, colors="k",       lw=1.5, alpha=0.7, zorder=2)
        ax.hlines(choice_ratio, start, bar_end, colors="#4A90D9", lw=1.5, alpha=1.0, zorder=2)

        if show_prob_labels:
            mid = (start + end) / 2
            ax.text(mid, 1.01, f"{r1:.0%}/{r0:.0%}", ha="left", va="bottom",
                    fontsize=5, color="#aaa", rotation=45,
                    transform=ax.get_xaxis_transform())

    ax.plot(sess.index, sess["rolling"], color="#4A90D9", lw=0.8, alpha=0.6)

    ax.set_ylim(-0.05, 1.05 if not show_prob_labels else 1.25)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"], fontsize=5)
    ax.set_xlabel("Trial", fontsize=7)
    ax.set_ylabel("P(choice 1)", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", which="both", length=2, width=0.5, pad=2)
    plt.tight_layout(pad=0.5)
    return ax


def get_block_ratios(result, min_block_len=50, skip=25):
    """Compute [reward_ratio, choice_ratio] per block. Works on model result_df."""
    rows = []
    for (sessid, block), blk in result.groupby(["sessionid", "block"]):
        if len(blk) <= min_block_len:
            continue
        r0 = blk["reward0"].iloc[0]
        r1 = blk["reward1"].iloc[0]
        reward_ratio = r1 / (r0 + r1)
        choice_ratio = blk["decision"].iloc[skip:].mean()
        rows.append([reward_ratio, choice_ratio])
    return np.array(rows)


def plot_choice_vs_reward(arr, ax=None):
    """Plot mean ± SD choice ratio vs reward ratio."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.2, 1.2))

    reward_ratios = np.unique(arr[:, 0])
    means = np.array([arr[arr[:, 0] == r, 1].mean() for r in reward_ratios])
    stds  = np.array([arr[arr[:, 0] == r, 1].std()  for r in reward_ratios])

    ax.fill_between(reward_ratios, means - stds, means + stds,
                    color="#4A90D9", alpha=0.2, lw=0)
    ax.plot(reward_ratios, means, color="#4A90D9", lw=1.2, marker="o",
            markersize=3, markerfacecolor="w", markeredgewidth=0.8)
    ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls="--", alpha=0.4)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, .5, 1.]); ax.set_yticks([0, .5, 1.])
    ax.set_xticklabels([0, .5, 1.], fontsize=5)
    ax.set_yticklabels([0, .5, 1.], fontsize=5)
    ax.set_xlabel("Reward ratio", fontsize=7)
    ax.set_ylabel("Choice ratio", fontsize=7)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, width=0.5, pad=2)
    plt.tight_layout(pad=0.5)
    return ax


def get_block_variance(result, min_block_len=50, skip=None):
    """Compute SD of choice ratio and mean switch rate per reward condition."""
    records = {}
    for (sessid, block), blk in result.groupby(["sessionid", "block"]):
        if len(blk) <= min_block_len:
            continue
        r0 = blk["reward0"].iloc[0]
        r1 = blk["reward1"].iloc[0]
        rr = round(r1 / (r0 + r1), 6)

        d = blk["decision"].values
        rec = dict(
            choice_ratio = d.mean(),
            switch_rate  = (d[1:] != d[:-1]).mean(),
        )
        if skip is not None:
            d_skip = d[skip:]
            rec["choice_ratio_skip"] = d_skip.mean()
            rec["switch_rate_skip"]  = (d_skip[1:] != d_skip[:-1]).mean()

        records.setdefault(rr, []).append(rec)

    out = {}
    for rr, recs in sorted(records.items()):
        choice_ratios = np.array([r["choice_ratio"] for r in recs])
        switch_rates  = np.array([r["switch_rate"]  for r in recs])
        entry = {
            "n_blocks"   : len(recs),
            "choice_sd"  : choice_ratios.std(),
            "switch_rate": switch_rates.mean(),
        }
        if skip is not None:
            cr_skip = np.array([r["choice_ratio_skip"] for r in recs])
            sw_skip = np.array([r["switch_rate_skip"]  for r in recs])
            entry["choice_sd_skip"]   = cr_skip.std()
            entry["switch_rate_skip"] = sw_skip.mean()
        out[rr] = entry
    return out


def plot_block_variance(result, min_block_len=50, axes=None):
    """Plot SD of choice ratio and mean switch rate vs reward ratio."""
    var = get_block_variance(result, min_block_len=min_block_len, skip=None)

    rrs = sorted(var.keys())

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(3, 1.5))

    metrics = [
        ("choice_sd",   "SD of choice ratio"),
        ("switch_rate", "mean switch rate"),
    ]
    scatter_kw = dict(color="#4A90D9", s=20, zorder=3, clip_on=False)

    for ax, (key, ylabel) in zip(axes, metrics):
        vals = np.array([var[r][key] for r in rrs])

        ax.plot(rrs, vals, color="#4A90D9", lw=0.8, zorder=2)
        ax.scatter(rrs, vals, **scatter_kw)
        ax.set_xlim([0, 1])
        ax.set_xticks(rrs)
        ax.set_xticklabels([f"{r:.1f}" for r in rrs], fontsize=5, rotation=45)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_xlabel("reward ratio", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, width=0.5, labelsize=5, pad=2)
        ax.set_ylim(bottom=0)

    plt.tight_layout(pad=0.6)
    return axes


def plot_choice_sd_vs_reward(result, min_block_len=50, ax=None):
    """Standalone panel: SD of block-level choice ratio vs reward ratio."""
    var = get_block_variance(result, min_block_len=min_block_len, skip=None)
    rrs = sorted(var.keys())
    vals = np.array([var[r]["choice_sd"] for r in rrs])

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.3, 1.2))

    ax.plot(rrs, vals, color="#4A90D9", lw=1.2, marker="o", markersize=3,
            markerfacecolor="w", markeredgewidth=0.8, zorder=3, clip_on=False)

    ax.set_xlim(0, 1)
    ax.set_xticks([0, .5, 1.])
    ax.set_xticklabels([0, .5, 1.], fontsize=5)
    ax.set_ylabel("SD of choice ratio", fontsize=7)
    ax.set_xlabel("Reward ratio", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, width=0.5, pad=2)
    ax.set_ylim(bottom=0)

    plt.tight_layout(pad=0.5)
    return ax


# ---------------------------------------------------------------------------
# Convenience: all three plots at once
# ---------------------------------------------------------------------------

def plot_all(
    metrics,
    min_block_len   = 30,
    skip            = 25,
    window          = 20,
    mean_block_len  = 80,
    rolling_trials  = 1_000,
    stats_trials    = 100_000,
    rng_seed        = 42,
):
    """Produce all three diagnostic plots for a model.

    The rolling-decisions plot uses a short sequence (~rolling_trials trials)
    so individual blocks stay legible, while the three summary plots
    (choice-vs-reward and block-variance) are computed from a much longer
    sequence (stats_trials trials) for good statistics.

    Subplots are laid out at the same native sizes as plots.py:
      - rolling decisions : (3, 1.)
      - choice vs reward  : (1.2, 1.2)
      - block variance    : (3, 1.5)  (two panels)

    Args:
        metrics       : dict from simulate_beron.
        rolling_trials: trials for the rolling-decisions panel.
        stats_trials  : trials for the summary panels.
        rng_seed      : base RNG seed; the stats sequence uses rng_seed + 1
                        so it is independent of the rolling sequence.
    """
    # Short sequence for the rolling-decisions panel
    result_rolling = make_model_result(
        metrics,
        mean_block_len = mean_block_len,
        rng_seed       = rng_seed,
        n_trials       = rolling_trials,
    )

    # Long sequence for the summary statistics
    result_stats = make_model_result(
        metrics,
        mean_block_len = mean_block_len,
        rng_seed       = rng_seed + 1,
        n_trials       = stats_trials,
    )

    # --- Figure 1: rolling decisions, native size (3, 1.) ---
    fig1, ax1 = plt.subplots(figsize=(3, 1.))
    plot_rolling_decisions(
        result_rolling, sessid=0, window=window, ax=ax1, show_prob_labels=True
    )
    ax1.set_title("Model: rolling decisions", fontsize=7)
    plt.tight_layout(pad=0.5)

    # --- Figure 2: choice vs reward, native size (1.2, 1.2) ---
    fig2, ax2 = plt.subplots(figsize=(1.2, 1.2))
    arr = get_block_ratios(result_stats, min_block_len=min_block_len, skip=skip)
    if len(arr) > 0:
        plot_choice_vs_reward(arr, ax=ax2)
    ax2.set_title("Choice vs reward", fontsize=7)
    plt.tight_layout(pad=0.5)

    # --- Figure 3: block variance, native size (3, 1.5) ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(3, 1.5))
    plot_block_variance(result_stats, min_block_len=min_block_len, axes=axes3)
    plt.tight_layout(pad=0.6)

    plt.show()
    return result_rolling, result_stats