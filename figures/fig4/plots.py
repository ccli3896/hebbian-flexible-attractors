import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "axes.labelsize": 7,
    "lines.linewidth": 1,
})

def plot_rolling_decisions(result, sessid, window=20, ax=None, show_prob_labels=False):
    sess = result[result["sessionid"] == sessid].copy().reset_index(drop=True)
    if sess.empty:
        raise ValueError(f"No data for sessionid={sessid}")

    sess["rolling"] = sess["decision"].rolling(window, min_periods=1).mean()

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 1.))
        standalone = True
    else:
        standalone = False

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
    """
    For every block with length > min_block_len, compute:
      - reward ratio : r1 / (r0 + r1)
      - choice ratio : mean(decision) over trials [skip:] within that block

    Returns
    -------
    arr : np.ndarray, shape (N, 2)
        columns: [reward_ratio, choice_ratio]
    """
    rows = []
    for (sessid, block), blk in result.groupby(["sessionid", "block"]):
        if len(blk) <= min_block_len:
            continue
        r0 = blk["reward0"].iloc[0]
        r1 = blk["reward1"].iloc[0]
        reward_ratio = r1 / (r0 + r1)
        choice_ratio = blk["decision"].iloc[skip:].mean()
        rows.append([reward_ratio, choice_ratio])

    return np.array(rows)   # shape (N, 2)


def plot_choice_vs_reward(arr, ax=None):
    """
    For each unique reward ratio in arr[:,0], plot mean ± std of choice ratio arr[:,1].

    Parameters
    ----------
    arr : np.ndarray (N, 2) — [reward_ratio, choice_ratio]
    ax  : matplotlib Axes, optional
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.2, 1.2))
        standalone = True
    else:
        standalone = False

    reward_ratios = np.unique(arr[:, 0])
    means = [arr[arr[:, 0] == r, 1].mean() for r in reward_ratios]
    stds  = [arr[arr[:, 0] == r, 1].std()  for r in reward_ratios]

    means, stds = np.array(means), np.array(stds)

    ax.fill_between(reward_ratios, means - stds, means + stds,
                    color="#4A90D9", alpha=0.2, lw=0)
    ax.plot(reward_ratios, means, color="#4A90D9", lw=1.2, marker="o",
            markersize=3, markerfacecolor="w", markeredgewidth=0.8)

    # identity line
    ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls="--", alpha=0.4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, .5, 1.])
    ax.set_yticks([0, .5, 1.])
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
    """
    For each unique reward ratio condition, compute two variance metrics
    across all qualifying blocks (length > min_block_len):

      1. SD of block-level choice ratios   (cross-block consistency)
      2. Mean trial-to-trial switch rate   (within-block erraticism)

    Both computed over the full block and, if skip is set, over trials [skip:].

    Parameters
    ----------
    result       : pd.DataFrame
    min_block_len: int   — minimum block length to include
    skip         : int or None — if set, also compute metrics on trials [skip:]

    Returns
    -------
    dict keyed by reward_ratio, each value a dict:
        {
          "n_blocks"       : int,
          "choice_sd"      : float,   # full block
          "switch_rate"    : float,   # full block
          "choice_sd_skip" : float,   # only if skip is set
          "switch_rate_skip: float,   # only if skip is set
        }
    """
    # collect per-block stats
    records = {}
    for (sessid, block), blk in result.groupby(["sessionid", "block"]):
        if len(blk) <= min_block_len:
            continue

        r0 = blk["reward0"].iloc[0]
        r1 = blk["reward1"].iloc[0]
        rr = round(r1 / (r0 + r1), 6)  # reward ratio, used as key

        d = blk["decision"].values
        choice_ratio  = d.mean()
        switch_rate   = (d[1:] != d[:-1]).mean()

        rec = dict(choice_ratio=choice_ratio, switch_rate=switch_rate)

        if skip is not None:
            d_skip = d[skip:]
            rec["choice_ratio_skip"] = d_skip.mean()
            rec["switch_rate_skip"]  = (d_skip[1:] != d_skip[:-1]).mean()

        records.setdefault(rr, []).append(rec)

    # aggregate per reward ratio
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
    var = get_block_variance(result, min_block_len=min_block_len, skip=None)

    rrs = sorted(var.keys())  # actual ratio values used as x positions

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(3, 1.5))
        standalone = True
    else:
        standalone = False

    metrics = [
        ("choice_sd",   "SD of choice ratio"),
        ("switch_rate", "mean switch rate"),
    ]

    scatter_kw = dict(color="#4A90D9", s=20, zorder=3, clip_on=False)

    for ax, (key, ylabel) in zip(axes, metrics):
        vals = np.array([var[r][key] for r in rrs])

        ax.plot(rrs, vals, color="#4A90D9", lw=0.8, zorder=2)
        ax.scatter(rrs, vals, **scatter_kw)
        ax.set_xlim([0,1])

        ax.set_xticks(rrs)
        ax.set_xticklabels([f"{r:.1f}" for r in rrs], fontsize=5, rotation=45)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_xlabel("reward ratio", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, width=0.5, labelsize=5, pad=2)
        ax.set_ylim(bottom=0)

    if standalone:
        plt.tight_layout(pad=0.6)

    return axes


def plot_choice_sd_vs_reward(result, min_block_len=50, ax=None):
    """
    Standalone panel: SD of block-level choice ratio vs reward ratio
    (cross-block consistency of probability matching).
    """
    var = get_block_variance(result, min_block_len=min_block_len, skip=None)
    rrs = sorted(var.keys())
    vals = np.array([var[r]["choice_sd"] for r in rrs])

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.3, 1.2))
        standalone = True
    else:
        standalone = False

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

    if standalone:
        plt.tight_layout(pad=0.5)

    return ax