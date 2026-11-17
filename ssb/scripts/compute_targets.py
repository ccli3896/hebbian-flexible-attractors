"""compute_targets.py — Compute animal behavioral target statistics.

Loads the Beron result dataframe, runs the same block-level analysis
used throughout the notebook, and saves the target statistics to a pkl
file for use by the Optuna optimizer.

Usage:
    python -m ssb.scripts.compute_targets \
        --data path/to/beron_result.pkl \
        --out  path/to/beron_targets.pkl \
        --skip 25 \
        --min-block-len 50

Output pkl contains a dict with keys matching compute_behavior_metrics output:
    reward_ratios, choice_ratio, choice_ratio_se,
    choice_ratio_skip, choice_ratio_skip_se,
    choice_sd, choice_sd_skip,
    switch_rate, switch_rate_skip
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def get_block_ratios_and_variance(
    result: pd.DataFrame,
    min_block_len: int = 50,
    skip: int = 25,
) -> dict:
    """Compute per-condition choice ratio, SD, and switch rate from animal data.

    Mirrors the logic of get_block_ratios / get_block_variance from the
    analysis notebook, but returns everything in a single dict structured
    to match compute_behavior_metrics output.

    Args:
        result       : Beron-format dataframe with columns
                       [sessionid, block, decision, reward0, reward1].
        min_block_len: Minimum block length to include.
        skip         : Trials to skip at start of each block for _skip metrics.

    Returns:
        dict matching the compute_behavior_metrics schema, keyed by
        reward_ratio values found in the data.
    """
    result = result.copy()
    result['reward_ratio'] = (
        result['reward1'] / (result['reward1'] + result['reward0'])
    ).round(6)

    reward_ratios = sorted(result['reward_ratio'].unique())

    records = {rr: [] for rr in reward_ratios}

    for (sessid, block), blk in result.groupby(['sessionid', 'block']):
        if len(blk) <= min_block_len:
            continue
        rr = blk['reward_ratio'].iloc[0]
        d  = blk['decision'].values

        records[rr].append({
            'choice_ratio'      : d.mean(),
            'choice_ratio_skip' : d[skip:].mean(),
            'switch_rate'       : (d[1:] != d[:-1]).mean(),
            'switch_rate_skip'  : (d[skip+1:] != d[skip:-1]).mean() if len(d) > skip + 1 else np.nan,
        })

    # Aggregate per reward ratio
    rrs_out         = []
    cr_out          = []
    cr_se_out       = []
    cr_skip_out     = []
    cr_skip_se_out  = []
    sd_out          = []
    sd_skip_out     = []
    sw_out          = []
    sw_skip_out     = []

    for rr in reward_ratios:
        recs = records[rr]
        if not recs:
            continue

        cr   = np.array([r['choice_ratio']      for r in recs])
        crsk = np.array([r['choice_ratio_skip']  for r in recs])
        sw   = np.array([r['switch_rate']        for r in recs])
        swsk = np.array([r['switch_rate_skip']   for r in recs])
        swsk = swsk[~np.isnan(swsk)]

        rrs_out.append(rr)
        cr_out.append(cr.mean())
        cr_se_out.append(cr.std() / np.sqrt(len(cr)))
        cr_skip_out.append(crsk.mean())
        cr_skip_se_out.append(crsk.std() / np.sqrt(len(crsk)))
        sd_out.append(cr.std())
        sd_skip_out.append(crsk.std())
        sw_out.append(sw.mean())
        sw_skip_out.append(swsk.mean() if len(swsk) > 0 else np.nan)

    return {
        'reward_ratios'        : np.array(rrs_out),
        'choice_ratio'         : np.array(cr_out),
        'choice_ratio_se'      : np.array(cr_se_out),
        'choice_ratio_skip'    : np.array(cr_skip_out),
        'choice_ratio_skip_se' : np.array(cr_skip_se_out),
        'choice_sd'            : np.array(sd_out),
        'choice_sd_skip'       : np.array(sd_skip_out),
        'switch_rate'          : np.array(sw_out),
        'switch_rate_skip'     : np.array(sw_skip_out),
    }


def main():
    parser = argparse.ArgumentParser(description='Compute Beron animal target statistics.')
    parser.add_argument('--data',          type=str, required=True, help='Path to beron_result.pkl')
    parser.add_argument('--out',           type=str, default='beron_targets.pkl')
    parser.add_argument('--skip',          type=int, default=25)
    parser.add_argument('--min-block-len', type=int, default=50)
    args = parser.parse_args()

    result = pd.read_pickle(args.data)
    print(f'Loaded {len(result):,} trials from {args.data}')

    targets = get_block_ratios_and_variance(
        result,
        min_block_len=args.min_block_len,
        skip=args.skip,
    )

    print('\nTarget statistics:')
    for rr, cr, sw in zip(targets['reward_ratios'],
                           targets['choice_ratio'],
                           targets['switch_rate']):
        print(f'  reward_ratio={rr:.3f}  choice_ratio={cr:.3f}  switch_rate={sw:.3f}')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(targets, f)
    print(f'\nSaved targets to {out_path}')


if __name__ == '__main__':
    main()
