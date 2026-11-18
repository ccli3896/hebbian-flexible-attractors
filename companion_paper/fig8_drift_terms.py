"""companion paper Fig. 8: drift on the wrong side, measured against the two terms of the expansion (gate, reward variability).
Reads the table printed by n1_derive.py (derive_results.txt beside this file).
Usage: python fig8_drift_terms.py"""
import os, re
from pathlib import Path
import numpy as np
from n1_style import plt, nature_ax

HERE = Path(__file__).resolve().parent
txt = (HERE / 'derive_results.txt').read_text().split('=== Drift')[1]
rows = re.findall(r'^\s*([\d.]+)\s+(far|middle|near)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', txt, re.M)
REGIONS = [('near', 'Near the boundary\n$-0.02 < W < 0$'), ('far', 'Far from the boundary\n$-0.6 < W < -0.3$')]
GATE, SHIFT = '#2b6aa6', '#a9c6e3'
fig, axs = plt.subplots(1, 2, figsize=(4.6, 2.0), sharey=True, gridspec_kw=dict(wspace=0.18))
for ax, (reg, title) in zip(axs, REGIONS):
    d = np.array([[float(r[0]), float(r[2]), float(r[3]), float(r[4])] for r in rows if r[1] == reg])
    s, meas, gate, shift = d.T
    x = np.arange(len(s))
    ax.bar(x, gate, 0.6, color=GATE, label='gate term')
    ax.bar(x, shift, 0.6, bottom=gate, color=SHIFT, label='reward-variability term')
    ax.scatter(x, meas, s=14, color='k', zorder=3, label='measured')
    print(reg, 'max |sum/measured - 1| =', np.abs((gate + shift) / meas - 1).max().round(3))
    ax.set_xticks(x); ax.set_xticklabels([f'{v:g}' for v in s])
    ax.set_title(title, fontsize=7.5, pad=3)
    nature_ax(ax, r'Reward $s$', r'Drift ($\times 10^{-5}$/trial)' if ax is axs[0] else '', ny=4)
    ax.set_xticks(x); ax.set_xticklabels([f'{v:g}' for v in s])
axs[0].legend(fontsize=6.5, frameon=False, loc='upper left', handlelength=1.0)
out = HERE / 'figs'; os.makedirs(out, exist_ok=True)
plt.savefig(out / 'n1_drift_terms.svg', dpi=300, bbox_inches='tight')
