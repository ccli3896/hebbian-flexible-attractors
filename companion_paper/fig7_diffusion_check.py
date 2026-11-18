"""companion paper Fig. 7: diffusion after learning, measured against D = (beta s eta / (2 alpha + beta s^2))^2.
Reads the measured/predicted table printed by n1_derive.py (derive_results.txt beside this file).
Usage: python fig7_diffusion_check.py"""
import os, re
from pathlib import Path
import numpy as np
from n1_style import plt, nature_ax, reward_color_abs
from n1_sim import ALPHA, BETA, ETA

HERE = Path(__file__).resolve().parent
txt = (HERE / 'derive_results.txt').read_text().split('=== Drift')[0]
rows = [tuple(map(float, m)) for m in re.findall(r'^\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$', txt, re.M)]
Dth = lambda r: (BETA * r * ETA / (2 * ALPHA + BETA * r ** 2)) ** 2 * 1e8
s = np.array([r[0] for r in rows])
meas = np.array([r[3] for r in rows]) * Dth(s)          # measured = (measured/predicted ratio) x predicted
print('max |measured/predicted - 1|:', np.abs(meas / Dth(s) - 1).max().round(3))
fig, ax = plt.subplots(figsize=(2.3, 1.9))
g = np.linspace(0, 1.05, 200)
ax.plot(g, Dth(g), color='#1f2d3a', lw=0.8, label='derived', zorder=1)
ax.scatter(s, meas, c=[reward_color_abs(r) for r in s], s=16, edgecolor='#1f2d3a', lw=0.4, zorder=2, label='measured')
ax.set_xlim(0, 1.05); ax.set_ylim(0, 7.6)
ax.legend(fontsize=7, frameon=False, loc='upper left', handlelength=1.2)
nature_ax(ax, r'Reward $s$', r'$D$ after learning ($\times 10^{-8}$/trial)', ny=4); ax.set_xticks([0, 0.5, 1])
ax.yaxis.label.set_size(8)
out = HERE / 'figs'; os.makedirs(out, exist_ok=True)
plt.savefig(out / 'n1_diffusion_check.svg', dpi=300, bbox_inches='tight')
