"""companion paper Fig 13: the line X = x0*theta in the large (1-50-2) PC network.

Reads big.npz written by large_network_line.py (run it first; about 15 s) and writes
figs/n1_large_network.svg.
a: X against theta for samples with x0 between its 40th and 60th percentiles, with a
   least-squares fit through the origin.
b: theta over time for four seeds. Seed colors are categorical hues that avoid the blue
   reserved for reward size, validated for colour-blind separation; each line is also
   labeled with its seed at its right end.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from n1_style import plt, nature_ax, side_legend, reward_color_abs

d = np.load(HERE / 'big.npz')
rec = d['rec']                                   # (time, seed, [x0, theta, X]), every 10th trial from 60,000
x0, th, X = (rec[..., k].ravel() for k in range(3))
lo, hi = np.percentile(x0, [40, 60]); m = (x0 >= lo) & (x0 <= hi)
slope = (th[m] @ X[m]) / (th[m] @ th[m])
r2 = 1 - ((X[m] - slope * th[m]) ** 2).sum() / ((X[m] - X[m].mean()) ** 2).sum()

SEEDS = [7, 9, 3, 2]                             # spread of mean theta: 3.9, 1.6, 0.8, 0.1
COLORS = ['#e34948', '#1baf7a', '#6250d6', '#eda100']   # red, aqua, violet, yellow (no blue)
t = (60_000 + 10 * np.arange(rec.shape[0])) / 1000

fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.4, 2.0), gridspec_kw=dict(wspace=0.45))
ax.scatter(th[m], X[m], s=2, color=reward_color_abs(1.0), alpha=0.15, lw=0, rasterized=True)
xs = np.array([min(0, th[m].min()), th[m].max()])
ax.plot(xs, slope * xs, color='k', lw=0.8)
ax.text(0.04, 0.96, f'slope {slope:.2f}\n$R^2$ = {r2:.3f}', transform=ax.transAxes, fontsize=7, va='top')
nature_ax(ax, r'$\theta = \sum_k (w_{2k} - w_{1k})\,w_k$', '$X = x_2 - x_1$')

for s, c in zip(SEEDS, COLORS):
    bx.plot(t, rec[:, s, 1], color=c, lw=0.6, label=str(s))
    bx.text(t[-1] + 0.8, rec[-1, s, 1], str(s), color='0.25', fontsize=6, va='center')
bx.axhline(0, color='0.85', lw=0.4, zorder=0)
nature_ax(bx, r'Trial ($\times$1000)', r'$\theta$')
bx.set_xlim(t[0], t[-1])
leg = side_legend(bx, title='Seed') or bx.get_legend()
for ln in leg.get_lines(): ln.set_linewidth(1.6)

for a_, l in zip((ax, bx), 'ab'):
    a_.text(-0.2, 1.05, l, transform=a_.transAxes, fontsize=10, fontweight='bold', va='bottom')
out = HERE / 'figs' / 'n1_large_network.svg'; out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'slope {slope:.4f}, R2 {r2:.4f}, mean x0 in band {x0[m].mean():.3f}; wrote {out}')
