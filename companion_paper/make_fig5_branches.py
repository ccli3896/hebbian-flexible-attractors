"""companion paper Fig 5: the two branches of resting states during learning.
X against W for all 200 networks over the first 20,000 trials of the attractor runs, at three reward sizes.
Prints the slope of each branch (least squares through the origin); the slopes are quoted in the text, not on the figure. Usage: python make_fig5_branches.py [cache_dir]"""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np
from n1_style import plt, nature_ax

C = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'cache'
REW = [0.2, 0.5, 1.0]
DARK = '#1f2d3a'
rng = np.random.default_rng(0)
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw=dict(wspace=0.42))
for ax, s in zip(axs, REW):
    d = np.load(C / f'att_s{s}.npz')
    sl = slice(0, 2000)                                  # stored every 10th trial -> first 20,000 trials
    w, X, x0 = (d[k][sl].ravel() for k in ('W', 'X', 'x0'))
    pos, neg = w > 0.05, w < -0.05
    sp = np.sum(w[pos] * X[pos]) / np.sum(w[pos] ** 2)
    sn = np.sum(w[neg] * X[neg]) / np.sum(w[neg] ** 2)
    print(f's = {s}: paying-side slope {sp:.4f} ({sp/s:.3f} s), mean x0 there {x0[pos].mean()/s:.3f} s;'
          f'  wrong-side slope {sn:.4f} ({sn/s:.3f} s), mean x0 there {x0[neg].mean()/s:.3f} s')
    k = rng.choice(w.size, 6000, replace=False)
    sc = ax.scatter(w[k], X[k] / s, c=x0[k] / s, cmap='Blues', vmin=-0.25, vmax=1.0, s=2.2, lw=0, rasterized=True)
    g = np.array([-0.8, 0.8])
    ax.plot(g, g, color=DARK, lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.axhline(0, color='0.88', lw=0.5, zorder=0); ax.axvline(0, color='0.88', lw=0.5, zorder=0)
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_aspect('equal')
    ax.set_title(f'$s = {s}$', fontsize=8.5, pad=3)
    if ax is axs[0]:                                     # branch labels on the first panel only
        ax.text(0.78, -0.4, 'paying side', fontsize=6.5, ha='right', color='#08519c')
        ax.text(-0.78, 0.22, 'wrong side', fontsize=6.5, ha='left', color='0.4')
    nature_ax(ax, r'$W$', r'$X / s$' if ax is axs[0] else '', nx=3, ny=3)
cb = fig.colorbar(sc, ax=axs, shrink=0.75, pad=0.02, ticks=[0, 0.5, 1])
cb.set_label('$x_0 / s$ on that trial', fontsize=7); cb.ax.tick_params(labelsize=6)
out = HERE / 'figs'; os.makedirs(out, exist_ok=True)
plt.savefig(out / 'n1_branches.svg', dpi=300, bbox_inches='tight')
