"""Supp. Note 1 Fig 6: what eps is, and why reward makes zero-mean noise push W in one direction (cartoon, not to scale)."""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np
from n1_style import plt, nature_ax

BLUE, DARK, GREY, RED = '#2171B5', '#1f2d3a', '#b4bec7', '#C44E52'
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw=dict(wspace=0.32))
ax, bx = axs

# ---- a: one weight, many trials ----
rng = np.random.default_rng(3)
slope, w = 0.8, -0.25
y0 = slope * w
ax.plot([-1, 1], [-slope, slope], color=BLUE, lw=2.2)
ax.text(0.55, 0.55 * slope + 0.08, r'$X = x_0W$', color=BLUE, fontsize=8, rotation=33, ha='center')
ax.axhline(0, color='0.55', lw=0.8, ls=(0, (3, 2)))
ax.text(0.98, 0.03, 'choice boundary', ha='right', va='bottom', fontsize=7, color='0.4')
ax.text(-0.97, 0.82, '$X > 0$: arm 2, rewarded', ha='left', va='top', fontsize=7, color=BLUE)
ax.text(0.97, -0.82, '$X < 0$: arm 1, not rewarded', ha='right', va='bottom', fontsize=7, color='0.45')
ax.axvline(w, color='0.88', lw=0.5, zorder=0)
ys = y0 + rng.normal(0, 0.17, 40)
xs = w + rng.uniform(-0.012, 0.012, 40)
ax.scatter(xs, ys, s=8, c=[BLUE if y > 0 else GREY for y in ys], zorder=3, lw=0)
ax.plot(w, y0, 'o', color='white', mec=BLUE, mew=1.2, ms=5.5, zorder=4)
ax.annotate('resting state', xy=(w, y0), xytext=(w + 0.2, y0 - 0.32), fontsize=7, color=DARK,
            arrowprops=dict(arrowstyle='-', color=DARK, lw=0.5))
yk = y0 + 0.38
ax.plot(w, yk, 'o', color=DARK, ms=4.5, zorder=5)
ax.annotate('one trial', xy=(w, yk), xytext=(w + 0.2, yk + 0.22), fontsize=7, color=DARK,
            arrowprops=dict(arrowstyle='-', color=DARK, lw=0.5))
bxp = w - 0.07
ax.annotate('', xy=(bxp, yk), xytext=(bxp, y0), arrowprops=dict(arrowstyle='<->', color=RED, lw=1.0, shrinkA=0, shrinkB=0))
for yy in (y0, yk):
    ax.plot([bxp, w], [yy, yy], color=RED, lw=0.5, ls=':')
ax.text(bxp - 0.03, (y0 + yk) / 2, r'$\varepsilon$', color=RED, fontsize=10, ha='right', va='center')
ax.set_xlim(-1, 1); ax.set_ylim(-0.9, 0.9)
ax.set_xticks([0]); ax.set_yticks([0])
nature_ax(ax, r'$W$', r'$X$'); ax.set_xticks([0]); ax.set_yticks([0])
ax.set_title('one weight, many trials', fontsize=8, pad=4)

# ---- b: the same trials as a distribution of eps (cartoon) ----
sd = 0.17; e = np.linspace(-0.6, 0.6, 400); pdf = np.exp(-e**2 / (2 * sd**2))
thr = -y0
bx.fill_between(e, pdf, where=e <= thr, color=GREY, lw=0)
bx.fill_between(e, pdf, where=e >= thr, color=BLUE, lw=0)
bx.axvline(0, ymax=0.8, color=DARK, lw=0.6, ls=(0, (3, 2)))
bx.axvline(thr, ymax=0.8, color='0.55', lw=0.8, ls=(0, (3, 2)))
bx.text(thr + 0.02, 1.02, '$X$ reaches 0', fontsize=7, color='0.4', va='bottom')
bx.text(0.45, 0.5, 'rewarded', fontsize=7, color=BLUE, ha='center')
bx.text(-0.4, 0.5, 'not rewarded', fontsize=7, color='0.45', ha='center')
bx.set_xlim(-0.6, 0.6); bx.set_ylim(0, 1.3)
nature_ax(bx, r'$\varepsilon$', ''); bx.set_yticks([]); bx.set_xticks([0]); bx.spines['left'].set_visible(False)
bx.set_title(r'the same trials, by $\varepsilon$', fontsize=8, pad=4)

for a, l in zip(axs, 'ab'):
    a.text(-0.16, 1.06, l, transform=a.transAxes, fontsize=10, fontweight='bold', va='bottom')
out = HERE / 'figs'; os.makedirs(out, exist_ok=True)
plt.savefig(out / 'n1_gate_cartoon.svg', bbox_inches='tight')
