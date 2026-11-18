"""companion paper Fig. 4a: the flow of the reduced system in the (W, X) plane, computed exactly.

With x0 held at s, the reduced system (§3) is linear:
    dX/dt = 2 alpha (s W - X),   dW/dt = beta s (X - s W).
Arrows show this vector field at the operating point (Table 3); the trace is the exact solution
from a state on the line displaced by delta along W (or along X). Along the flow beta*s*X + 2*alpha*W is conserved, so
the trace returns to the line at W* + 2*alpha*delta / (2 alpha + beta s^2) (W displaced) or beta*s*delta / (2 alpha + beta s^2) (X displaced).
"""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from scipy.linalg import expm
from n1_style import plt, nature_ax, mticker

ALPHA, BETA, S = 0.0848342898244072, 0.016378937069540647, 1.0
J = np.array([[-BETA * S * S, BETA * S],          # state order (W, X)
              [2 * ALPHA * S, -2 * ALPHA]])
BLUE, DARK, GREY = '#2171B5', '#1f2d3a', '#9aa6b0'
L = 0.5                                            # axis half-width

fig, ax = plt.subplots(figsize=(3.0, 3.0))
# vector field
g = np.linspace(-0.45, 0.45, 10)
Wg, Xg = np.meshgrid(g, g)
dW = BETA * S * (Xg - S * Wg)
dX = 2 * ALPHA * (S * Wg - Xg)
ax.quiver(Wg, Xg, dW, dX, color=GREY, angles='xy', scale_units='xy', scale=2.6, width=0.005,
          headwidth=4, headlength=4.5, headaxislength=4, zorder=1)
# axes and the line of fixed points
ax.axhline(0, color='0.75', lw=0.5, zorder=0); ax.axvline(0, color='0.75', lw=0.5, zorder=0)
ax.plot([-L, L], [-S * L, S * L], color=BLUE, lw=2.2, zorder=2, solid_capstyle='butt')
box = dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.6)
ang = np.degrees(np.arctan(S))
ax.text(-0.47, -0.47 + 0.035, r'line of resting states: $X = \langle x_0\rangle W$', rotation=ang,
        rotation_mode='anchor', transform_rotates_text=True, ha='left', va='bottom', color=BLUE,
        fontsize=7.5, bbox=box, zorder=6)
ax.text(-L + 0.02, L - 0.02, r'$X > 0$: chooses arm 2', ha='left', va='top', color='0.35', fontsize=7.5, bbox=box, zorder=6)
ax.text(L - 0.02, -L + 0.02, r'$X < 0$: chooses arm 1', ha='right', va='bottom', color='0.35', fontsize=7.5, bbox=box, zorder=6)


def perturb(w0, d, bracket, label):
    """Displace the resting state (w0, s*w0) by d = (dW, dX); draw the displacement, its bracket and
    the exact trajectory back to the line. Returns the W at which the trajectory rejoins the line."""
    rest = np.array([w0, S * w0])
    start = rest + np.asarray(d)
    traj = np.array([expm(J * t) @ start for t in np.linspace(0, 60, 400)])
    ax.plot(*zip(rest, start), color=DARK, lw=0.5, ls=(0, (1.5, 1.5)), zorder=3)
    (bx0, by0), (bx1, by1) = rest + bracket, start + bracket          # bracket offset from the displacement
    ax.plot([bx0, bx1], [by0, by1], color=DARK, lw=0.7, zorder=3)
    tick = np.asarray(bracket) / np.hypot(*bracket) * 0.008      # end ticks, across the bracket
    for x, y in ((bx0, by0), (bx1, by1)):
        ax.plot([x - tick[0], x + tick[0]], [y - tick[1], y + tick[1]], color=DARK, lw=0.7, zorder=3)
    mid = (rest + start) / 2 + 1.9 * np.asarray(bracket)
    ax.text(*mid, label, ha='center', va='center', fontsize=9, color=DARK)
    ax.plot(traj[:, 0], traj[:, 1], color=DARK, lw=1.4, zorder=4)
    ax.annotate('', xy=traj[45], xytext=traj[25], arrowprops=dict(arrowstyle='-|>', color=DARK, lw=1.4, mutation_scale=9), zorder=5)
    ax.plot(*rest, 'o', color='white', mec=DARK, mew=0.8, ms=4, zorder=5)
    ax.plot(*start, 'o', color=DARK, ms=4, zorder=5)
    ax.plot(*traj[-1], 'o', color=BLUE, ms=4, mec='white', mew=0.6, zorder=5)
    return traj[-1, 0]


DELTA = 0.28
cases = {'W': (-0.24, (DELTA, 0.0), (0.0, -0.04)),    # weight displaced: the state keeps most of it
         'X': (0.18, (0.0, DELTA), (-0.04, 0.0))}      # activity displaced: the state returns almost to where it was
ends = {k: perturb(w0, d, br, r'$\delta$') for k, (w0, d, br) in cases.items()}
ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect('equal')
nature_ax(ax, r'$W = w_2 - w_1$', r'$X = x_2 - x_1$', nx=3, ny=3)
plt.tight_layout()
out = HERE / 'figs'; os.makedirs(out, exist_ok=True)
plt.savefig(out / 'n1_line_attractor.svg', dpi=300, bbox_inches='tight')
k = 2 * ALPHA + BETA * S * S
for name, pred in (('W', 2 * ALPHA * DELTA / k), ('X', BETA * S * DELTA / k)):
    w0 = cases[name][0]
    print(f'{name} displaced: rejoins line at W = {ends[name]:.4f}, predicted {w0 + pred:.4f} (resting W = {w0})')
