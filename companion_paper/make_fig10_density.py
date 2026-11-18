"""companion paper Fig 10, right panel: P(optimal) as an area under the density of eps.

For two values of m, draws the Gaussian density of eps (s.d. sigma_eps) and shades the area beyond
the threshold -P m, where P solves P = Phi(P m / sigma_eps). Then appends this panel to the right of
figs/n1_collapse.svg (made by n1_analysis.py, 'collapse') and writes figs/n1_collapse_v2.svg.
Needs only numpy and matplotlib.
"""
import os, re, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np
from n1_style import plt, nature_ax

ALPHA, ETA = 0.0848342898244072, 0.003          # as in n1_sim.py
GAMMA = 1 - 2 * ALPHA
SIG = math.sqrt(2) * ETA / math.sqrt(1 - GAMMA**2)   # sigma_eps = 0.0076
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))


def P_sc(m, lo=1e-9, hi=1.0):
    """Unique root of P - Phi(P m / sigma_eps) on (0, 1], by bisection."""
    f = lambda p: p - Phi(p * m / SIG)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


BLUE, DARK, GREY = '#2171B5', '#1f2d3a', '#b4bec7'
CASES = [(-0.02, 'wrong side'), (0.01, 'past the boundary')]

fig, axs = plt.subplots(2, 1, figsize=(1.85, 2.1), gridspec_kw=dict(hspace=1.0))
e = np.linspace(-0.026, 0.026, 600)
pdf = np.exp(-e**2 / (2 * SIG**2)) / (SIG * math.sqrt(2 * math.pi))
for k, (ax, (m, lab)) in enumerate(zip(axs, CASES), 1):
    P = P_sc(m); thr = -P * m
    ax.fill_between(e, pdf, where=e <= thr, color=GREY, lw=0)
    ax.fill_between(e, pdf, where=e >= thr, color=BLUE, lw=0)
    ax.plot(e, pdf, color=DARK, lw=0.5)
    ax.axvline(thr, ymax=1.0, color='0.35', lw=0.7, ls=(0, (3, 2)))
    ax.text(thr + (0.0012 if m < 0 else -0.0012), pdf.max() * 1.0, r'$-Pm$', fontsize=7, color='0.3',
            ha='left' if m < 0 else 'right', va='top')
    ax.text(0.0265, pdf.max() * 0.55, f'$P$ = {P:.2f}', fontsize=7, color=BLUE, ha='right')
    ax.set_title(f'$m = {m:+.2f}$ ({lab})', fontsize=7, pad=4)
    # numbered dot matching the large dot on the left panel
    ax.plot([-0.08], [1.17], 'o', ms=9, color=DARK, mec='white', mew=0.8, transform=ax.transAxes, clip_on=False)
    ax.text(-0.08, 1.17, str(k), fontsize=6.5, color='white', fontweight='bold', ha='center', va='center',
            transform=ax.transAxes)
    ax.set_xlim(e[0], e[-1]); ax.set_ylim(0, pdf.max() * 1.05)
    nature_ax(ax, '', ''); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xticks([-0.02, 0, 0.02]); ax.set_xticklabels(['−0.02', '0', '0.02'])
axs[1].set_xlabel(r'$\varepsilon$', fontsize=9)
out = HERE / 'figs'; os.makedirs(out, exist_ok=True)
panel = out / 'n1_collapse_density.svg'
plt.savefig(panel, bbox_inches='tight'); plt.close()


# ---- append the panel to the right of n1_collapse.svg ----
def load(p, prefix):
    s = Path(p).read_text()
    s = re.sub(r'<\?xml.*?\?>|<!DOCTYPE.*?>|<metadata>.*?</metadata>', '', s, flags=re.S)
    w, h = (float(v) for v in re.search(r'width="([\d.]+)pt" height="([\d.]+)pt"', s).groups())
    s = re.sub(r'id="([^"]+)"', lambda mm: f'id="{prefix}{mm.group(1)}"', s)
    s = re.sub(r'(xlink:href|href)="#([^"]+)"', lambda mm: f'{mm.group(1)}="#{prefix}{mm.group(2)}"', s)
    s = re.sub(r'url\(#([^)]+)\)', lambda mm: f'url(#{prefix}{mm.group(1)})', s)
    return s.strip(), w, h


svgs = HERE / 'figs'
a, wa, ha = load(svgs / 'n1_collapse.svg', 'L')

# large dots on the left panel at the (m, P) of the two densities. Axis calibration read from the
# tick positions in n1_collapse.svg: x = -0.06 -> 36.376, 0.06 -> 137.035; y = P 0 -> 102.354, 0.9 -> 18.395
X0, X1, Y0, Y9 = 36.376289, 137.035113, 102.354231, 18.394615
to_svg = lambda m, P: (X0 + (m + 0.06) / 0.12 * (X1 - X0), Y0 + P / 0.9 * (Y9 - Y0))
dots = ''.join(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4.6" style="fill: {DARK}; stroke: #ffffff; stroke-width: 0.8"/>'
               f'<text x="{x:.3f}" y="{y + 2.2:.3f}" text-anchor="middle" style="font: bold 6.5px Arial, \'Liberation Sans\', sans-serif; fill: #ffffff">{k}</text>'
               for k, (x, y) in enumerate((to_svg(m, P_sc(m)) for m, _ in CASES), 1))
a = a[:a.rindex('</svg>')] + dots + '</svg>'
b, wb, hb = load(panel, 'R')
gap = 6.0
W, H = wa + gap + wb, max(ha, hb)
place = lambda s, x, y, w, h: re.sub(r'<svg ', f'<svg x="{x}" y="{y}" ', re.sub(r'width="[\d.]+pt" height="[\d.]+pt"', f'width="{w}" height="{h}"', s, count=1), count=1)
doc = (f'<?xml version="1.0" encoding="utf-8" standalone="no"?>\n'
       f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" '
       f'width="{W:.2f}pt" height="{H:.2f}pt" viewBox="0 0 {W:.2f} {H:.2f}">\n'
       + place(a, 0, (H - ha) / 2, wa, ha) + '\n'
       + place(b, wa + gap, (H - hb) / 2, wb, hb) + '\n</svg>\n')
(svgs / 'n1_collapse_v2.svg').write_text(doc)
print(f'sigma_eps = {SIG:.5f}; ' + '; '.join(f'm={m:+.2f}: P={P_sc(m):.3f}, -Pm={-P_sc(m)*m:+.5f}' for m, _ in CASES))
