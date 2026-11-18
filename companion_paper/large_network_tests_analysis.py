"""Analysis of big_tests.npz (large_network_tests.py): noise control, drift/diffusion of theta,
behavior collapse against m = s*theta, and a held-out Fokker-Planck prediction."""
import numpy as np
d = np.load('big_tests.npz')
R = (0.2, 0.5, 1.0); LAG = 4                     # th sampled every 100 trials -> lag 400
FIT, HOLD = slice(0, 120), slice(120, 200)

# 1. noise control
th = d['th_noiseless']; b = d['beh_noiseless']
print('NOISELESS: mean |theta(50k)-theta(1k)| =', np.abs(th[-1] - th[10]).mean().round(4),
      ' mean |theta| =', np.abs(th[-1]).mean().round(3),
      ' P(opt) first 5k / last 5k:', b[:500, :, 1].mean().round(3), b[-500:, :, 1].mean().round(3))
th1 = d['th_1.0']
print('  with noise (s=1): mean |theta(50k)-theta(1k)| =', np.abs(th1[500] - th1[10]).mean().round(3))

# 2. drift and diffusion
edges = np.linspace(-1.5, 3, 46); cen = (edges[1:] + edges[:-1]) / 2
VD = {}
for r in R:
    th = d[f'th_{r}'][:, FIT]
    a, dlt = th[:-LAG].ravel(), (th[LAG:] - th[:-LAG]).ravel()
    idx = np.digitize(a, edges) - 1
    v = np.array([dlt[idx == k].mean() / 400 if (idx == k).sum() > 200 else np.nan for k in range(len(cen))])
    D = np.array([dlt[idx == k].var() / 800 if (idx == k).sum() > 200 else np.nan for k in range(len(cen))])
    VD[r] = (v, D)
    w = a < -0.05; c = a > 0.3
    print(f'DRIFT s={r}: wrong side v = {dlt[w].mean()/400:.2e} (v/s {dlt[w].mean()/400/r:.2e}, n={w.sum()}); '
          f'past boundary v = {dlt[c].mean()/400:.2e}; D wrong {dlt[w].var()/800:.2e}, D past {dlt[c].var()/800:.2e}')

# 3. behavior against m = s*theta
medges = np.linspace(-0.6, 0.6, 49); mc = (medges[1:] + medges[:-1]) / 2
curves = {}
for r in R:
    bb = d[f'beh_{r}']; m = (r * bb[..., 0]).ravel(); ch = bb[..., 1].ravel()
    i = np.digitize(m, medges) - 1
    curves[r] = np.array([ch[i == k].mean() if (i == k).sum() > 500 else np.nan for k in range(len(mc))])
C = np.array([curves[r] for r in R]); ok = np.all(np.isfinite(C), 0)
print('COLLAPSE vs m: RMS across rewards', np.sqrt(np.nanmean((C[:, ok] - C[:, ok].mean(0)) ** 2)).round(4),
      ' (vs theta:', end=' ')
cth = {}
for r in R:
    bb = d[f'beh_{r}']; m = bb[..., 0].ravel(); ch = bb[..., 1].ravel()
    i = np.digitize(m, medges) - 1
    cth[r] = np.array([ch[i == k].mean() if (i == k).sum() > 500 else np.nan for k in range(len(mc))])
C2 = np.array([cth[r] for r in R]); ok2 = np.all(np.isfinite(C2), 0)
print(np.sqrt(np.nanmean((C2[:, ok2] - C2[:, ok2].mean(0)) ** 2)).round(4), ')')
for k in range(0, len(mc), 4):
    print(f'   m={mc[k]:+.3f}: ' + '  '.join(f'{curves[r][k]:.2f}' for r in R))
# pooled behavior curve from fit seeds
mm = np.concatenate([(r * d[f'beh_{r}'][:, FIT, 0]).ravel() for r in R])
cc = np.concatenate([d[f'beh_{r}'][:, FIT, 1].ravel() for r in R])
i = np.digitize(mm, medges) - 1
Pm = np.array([cc[i == k].mean() if (i == k).sum() > 200 else np.nan for k in range(len(mc))])
g = np.isfinite(Pm); Pm_f = lambda m: np.interp(m, mc[g], Pm[g])

# 4. Fokker-Planck, held-out seeds
grid = np.linspace(-3, 6, 451); hg = grid[1] - grid[0]
errs = []
for r in R:
    v, D = VD[r]; gv = np.isfinite(v)
    vg = np.interp(grid, cen[gv], v[gv]); Dg = np.interp(grid, cen[gv], D[gv])
    th = d[f'th_{r}'][:, HOLD]; beh = d[f'beh_{r}'][:, HOLD]
    rho, _ = np.histogram(th[0], bins=np.append(grid - hg / 2, grid[-1] + hg / 2)); rho = rho / (rho.sum() * hg)
    rho = np.convolve(rho, np.ones(5) / 5, 'same')
    dt = 0.2 * hg**2 / max(Dg.max(), 1e-12); dt = min(dt, 0.2 * hg / np.abs(vg).max())
    times = [2_000, 5_000, 10_000, 20_000, 40_000, 80_000, 149_000]; t = 0.0; out = []
    for T in times:
        while t < T:
            # upwind drift flux at faces, diffusion flux -(d/dW)(D rho)
            vf = 0.5 * (vg[1:] + vg[:-1]); up = np.where(vf > 0, rho[:-1], rho[1:])
            F = vf * up - (Dg[1:] * rho[1:] - Dg[:-1] * rho[:-1]) / hg
            F = np.concatenate([[0], F, [0]])
            rho = rho - dt * (F[1:] - F[:-1]) / hg; t += dt
        pred = np.sum(Pm_f(r * grid) * rho) * hg / (rho.sum() * hg)
        k = T // 10; meas = beh[max(0, k - 100):k + 100, :, 1].mean()
        out.append((T, meas, pred)); errs.append(pred - meas)
    print(f'FP s={r}: ' + '  '.join(f'{T//1000}k {a:.3f}/{b:.3f}' for T, a, b in out))
errs = np.array(errs); print('FP RMS error', np.sqrt((errs**2).mean()).round(4), 'max', np.abs(errs).max().round(4))
