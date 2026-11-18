"""companion paper §6, "Deriving the drift and diffusion": checks of the expansion around the line.

Diffusion after learning: D = (beta s eta / (2 alpha + beta s^2))^2, against the measured D(W) for 0.1 < W < 0.8
(attractor runs in cache/, seeds 0-119, 400-trial increments, as in §6).
Drift on the wrong side: v = beta <x0 eps> against the two-term expansion
    v_gate  = beta (1 - 2 alpha) / (2 (1 - alpha)) Cov(s_t, eps_{t-1})
    v_shift = -beta alpha W Var(s_t) / (2 (1 - alpha)^2)
from new runs that record every trial (200 seeds x 60,000 trials per reward).
Usage: python n1_derive.py [cache_dir]   (cache_dir defaults to ./cache, written by n1_run.py att)"""
import sys
from pathlib import Path
import numpy as np
from n1_sim import simulate, ALPHA, BETA, ETA

HERE = Path(__file__).resolve().parent
C = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'cache'
REWARDS = [0.2, 0.3, 0.5, 0.7, 1.0]
RHO = 1 - 2 * ALPHA
P = lambda *a: print(*a, flush=True)

P('=== Diffusion after learning (x 1e-8 per trial) ===')
P(f"{'s':>4} {'measured':>9} {'predicted':>10} {'ratio':>6}")
for s in REWARDS:
    W = np.load(C / f'att_s{s}.npz')['W'][:, :120]
    lag = 40                                             # stored every 10th trial -> 400 trials
    w0, dW = W[:-lag], W[lag:] - W[:-lag]
    m = (w0 > 0.1) & (w0 < 0.8)
    Dm = dW[m].var() / (2 * lag * 10)
    Dp = (BETA * s * ETA / (2 * ALPHA + BETA * s * s)) ** 2
    P(f'{s:4.1f} {Dm*1e8:9.2f} {Dp*1e8:10.2f} {Dm/Dp:6.3f}')

P('\n=== Drift on the wrong side (x 1e-5 per trial) ===')
BINS = {'far': (-0.6, -0.3), 'middle': (-0.15, -0.05), 'near': (-0.02, 0.0)}
P(f"{'s':>4} {'region':>7} {'measured':>9} {'gate':>7} {'shift':>7} {'sum':>7} {'mean-eps part':>14} {'P(opt)':>7} {'sd(eps)':>8}")
for s in REWARDS:
    o = simulate(np.arange(200), s, 60_000, stride=1, chunk=10_000)
    W, x0, X, A = o['W'], o['x0'], o['X'], o['A']
    Wp = np.roll(W, 1, 0)                                # the weight used in the update on trial t
    eps = X - x0 * Wp                                    # deviation the update on trial t sees
    eps_prev = np.roll(X, 1, 0) - np.roll(x0, 1, 0) * Wp  # deviation at the time of the choice on trial t
    sl = slice(2, None)
    w, e, ep, xx, a = (v[sl].ravel() for v in (Wp, eps, eps_prev, x0, A))
    for name, (lo, hi) in BINS.items():
        k = (w > lo) & (w < hi)
        r = s * a[k]; Wc = w[k].mean()
        v = BETA * np.mean(xx[k] * e[k])
        vmean = BETA * xx[k].mean() * e[k].mean()
        G = np.cov(r, ep[k])[0, 1]; Vr = r.var()
        vg = BETA * RHO * G / (2 * (1 - ALPHA))
        vs = -BETA * ALPHA * Wc * Vr / (2 * (1 - ALPHA) ** 2)
        P(f'{s:4.1f} {name:>7} {v*1e5:9.3f} {vg*1e5:7.3f} {vs*1e5:7.3f} {(vg+vs)*1e5:7.3f} {vmean*1e5:14.3f} {a[k].mean():7.3f} {e[k].std():8.4f}')
