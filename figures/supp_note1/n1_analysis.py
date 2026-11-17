"""Supplementary Note 1: analysis and figures at the asymptote-flat operating point.

Reads the simulation caches written by n1_run.py / n1_noise_run.py, prints every number quoted in
the note, and writes the figures to figs/. Sections follow the note.
"""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[2]))   # this folder; repo root (contains ssb/)
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
from scipy.stats import norm
from n1_style import plt, mticker, REWARDS, reward_color_abs, nature_ax, side_legend, kfmt
from n1_sim import ALPHA, BETA, ETA

C = str(HERE / 'cache'); F = str(HERE / 'figs'); os.makedirs(F, exist_ok=True)
ST = 10                      # attractor runs store every 10th trial
LAG_STEPS = 400; LAG = LAG_STEPS // ST
NFIT = 120                   # seeds 0-119 estimate v, D, P(m); 120-199 are held out
WINDOW = 100
only = set(sys.argv[1:])     # optionally run only some sections
want = lambda k: not only or k in only


def P(*a, **k):
    print(*a, **k, flush=True)


# =====================================================================================
# §1  Behaviour: learning curves, rate, asymptote
# =====================================================================================
BEH = {r: np.load(f'{C}/beh_s{r}.npz')['B'] for r in REWARDS}       # (10 batches, bins, 80 seeds)
T_AXIS = (np.arange(BEH[0.2].shape[1]) + 1) * WINDOW


def metrics(curve):
    """Asymptote = mean of the final 10% of bins; t_crit = first bin reaching 0.9 x asymptote
    (the main-text definition, Methods 'Rate and asymptote sensitivity'); rate = 1/t_crit."""
    A = curve[-len(curve) // 10:].mean()
    tc = (int(np.argmax(curve >= 0.9 * A)) + 1) * WINDOW
    return A, tc


if want('beh'):
    fig, ax = plt.subplots(figsize=(2.6, 1.35))
    for r in REWARDS:
        ax.plot(T_AXIS, BEH[r].mean(-1).mean(0), color=reward_color_abs(r), lw=0.9)
    ax.axhline(0.5, color='0.85', lw=0.4, zorder=0)
    nature_ax(ax, 'Trial', 'P(optimal)')
    ax.set_xlim(0, T_AXIS[-1]); ax.set_ylim(0.45, 1.02)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(100_000)); kfmt(ax)
    side_legend(ax, title='Reward', handles=[plt.Line2D([], [], color=reward_color_abs(r), lw=1.0,
                                                        label=f'{r:.1f}') for r in REWARDS])
    plt.tight_layout(); plt.savefig(f'{F}/n1_learningcurves.svg', dpi=300, bbox_inches='tight')
    plt.close()

    A = np.zeros((5, 10)); TC = np.zeros((5, 10))
    for i, r in enumerate(REWARDS):
        for k in range(10):
            A[i, k], TC[i, k] = metrics(BEH[r][k].mean(-1))
    RATE = 1e4 / TC
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 1.35), gridspec_kw=dict(wspace=0.6))
    for ax, vals, ylab in [(axes[0], RATE, 'Rate\n' r'($10^4$ / trials to criterion)'),
                           (axes[1], A, 'Asymptote')]:
        for i, r in enumerate(REWARDS):
            jit = np.random.default_rng(i).normal(0, 0.012, size=vals.shape[1])
            ax.scatter(np.full(vals.shape[1], r) + jit, vals[i], facecolors='none',
                       edgecolors=reward_color_abs(r), s=7, lw=0.4, alpha=0.8, zorder=2)
        m, se = vals.mean(1), vals.std(1) / np.sqrt(vals.shape[1])
        ax.errorbar(REWARDS, m, yerr=se, fmt='-', color='0.15', lw=0.9, marker='o', markersize=2.6,
                    capsize=1.6, elinewidth=0.7, zorder=3)
        nature_ax(ax, 'Reward size', ylab, nx=5, ny=4)
    axes[0].set_ylim(0, None)
    axes[1].set_ylim(0.5, 1.02); axes[1].axhline(1.0, color='0.85', lw=0.4, zorder=0)
    plt.tight_layout(); plt.savefig(f'{F}/n1_rate_asymptote.svg', dpi=300, bbox_inches='tight')
    plt.close()

    P('\n=== §2 Fig. 3: rate and asymptote (mean ± s.e.m. over 10 batches of 80 seeds) ===')
    P(f"{'s':>4} {'t_crit':>16} {'rate (1e4/t)':>16} {'asymptote':>16} {'P at 10k':>9} {'P at 60k':>9}")
    for i, r in enumerate(REWARDS):
        c = BEH[r].mean(-1).mean(0)
        P(f'{r:4.1f} {TC[i].mean():8,.0f} ± {TC[i].std()/np.sqrt(10):5,.0f} '
          f'{RATE[i].mean():8.3f} ± {RATE[i].std()/np.sqrt(10):.3f} '
          f'{A[i].mean():8.4f} ± {A[i].std()/np.sqrt(10):.4f} {c[99]:9.3f} {c[599]:9.3f}')
    lr = np.polyfit(np.log(REWARDS), np.log(RATE.mean(1)), 1)[0]
    la = np.polyfit(REWARDS, A.mean(1), 1)[0]
    P(f'rate ~ s^{lr:.2f};  rate ratio s=1/s=0.2: {RATE[-1].mean()/RATE[0].mean():.1f}x;  '
      f'asymptote linear slope d(A)/ds = {la:+.4f};  range {A.mean(1).min():.4f}-{A.mean(1).max():.4f}')
    P(f'start of curves (first 1k trials): ' + ', '.join(
        f'{BEH[r].mean(-1).mean(0)[:10].mean():.3f}' for r in REWARDS))

# =====================================================================================
# Attractor runs
# =====================================================================================
ATT = {}
for r in REWARDS:
    d = np.load(f'{C}/att_s{r}.npz')
    ATT[r] = {k: d[k] for k in ('W', 'x0', 'X', 'A', 'W0')}
NT = ATT[0.2]['W'].shape[0]
SS = slice(NT - 50_000 // ST, NT)                     # steady-state window: last 50k trials


def cross_section_slope(xi, W, X):
    lo, hi = np.percentile(xi, [25, 75])
    m = (xi >= lo) & (xi <= hi) & (W > 0)
    return float(np.sum(W[m] * X[m]) / np.sum(W[m] ** 2)), m


SLOPE, DRIVE, WDRIVE = {}, {}, {}
for r in REWARDS:
    xi, W, X = (ATT[r][k][SS].ravel() for k in ('x0', 'W', 'X'))
    SLOPE[r], msk = cross_section_slope(xi, W, X)
    DRIVE[r] = float(xi[msk].mean()); WDRIVE[r] = float(np.sum(W[msk] ** 2 * xi[msk]) / np.sum(W[msk] ** 2))

# =====================================================================================
# §4-6  Constants of the operating point
# =====================================================================================
SIGMA_DEC = np.sqrt(2) * ETA
SIGMA_OU = np.sqrt(2) * ETA / np.sqrt(1 - (1 - 2 * ALPHA) ** 2)
if want('const'):
    P('\n=== §4-6 constants ===')
    P(f'alpha={ALPHA:.6g} beta={BETA:.6g} eta={ETA} beta/alpha={BETA/ALPHA:.4f} 1/2alpha={1/(2*ALPHA):.2f}')
    for s in (0.2, 1.0):
        P(f'  s={s}: contracting eigenvalues -2a = {-2*ALPHA:.4f}, -(2a+b s^2) = {-(2*ALPHA+BETA*s*s):.4f}; '
          f'beta*s = {BETA*s:.4f} vs 2alpha = {2*ALPHA:.4f}')
    P(f'sigma_dec = sqrt2 eta = {SIGMA_DEC:.5f};  sigma_OU = {SIGMA_OU:.5f} '
      f'(sqrt(1-(1-2a)^2) = {np.sqrt(1-(1-2*ALPHA)**2):.4f}; ratio to one kick {SIGMA_OU/SIGMA_DEC:.2f})')
    P(f'max activity in any run: see n1_run logs')
    for r in REWARDS:
        eps = (ATT[r]['X'] - ATT[r]['x0'] * ATT[r]['W'])[SS]
        P(f'  s={r}: sd of off-line deviation X - x0 W at steady state = {eps.std():.5f}')

# =====================================================================================
# §5  Steady-state clouds and slopes (Figs 5-6)
# =====================================================================================
CLOUD_R = [0.2, 0.5, 1.0]
if want('cloud'):
    P('\n=== §5 Figs 5-6: steady-state slope (last 50k trials, 200 seeds) ===')
    P(f"{'s':>4} {'slope':>7} {'<x0>':>7} {'W2-wtd x0':>10} {'slope/wtd':>10} {'slope/s':>8}")
    for r in REWARDS:
        P(f'{r:4.1f} {SLOPE[r]:7.3f} {DRIVE[r]:7.3f} {WDRIVE[r]:10.3f} {SLOPE[r]/WDRIVE[r]:10.3f} {SLOPE[r]/r:8.3f}')
    # during learning: drive vs behaviour, binned in m = s W
    P('\nduring learning, <x0>/s against P(optimal) in bins of m = s*W (fit seeds, all times):')
    edges = np.array([-0.2, -0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05])
    for r in REWARDS:
        m = (r * ATT[r]['W'])[:, :NFIT].ravel(); x0 = ATT[r]['x0'][:, :NFIT].ravel()
        a = ATT[r]['A'][:, :NFIT].ravel(); idx = np.digitize(m, edges) - 1
        row = [(x0[idx == k].mean() / r, a[idx == k].mean()) for k in range(len(edges) - 1)
               if (idx == k).sum() > 2000]
        P(f'  s={r}: ' + '  '.join(f'{u:.2f}/{p:.2f}' for u, p in row))

    # Fig 5: 3-D clouds with planes
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(3.3, 2.6)); ax = fig.add_subplot(111, projection='3d')
    cl = {r: tuple(ATT[r][k][SS][:, :60].ravel() for k in ('x0', 'W', 'X')) for r in CLOUD_R}
    for r in CLOUD_R:
        xi, W, X = cl[r]; idx = rng.choice(len(xi), 2000, replace=False)
        ax.scatter(xi[idx], W[idx], X[idx], s=3, c=[reward_color_abs(r)], alpha=0.5, linewidths=0,
                   rasterized=True, label=f'$s$={r}')
    Whi = np.percentile(np.concatenate([cl[r][1] for r in CLOUD_R]), 99)
    Xhi = np.percentile(np.concatenate([cl[r][2] for r in CLOUD_R]), 99)
    WW, XX = np.meshgrid([0.0, Whi], [0.0, Xhi])
    for r in CLOUD_R:
        c = float(np.median(cl[r][0]))
        ax.plot_surface(np.full_like(WW, c), WW, XX, color=reward_color_abs(r), alpha=0.20, linewidth=0, shade=False)
        ax.plot([c] * 5, [0, Whi, Whi, 0, 0], [0, 0, Xhi, Xhi, 0], color=reward_color_abs(r), lw=1.4)
    ax.legend(fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0), markerscale=3)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor('#CCCCCC'); pane.set_linewidth(0.25)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(mticker.MaxNLocator(3)); axis.line.set_linewidth(0.25)
    ax.grid(False); ax.tick_params(labelsize=7, pad=-2, colors='#555555', width=0.25)
    ax.set_xlabel('$x_{in}$', fontsize=9, labelpad=-5); ax.set_ylabel('$w_2-w_1$', fontsize=9, labelpad=-5)
    ax.set_zlabel('$x_2-x_1$', fontsize=9, labelpad=-3); ax.view_init(elev=16, azim=-60)
    fig.subplots_adjust(left=0.04, right=0.74, top=1.0, bottom=0.0)
    plt.savefig(f'{F}/n1_clouds_planes.svg', dpi=300); plt.close()

    # Fig 6: cross-sections
    fig, ax = plt.subplots(figsize=(2.6, 1.75))
    for r in CLOUD_R:
        xi, W, X = (ATT[r][k][SS].ravel() for k in ('x0', 'W', 'X'))
        sl, msk = cross_section_slope(xi, W, X)
        sub = np.flatnonzero(msk); sub = np.random.default_rng(1).choice(sub, min(30000, len(sub)), replace=False)
        ax.scatter(W[sub], X[sub], s=2, c=[reward_color_abs(r)], alpha=0.03, linewidths=0, rasterized=True)
        xs = np.linspace(0, np.percentile(W[msk], 99), 20)
        ax.plot(xs, sl * xs, color=reward_color_abs(r), lw=1.5, label=f'$s$={r}:  {sl:.2f}')
    ax.axhline(0, color='0.85', lw=0.5, zorder=0); ax.axvline(0, color='0.85', lw=0.5, zorder=0)
    nature_ax(ax, '$w_2-w_1$', '$x_2-x_1$'); side_legend(ax, title='cross-section slope')
    plt.tight_layout(); plt.savefig(f'{F}/n1_crosssections.svg', dpi=300, bbox_inches='tight'); plt.close()

# =====================================================================================
# §8  Drift and diffusion
# =====================================================================================
W_EDGES = np.unique(np.concatenate([np.arange(-0.70, -0.05, 0.05), np.arange(-0.05, 0.10, 0.01),
                                    np.arange(0.10, 1.0001, 0.05)]).round(4))


def drift_diffusion(W, edges=W_EDGES, lag=LAG, lag_steps=LAG_STEPS, min_n=1500):
    W0 = W[:-lag].ravel(); d = (W[lag:] - W[:-lag]).ravel()
    ctr = (edges[1:] + edges[:-1]) / 2; idx = np.digitize(W0, edges) - 1
    v = np.full(len(ctr), np.nan); D = np.full(len(ctr), np.nan); n = np.zeros(len(ctr), int)
    for k in range(len(ctr)):
        sel = idx == k; n[k] = sel.sum()
        if n[k] >= min_n:
            v[k] = d[sel].mean() / lag_steps; D[k] = d[sel].var() / (2 * lag_steps)
    return ctr, v, D, n


DD = {r: drift_diffusion(ATT[r]['W'][:, :NFIT]) for r in REWARDS}


def drift_in_m(r, W=None, A=None, edges=None, lag=LAG, min_n=1500):
    """Drift of m = slope*W (slope = s to three decimals), binned in m (fit seeds)."""
    W = ATT[r]['W'][:, :NFIT] if W is None else W
    edges = np.array([-0.3, -0.1, -0.05, -0.03, -0.02, -0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015,
                      0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.4]) if edges is None else edges
    m = SLOPE[r] * W
    m0 = m[:-lag].ravel(); dm = ((m[lag:] - m[:-lag]) / (lag * ST)).ravel()
    idx = np.digitize(m0, edges) - 1; ctr = (edges[1:] + edges[:-1]) / 2
    vm = np.array([dm[idx == k].mean() if (idx == k).sum() >= min_n else np.nan for k in range(len(ctr))])
    return ctr, vm


if want('dd'):
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.55), gridspec_kw=dict(wspace=0.65))
    for r in REWARDS:
        ctr, v, D, _ = DD[r]; g = np.isfinite(v)
        axes[0].plot(ctr[g], v[g] * 1e5, color=reward_color_abs(r), lw=1.0)
        axes[1].plot(ctr[g], D[g] * 1e8, color=reward_color_abs(r), lw=1.0)
        cm, vm = drift_in_m(r); gm = np.isfinite(vm)
        axes[2].plot(cm[gm], vm[gm] * 1e5, '-o', ms=1.6, mew=0, color=reward_color_abs(r), lw=0.9, label=f'{r}')
    for ax in axes:
        ax.axhline(0, color='0.8', lw=0.5, zorder=0); ax.axvline(0, color='0.8', lw=0.5, zorder=0)
    nature_ax(axes[0], '$W = w_2-w_1$', r'Drift $v(W)$' '\n' r'($\times 10^{-5}$/trial)')
    nature_ax(axes[1], '$W = w_2-w_1$', r'Diffusion $D(W)$' '\n' r'($\times 10^{-8}$/trial)')
    nature_ax(axes[2], r'$m = sW$', r'Drift of $m$' '\n' r'($\times 10^{-5}$/trial)')
    axes[2].set_xlim(-0.1, 0.1)
    side_legend(axes[2], title='Reward')
    plt.tight_layout(); plt.savefig(f'{F}/n1_drift_diffusion.svg', dpi=300, bbox_inches='tight'); plt.close()

    P('\n=== §8 drift and diffusion (fit seeds, lag 400) ===')
    for r in REWARDS:
        ctr, v, D, n = DD[r]
        wrong = (ctr < -0.05) & np.isfinite(v); right = (ctr > 0.1) & np.isfinite(v)
        P(f'  s={r}: wrong side (W<-0.05) v = {np.nanmean(v[wrong])*1e5:.3f}e-5 '
          f'[{np.nanmin(v[wrong])*1e5:.2f}-{np.nanmax(v[wrong])*1e5:.2f}], D = {np.nanmean(D[wrong])*1e8:.2f}e-8; '
          f'right side (W>0.1) v = {np.nanmean(v[right])*1e5:+.3f}e-5, D = {np.nanmean(D[right])*1e8:.2f}e-8')
    vw = np.array([np.nanmean(DD[r][1][(DD[r][0] < -0.05) & (DD[r][0] > -0.3)]) for r in REWARDS])
    P(f'  wrong-side drift (-0.3<W<-0.05) vs s: exponent {np.polyfit(np.log(REWARDS), np.log(vw), 1)[0]:.2f}; '
      f'v/s = ' + ', '.join(f'{x*1e5/s:.2f}' for x, s in zip(vw, REWARDS)) + ' (x1e-5)')
    P('  drift of m (x1e-5/trial) by m bin:')
    cm, _ = drift_in_m(0.2)
    P('   m:    ' + ' '.join(f'{c:+.4f}' for c in cm))
    for r in REWARDS:
        P(f'   s={r}: ' + ' '.join(f'{x*1e5:+7.3f}' if np.isfinite(x) else '    nan' for x in drift_in_m(r)[1]))
    # where does the drift of m vanish?  first m (from the left) at which v_m drops below 5% of its wrong-side value
    P('  m at which the drift falls to 10% / 0 of its wrong-side value:')
    fine = np.arange(-0.05, 0.06, 0.0025)
    for r in REWARDS:
        c, vm = drift_in_m(r, edges=fine, min_n=300)
        ref = np.nanmean(vm[c < -0.01]); g = np.isfinite(vm) & (c > 0)
        m10 = c[g][np.argmax(vm[g] < 0.1 * ref)]; m0 = c[g][np.argmax(vm[g] <= 0)]
        P(f'   s={r}: wrong-side v_m {ref*1e5:.3f}e-5; 10% at m = {m10:.4f}; zero at m = {m0:.4f}; '
          f'P(m) there = {np.mean(ATT[r]["A"][:, :NFIT].ravel()[np.abs((SLOPE[r]*ATT[r]["W"][:, :NFIT]).ravel()-m0)<0.00125]):.3f}')

# =====================================================================================
# §9  Behaviour against m = sW and against the output difference X
# =====================================================================================


def pooled_curve(coord, action, n_bins=40, min_n=500, edges=None):
    e = edges if edges is not None else np.unique(np.percentile(coord, np.linspace(0.5, 99.5, n_bins + 1)))
    c = (e[1:] + e[:-1]) / 2; idx = np.digitize(coord, e) - 1
    y = np.array([action[idx == k].mean() if (idx == k).sum() >= min_n else np.nan for k in range(len(c))])
    return c, y


def best_width(c, y):
    g = np.geomspace(0.0005, 0.3, 1500)
    rms = [np.sqrt(np.nanmean((y - norm.cdf(c / s)) ** 2)) for s in g]
    k = int(np.argmin(rms)); return g[k], rms[k]


M_EDGES = np.linspace(-0.06, 0.06, 49)
m_fit = np.concatenate([(SLOPE[r] * ATT[r]['W'])[:, :NFIT].ravel() for r in REWARDS])
a_fit = np.concatenate([ATT[r]['A'][:, :NFIT].ravel() for r in REWARDS])
MC, PCm = pooled_curve(m_fit, a_fit, edges=M_EDGES)
_ok = np.isfinite(PCm); MC, PCm = MC[_ok], PCm[_ok]
P_of_m = lambda m: np.interp(m, MC, PCm, left=PCm[0], right=PCm[-1])

if want('collapse'):
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 1.75), gridspec_kw=dict(wspace=0.55))
    for r in REWARDS:
        c, y = pooled_curve((SLOPE[r] * ATT[r]['W']).ravel(), ATT[r]['A'].ravel(), edges=np.linspace(-0.06, 0.06, 25))
        g = np.isfinite(y); axes[0].plot(c[g], y[g], 'o', ms=2.6, color=reward_color_abs(r), mew=0)
    axes[0].plot(MC, PCm, 'k-', lw=1.0, zorder=3)
    nature_ax(axes[0], r'$m = sW$', 'P(optimal)'); axes[0].set_xlim(-0.06, 0.06)
    X_all = np.concatenate([ATT[r]['X'].ravel() for r in REWARDS]); A_all = np.concatenate([ATT[r]['A'].ravel() for r in REWARDS])
    for r in REWARDS:
        c, y = pooled_curve(ATT[r]['X'].ravel(), ATT[r]['A'].ravel(), edges=np.linspace(-6 * SIGMA_DEC, 6 * SIGMA_DEC, 25))
        g = np.isfinite(y); axes[1].plot(c[g], y[g], 'o', ms=2.6, color=reward_color_abs(r), mew=0, label=f'{r}')
    xx = np.linspace(-6 * SIGMA_DEC, 6 * SIGMA_DEC, 200)
    axes[1].plot(xx, norm.cdf(xx / SIGMA_DEC), 'k-', lw=1.2, zorder=3)
    axes[1].set_xlim(-6 * SIGMA_DEC, 6 * SIGMA_DEC)
    nature_ax(axes[1], '$X = x_2-x_1$', 'P(optimal)', nx=3)
    for ax in axes:
        ax.axhline(0.5, color='0.85', lw=0.4, zorder=0); ax.axvline(0, color='0.85', lw=0.4, zorder=0); ax.set_ylim(-0.02, 1.02)
    side_legend(axes[1], title='Reward')
    plt.tight_layout(); plt.savefig(f'{F}/n1_collapse.svg', dpi=300, bbox_inches='tight'); plt.close()

    def collapse_rms(key, edges):
        crd = (lambda r: SLOPE[r] * ATT[r]['W']) if key == 'W' else (lambda r: ATT[r]['X'])
        allc = np.concatenate([crd(r)[:, :NFIT].ravel() for r in REWARDS])
        shared = pooled_curve(allc, a_fit, edges=edges)[1]
        dev = []
        for r in REWARDS:
            y = pooled_curve(crd(r)[:, :NFIT].ravel(), ATT[r]['A'][:, :NFIT].ravel(), edges=edges)[1]
            g = np.isfinite(y) & np.isfinite(shared); dev += list(y[g] - shared[g])
        return float(np.sqrt(np.mean(np.square(dev)))), len(dev)

    rW = collapse_rms('W', np.linspace(-0.06, 0.06, 31)); rX = collapse_rms('X', np.linspace(-6 * SIGMA_DEC, 6 * SIGMA_DEC, 31))
    cx, yx = pooled_curve(X_all, A_all, edges=np.linspace(-6 * SIGMA_DEC, 6 * SIGMA_DEC, 49)); gx = np.isfinite(yx)
    wX = best_width(cx[gx], yx[gx]); wM = best_width(MC, PCm)
    P('\n=== §9 behaviour curves ===')
    P(f'collapse RMS across rewards: P vs m = {rW[0]:.4f} ({rW[1]} bins);  P vs X = {rX[0]:.4f} ({rX[1]} bins)')
    P(f'width vs X: predicted sqrt2 eta = {SIGMA_DEC:.5f}, best fit {wX[0]:.5f} (ratio {wX[0]/SIGMA_DEC:.2f}, RMS {wX[1]:.4f})')
    P(f'width vs m (pooled): {wM[0]:.5f} (RMS {wM[1]:.4f}) = {wM[0]/wX[0]:.2f} x the decision width; sigma_OU = {SIGMA_OU:.5f}')
    for r in REWARDS:
        c, y = pooled_curve((SLOPE[r] * ATT[r]['W'])[:, :NFIT].ravel(), ATT[r]['A'][:, :NFIT].ravel(), edges=M_EDGES)
        g = np.isfinite(y); w = best_width(c[g], y[g])
        P(f'   per-reward width in m, s={r}: {w[0]:.5f}')
    for mm in (0.01, 0.02, 0.03):
        P(f'  pooled P(m={mm}) = {P_of_m(mm):.4f}')

# =====================================================================================
# §10  Fokker-Planck prediction
# =====================================================================================


def fp_integrate(grid, v, D, rho0, t_eval, cfl=0.4):
    h = grid[1] - grid[0]
    dt = cfl * min(h * h / (2 * max(D.max(), 1e-14)), h / max(np.abs(v).max(), 1e-14))
    vf = 0.5 * (v[1:] + v[:-1]); rho = rho0.copy(); want_t = sorted(t_eval); out, wi, T = {}, 0, 0.0
    while wi < len(want_t):
        while wi < len(want_t) and T >= want_t[wi] - dt / 2:
            out[want_t[wi]] = rho.copy(); wi += 1
        if wi >= len(want_t):
            break
        adv = np.where(vf > 0, vf * rho[:-1], vf * rho[1:])
        dif = -(D[1:] * rho[1:] - D[:-1] * rho[:-1]) / h
        Fx = np.zeros(len(rho) + 1); Fx[1:-1] = adv + dif
        rho = rho - dt * (Fx[1:] - Fx[:-1]) / h; T += dt
    return out, dt


GRID = np.arange(-1.2, 1.2, 0.0025)
T_EVAL = [2000, 5000, 10_000, 20_000, 40_000, 80_000, 150_000, 300_000]
RHO_SD = 0.01


def fp_run(r, times, lag=LAG, lag_steps=LAG_STEPS):
    ctr, v, D, _ = drift_diffusion(ATT[r]['W'][:, :NFIT], lag=lag, lag_steps=lag_steps)
    g = np.isfinite(v) & np.isfinite(D)
    v_g = np.interp(GRID, ctr[g], v[g]); D_g = np.maximum(np.interp(GRID, ctr[g], D[g]), 1e-12)
    W0 = ATT[r]['W'][0, NFIT:]
    rho0 = np.exp(-0.5 * ((GRID[:, None] - W0[None, :]) / RHO_SD) ** 2).sum(1); rho0 /= rho0.sum() * (GRID[1] - GRID[0])
    return fp_integrate(GRID, v_g, D_g, rho0, times)


def measured(r, t):
    Ah = ATT[r]['A'][:, NFIT:]; half = 500 // ST
    i0 = min(max(0, t // ST - half), Ah.shape[0] - 2 * half)
    return Ah[i0:i0 + 2 * half].mean()


if want('fp'):
    pred, meas = {}, {}
    for r in REWARDS:
        rh, FP_DT = fp_run(r, T_EVAL)
        pred[r] = np.array([np.trapezoid(rh[t] * P_of_m(SLOPE[r] * GRID), GRID) for t in T_EVAL])
        meas[r] = np.array([measured(r, t) for t in T_EVAL])
    fig, ax = plt.subplots(figsize=(2.7, 1.75))
    for r in REWARDS:
        ax.plot(T_EVAL, meas[r], 'o', ms=3, color=reward_color_abs(r), mew=0, zorder=3)
        tt = sorted(set(list(range(0, 300_001, 5000)) + T_EVAL)); tt[0] = 1000
        rh, _ = fp_run(r, tt)
        ax.plot(tt, [np.trapezoid(rh[t] * P_of_m(SLOPE[r] * GRID), GRID) for t in tt], '-', lw=1.0, color=reward_color_abs(r))
    nature_ax(ax, 'Trial', 'P(optimal)'); ax.set_xlim(0, 300_000); ax.set_ylim(0.48, 1.01)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(100_000)); kfmt(ax)
    side_legend(ax, handles=[plt.Line2D([], [], color=reward_color_abs(r), lw=1.4, label=f'{r}') for r in REWARDS]
                + [plt.Line2D([], [], color='0.3', marker='o', ms=3, lw=0, label='measured'),
                   plt.Line2D([], [], color='0.3', lw=1.0, label='Fokker–Planck')], title='Reward')
    plt.tight_layout(); plt.savefig(f'{F}/n1_fp_reconstruction.svg', dpi=300, bbox_inches='tight'); plt.close()
    P('\n=== §10 Fokker-Planck, held-out seeds: measured/predicted ===')
    P(f"{'s':>4} " + ' '.join(f'{t//1000:>10}k' for t in T_EVAL))
    for r in REWARDS:
        P(f'{r:4.1f} ' + ' '.join(f'{a:.3f}/{b:.3f}' for a, b in zip(meas[r], pred[r])))
    err = np.concatenate([pred[r] - meas[r] for r in REWARDS])
    rh, FP_DT = fp_run(1.0, [300_000])
    P(f'RMS {np.sqrt(np.mean(err**2)):.4f}, max |err| {np.abs(err).max():.4f}; timestep {FP_DT:.1f} trials; '
      f'mass error {abs(np.trapezoid(rh[300_000], GRID)-1):.1e}')
    P('predicted asymptote (300k) per reward: ' + ', '.join(f'{pred[r][-1]:.3f}' for r in REWARDS))

# self-consistent behaviour curve: the output difference is x0*W = s*P*W, so P(m) should solve P = Phi(P m / sigma)
if want('collapse'):
    P('\nself-consistent curve P = Phi(P*m/sigma_OU) vs pooled P(m):')
    from scipy.optimize import brentq
    def P_sc(m, sig):
        f = lambda p: p - norm.cdf(p * m / sig)
        return brentq(f, 1e-9, 1.0)
    for mm in (-0.03, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02, 0.03):
        P(f'   m={mm:+.3f}: pooled {P_of_m(mm):.3f}   self-consistent {P_sc(mm, SIGMA_OU):.3f}')
    g = (MC > -0.05) & (MC < 0.05)
    sc = np.array([P_sc(m, SIGMA_OU) for m in MC[g]])
    P(f'   RMS over |m|<0.05: {np.sqrt(np.mean((sc-PCm[g])**2)):.4f}  (no free parameter)')

# =====================================================================================
# §11  Checks on the one-dimensional reduction (Fig 10)
# =====================================================================================


def ks_stat(rho, sample, grid=GRID):
    h = grid[1] - grid[0]; cdf = np.cumsum(rho) * h; cdf /= cdf[-1]; s = np.sort(sample)
    return float(np.abs(np.interp(s, grid, cdf) - np.arange(1, len(s) + 1) / len(s)).max())


if want('checks'):
    DENS_R = [0.2, 0.5, 1.0]; DENS_T = [10_000, 40_000, 100_000, 300_000]
    fig, axes = plt.subplots(3, 4, figsize=(6.4, 2.9), sharex=True, gridspec_kw=dict(hspace=0.55, wspace=0.35))
    bins = np.arange(-1.0, 1.0001, 0.04); ks_all = []
    for i, r in enumerate(DENS_R):
        rh, _ = fp_run(r, DENS_T); Wh = ATT[r]['W'][:, NFIT:]
        for j, t in enumerate(DENS_T):
            ax = axes[i, j]; mw = Wh[min(t // ST, Wh.shape[0]) - 1]
            ax.hist(mw, bins=bins, density=True, color=reward_color_abs(r), alpha=0.45, lw=0)
            # display the density smoothed to the histogram bin width
            k = int(round(0.04 / (GRID[1] - GRID[0]))); sm = np.convolve(rh[t], np.ones(k) / k, 'same')
            ax.plot(GRID, sm, 'k-', lw=0.9)
            kv = ks_stat(rh[t], mw); ks_all.append(kv)
            ax.set_title(f'$s$={r}, {t//1000}k   KS={kv:.2f}', fontsize=6.5, pad=2)
            nature_ax(ax, '$W$' if i == 2 else '', 'density' if j == 0 else '', nx=3, ny=3)
            ax.set_xlim(-1.0, 1.0)
    plt.savefig(f'{F}/n1_fp_density.svg', dpi=300, bbox_inches='tight'); plt.close()
    P('\n=== §11 checks ===')
    P(f'CHECK 1 density: {len(ks_all)} panels, KS max {max(ks_all):.3f}, mean {np.mean(ks_all):.3f}; '
      f'n=80 critical value {1.36/np.sqrt(80):.3f}; per panel ' + ' '.join(f'{k:.3f}' for k in ks_all))
    P('CHECK 2 lag:  lag   P RMS   FP sd / measured sd @300k')
    for lag_s in (100, 200, 400, 800, 1600, 3200):
        errs, sdr = [], []
        for r in REWARDS:
            rh, _ = fp_run(r, T_EVAL, lag=lag_s // ST, lag_steps=lag_s)
            pr = np.array([np.trapezoid(rh[t] * P_of_m(SLOPE[r] * GRID), GRID) for t in T_EVAL])
            errs.append(pr - np.array([measured(r, t) for t in T_EVAL]))
            rf = rh[T_EVAL[-1]]; fm = np.trapezoid(rf * GRID, GRID); fs = np.sqrt(np.trapezoid(rf * (GRID - fm) ** 2, GRID))
            sdr.append(fs / ATT[r]['W'][-1, NFIT:].std())
        e = np.concatenate(errs)
        P(f'      {lag_s:5d}  {np.sqrt((e**2).mean()):.4f}   {np.mean(sdr):.3f}')
    P('CHECK 3 time homogeneity (first vs last third of 0-150k, wrong side and boundary bins):')
    rng = np.random.default_rng(0)
    for r in REWARDS:
        W = ATT[r]['W'][:15000, :NFIT]; n = W.shape[0]
        _, ve, _, _ = drift_diffusion(W[:n // 3], min_n=400); _, vl, _, _ = drift_diffusion(W[2 * n // 3:], min_n=400)
        g = np.isfinite(ve) & np.isfinite(vl)
        scale = np.max(np.abs(np.concatenate([ve[g], vl[g]]))); rel = np.median(np.abs(ve[g] - vl[g])) / scale
        sh = rng.permutation(NFIT); early = W[:n // 3]
        _, va, _, _ = drift_diffusion(early[:, sh[:NFIT // 2]], min_n=400); _, vb, _, _ = drift_diffusion(early[:, sh[NFIT // 2:]], min_n=400)
        g2 = np.isfinite(va) & np.isfinite(vb); nf = np.median(np.abs(va[g2] - vb[g2])) / np.max(np.abs(np.concatenate([va[g2], vb[g2]])))
        P(f'   s={r}: bins {g.sum()}, |dv|/max|v| {rel:.3f}, noise floor {nf:.3f} -> '
          f'{"within sampling noise" if rel <= nf else f"{rel/nf:.1f}x the noise floor"}')
    P('CHECK 4a increment autocorrelation (lags in trials):')
    lags = [1, 2, 4, 10, 20, 40]
    for r in (0.5, 1.0):
        inc = np.diff(ATT[r]['W'][:, :NFIT], axis=0); inc = inc - inc.mean(0, keepdims=True)
        P(f'   s={r}: ' + ' '.join(f'{k*ST}:{(inc[:-k]*inc[k:]).mean()/inc.var():.3f}' for k in lags))
    P('CHECK 4b matched W, low vs high recent reward rate (trailing 1000 trials), wrong side W in [-0.3,-0.1]:')
    H = 1000 // ST
    for r in (0.3, 0.5, 1.0):
        W = ATT[r]['W'][:, :NFIT]; A = ATT[r]['A'][:, :NFIT].astype(np.float64)
        cs = np.cumsum(np.vstack([np.zeros((1, A.shape[1])), A]), 0); rr = (cs[H:-1] - cs[:-H - 1]) / H
        Wv = W[H:-LAG].ravel(); dv = ((W[H + LAG:] - W[H:-LAG]) / LAG_STEPS).ravel(); rv = rr[:-LAG].ravel()
        sel = (Wv > -0.3) & (Wv < -0.1)
        q = np.percentile(rv[sel], [33, 67]); lo = rv[sel] < q[0]; hi = rv[sel] > q[1]
        vlo, vhi = dv[sel][lo].mean(), dv[sel][hi].mean()
        P(f'   s={r}: rr {rv[sel][lo].mean():.2f} vs {rv[sel][hi].mean():.2f}; v {vlo*1e5:.3f}e-5 vs {vhi*1e5:.3f}e-5; '
          f'(hi-lo)/mean = {(vhi-vlo)/abs((vhi+vlo)/2):.2f}')

# =====================================================================================
# §12  The two-factor law
# =====================================================================================
if want('law'):
    P('\n=== §13 Table 1: drift at matched behaviour ===')
    P(f"{'s':>4} {'slope':>6} {'drive':>6} {'P(opt)':>7} {'v_W (1e-5)':>11} {'slope*v (1e-5)':>15}   [band -0.04 < m < -0.01]")
    v0, dr0 = [], []
    for r in REWARDS:
        W = ATT[r]['W'][:, :NFIT]; sl = SLOPE[r]; m = (sl * W)[:-LAG].ravel()
        dW = ((W[LAG:] - W[:-LAG]) / LAG_STEPS).ravel(); x0 = ATT[r]['x0'][:, :NFIT][:-LAG].ravel()
        a = ATT[r]['A'][:, :NFIT][:-LAG].ravel()
        sel = (m > -0.04) & (m < -0.01)
        v0.append(dW[sel].mean()); dr0.append(x0[sel].mean())
        P(f'{r:4.1f} {sl:6.3f} {dr0[-1]:6.3f} {a[sel].mean():7.3f} {v0[-1]*1e5:11.3f} {sl*v0[-1]*1e5:15.3f}   (n={sel.sum()})')
    sl_a = np.array([SLOPE[r] for r in REWARDS]); v0 = np.array(v0); dr0 = np.array(dr0)
    ex = lambda x, y: np.polyfit(np.log(x), np.log(y), 1)[0]
    P(f'  v ~ slope^{ex(sl_a, v0):.2f};  slope*v ~ slope^{ex(sl_a, sl_a*v0):.2f};  v ~ drive^{ex(dr0, v0):.2f};  '
      f'slope*v ~ drive^{ex(dr0, sl_a*v0):.2f}; drive/s = ' + ', '.join(f'{d/s:.3f}' for d, s in zip(dr0, REWARDS)))

# =====================================================================================
# §13  Noise sweep (Table 2) and §14 population
# =====================================================================================


def run_metrics(d, r, eta):
    W, A = d['W'], d['A']
    ssW, ssx, ssX = d['W'][-5000:].ravel(), d['x0'][-5000:].ravel(), d['X'][-5000:].ravel()
    sl, _ = cross_section_slope(ssx, ssW, ssX)
    m = sl * W
    c, y = pooled_curve(m.ravel(), A.ravel(), edges=np.linspace(-15 * eta, 15 * eta, 49), min_n=300)
    g = np.isfinite(y); w = best_width(c[g], y[g])[0]
    dW = ((W[LAG:] - W[:-LAG]) / LAG_STEPS); mm = m[:-LAG]
    sel = (mm > -2.0 * w) & (mm < -0.5 * w)
    num, den = (dW * sel).sum(0), sel.sum(0); v = num.sum() / den.sum()
    rng = np.random.default_rng(0); bs = []
    for _ in range(300):
        i = rng.integers(0, len(num), len(num)); bs.append(num[i].sum() / den[i].sum())
    # population: bins of 1000 trials from the every-10th-trial record
    curve = A.reshape(-1, 100, A.shape[1]).mean(1).mean(1); tt = (np.arange(len(curve)) + 1) * 1000
    Aas = curve[-len(curve) // 10:].mean(); tc = tt[np.argmax(curve >= 0.9 * Aas)]
    return dict(slope=sl, w=w, v=v, v_se=float(np.std(bs)), rate=sl * v * norm.pdf(0) / w * 1e4,
                pop=1e4 / tc, asym=Aas, tc=tc)


if want('noise'):
    P('\n=== §13 Table 2: noise sweep (200 seeds x 300k trials) ===')
    P(f"{'s':>4} {'eta':>7} {'slope':>6} {'v (1e-5)':>14} {'width w':>8} {'net. rate (1e-4)':>17} {'pop. rate':>9} {'t_crit':>8} {'asym':>6}")
    RES = {}
    for r in (0.2, 0.5, 1.0):
        for eta in (0.0015, 0.003, 0.006):
            f = f'{C}/att_s{r}.npz' if eta == 0.003 else f'{C}/noise_s{r}_e{eta}.npz'
            d = np.load(f); d = {k: d[k] for k in ('W', 'x0', 'X', 'A')}
            if eta == 0.003:
                pass
            q = run_metrics(d, r, eta); RES[(r, eta)] = q
            P(f"{r:4.1f} {eta:7.4f} {q['slope']:6.3f} {q['v']*1e5:7.3f} ± {q['v_se']*1e5:.3f} {q['w']:8.4f} "
              f"{q['rate']:17.3f} {q['pop']:9.3f} {q['tc']:8,d} {q['asym']:6.3f}")
    E = np.array([0.0015, 0.003, 0.006]); ex = lambda y: np.polyfit(np.log(E), np.log(y), 1)[0]
    P('exponents in eta (0 = noise-independent):')
    for r in (0.2, 0.5, 1.0):
        col = lambda k: np.array([RES[(r, e)][k] for e in E])
        P(f"  s={r}: v {ex(col('v')):.2f}  width {ex(col('w')):.2f}  network rate {ex(col('rate')):.2f}  "
          f"population rate {ex(col('pop')):.2f};  network rate range {col('rate').min():.3f}-{col('rate').max():.3f} "
          f"({(col('rate').max()/col('rate').min()-1)*100:.0f}%)")
    P('exponent of the network rate in the slope, s=0.2 -> 1.0, at each eta:')
    for e in E:
        sl = np.array([RES[(r, e)]['slope'] for r in (0.2, 0.5, 1.0)]); rt = np.array([RES[(r, e)]['rate'] for r in (0.2, 0.5, 1.0)])
        pp = np.array([RES[(r, e)]['pop'] for r in (0.2, 0.5, 1.0)])
        P(f"  eta={e}: network {np.polyfit(np.log(sl), np.log(rt), 1)[0]:.2f}   population {np.polyfit(np.log(sl), np.log(pp), 1)[0]:.2f}")

if want('pop'):
    P('\n=== §13 population ===')
    W0 = ATT[1.0]['W0']; P(f'initial W: sd {W0.std():.3f}, fraction W<0 {np.mean(W0<0):.3f}, mean |W| of W<0 seeds {np.abs(W0[W0<0]).mean():.3f}')
    for r in REWARDS:
        W = ATT[r]['W']; neg = W[0] < 0
        # crossing time: first record with W > 0.02/s (behaviour saturated)
        thr = 0.02 / SLOPE[r]
        cross = np.array([(np.argmax(W[:, j] > thr) + 1) * ST if (W[:, j] > thr).any() else np.nan for j in range(W.shape[1])])
        pos = ~neg
        early = ATT[r]['A'][:200]                     # first 2000 trials
        P(f"  s={r}: start-W>0 seeds P(first 2k) {early[:, pos].mean():.2f}, start-W<0 seeds {early[:, neg].mean():.2f};"
          f" W<0 seeds reach m=0.02 after median {np.nanmedian(cross[neg]):,.0f} trials;"
          f" predicted |W0|/v_wrong median {np.median(np.abs(W[0, neg]) / np.nanmean(DD[r][1][(DD[r][0] < -0.05) & (DD[r][0] > -0.3)])):,.0f};"
          f" W>0 seeds below m=0.02 at start: {np.mean(W[0, pos] < thr):.2f}")

# =====================================================================================
# §9  Why the population rate is set by the time to reach the boundary (Fig. 11)
# =====================================================================================
if want('pop'):
    P('\n=== §9 population: crossing the boundary (Fig. 11) ===')
    TCm = {}
    for r in REWARDS:
        TCm[r] = np.mean([metrics(BEH[r][k].mean(-1))[1] for k in range(BEH[r].shape[0])])
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.7), gridspec_kw=dict(wspace=0.45))
    for r in REWARDS:
        W = ATT[r]['W']; A = ATT[r]['A']; neg = W[0] < 0
        t = (np.arange(W.shape[0]) + 1) * ST
        crossed = (W[:, neg] > 0).mean(1)
        k = int(TCm[r] // ST) - 1
        for ax, x in ((axes[0], t), (axes[1], t * r)):
            ax.plot(x, crossed, color=reward_color_abs(r), lw=1.0, label=f'{r}')
            ax.plot(x[k], crossed[k], 'o', color=reward_color_abs(r), ms=3.2, mec='k', mew=0.4, zorder=3)
        done = (W[k, neg] > 0.02 / SLOPE[r]).mean()
        Pneg = A[max(0, k - 50):k + 50][:, neg].mean()
        P(f'  s={r}: criterion at {TCm[r]:,.0f} trials (x reward {TCm[r]*r:,.0f}); wrong-side networks past the boundary '
          f'{crossed[k]:.2f}, past m=0.02 {done:.2f}, their P(optimal) {Pneg:.2f}')
    for ax in axes:
        ax.set_ylim(0, 1.02); ax.axhline(0.8, color='0.85', lw=0.5, ls='--', zorder=0)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k' if x else '0'))
    axes[0].set_xlim(0, 200_000); axes[1].set_xlim(0, 40_000)
    nature_ax(axes[0], 'Trial', 'Fraction past\nthe boundary')
    nature_ax(axes[1], 'Trial × reward', '')
    side_legend(axes[1], title='Reward')
    plt.tight_layout(); plt.savefig(f'{F}/n1_population_crossing.svg', dpi=300, bbox_inches='tight'); plt.close()
