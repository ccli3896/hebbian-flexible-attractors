"""Attractor-geometry probe: is the rule's fixed set a line or a point?

The paper's mechanism claim (F6) is that a local Hebbian rule produces
reward-seeking because its update vanishes on a *line* rather than at a point:
behaviour is a fixed function of position along that line, and reward magnitude
sets the line's **slope**. RPE, by contrast, collapses to a point. This module
packages that measurement so it can be run inside a sweep job at every
hyperparameter cell, instead of by hand at one operating point in a notebook.

Adapted from the attractor-cloud measurements in the figure notebooks, with three
changes needed to make it sweep-safe:

1. **``beta``, the weight learning rate.** Probes written before the rename set
   ``hps['omega']``, which no longer exists; ``batchify_hps`` now raises on it
   rather than letting ``beta`` sit silently at its default.

2. **Architecture-general readout.** ``w2-w1`` as ``weights[0, 2] - weights[0, 1]``
   only means anything for the 3-neuron small net, where the input projects
   straight onto the two motor units. In a hidden-layer net that entry is not a
   connection at all (``tern`` masks it) and its value is init noise. Here the
   differential readout is summed over the motor units' *actual* presynaptic
   partners, masked by ``tern``, which reduces exactly to ``w2-w1`` on the small
   net and stays meaningful on ``big`` / ``N50_L3`` / ``N200_L2``.

3. **A measurable claim, not an eyeballed one.** ``attractor_metrics`` reduces a
   cloud to storable scalars, and ``slope_reward_sensitivity`` tests the
   mechanism claim *across* reward levels. Note what is deliberately NOT here:
   an automatic line-vs-point verdict from a single cloud's shape. That was
   tried and did not survive validation — an elongated cloud is equally the
   signature of a trajectory that hasn't settled, and on HR's det-best cell
   (a known weight-blow-up regime) shape alone calls a divergence a "line".
   The line-vs-point distinction is recoverable from how the slope responds to
   reward, which does separate the rules correctly.

Coordinates (per step, per seed), matching the paper's panels:

    x_in : input-unit activity            (the reward-carrying input)
    dW   : differential readout weight    (the line-attractor coordinate)
    dX   : x_last - x_second_last         (motor difference; choice A <=> dX < 0)
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax import vmap

from ssb.models import ssb, rules
from ssb.envs.bandits import bandit_det
from ssb.utils.architectures import get_sizes
from ssb.utils.batch_inits import make_default_hps, batchify_hps, init_params_batch

# Activity clip used throughout the paper's probes (F5/F6 CONFIG).
CLIP = 3.0

# |dW| above which the readout is treated as having run away rather than
# settled. Calibrated against the known-pathological regime: HR at its det-best
# cell (beta=0.44, the setting the F3 notebook's DISPLAY_PARS comment describes
# as blowing the weights up) sits at |dW| ~ 27 and still growing after 8k burn
# steps, whereas every settled cell measured stayed below ~2. Cells flagged
# divergent must not be read as attractor geometry — the cloud is a trajectory.
DIVERGENCE_DW = 5.0


def _probe_hps(rule, alpha, beta, eta_a, tau, arch, n_seeds, seed, init_scale=0.25):
    """Batched hps for the probe. Sets `beta`, this package's weight-rate key."""
    hps = make_default_hps(seed)
    hps['alpha'] = jnp.array([float(alpha)])
    hps['beta'] = jnp.array([float(beta)])
    hps['eta_a'] = jnp.array([float(eta_a)])
    hps['init_scale'] = jnp.array([float(init_scale)])
    if tau is not None:
        hps['tau'] = jnp.array([float(tau)])
    # bandit_det is a decision env: 1 input unit, 2 motor units.
    hps['sizes'] = jnp.array(get_sizes(arch, 1, 2))
    return batchify_hps(hps, n_seeds)


def run_attractor_cloud(rule, alpha, beta, eta_a, reward, arch='small', tau=None,
                        n_seeds=20, burn=3000, sample=1500, seed=0):
    """Run to steady state on the deterministic bandit and return the cloud.

    ``burn`` steps are discarded so the network has settled onto whatever
    invariant set the rule has; the following ``sample`` steps are the cloud.

    Returns a dict of flat float arrays (x_in, dW, dX), each of length
    ``sample * n_seeds``.
    """
    hb = _probe_hps(rule, alpha, beta, eta_a, tau, arch, n_seeds, seed)
    uw, us = rules.rule2fns(rule)
    UA = vmap(ssb.update_acts); UW = vmap(uw); US = vmap(us)
    NO = vmap(ssb.act_noise);   EX = vmap(ssb.expand_stimulus)
    env = vmap(bandit_det, in_axes=(0, 0, 0, 0))

    acts, states, weights, tern, key = init_params_batch(hb)

    # Differential readout mask. The last two units are the motor pair; a unit
    # counts as presynaptic to a motor unit only where `tern` says the synapse
    # exists, so unconnected entries (which never update and hold init noise)
    # are excluded. On the small net only the input unit survives, making
    # dW == weights[0, 2] - weights[0, 1] exactly as in the notebooks.
    m_last = jnp.abs(tern[:, :, -1])
    m_prev = jnp.abs(tern[:, :, -2])

    rewards_b = jnp.tile(jnp.array([0., float(reward)])[None], (n_seeds, 1))
    dummy = jnp.zeros((n_seeds, 1))

    @jax.jit
    def step(carry, _):
        acts, states, weights, key = carry
        rb, action, _, key = env(dummy, acts[:, -2:], rewards_b, key)
        stim = EX(rb, acts)
        acts = UA(stim, acts, states, weights, tern, hb)
        acts, key = NO(acts, key, hb)
        acts = jnp.clip(acts, 0., CLIP)
        weights = UW(stim, acts, states, weights, tern, hb)
        states = US(acts, states, hb)
        dW = jnp.sum(weights[:, :, -1] * m_last, axis=1) \
            - jnp.sum(weights[:, :, -2] * m_prev, axis=1)
        dX = acts[:, -1] - acts[:, -2]
        return (acts, states, weights, key), (acts[:, 0], dW, dX)

    carry = (acts, states, weights, key)
    if burn:
        carry, _ = jax.lax.scan(step, carry, None, length=burn)
    _, out = jax.lax.scan(step, carry, None, length=sample)

    return {
        'x_in': np.asarray(out[0]).ravel().astype(np.float32),
        'dW': np.asarray(out[1]).ravel().astype(np.float32),
        'dX': np.asarray(out[2]).ravel().astype(np.float32),
    }


def cross_section_slope(x_in, dW, dX):
    """Through-origin slope of dX on dW, in the central x_in band, for dW > 0.

    This is the quantity the paper identifies with reward magnitude: the slope
    of the attractor. Restricting to the interquartile band of ``x_in`` takes a
    cross-section at roughly constant input drive (so the slope isn't
    contaminated by input variation), and to ``dW > 0`` because the fit is
    through the origin on the branch the paper plots.

    Returns NaN when the cross-section is too small to fit.
    """
    x_in = np.asarray(x_in); dW = np.asarray(dW); dX = np.asarray(dX)
    if x_in.size == 0:
        return float('nan')
    lo, hi = np.percentile(x_in, [25, 75])
    m = (x_in >= lo) & (x_in <= hi) & (dW > 0)
    if m.sum() < 10:
        return float('nan')
    denom = float(np.sum(dW[m] * dW[m]))
    if denom <= 0:
        return float('nan')
    return float(np.sum(dW[m] * dX[m]) / denom)


def cloud_geometry(dW, dX):
    """PCA of the (dW, dX) cloud -> extent and anisotropy.

    A line attractor is an elongated cloud: variance concentrates on one
    principal axis. A point attractor is an isotropic blob whose extent is set
    only by activity noise. Returns:

      length      2*sqrt(lambda1)  -- extent along the principal axis
      width       2*sqrt(lambda2)  -- extent across it
      anisotropy  sqrt(lambda1/lambda2) -- >> 1 line, ~ 1 point
      angle       principal-axis angle in the (dW, dX) plane, radians
    """
    dW = np.asarray(dW, dtype=float); dX = np.asarray(dX, dtype=float)
    out = {'length': float('nan'), 'width': float('nan'),
           'anisotropy': float('nan'), 'angle': float('nan')}
    if dW.size < 10:
        return out
    pts = np.vstack([dW, dX]).T
    pts = pts - pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return out
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    lam2, lam1 = float(vals[0]), float(vals[1])     # eigh returns ascending
    out['length'] = 2.0 * np.sqrt(lam1)
    out['width'] = 2.0 * np.sqrt(lam2)
    # Guard the degenerate case: a cloud with no measurable spread across the
    # principal axis would divide by ~0 and report spurious infinite
    # anisotropy, so treat sub-floating-point width as unresolvable.
    out['anisotropy'] = float(np.sqrt(lam1 / lam2)) if lam2 > 1e-24 else float('inf')
    out['angle'] = float(np.arctan2(vecs[1, 1], vecs[0, 1]))
    return out


def attractor_metrics(cloud, divergence_dw=DIVERGENCE_DW):
    """Reduce a cloud dict to the scalars a sweep stores per cell.

    Deliberately reports NO line-vs-point verdict. A single cloud's anisotropy
    does not identify a line attractor: an elongated cloud is equally the
    signature of a trajectory that has not settled, and on the known-divergent
    HR cell the two are indistinguishable by shape alone. The line-vs-point
    claim is instead tested across reward levels by `slope_reward_sensitivity`
    below, which is the form the paper actually states it in. `anisotropy` is
    still stored as a descriptor, but must be read together with `divergent`.
    """
    x_in, dW, dX = cloud['x_in'], cloud['dW'], cloud['dX']
    geom = cloud_geometry(dW, dX)
    out = {
        'slope': cross_section_slope(x_in, dW, dX),
        'dW_mean': float(np.mean(dW)) if dW.size else float('nan'),
        'dW_std': float(np.std(dW)) if dW.size else float('nan'),
        'dW_absmean': float(np.mean(np.abs(dW))) if dW.size else float('nan'),
        'dX_mean': float(np.mean(dX)) if dX.size else float('nan'),
        'n_points': int(dW.size),
    }
    out.update(geom)
    out['divergent'] = bool(np.isfinite(out['dW_absmean'])
                            and out['dW_absmean'] > divergence_dw)
    return out


def probe_cell(rule, alpha, beta, eta_a, reward, arch='small', tau=None,
               n_seeds=20, burn=3000, sample=1500, seed=0,
               return_cloud=False):
    """Run the probe at one cell/reward and return its metrics.

    Set ``return_cloud=True`` to also get the raw points back (for plotting a
    representative cell); sweeps leave it off so they store only scalars.
    """
    cloud = run_attractor_cloud(rule, alpha, beta, eta_a, reward, arch=arch,
                                tau=tau, n_seeds=n_seeds, burn=burn,
                                sample=sample, seed=seed)
    metrics = attractor_metrics(cloud)
    if return_cloud:
        return metrics, cloud
    return metrics


def slope_reward_sensitivity(rewards, slopes):
    """Fit slope = d_slope_dR * R + intercept across reward levels.

    This is the measurable form of the paper's mechanism claim: *reward
    magnitude is the attractor's slope*. A rule whose attractor slope is set by
    reward gives a large ``d_slope_dR`` with high ``r2``; a rule that collapses
    to a reward-insensitive point gives ``d_slope_dR`` near zero.

    Measured at each rule's det-best cell on the small net (5 reward levels,
    0.2-1.0), this separates the rules as the paper describes:

        PC    slope =  1.127 R - 0.119   r2 = 0.996
        OJA   slope =  2.921 R + 0.019   r2 = 0.999
        HR    slope = -0.147 R + 0.124   r2 = 0.797  (collapses toward 0)
        COV   slope =  0.000 R - 0.081   r2 = 0.999  (flat: reward-insensitive)
        HZ    slope = -0.005 R + 0.004   r2 = 0.680  (flat, near zero)

    Returns dict with d_slope_dR, intercept, r2, n_used. Needs >= 3 finite
    slopes; otherwise all-NaN.
    """
    rewards = np.asarray(rewards, dtype=float)
    slopes = np.asarray(slopes, dtype=float)
    ok = np.isfinite(slopes) & np.isfinite(rewards)
    out = {'d_slope_dR': float('nan'), 'intercept': float('nan'),
           'r2': float('nan'), 'n_used': int(ok.sum())}
    if ok.sum() < 3:
        return out
    x, y = rewards[ok], slopes[ok]
    coef = np.polyfit(x, y, 1)
    pred = np.polyval(coef, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out['d_slope_dR'] = float(coef[0])
    out['intercept'] = float(coef[1])
    out['r2'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return out


# Scalar metric names, in a fixed order, so sweep drivers can store results as
# a plain float array and aggregate them without carrying dicts around.
METRIC_KEYS = ['slope', 'length', 'width', 'anisotropy', 'angle',
               'dW_mean', 'dW_std', 'dW_absmean', 'dX_mean', 'divergent']


def metrics_to_vector(metrics) -> np.ndarray:
    """Pack metrics into a float32 vector ordered by ``METRIC_KEYS``.

    ``divergent`` is stored as 1.0/0.0 so the whole record stays one numeric
    array (bools would force an object dtype and break plain .npy stacking).
    """
    return np.array([float(metrics.get(k, np.nan)) for k in METRIC_KEYS],
                    dtype=np.float32)
