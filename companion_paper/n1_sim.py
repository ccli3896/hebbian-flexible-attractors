"""Simulations for the companion paper (Li & Erlich) (asymptote-flat operating point).

Three-neuron PC network (1 input -> 2 motor units) on the deterministic two-armed bandit.
Same step order as ssb.scripts.run.run_env with settle_time=1:
    env (choose arm from current motor units, deliver reward) -> stimulus -> one activity step
    -> noise (+ rectification) -> PC weight step.
No activity cap (identical to run.run_quick). Each simulated network ("seed") has its own
PRNG key from ssb's init_params, so results are reproducible seed by seed.
"""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))   # this folder; repo root (contains ssb/)
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np, jax, jax.numpy as jnp
from jax import vmap
from ssb.models import ssb, rules
from ssb.envs import bandits
from ssb.utils import batch_inits
from ssb.analysis.cell_select import ALPHAS, BETAS

_snap = lambda v, grid: float(grid[np.argmin(np.abs(np.log(grid) - np.log(v)))])
ALPHA = _snap(0.08483, ALPHAS)     # 0.0848342898244072
BETA = _snap(0.01638, BETAS)       # 0.016378937069540647
ETA = 0.003


def make_rows(seeds, alpha=ALPHA, beta=BETA, eta=ETA):
    n = len(seeds)
    hps = batch_inits.make_default_hps(0)
    hps['tau'] = jnp.array([0.])
    hb = batch_inits.batchify_hps(hps, batch_size=n)
    col = lambda v: jnp.broadcast_to(jnp.asarray(v, dtype=jnp.float32), (n,)).reshape(n, 1)
    hb['alpha'], hb['beta'], hb['eta_a'] = col(alpha), col(beta), col(eta)
    hb['seed'] = jnp.asarray(seeds).reshape(n, 1)
    return hb


def simulate(seeds, reward, timesteps, alpha=ALPHA, beta=BETA, eta=ETA, window=None,
             stride=None, burn=0, chunk=50_000):
    """window: return per-seed P(optimal) in non-overlapping bins of `window` trials.
    stride:  return (W, x0, X, A) every `stride`-th trial (A = 1 if the paying arm was chosen).
    burn:    trials run (and discarded) before recording."""
    hb = make_rows(seeds, alpha, beta, eta)
    n = len(seeds)
    upd_w, upd_s = rules.rule2fns('PC')
    ua, uw = vmap(ssb.update_acts), vmap(upd_w)
    an, us = vmap(ssb.act_noise), vmap(upd_s)
    envb, expb = vmap(bandits.bandit_det), vmap(ssb.expand_stimulus)
    env_par = jnp.stack([jnp.zeros(n), jnp.broadcast_to(jnp.asarray(reward, jnp.float32), (n,))], 1)

    def step(state, _):
        es, acts, st, w, tern, key = state
        rew, act, es, key = envb(es, acts[:, -2:], env_par, key)
        stim = expb(rew, acts)
        acts = ua(stim, acts, st, w, tern, hb)
        acts, key = an(acts, key, hb)
        w = uw(stim, acts, st, w, tern, hb)
        st = us(acts, st, hb)
        return (es, acts, st, w, tern, key), (act, acts[:, 0], acts[:, 2] - acts[:, 1],
                                              w[:, 0, 2] - w[:, 0, 1], acts.max(1))

    def make_chunk(L):
        @jax.jit
        def run_chunk(state):
            state, (act, x0, X, W, amax) = jax.lax.scan(step, state, None, length=L)
            outs = {'amax': amax.max(0)}
            if window:
                outs['B'] = act.reshape(L // window, window, n).mean(1)
            if stride:
                sl = slice(stride - 1, None, stride)
                outs['W'], outs['x0'], outs['X'], outs['A'] = W[sl], x0[sl], X[sl], act[sl]
            return state, outs
        return run_chunk

    acts, st, w, tern, key = batch_inits.init_params_batch(hb)
    state = (jnp.zeros((n, 2)), acts, st, w, tern, key)
    rc = make_chunk(chunk)
    for _ in range(burn // chunk):
        state, _ = rc(state)
    res = {k: [] for k in ('B', 'W', 'x0', 'X', 'A')}
    amax = 0.
    W0 = np.asarray(state[3][:, 0, 2] - state[3][:, 0, 1])
    for _ in range(timesteps // chunk):
        state, o = rc(state)
        amax = max(amax, float(np.asarray(o['amax']).max()))
        for k in res:
            if k in o:
                res[k].append(np.asarray(o[k]))
    out = {k: np.concatenate(v, 0).astype(np.float32) for k, v in res.items() if v}
    if 'A' in out:
        out['A'] = (out['A'] == 1).astype(np.float32)
    out['W0'] = W0.astype(np.float32)
    out['amax'] = amax
    return out
