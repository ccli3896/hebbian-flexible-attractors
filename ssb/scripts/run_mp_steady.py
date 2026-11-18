"""Steady-state matching-pennies sweep (per-step task).

Why this exists
---------------
The original matching-pennies sweep (run_sweep.py, env='matchingpennies') ran three 20k-step
phases (computer -> coin flip -> computer) and scored the 3-bit entropy of each. Phases that short
mostly measure how fast the *initial* weight difference decays: in PC at the old SFig 1d cell the
coin-flip "bias" is leftover from initialization and never returns on later cycles, so the sweep
does not show settled behaviour. This driver alternates the two environments for several cycles
and scores every phase, so settled, repeatable responses can be told apart from transients.

Protocol (per parameter cell, network state carried across phases exactly as in run_sweep):
    phase 0           : computer opponent   (ssb.envs.matchingpennies.play_mp, p = 0.05)
    phases 1, 2, ...  : coin flip (bandits.bandit_unbait, [0.5, 0.5]) and computer, alternating
    n_phases = 1 + 2 * n_cycles  (default n_cycles = 3 -> 7 phases), `timesteps` steps per phase
    The opponent's history buffer resets at the start of every phase; weights / activities /
    running statistics carry over (run_env checkpoint_log).
    Shared noise: run_env re-derives the PRNG key from hps['seed'] at every call, so every phase of
    the same type receives the same activity-noise and opponent draws (only the carried-over state
    differs). This is the version that produced the paper's sweep data, and it is kept as is so the
    data can be reproduced. It makes repeat phases look somewhat more alike than independent draws
    would (the settled-behaviour filter is correspondingly lenient); for the PC cell the settled
    medians were unchanged when tested with fresh noise per phase. The 50-seed re-runs behind
    Fig. 3h-k (main_paper/fig3/f4_matchingpennies.ipynb) carry the random state across phases instead.

Scores (per seed, over the second half of every phase)
    metric 0  'entropy' : 3-bit block entropy of the choice sequence (bits, max 3)
    metric 1  'bias'    : side bias |P(choice 1) - 0.5|
    metric 2  'switch'  : P(choice_t != choice_{t-1})
    metric 3  'win'     : mean reward (computer phases: win rate; coin-flip phases: ~0.5)

Output (.npz)
    scores : float32, shape (n_eta, n_tau, n_alpha, n_beta, n_phases, 4, n_seeds)
             (n_eta / n_tau / n_alpha are 1 when sliced with --eta-idx / --tau-idx / --alpha-idx;
              n_tau is 1 for rules without tau)
    plus the grid values, slice indices, phase labels and metric names.
Aggregate the slices with:  python -m ssb.analysis.aggregate_mp_steady --indir <outdir>/results

Config (JSON): {"rule": "PC", "arch": "small", "timesteps": 20000, "n_cycles": 3}
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from ssb.models import ssb
from ssb.envs import bandits
from ssb.envs import matchingpennies as mp
from ssb.scripts.run import run_env
from ssb.utils.batch_inits import make_default_hps, batchify_hps, expand_hps_batch
from ssb.utils.architectures import get_sizes

RULES_WITH_TAU = {'COV', 'BCM', 'LC', 'HZ'}
METRICS = ['entropy', 'bias', 'switch', 'win']


def entropy3(actions):
    """3-bit block entropy per column of a (T, lanes) 0/1 array."""
    a = actions.astype(np.int64)
    b = a[:-2] * 4 + a[1:-1] * 2 + a[2:]
    p = np.stack([(b == k).mean(0) for k in range(8)])
    with np.errstate(divide='ignore', invalid='ignore'):
        return -np.where(p > 0, p * np.log2(p), 0.).sum(0)


def phase_scores(actions, rewards):
    """(T, lanes) arrays from one phase -> (4, lanes) scores over the second half."""
    T = actions.shape[0]
    a, r = actions[T // 2:], rewards[T // 2:]
    return np.stack([entropy3(a),
                     np.abs(a.mean(0) - 0.5),
                     (a[1:] != a[:-1]).mean(0),
                     r.mean(0)]).astype(np.float32)


def run_mp_steady(config, seed=825, seeds=20, eta_idx=None, tau_idx=None, alpha_idx=None, beta_idx=None):
    rule = config['rule']
    arch = config.get('arch', 'small')
    T = int(config.get('timesteps', 20000))
    n_cycles = int(config.get('n_cycles', 3))
    n_alpha, n_beta = int(config.get('n_alpha', 15)), int(config.get('n_beta', 15))

    # Same grids as run_sweep.py
    alphas_full = np.logspace(-5, 0, n_alpha)
    betas_full = np.logspace(-5, 0, n_beta)
    etas_full = np.logspace(-6, -1, 6)
    taus_full = np.logspace(-3, -1, 4)
    uses_tau = rule in RULES_WITH_TAU

    sl = lambda full, i: full if i is None else full[i:i + 1]
    etas, alphas, betas = sl(etas_full, eta_idx), sl(alphas_full, alpha_idx), sl(betas_full, beta_idx)
    taus = sl(taus_full, tau_idx) if uses_tau else np.array([0.01])

    phases = ['computer'] + ['coinflip', 'computer'] * n_cycles
    envs = {'computer': (mp.play_mp, jnp.array([0.05])),
            'coinflip': (bandits.bandit_unbait, jnp.array([0.5, 0.5]))}

    hps = make_default_hps(seed)
    hps['sizes'] = jnp.array(get_sizes(arch, 1, 2))
    out = np.zeros((len(etas), len(taus), len(alphas), len(betas), len(phases), len(METRICS), seeds), np.float32)

    t0 = time.time()
    for i_e, eta in enumerate(etas):
        for i_t, tau in enumerate(taus):
            hps['eta_a'] = jnp.array([eta]); hps['tau'] = jnp.array([tau])
            hb = batchify_hps(hps, seeds)
            hb = expand_hps_batch(hb, jnp.asarray(alphas), 'alpha')
            hb = expand_hps_batch(hb, jnp.asarray(betas), 'beta')
            ckpt = None
            for i_p, ph in enumerate(phases):
                fn, par = envs[ph]
                log, ckpt = run_env(hb, rule, fn, par, T, settle_time=1, update_acts=ssb.update_acts,
                                    env_state=mp.init_mp_state(100), n_motor=2, checkpoint_log=ckpt)
                act = np.asarray(log[0]); rew = np.asarray(log[1], dtype=np.float32)
                # lane order from batchify/expand: lane = (seed * n_alpha + alpha) * n_beta + beta
                sc = phase_scores(act, rew).reshape(len(METRICS), seeds, len(alphas), len(betas))
                out[i_e, i_t, :, :, i_p] = sc.transpose(2, 3, 0, 1)
                del log, act, rew
                print(f'eta {eta:.0e} tau {tau:.3g} phase {i_p} ({ph}) done, {time.time() - t0:.0f}s', flush=True)
    meta = dict(rule=rule, arch=arch, timesteps=T, n_cycles=n_cycles, seed=seed, seeds=seeds, phases=np.array(phases),
                metrics=np.array(METRICS), etas=etas, taus=taus, alphas=alphas, betas=betas,
                etas_full=etas_full, taus_full=taus_full if uses_tau else np.array([0.01]),
                alphas_full=alphas_full, betas_full=betas_full,
                eta_idx=-1 if eta_idx is None else eta_idx, tau_idx=-1 if (tau_idx is None or not uses_tau) else tau_idx,
                alpha_idx=-1 if alpha_idx is None else alpha_idx, beta_idx=-1 if beta_idx is None else beta_idx)
    return out, meta


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True, help='output .npz')
    p.add_argument('--seed', type=int, default=825, help='base seed (run_sweep default)')
    p.add_argument('--seeds', type=int, default=20)
    p.add_argument('--eta-idx', type=int, default=None)
    p.add_argument('--tau-idx', type=int, default=None)
    p.add_argument('--alpha-idx', type=int, default=None)
    p.add_argument('--beta-idx', type=int, default=None)
    p.add_argument('--overwrite', action='store_true', help='rerun even if --out exists')
    a = p.parse_args()
    out_path = Path(a.out)
    if out_path.exists() and not a.overwrite:
        print(f'{out_path} exists; skipping (use --overwrite to rerun)'); raise SystemExit(0)
    config = json.loads(Path(a.config).read_text())
    scores, meta = run_mp_steady(config, a.seed, a.seeds, a.eta_idx, a.tau_idx, a.alpha_idx, a.beta_idx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + '.tmp.npz')      # atomic write: a killed job leaves no file
    np.savez_compressed(tmp, scores=scores, **meta)
    tmp.replace(out_path)
    print('saved', out_path, scores.shape)
