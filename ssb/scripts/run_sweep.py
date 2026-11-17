from ssb.models import ssb
from ssb.envs import streams, bandits
from ssb.envs import matchingpennies as mp
from ssb.scripts.run import run_env
from ssb.utils.batch_inits import *

from ssb.analysis.scorers import *

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import vmap, jit

import numpy as np
import time


# Rules whose state update depends on a running-average timescale `tau`.
# For these rules we sweep tau as well. For others, tau is left at the
# default from make_default_hps and is not part of the output shape.
RULES_WITH_TAU = {'COV', 'BCM', 'LC', 'HZ'}


def run_sweep(config, seed=825, savefile=None, eta_idx=None, tau_idx=None,
              alpha_idx=None, beta_idx=None, seeds=20):
    """Run an alpha x beta x eta (x tau) sweep.

    The eta, tau, alpha and beta axes can each be sliced down to a single
    value via ``eta_idx`` / ``tau_idx`` / ``alpha_idx`` / ``beta_idx``,
    which is how the SLURM array jobs parallelize across the cluster. When
    sliced, the output retains its full rank (with size 1 along each sliced
    axis) so per-slice files can be aggregated into a single tensor
    afterward.

    The alpha/beta slicing is the grid-sharding lever: a single
    (eta, tau) slice runs the whole 15x15 alpha x beta grid at a 4500-wide
    batch, and both wall time (CPU) and XLA compile time (GPU) scale with
    that batch. Sharding alpha (and/or beta) across array tasks shrinks the
    per-task batch proportionally, turning one long slice into many short,
    parallelizable ones at identical total compute. Slicing is purely a
    sub-selection of the SAME logspace grid, so a sharded-then-aggregated
    run is elementwise identical to an unsharded one at every coordinate.

    Args:
        config: dict with keys 'env', 'rule', 'size', 'timesteps'.
        seed: base random seed.
        savefile: output path (used to derive the data/DetSweep filename).
        eta_idx: if int, restrict eta sweep to the single value at this
            index. Used by SLURM array jobs for big-network parallelization.
        tau_idx: if int, restrict tau sweep to the single value at this
            index. Used by SLURM array jobs for tau-having rules.
            Ignored for rules NOT in RULES_WITH_TAU.
        alpha_idx: if int, restrict the alpha sweep to the single value at
            this index (grid-sharding). Range checked against n_alpha.
        beta_idx: if int, restrict the beta sweep to the single value at
            this index (grid-sharding). Range checked against n_beta.
        seeds: number of random seeds averaged over per parameter point.
            Defaults to 20 (the production value). Set to 1 for fast local
            smoke tests. The output tensor shape is independent of this.
    """
    save_path = Path(savefile) if savefile is not None else Path("results.npy")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Full parameter sweep ranges. The alpha/beta grid RESOLUTION is
    # configurable per (env,rule,arch) via optional config keys 'n_alpha' /
    # 'n_beta' (endpoints unchanged). This lets the expensive GPU-only arch
    # (N200_L2) run a coarser 9x9 grid while everything else stays 15x15,
    # cutting per-slice cost ~3x on the bottleneck without touching the
    # eta/tau slicing that the SLURM array uses. All slices of a given
    # (env,rule,arch) MUST share the same resolution or aggregation will
    # refuse to stitch them (shape mismatch) — so set it in the config once
    # per combo, which the launcher/timing path does.
    n_alpha = int(config.get('n_alpha', 15))
    n_beta = int(config.get('n_beta', 15))
    alphas_full = jnp.logspace(-5, 0, n_alpha)
    betas_full = jnp.logspace(-5, 0, n_beta)
    etas_full = jnp.logspace(-6, -1, 6)
    taus_full = jnp.logspace(-3, -1, 4)   # only meaningful for RULES_WITH_TAU

    # Apply slicing from CLI flags. We use full-range arrays for shape
    # computation (so output tensors have the same shape regardless of
    # which slice this job ran) and a 1-element sub-array for the loop.
    rule_uses_tau = config['rule'] in RULES_WITH_TAU

    if eta_idx is not None:
        if not (0 <= eta_idx < len(etas_full)):
            raise ValueError(f"eta_idx {eta_idx} out of range [0, {len(etas_full)})")
        etas_loop = etas_full[eta_idx:eta_idx + 1]
    else:
        etas_loop = etas_full

    if rule_uses_tau:
        if tau_idx is not None:
            if not (0 <= tau_idx < len(taus_full)):
                raise ValueError(f"tau_idx {tau_idx} out of range [0, {len(taus_full)})")
            taus_loop = taus_full[tau_idx:tau_idx + 1]
        else:
            taus_loop = taus_full
    else:
        taus_loop = jnp.array([0.01])   # 1-element placeholder; never sliced

    # Grid-sharding: slice alpha/beta down to a single index when requested.
    # This is a pure sub-selection of the SAME logspace grid, so the batch,
    # the scores-array shape, and the scoring loops below all stay consistent
    # (they all read `alphas`/`betas`), and the sliced axis ends up size 1 in
    # the per-slice output — exactly like eta/tau. The aggregator places each
    # size-1 tile back at alpha_idx/beta_idx.
    if alpha_idx is not None:
        if not (0 <= alpha_idx < n_alpha):
            raise ValueError(f"alpha_idx {alpha_idx} out of range [0, {n_alpha})")
        alphas = alphas_full[alpha_idx:alpha_idx + 1]
    else:
        alphas = alphas_full

    if beta_idx is not None:
        if not (0 <= beta_idx < n_beta):
            raise ValueError(f"beta_idx {beta_idx} out of range [0, {n_beta})")
        betas = betas_full[beta_idx:beta_idx + 1]
    else:
        betas = betas_full

    # Backwards-compat: keep `etas` name pointing at what we actually
    # loop over (since scoring code below uses len(etas) and i_e).
    etas = etas_loop

    # Set hyperparameters according to config.
    # Note: hps['sizes'] is set later (after INPUT_DIM is known per env)
    # via the architecture registry.
    hps = make_default_hps(seed)


    '''------------------------------------------------------------------------
    ENVIRONMENT SETUP
    ------------------------------------------------------------------------'''

    # Set fixed variables
    if config['env']=='selectivity':
        INPUT_DIM = 37
        CKPT_FLAG = True # Whether to do checkpoints over loops. This one is true because ckpt is used for scoring

        # Env state
        env_state = jnp.array([0., 0., 1.]) # step count, previous action, hold length

        # Env pars
        filters = streams.make_filters(INPUT_DIM, 15, sharpness=20)
        horizontal = np.tile(filters[11], (10, 1)) 
        vertical = np.tile(filters[4], (10, 1))
        filters_augmented = np.concatenate([filters, horizontal, vertical], axis=0) # Shape 10+10+k, k
        
        # Augmented filters are only for LC proportionality tests. Otherwise use regular filters
        env_params_list = [(streams.sample_inputs, filters)]

        # Scores shape (with optional tau axis for tau-having rules)
        if rule_uses_tau:
            scores = np.zeros((len(etas), len(taus_loop), len(alphas), len(betas)))
        else:
            scores = np.zeros((len(etas), len(alphas), len(betas)))

    elif config['env']=='slowfeature':
        INPUT_DIM = 2
        CKPT_FLAG = False 

        # Set run.py output to (reward_batch[:, 0], acts[:, -1])

        # Env state
        env_state = jnp.array([0.])

        # Env pars
        prob_switch = 0.05
        env_params_list = list(zip([streams.stream_slow_feature]*2, jnp.array([[prob_switch, 0.1],[prob_switch, 10.]])))

        # Scores shape (with optional tau axis)
        if rule_uses_tau:
            scores = np.zeros((len(etas), len(taus_loop), len(alphas), len(betas), 2))
        else:
            scores = np.zeros((len(etas), len(alphas), len(betas), 2))

    elif config['env']=='det':
        INPUT_DIM = 1
        CKPT_FLAG = False 

        # Env state
        env_state = None

        # Env pars and zipped list
        env_params_list = list(zip([bandits.bandit_unbait], [jnp.array([0., 1.])]))
        
        # No scores matrix here because we're just saving actions
        scores = []

    elif config['env']=='bandit_bait':
        INPUT_DIM = 1
        CKPT_FLAG = False 

        # Env state
        env_state = None

        # Env pars and zipped list
        prob_num = 7
        probs = jnp.vstack([jnp.linspace(0., .3, prob_num), jnp.linspace(.3, 0., prob_num)]).T
        env_params_list = list(zip([bandits.bandit_bait]*prob_num, probs))

        # Scores shape (with optional tau axis)
        if rule_uses_tau:
            scores = np.zeros((len(etas), len(taus_loop), len(alphas), len(betas)))
        else:
            scores = np.zeros((len(etas), len(alphas), len(betas)))

    elif config['env']=='bandit_unbait':
        # Unbaited 2-armed bandit. Unlike bandit_bait (rewards persist until
        # collected), rewards here are redrawn Bernoulli every trial, so the
        # optimal policy is to fixate the better arm and matching is NOT
        # incentivised. We sweep the SAME symmetric reward-ratio grid as
        # bandit_bait and, per (alpha, beta), record BOTH behavioural
        # summaries used in data/Processing.ipynb:
        #   [..., 0] probability-matching index  (over/under-matching of the
        #            choice ratio relative to the reward ratio; identical
        #            construction to the bandit_bait ou_index)
        #   [..., 1] SD of the choice ratio      (cross-seed spread of the
        #            per-block choice ratio, averaged over conditions — the
        #            choice_sd metric in behavior_metrics / plot_block_variance)
        INPUT_DIM = 1
        CKPT_FLAG = False

        # Env state (unbaited bandit is stateless; the `_` slot is ignored)
        env_state = None

        # Env pars and zipped list. Reuse bandit_bait's symmetric reward grid
        # so the two envs are directly comparable: reward ratio spans 1 -> 0
        # across the conditions, and probs[i] = (p_arm0, p_arm1).
        prob_num = 7
        probs = jnp.vstack([jnp.linspace(0., .3, prob_num), jnp.linspace(.3, 0., prob_num)]).T
        env_params_list = list(zip([bandits.bandit_unbait]*prob_num, probs))

        # Scores shape: trailing axis of size 2 -> [pm_index, choice_sd]
        if rule_uses_tau:
            scores = np.zeros((len(etas), len(taus_loop), len(alphas), len(betas), 2))
        else:
            scores = np.zeros((len(etas), len(alphas), len(betas), 2))

    elif config['env']=='matchingpennies':
        INPUT_DIM = 1
        CKPT_FLAG = True

        # Env state
        env_state = mp.init_mp_state(100)

        # Env pars and zipped list
        functions = [mp.play_mp, bandits.bandit_unbait, mp.play_mp]
        env_par = [jnp.array([0.05]), jnp.array([0.5, 0.5]), jnp.array([0.05])]
        env_params_list = list(zip(functions, env_par))

        # Scores shape (with optional tau axis)
        if rule_uses_tau:
            scores = np.zeros((len(etas), len(taus_loop), len(alphas), len(betas), 3))
        else:
            scores = np.zeros((len(etas), len(alphas), len(betas), 3))
    


    '''------------------------------------------------------------------------
    RUN SIMULATIONS
    ------------------------------------------------------------------------'''

    # Change sizes via the architecture registry. Motor count is
    # determined by the environment (decision envs need 2 motor neurons
    # for argmax; perceptual envs need 1).
    from ssb.utils.architectures import get_sizes, get_n_motor
    arch_name = config.get('arch', config.get('size', 'small'))  # accept either key
    n_motor = get_n_motor(config['env'])
    hps['sizes'] = jnp.array(get_sizes(arch_name, INPUT_DIM, n_motor))

    # Start timing
    t_start_sweep = time.time()

    # ---- Run sweep ----
    # Outer: eta (sliced down to 1 element if --eta-idx given).
    # Inner: tau (sliced down to 1 element if --tau-idx given OR rule doesn't use tau).
    total_iters = len(etas) * len(taus_loop)
    iter_idx = 0

    def _score_prefix(i_e, i_t):
        """Index prefix into the scores array.

        For tau-having rules the array has a tau axis after eta; for
        other rules it doesn't. Returning a tuple lets us index uniformly
        as scores[prefix + (i_a, i_w, ...)] in the scoring blocks below.
        """
        if rule_uses_tau:
            return (i_e, i_t)
        return (i_e,)

    # Filename suffix encoding the absolute slice indices (NOT the loop
    # indices, which start at 0 inside the script). Used by 'det' env
    # raw-file naming and by aggregator code.
    abs_eta_idx = eta_idx if eta_idx is not None else None
    abs_tau_idx = tau_idx if (rule_uses_tau and tau_idx is not None) else None

    for i_e, eta in enumerate(etas):
        for i_t, tau in enumerate(taus_loop):
            iter_idx += 1
            iter_t0 = time.time()
            if rule_uses_tau:
                print(f"Sweep iter {iter_idx}/{total_iters} (eta_idx={i_e}, tau_idx={i_t}, tau={float(tau):.4g})")
            else:
                print(f"Sweep iter {iter_idx}/{total_iters} (eta_idx={i_e})")

            # Build batched hyperparameters
            hps['eta_a'] = eta
            hps['tau'] = tau  # harmless for non-tau rules
            print(hps)
            hps_batch = batchify_hps(hps, seeds)
            hps_batch = expand_hps_batch(hps_batch, alphas, 'alpha')
            hps_batch = expand_hps_batch(hps_batch, betas, 'beta')

            # Run all environments
            logs = []
            ckpt = None

            for i_p, (environs, env_par_i) in enumerate(env_params_list):
                log_i, ckpt = run_env(
                    hps_batch,
                    config['rule'],
                    environs,
                    env_par_i,
                    config['timesteps'],
                    settle_time=1,
                    update_acts=ssb.update_acts,
                    env_state=env_state,
                    n_motor=2, # Irrelevant in perceptual tasks
                    checkpoint_log=ckpt,
                )

                # Make sure checkpoint isn't used, unless set to True
                if not CKPT_FLAG:
                    ckpt = None

                print(f"  ran prob {i_p+1}/{len(env_params_list)}")


                # Preprocessing for memory
                # Output of run.py is (action_batch, reward_batch[:, 0], acts[:, -1]). Cut this down if not needed and transfer device
                if config['env'] in ('bandit_bait', 'bandit_unbait'):
                    log_i = log_i[0]
                    log_i = jnp.mean(log_i[config['timesteps'] // 2 :, :], axis=0) # log_i is shape (timesteps, seeds*alpha*beta). (actions -> per-seed choice ratio)

                elif config['env'] == 'det':
                    log_i = log_i[0][config['timesteps'] // 2 :, :]

                elif config['env'] == 'matchingpennies':
                    log_i = log_i[0]
                    log_i = mp.get_3bit_entropy(log_i[config['timesteps'] // 2 :, :]) # Returns shape (entropy [1], seeds) (actions)

                elif config['env'] == 'selectivity':
                    log_i = None
                    log_i = get_selectivity_unbiased(filters, ckpt[1], ckpt[3], ckpt[4]) # Shape (seeds,). Log irrelevant
                
                elif config['env'] == 'slowfeature':
                    log_i = batched_get_selectivity(log_i[1][-1000:, :], log_i[2][-1000:, :]) # log_i is (reward_batch[:,0], acts[:, -1])
                    # log_i is now shape (seeds,)

                log_i = jax.device_get(log_i)
                logs.append(log_i)


            '''------------------------------------------------------------------------
            SCORE OUTPUTS
            ------------------------------------------------------------------------'''
            pref = _score_prefix(i_e, i_t)

            # ---- Scoring for det ----
            if config['env'] == 'det':
                actions = np.array(logs[0])
                # shape: (T/2, seeds * alphas * betas) -> (T/2, seeds, alphas, betas)
                actions = actions.reshape(-1, seeds, len(alphas), len(betas))
                raw_dir = Path("data/DetSweep/raw")
                raw_dir.mkdir(parents=True, exist_ok=True)
                # Build a bare stem without any slice tags (eta/tau/alpha/beta
                # indices already appear in the prefix, so strip them from the
                # stem to avoid duplication like eta3_det_PC_big_eta3.npz).
                import re as _re
                bare_stem = _re.sub(r'_eta\d+(_tau\d+)?(_a\d+)?(_w\d+)?$', '', save_path.stem)
                eta_tag = f"eta{abs_eta_idx}" if abs_eta_idx is not None else f"eta{i_e}"
                tau_tag = f"_tau{abs_tau_idx}" if abs_tau_idx is not None else ""
                a_tag = f"_a{alpha_idx}" if alpha_idx is not None else ""
                w_tag = f"_w{beta_idx}" if beta_idx is not None else ""
                np.savez_compressed(raw_dir / f'{eta_tag}{tau_tag}{a_tag}{w_tag}_{bare_stem}.npz', x=actions.astype(bool))

            # ---- Scoring for bandit_bait ----
            elif config['env'] == 'bandit_bait':
                actions = np.array(logs) # (probs, seeds * alpha * beta)

                for i_a, alpha in enumerate(alphas):
                    for i_w, beta in enumerate(betas):
                        mu_seeds = actions[:, (i_a*len(betas))+i_w :: len(alphas)*len(betas)] # Shape (7, seeds)
            
                        nprobs = mu_seeds.shape[0]
                        matching = np.linspace(1, 0, nprobs)[:, None]
                        ou_index = -np.sum(
                            (matching[: nprobs // 2] - mu_seeds[: nprobs // 2]) +
                            (mu_seeds[-(nprobs // 2) :] - matching[-(nprobs // 2) :]),
                            axis=0,
                        )
                        scores[pref + (i_a, i_w)] = np.mean(ou_index / ((nprobs + 1) / 4))

            # ---- Scoring for bandit_unbait ----
            # Two summaries per (alpha, beta), matching data/Processing.ipynb:
            #   [..., 0] probability-matching index (same over/under-matching
            #            construction as bandit_bait's ou_index)
            #   [..., 1] SD of choice ratio (cross-seed spread per condition,
            #            averaged over conditions — the choice_sd metric)
            elif config['env'] == 'bandit_unbait':
                actions = np.array(logs) # (probs, seeds * alpha * beta)

                for i_a, alpha in enumerate(alphas):
                    for i_w, beta in enumerate(betas):
                        mu_seeds = actions[:, (i_a*len(betas))+i_w :: len(alphas)*len(betas)] # Shape (7, seeds)

                        nprobs = mu_seeds.shape[0]
                        matching = np.linspace(1, 0, nprobs)[:, None]
                        ou_index = -np.sum(
                            (matching[: nprobs // 2] - mu_seeds[: nprobs // 2]) +
                            (mu_seeds[-(nprobs // 2) :] - matching[-(nprobs // 2) :]),
                            axis=0,
                        )
                        pm_index = np.mean(ou_index / ((nprobs + 1) / 4))

                        # SD of the choice ratio: per condition, spread across
                        # seeds (each seed is one independent "block"); then
                        # average over conditions for a single scalar.
                        choice_sd = np.mean(np.std(mu_seeds, axis=1))

                        scores[pref + (i_a, i_w, 0)] = pm_index
                        scores[pref + (i_a, i_w, 1)] = choice_sd

            # ---- Scoring for matching pennies ----
            elif config['env'] == 'matchingpennies':
                actions = np.array(logs) # (3, seeds * alpha * beta)

                for i_a, alpha in enumerate(alphas):
                    for i_w, beta in enumerate(betas):
                        ent_seeds = actions[:, (i_a*len(betas))+i_w :: len(alphas)*len(betas)]
                        scores[pref + (i_a, i_w, slice(None))] = np.mean(ent_seeds, axis=1)

            # ---- Scoring for BCM selectivity ----
            elif config['env'] == 'selectivity':
                for i_a, alpha in enumerate(alphas):
                    for i_w, beta in enumerate(betas):
                        scores[pref + (i_a, i_w)] = np.mean(np.array(logs[0])[(i_a*len(betas))+i_w :: len(alphas)*len(betas)])

            # ---- Scoring for slow feature extraction ----
            elif config['env'] == 'slowfeature':
                for i_a, alpha in enumerate(alphas):
                    for i_w, beta in enumerate(betas):
                        for i_l in range(len(logs)):
                            scores[pref + (i_a, i_w, i_l)] = np.mean(np.array(logs[i_l])[(i_a*len(betas))+i_w :: len(alphas)*len(betas)])



            print(f"Completed iter {iter_idx}/{total_iters} in {time.time() - iter_t0:.2f}s")

    print(f"Sweep finished in {time.time() - t_start_sweep:.1f}s")

                
    return scores                    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON config file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results.npy")
    parser.add_argument("--eta-idx", type=int, default=None,
                        help="If set, run only this eta index from the sweep range. "
                             "Used by SLURM array jobs for big-network parallelization.")
    parser.add_argument("--tau-idx", type=int, default=None,
                        help="If set (and rule uses tau), run only this tau index "
                             "from the sweep range. Used by SLURM array jobs.")
    parser.add_argument("--alpha-idx", type=int, default=None,
                        help="If set, run only this alpha index from the grid "
                             "(grid-sharding). Shrinks the per-task batch so a "
                             "long slice becomes many short, parallel ones.")
    parser.add_argument("--beta-idx", type=int, default=None,
                        help="If set, run only this beta index from the grid "
                             "(grid-sharding). Combine with --alpha-idx to shard "
                             "down to a single grid point.")
    parser.add_argument("--seeds", type=int, default=20,
                        help="Number of random seeds averaged per parameter point. "
                             "Defaults to 20 (production). Set to 1 for fast local "
                             "smoke tests; output shape is unaffected.")

    args = parser.parse_args()
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        config = json.load(f)

    scores = run_sweep(
        config,
        seed=args.seed,
        savefile=args.out,
        eta_idx=args.eta_idx,
        tau_idx=args.tau_idx,
        alpha_idx=args.alpha_idx,
        beta_idx=args.beta_idx,
        seeds=args.seeds,
    )

    if config['env'] != 'det':
        np.save(args.out, scores)