from ssb.models import ssb, rules
from ssb.utils.batch_inits import *

import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
from ssb.utils import batch_inits

from functools import partial

def run_env(hps_batch,
        weight_rule,
        env,
        env_par,
        timesteps,
        settle_time=1,
        update_acts=ssb.update_acts,
        env_state=None,
        n_motor=2,
        checkpoint_log=None,
        log_weights=False
           ):
    '''
    Defines a jitted function that runs the sim. Designed for alpha/beta sweeps. 2 neurons.
    Inputs:
        hps_batch: batched hyperparameters. This input decides the parallelization; must be formatted before run()
        weight_rule: String corresponding to a weight learning rule
        env: an environment that determines stimuli.
        env_par: a setting for the environment. For bandits this is usually reward. For others it's the total set of possible stimuli.
            This is the rewards input.
        timesteps: per prob.
        settle_time: how many steps activities settle before each weight update
        update_acts: either the usual update_acts OR update_acts_ff
        log_weights: if True, the per-step log gains a 4th element: the full
            weight matrix at every timestep, shape (timesteps, batch, nrns, nrns).
            Opt-in (default False) so existing callers that index log[0..2] are
            unaffected. Lets you watch weights evolve / "jump" over a run without
            stitching together many checkpointed calls.

    Outputs:
        log of (actions, rewards, last-layer acts[, weights if log_weights]).
    '''
    batch_size = len(hps_batch['seed'])
    update_weights, update_states = rules.rule2fns(weight_rule)

    # Batchify core functions
    update_acts_batch    = vmap(update_acts)
    update_weights_batch = vmap(update_weights)
    act_noise_batch      = vmap(ssb.act_noise)
    update_states_batch  = vmap(update_states)
    env_batch            = vmap(env)
    expand_batch         = vmap(ssb.expand_stimulus)

    # --- NEW: Tile outside the step function ---
    # This creates the full batch of environment parameters once.
    env_par_batch = jnp.tile(env_par[None, ...], (batch_size, *([1] * env_par.ndim)))

    @partial(jax.jit, static_argnums=(1,))
    def step_fn(state, t):
        # env_par_batch is now passed in the state or captured via closure
        env_state_batch, acts, states, weights, tern, key, env_par_batch = state

        # Get stimulus from environment using the pre-tiled batch
        reward_batch, action_batch, env_state_batch, key = env_batch(
            env_state_batch, acts[:, -n_motor:], env_par_batch, key
        )
        
        stimuli = expand_batch(reward_batch, acts)

        # Activity settling & Plasticity updates
        def full_phase(carry, stimulus_input):
            acts, states, weights, key = carry
            def settle_step(c, _):
                a, s, w, k = c
                a = update_acts_batch(stimulus_input, a, s, w, tern, hps_batch)
                return (a, s, w, k), None

            (acts, states, weights, key), _ = jax.lax.scan(
                settle_step, (acts, states, weights, key), jnp.arange(settle_time)
            )

            acts, key = act_noise_batch(acts, key, hps_batch)
            weights   = update_weights_batch(stimulus_input, acts, states, weights, tern, hps_batch)
            states    = update_states_batch(acts, states, hps_batch)
            return (acts, states, weights, key), None

        carry = acts, states, weights, key
        carry, _ = jax.lax.scan(lambda c, _: full_phase(c, stimuli), carry, None, length=1)
        acts, states, weights, key = carry

        # Return updated state and log the actions. When log_weights is set we
        # additionally emit the full weight matrix this step (decided at trace
        # time, so the log structure is static for the jitted scan).
        per_step = (action_batch, reward_batch[:, 0], acts[:, -1])
        if log_weights:
            per_step = per_step + (weights,)
        return (env_state_batch, acts, states, weights, tern, key, env_par_batch), per_step

    # --- Robust Initialization ---
    if env_state is None:
        # Default fallback
        env_state_batch = jnp.zeros((batch_size, 2))
    else:
        # This handles BOTH dicts and arrays. 
        # tree_map applies the tiling to every leaf (array) inside the state.
        env_state_batch = jax.tree_util.tree_map(
            lambda x: jnp.tile(x[None, ...], (batch_size, *([1] * jnp.ndim(x)))),
            env_state
        )

    acts, states, weights, tern, key = init_params_batch(hps_batch)

    if checkpoint_log is not None:
        acts, states, weights = checkpoint_log[1], checkpoint_log[2], checkpoint_log[3]
    
    # Run the scan
    init_state = (env_state_batch, acts, states, weights, tern, key, env_par_batch)
    final_state, log = jax.lax.scan(step_fn, init_state, None, length=timesteps)
    # (timesteps, seeds, -1)
    return log, final_state ##############


def run_quick(rule, alpha, beta, eta_a, env, timesteps, size='small', seed=235, n_seeds=20, tau=0., reward=1.):
    hps = batch_inits.make_default_hps(seed)
    hps['alpha'] = jnp.array([alpha])
    hps['beta'] = jnp.array([beta])
    hps['eta_a'] = jnp.array([eta_a])
    hps['tau']   = jnp.array([tau])
    if size=='big':
        hps['sizes'] = jnp.array([[1, 0], [50,0], [2, 0]])
    
    hps_batch = batch_inits.batchify_hps(hps, batch_size=n_seeds)
    log, final_state = run_env(hps_batch, rule, env, jnp.array([0., reward]), timesteps=timesteps)
    return log