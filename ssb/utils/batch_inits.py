import jax.numpy as jnp
import numpy as np
from ssb.models.ssb import init_params

'''
Utilities for initializing networks and setting up batches.
'''
def make_default_hps(seed=5232):
    ''' For convenience: makes default hyperparameter dictionary
    '''
    hps = {
        'sizes' : [[1,0], [2,0]],
        'init_scale': 1.,
        'randinit'  : 1.,
        'posinit'   : 1.,
        'seed'  : seed,

        'alpha' : .01,
        'beta' : .1, # Default for RPE should be 1. based on NB 49.
        'eta_a' : .01,

        'cap'   : 1.,   # Used in ALL RULES in weight_clip function
        'rho'   : 5,    # Used in cov rule only. Decided after testing on COV prob matching task.
        'tau'   : .01, # Used in cov rule only
    }
    hps = {k: jnp.array([v]) for k,v in hps.items()}
    hps['sizes'] = hps['sizes'].squeeze()
    return hps

'''
For batched trials
'''
#: Hyperparameter keys that were renamed, mapped to their current name.
#: The weight learning rate was called 'omega' until the rename to 'beta';
#: both trees now use 'beta'.
_RENAMED_HPS_KEYS = {'omega': 'beta'}


def check_hps_keys(hps):
    '''Reject hyperparameter dicts that still use a pre-rename key name.

    The weight learning rate is ``hps['beta']``. Scripts written before the
    rename set ``hps['omega']``, which is now simply an unused key: the rules
    would silently read the DEFAULT beta instead, producing plausible-looking
    but wrong results with no error anywhere. That silence is the whole reason
    this check exists — fail loudly at setup instead.
    '''
    for old, new in _RENAMED_HPS_KEYS.items():
        if old in hps:
            raise KeyError(
                f"hps['{old}'] is no longer used — it was renamed to "
                f"'{new}'. Setting '{old}' would leave '{new}' at its default "
                f"and be silently ignored. Update the caller to set "
                f"hps['{new}'] instead."
            )


def batchify_hps(hps, batch_size):
    '''Turns an hps dictionary into one for batched trials.
    Tiles over seeds by incrementing them.
    '''
    check_hps_keys(hps)
    hps_batch = {
        k: jnp.tile(v, (batch_size,) + (1,) * v.ndim) for k, v in hps.items()
    }
    hps_batch['seed'] += jnp.arange(batch_size).reshape(-1, 1)
    return hps_batch


def init_params_batch(hps_batch):
    '''Takes a batched hps and makes batched state variables.'''
    
    def get_hps(hps_batch, i):
        return {k: v[i] for k, v in hps_batch.items()}
        
    batch_size = hps_batch['sizes'].shape[0]

    acts, states, weights, tern, key = [], [], [], [], []

    for i in range(batch_size):
        hps_i = get_hps(hps_batch, i)
        acts_i, states_i, weights_i, tern_i, key_i = init_params(hps_i)
        
        acts.append(acts_i)
        states.append(states_i)
        weights.append(weights_i)
        tern.append(tern_i)
        key.append(key_i)

    acts, states, weights, tern, key = jnp.array(acts), jnp.array(states), jnp.array(weights), jnp.array(tern), jnp.array(key)
    return acts, states, weights, tern, key

def expand_hps_batch(hps_batch, sweep_vector, sweep_key):
    ''' Takes an existing hps_batch and expands it to sweep over sweep_vector, which is parameter sweep_key.
    Tiles sweep_vector and repeats everything else.

    E.g. seeds: [0, 1, 2]. Sweep_vector: 'alpha': [.1, .2, .3]
    Returns hps['seed']  = [ 0,  0,  0,  1,  1,  1,  2,  2,  2]
            hps['alpha'] = [.1, .2, .3, .1, .2, .3, .1, .2, .3]
    '''
    base_len = len(next(iter(hps_batch.values())))  # number of original entries
    sweep_len = len(sweep_vector)

    expanded_hps = {}

    # Repeat everything in the hps_batch
    for key, values in hps_batch.items():
        expanded_hps[key] = np.repeat(values, sweep_len, axis=0)

    # Write over sweep_key entries
    expanded_hps[sweep_key] = np.tile(sweep_vector, base_len)

    return expanded_hps