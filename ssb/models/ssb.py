""" 
Self-supervised behavior, core functions and logger.
See run_ssb.py for how to use functions.
"""

# Core library
import jax
import jax.numpy as jnp
from jax import grad, jit, random
jax.config.update('jax_platform_name','cpu')

# Quick function
sqsum = lambda x: jnp.sum(jnp.square(x))

""" SSB init and update functions """

def init_params(hps):
    ''' Initializes all variables for an SSB network.
    Input: a hyperparameter dictionary. Example:
        hps = {
            'sizes' : [[1,0], [2,0]], # [[layer 0 E neurons, layer 0 I neurons], [layer 1 E, layer 1 I], [...] ... ]
            'init_scale': 0.,         # Where weights are initalized (both same)
            'randinit'  : 1.,       # Whether or not initializations are random. If False, then all 0's.
            'posinit'   : 1.,       # Whether or not initializations are positive.
            'seed'  : 7835,           # Random seed.

            'alpha' : .01,            # Neuron activity update rate.
            'beta' : .01,            # Neuron weight update rate.
            'eta_a' : .01,            # Scale of noise added to activities at each step.

            'cap'   : 1.,             # Used in ALL RULES to clip weights at max cap.

            'rho'   : 5,              # Used in cov rule only. Sharpness of weight decay. rho = inf is step fn.
            'tau'   : .01,            # Time constant for running estimates.
        }

    Outputs:
    1. neuron activities
    2. neuron activity states (depends on the learning rule; initializes 3 trackers (not all used for some rules))
    3. weights
    4. a ternary matrix describing network connectivity, size nxn
        Defined as: element (i->j): +1 connected, positive
                                   0 no connection
                                   -1 connected, negative (for inhibition)
    5. random key
    '''

    # First, count up number of neurons and get useful values for making the ternary matrix (tern)
    flat = jnp.array([0, *[nrntype for layer in hps['sizes'] for nrntype in layer]])
        # Flattens out the network size array and prepends a 0.
    nrns = sum(flat)
    indices = jnp.cumsum(flat)

    key = random.PRNGKey(hps['seed'].item())
    states = jnp.ones((3,nrns)) * hps['init_scale']

    if not hps['randinit']:
    # Initialize activities, states, and weights to 0.
        acts = jnp.zeros(nrns)*hps['init_scale']
        weights = hps['init_scale'] * jnp.zeros((nrns, nrns))
    else:
    # Initialize everything to randoms, except states
        # Split key so acts and weights draw from independent entropy.
        # Without this split, JAX's counter-based PRNG makes weights[0,:] == acts[:],
        # which creates a pathological initialization (input weights = activities).
        key, wkey = random.split(key)
        acts = (random.uniform(key, shape=(nrns,)) - 0.5) * 2 * hps['init_scale']
        weights = hps['init_scale'] * (random.uniform(wkey, (nrns, nrns)) - 0.5) * 2

        if hps['posinit']:
            acts = jnp.abs(acts)
            weights = jnp.abs(weights)


    # Set up ternary matrix
    tern = jnp.zeros((nrns, nrns))

    # For each layer, fill in the E & I connections.
    for l in range(len(hps['sizes'])):

        # Excitatory to excitatory
        if l < (len(hps['sizes']) - 1):
                    # Num layers - 1
            tni = l * 2 # Temporary neuron index
            tern = tern.at[indices[tni]:indices[tni]+flat[tni+1], 
                           indices[tni+2]:indices[tni+2]+flat[tni+3]].set(jnp.ones((flat[tni+1], 
                                                                                    flat[tni+3])))
    
        # Excitatory to inhibitory, same layer
        tni = l * 2
        tern = tern.at[indices[tni]:indices[tni]+flat[tni+1], 
                       indices[tni+1]:indices[tni+1]+flat[tni+2]].set(jnp.ones((flat[tni+1], 
                                                                                flat[tni+2])))
        
        # Inhibitory to excitatory, same layer
        tni = l * 2 + 1
        tern = tern.at[indices[tni]:indices[tni]+flat[tni+1], 
                       indices[tni-1]:indices[tni-1]+flat[tni]].set(-1*jnp.ones((flat[tni+1], 
                                                                                 flat[tni])))

    # Width-aware weight normalization: scale each column of W by 1/N_pre(j),
    # where N_pre(j) is the number of presynaptic neurons feeding neuron j
    # (counted from the ternary connectivity mask). This keeps the typical
    # pre-activation s + W^T x roughly O(1) regardless of layer width, so
    # rules trained at one width transfer to another with the same alpha/beta.
    # Note: at the toy 2-neuron scale (one presynaptic neuron per postsynaptic),
    # presyn_count = 1 and this is a no-op.

    # presyn_count = jnp.sum(jnp.abs(tern), axis=0)  # shape (nrns,)
    # presyn_scale = 1.0 / jnp.maximum(presyn_count, 1.0)
    # weights = weights * presyn_scale  # broadcasts across rows

    return acts, states, weights, tern, key

# Add noise using an optional mask, for experiments where noise is only added to one part
@jit
def _act_noise(acts, key, hps, mask, relu_toggle):
    ''' Adds noise to each neuron, scaled by hps['eta'].
    This function is with an explicit mask; internal function.
    Inputs:
        acts, key, hps, mask, relu. Relu works like: either -1e6 or 0. If 0, then relu is effectively on and network is nonlinear.
    Outputs:
        acts, key
    '''
    key, subkey = random.split(key)
    noise = random.normal(subkey, acts.shape) * hps['eta_a']
    new_acts = jnp.maximum(relu_toggle, acts + noise)

    return new_acts, key

def act_noise(acts, key, hps, mask=None, relu_toggle=0.):
    ''' Wrapper around _act_noise() to allow jit with optional mask.
    Default is noise added to all neurons.
    '''
    if mask is None:
        mask = jnp.ones_like(acts, dtype=bool)
    return _act_noise(acts, key, hps, mask, relu_toggle)

# Weight update rules with helper functions

@jit 
def pred_loss(stimuli, acts, states, weights, tern, hps):
    ''' Calculates prediction error.
    Inputs:
        stim, acts, weights, tern, hps.
        stim needs to be len(acts).
    
    Output:
        scalar prediction loss.
    '''
    predloss = 0
    predloss += sqsum(acts - 
                      (stimuli + jnp.matmul(jax.lax.stop_gradient(acts), jnp.multiply(weights, tern))))
    return predloss 

@jit
def update_acts(stimuli, activities, states, weights, tern, hps, grad_clip=10.):
    ''' Updates activities by taking one step toward feedforward pass.
    '''
    act_grads = grad(pred_loss, argnums=1)(stimuli, activities, states, weights, tern, hps)
    return activities - hps['alpha'] * jnp.clip(act_grads, -grad_clip, grad_clip)


@jit
def update_acts_ff(stimuli, activities, states, weights, tern, hps, grad_clip=10.):
    ''' Updates activities by setting them to feedforward value. 
    '''
    return stimuli + jnp.matmul(activities, jnp.multiply(weights, tern))

@jit
def weight_clip(weights):
    return jnp.clip(weights, -1, 1)

@jit
def expand_stimulus(stimuli, acts):
    ''' Makes stimuli the right shape.
    '''
    inputs = jnp.zeros_like(acts)
    return inputs.at[0:len(stimuli)].set(stimuli)



