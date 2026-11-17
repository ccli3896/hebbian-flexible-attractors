# Core rules library
''' Rule functions, corresponding state updates, and some utilities.
Default is nonlinear in every script.
'''
import jax
import jax.numpy as jnp
from jax import jit, lax, grad

# Quick function
sqsum = lambda x: jnp.sum(jnp.square(x))

################################################################################################
# The original 3
################################################################################################

@jit
def update_weights(stimuli, activities, states, weights, tern, hps, grad_clip=10., relu_toggle=0.):
    ''' Updates weights according to predictive coding rule.
    relu_toggle is 0 (nonlinear) or -1e6 (linear)
    '''
    w_grads = jnp.outer(activities, (activities - jnp.matmul(activities, jnp.multiply(weights, tern))))
    return jnp.maximum(relu_toggle, weights + hps['beta'] * (jnp.clip(w_grads, -grad_clip, grad_clip) ))

@jit
def update_weights_cov(stimuli, activities, states, weights, tern, hps, grad_clip=10., relu_toggle=0.):
    ''' Updates weights according to covariance-based plasticity.
    '''
    means = states[0]
    w_grads = jnp.outer((activities - means), activities)
    return jnp.maximum(relu_toggle, (weights + hps['beta'] * (jnp.clip(w_grads, -grad_clip, grad_clip) 
                                          - (weights / hps['cap'])**hps['rho'])))

@jit
def update_weights_hrpe(stimuli, activities, states, weights, tern, hps, grad_clip=10., relu_toggle=0.):
    ''' Updates weights according to Hebbian learning with prediction error at sensory layer.
    '''
    input_neurons = hps['sizes'][0][0]
    mask = jnp.arange(stimuli.shape[0]) < input_neurons
    sensory_error = jnp.sum((stimuli - activities) * mask) / input_neurons # This is a scalar

    w_grads = jnp.outer(activities, activities) * sensory_error
    return jnp.maximum(relu_toggle, (weights + hps['beta'] * (jnp.clip(w_grads, -grad_clip, grad_clip) )))



################################################################################################
# Classic rules
################################################################################################

@jit
def update_weights_oja(stimuli, activities, states, weights, tern, hps, grad_clip=10., relu_toggle=0.):
    ''' Updates weights according to Oja's rule. https://en.wikipedia.org/wiki/Oja%27s_rule
    Fixed 3.6.26
    '''
    hebb = jnp.outer(activities, activities)
    norm = weights * (activities**2)[None, :]
    w_grads = hebb - norm
    return jnp.maximum(relu_toggle, (weights + hps['beta'] * (jnp.clip(w_grads, -grad_clip, grad_clip) )))


@jit 
def update_weights_bcm(stimuli, activities, states, weights, tern, hps, grad_clip=10., relu_toggle=0., c0=.1):
    '''According to BCM rule, 1982. No decay in this version.
    Shouldn't change relu_toggle because of stability conditions.
    Fixed 3.5.26
    '''
    means = states[0]
    w_grads = jnp.outer(activities, activities * (activities - (means**2)/c0))
    return jnp.maximum(relu_toggle, (weights + hps['beta'] * (jnp.clip(w_grads, -grad_clip, grad_clip) )))


@jit 
def update_weights_lc(stimuli, activities, states, weights, tern, hps, grad_clip=10., relu_toggle=0.):
    '''According to Law & Cooper, http://www.scholarpedia.org/article/BCM_theory
    Shouldn't change relu_toggle because of stability conditions.
    '''
    meansq = states[0]
    w_grads = jnp.outer(activities, activities * (activities - meansq)/(meansq+1e-6))
    return jnp.maximum(relu_toggle, (weights + hps['beta'] * (jnp.clip(w_grads, -grad_clip, grad_clip) )))

@jit 
def update_weights_hz(stimuli, activities, states, weights, tern, hps, grad_clip=10., relu_toggle=0.):
    '''According to Halvagal & Zenke 2023, https://www.nature.com/articles/s41593-023-01460-y#additional-information
    Shouldn't change relu_toggle because of stability conditions.
    "Using stale mean and variance estimates from previous mini-batches in our DNN simulations did cause a drop in readout performance (Supplementary Table 2)."
    3.6.26
    '''
    means = states[0]
    sig2s = states[1]
    pasts = states[2]
    w_grads = jnp.outer(activities, (1/(sig2s+1e-6)) * (activities - means + sig2s * (pasts - activities)))
    return jnp.maximum(relu_toggle, (weights + hps['beta'] * (jnp.clip(w_grads, -grad_clip, grad_clip) )))


################################################################################################
# State update functions
################################################################################################

@jit
def update_states_null(acts, states, hps):
    ''' Does nothing. For PC, H, Oja's rules.
    '''
    return states

@jit 
def update_states_means(acts, states, hps):
    ''' For covariance and BCM rule. Keeps track of means.
    Inputs:
        acts, (means, zeros, zeros), hps
    Outputs:
        (means, zeros, zeros)
    '''
    tau = hps['tau']
    new_means = (1-tau) * states[0] + tau * acts 
    return jnp.vstack([new_means, jnp.zeros_like(acts), jnp.zeros_like(acts)])

@jit 
def update_states_meansq(acts, states, hps):
    ''' For Law & Cooper rule. Keeps track of means of squared activities.
    Inputs:
        acts, (means of squares, zeros, zeros), hps
    Outputs:
        (means of squares, zeros, zeros)
    '''
    tau = hps['tau']
    new_meansq = (1-tau) * states[0] + tau * acts**2 
    return jnp.vstack([new_meansq, jnp.zeros_like(acts), jnp.zeros_like(acts)])

@jit 
def update_states_hz(acts, states, hps):
    ''' For Halvagal & Zenke LPL rule.
    Updates rolling estimates of means and variances using timescale tau. Also saves latest activities.
    Inputs:
        acts, (means, sig2s, last activities), hps
    Outputs:
        (means, sig2s, last activities)
    '''
    tau = hps['tau']
    new_means = (1-tau) * states[0] + tau * acts 
    new_sig2s = (1-tau) * states[1] + tau * (acts - states[0])**2
    new_past = acts.copy()
    return jnp.vstack([new_means, new_sig2s, new_past])


################################################################################################
# Utilities
################################################################################################

def rule2fns(weight_rule):
    # Get the corresponding weight rule and update state functions
    if weight_rule == 'PC':
        _update_weights = update_weights
        _update_states = update_states_null
    elif weight_rule == 'HR':
        _update_weights = update_weights_hrpe
        _update_states = update_states_null
    elif weight_rule == 'COV':
        _update_weights = update_weights_cov
        _update_states = update_states_means
    elif weight_rule == 'OJA':
        _update_weights = update_weights_oja
        _update_states = update_states_null
    elif weight_rule == 'BCM':
        _update_weights = update_weights_bcm
        _update_states = update_states_means
    elif weight_rule == 'LC':
        _update_weights = update_weights_lc
        _update_states = update_states_meansq
    elif weight_rule == 'HZ':
        _update_weights = update_weights_hz
        _update_states = update_states_hz
    else:
        raise ValueError('Invalid weight rule.')
    
    return _update_weights, _update_states

def set_pars(hps_batch, alpha, beta, eta):
    hps_batch['alpha'] = jnp.ones_like(hps_batch['alpha']) * alpha
    hps_batch['beta'] = jnp.ones_like(hps_batch['beta']) * beta
    hps_batch['eta_a'] = jnp.ones_like(hps_batch['eta_a']) * eta
    return hps_batch