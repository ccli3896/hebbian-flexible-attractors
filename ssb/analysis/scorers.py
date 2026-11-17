from functools import partial
import jax.numpy as jnp
import numpy as np
from jax import jit
import jax

@partial(jit, static_argnums=(1,))
def score_probmatching(actions, window=(30_000, 50_000), psum=0.3):
    '''
    Scores how far behavior deviates from probability matching, signed by direction.

    Inputs:
        actions: shape [probs, seeds, timesteps], binary action stream.
        window: (start, end) timesteps to average actions over (post-learning).
        psum: included for API compatibility; unused in the computation.

    Returns:
        score: shape [seeds]. Interpretation:
            score ≈  0   -> exact probability matching (action ratio ≡ target ratio)
            score >  0   -> overshooting matching (more extreme than the target)
                            +1 corresponds to a near-step-function policy
            score <  0   -> undershooting matching (flatter than the target)
                            ≈ −1 corresponds to a flat 0.5 policy

    NOTE: The naming "score_probmatching" suggests "higher is more matching,"
    but the metric actually measures matching RESIDUAL (signed): zero at
    matching, positive for overshoot, negative for undershoot. Use the sign
    and magnitude jointly, not just magnitude.

    METHOD: For each prob condition the target action-ratio comes from
    linspace(1, 0, nprobs). The score sums signed residuals from the first
    and last halves (skipping the middle to avoid divide-by-symmetry), then
    negates so that overshoot is positive. Normalized by (nprobs+1)/4 so
    that perfect step behavior approaches +1.
    '''
    # Get probabilities for action 1.
    nprobs = actions.shape[0]
    
    # Compute average action per condition within time window    
    mu_seeds = jnp.mean(actions[:, :, window[0]:window[1]], axis=2) # Left with [probs, seeds].

    # Target action ratio per condition: 1 → 0 across the nprobs sweep
    matching = jnp.linspace(1, 0, nprobs)
    matching = matching[:, None] # Expand to [nprobs, 1]
    # Sum signed residuals from each half: first half should overshoot upward,
    # second half should overshoot downward. Negation makes overshoot positive.
    ou_index = - np.sum((matching[:nprobs//2] - mu_seeds[:nprobs//2, :]) \
            + (mu_seeds[-(nprobs//2):, :] - matching[-(nprobs//2):, :]), axis=0)
    
    return ou_index / ((nprobs+1) / 4)


'''############################################################################
###############################################################################################
 SCORING FUNCTIONS
###############################################################################################
###############################################################################################
'''

@jax.jit
def get_hz_selectivity(cluster_identity, act_history):
    ''' Get selectivity as defined by methods in Halvagal & Zenke 2023.
    Z1 is response to cluster 1 and Z2 is response to cluster 2. 
    Z_max is maximum over time series and Z_min is minimum over time series.
        chi = |<Z1> - <Z2>| / (Z_max - Z_min)

    This function does this calculation over the number of timesteps input.
    '''
    # Sums for each cluster
    sum1 = jnp.sum(act_history * cluster_identity)
    sum0 = jnp.sum(act_history * (1.0 - cluster_identity))

    # Counts for each cluster
    count1 = jnp.sum(cluster_identity)
    count0 = jnp.sum(1.0 - cluster_identity)

    # Avoid division by zero
    Z1 = jnp.where(count1 > 0, sum1 / count1, 0.0)
    Z2 = jnp.where(count0 > 0, sum0 / count0, 0.0)

    return jnp.abs(Z1 - Z2) / (jnp.max(act_history) - jnp.min(act_history) + 1e-6)

batched_get_selectivity = jax.vmap(
    get_hz_selectivity,
    in_axes=(1, 1)  # Cluster identity and act history
)

@jax.jit
def get_selectivity_unbiased(stimulus_set, acts, weights, tern):
    # This is for original BCM selectivity metric. 

    pad_width = acts.shape[-1] - stimulus_set.shape[-1]
    stimulus_set = jnp.pad(stimulus_set, ((0, 0), (0, pad_width)))

    def get_selectivity_single_net(acts_s, weights_s, tern_s):

        def single_response(s):
            _acts = jnp.zeros_like(acts_s)
            for _ in range(3):
                _acts = s + jnp.matmul(_acts, weights_s * tern_s)
            _acts = jnp.maximum(0, _acts)
            return _acts[-1]

        responses = jax.vmap(single_response)(stimulus_set)
        return 1.0 - (jnp.mean(responses) + 1e-6) / (jnp.max(responses) + 1e-6)

    return jax.vmap(get_selectivity_single_net)(acts, weights, tern)

@jax.jit
def get_selectivity(stimulus_set, acts, weights, tern):
    # This metric checks for selectivity to the two augmented filters (horizontal and vertical).
    # It is basically the same as get_selectivity_unbiased, but it only checks the responses to the two augmented filters, and ignores the rest.
    
    pad_width = acts.shape[-1] - stimulus_set.shape[-1]
    stimulus_set = jnp.pad(stimulus_set, ((0, 0), (0, pad_width)))

    def get_selectivity_single_net(acts_s, weights_s, tern_s):

        def single_response(s):
            _acts = jnp.zeros_like(acts_s)
            for _ in range(3):
                _acts = s + jnp.matmul(_acts, weights_s * tern_s)
            _acts = jnp.maximum(0, _acts)
            return _acts[-1]

        responses = jax.vmap(single_response)(stimulus_set)

        # Relevant responses are to indices 4 and 11.
        # We want the difference between these and the rest of the responses to be large. 
        # If anti-selective, then the mean of the relevant responses will be smaller than the mean of the irrelevant responses, and the score will be negative.
        relevant = jnp.array([responses[4], responses[11]])
        irrelevant = jnp.concatenate([responses[:4], responses[5:11], responses[12:]])
        mean_relevant = jnp.mean(relevant)
        mean_irrelevant = jnp.mean(irrelevant)
        return (mean_relevant - mean_irrelevant) / (mean_relevant + mean_irrelevant + 1e-6)

    return jax.vmap(get_selectivity_single_net)(acts, weights, tern)

