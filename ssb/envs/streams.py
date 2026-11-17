"""Streamed-input environments.

These environments do not respond to motor outputs — they generate
inputs autonomously. Used for representation-learning tasks (slow-
feature selectivity, etc.) where the network is judged on what its
neurons select for, not on its actions.

The signature matches the bandit/foraging envs so all envs are
interchangeable in ``ssb.utils.run.run_env``:

    env(state, motor, env_par, key) -> (reward, action, new_state, new_key)

For streams, ``motor`` is unused, and "reward" is overloaded to mean
"next input vector" — not ideal naming, but it keeps the contract
uniform across env types. The "action" return is the index of the
sample (when applicable) or the current latent state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random


# ============================================================================
# Generic random-sample stream
# ============================================================================

@jit
def sample_inputs(state, motor, rewards, key):
    """Sample one input uniformly from a pre-computed bank ``rewards``.

    ``rewards`` here is shape (n_samples, input_dim) — the bank of
    inputs to draw from. The function picks one at random per call
    (no dependence on motor outputs).

    Returns:
        sample: the chosen input row.
        action: the chosen sample index.
        state: unchanged.
        new_key: split key.
    """
    key, subkey = random.split(key)
    action = random.randint(subkey, 1, 0, len(rewards))
    return rewards[action].squeeze(), action, state, key


# ============================================================================
# Stimulus banks (called once at config time, not in the hot loop)
# ============================================================================

def make_filters(N: int, k: int, freq: float = 1.0, sharpness: int = 1) -> np.ndarray:
    """Make k circularly-shifted rectified sine filters of length N.

    Used as a synthetic orientation-tuning input bank. Each row is a
    shifted half-rectified sinusoid.
    """
    n = np.arange(N)
    x = np.maximum(0, np.sin(2 * np.pi * freq * n / N))
    shifts = np.round(np.linspace(0, N, k, endpoint=False)).astype(int)
    X = np.column_stack([np.roll(x**sharpness, s) for s in shifts])
    return X.T


def make_cloud(N: int, k: int, seed: int = 358927, decay_rate: float = 0.8) -> jnp.ndarray:
    """Sample k points from a Gaussian cloud with exponentially decaying eigenvalues.

    Used as a synthetic stimulus distribution for PCA-style selectivity
    analysis (rules that find principal components should align with
    the top eigenvector).

    Args:
        N: dimensionality.
        k: number of samples.
        seed: RNG seed.
        decay_rate: eigenvalue decay base (eig_i = decay_rate**i).

    Returns:
        Array of shape (k, N).
    """
    key = random.PRNGKey(seed)

    # Exponentially decaying eigenvalues
    eigenvalues = decay_rate ** jnp.arange(N)

    # Random orthogonal rotation (sign-flipped to ensure det(Q) = 1)
    key, subkey = random.split(key)
    A = random.normal(subkey, (N, N))
    Q, R = jnp.linalg.qr(A)
    Q = Q * jnp.sign(jnp.diag(R))

    # Sample, scale by sqrt(eig), rotate
    key, subkey = random.split(key)
    Z = random.normal(subkey, (k, N))
    X = Z * jnp.sqrt(eigenvalues)
    X = X @ Q.T
    return X


# ============================================================================
# Live streams (no interaction with network)
# ============================================================================

@jit
def stream_slow_feature(state, motor, rewards, key):
    """Two-channel input: slow binary signal + fast Gaussian noise.

    From Halvagal & Zenke 2023. Tests whether a rule preferentially
    represents the slow-varying channel rather than the noisy one.

    Args:
        state: int scalar in {0, 1} — current slow-channel state.
        motor: unused (kept for API compatibility).
        rewards: jnp.array([prob_switch, variance]) — slow-channel
            switch probability per step, and noise variance.
        key: PRNG key.

    Returns:
        rewards_out: shape (2,) — [slow_input, noise_input].
        action: current slow-channel state (after possible switch).
        new_state: updated slow state.
        new_key: split key.
    """
    prob_switch, variance = rewards
    key, subkey1, subkey2 = random.split(key, 3)

    # Possibly toggle the slow channel
    switch = random.bernoulli(subkey1, prob_switch)
    new_state = jnp.where(switch, 1 - state, state)

    slow_input = new_state.astype(jnp.float32)
    noise = jnp.abs(random.normal(subkey2) * jnp.sqrt(variance))

    rewards_out = jnp.array([slow_input[0], noise])
    action = new_state
    return rewards_out, action, new_state, key


def stream_expect_reverse(state, motor, rewards, key):
    """From Mainen & Sejnowski. Returns (reward, action, state, key).

    NOT YET IMPLEMENTED — placeholder for the expectation-reversal task
    needed for F7 predictions (or possibly cut from the paper). Decide
    by week 7.
    """
    raise NotImplementedError(
        "stream_expect_reverse is a placeholder. "
        "Decide by week 7 whether this task is needed for F7 (predictions)."
    )
