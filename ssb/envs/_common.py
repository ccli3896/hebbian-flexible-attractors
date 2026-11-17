"""Shared utilities for environments.

Pulled out of bandits.py because the same logic — "argmax with uniform
tie-breaking" — appears in bandit_unbait, bandit_bait, bandit_peak, and
the matching-pennies env. Centralizing it (a) removes the inconsistency
between the lax.cond version and the additive-noise version and (b)
makes it easy to keep the random key discipline straight, which had
gotten muddled in a few spots.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random


def argmax_with_random_tiebreak(motor: jnp.ndarray, key: jax.Array) -> tuple[jnp.ndarray, jax.Array]:
    """Pick the argmax of ``motor`` with uniform random tie-breaking.

    Uses the additive-noise trick — adds a tiny uniform perturbation to
    the *tied* entries only, then argmaxes. This is jit/vmap friendly
    and avoids the ``lax.cond`` branching in the original bandits.py
    implementations.

    The returned key is split from the input key, so callers can chain
    further random operations without correlation.

    Args:
        motor: shape (..., n_actions). Action scores.
        key: PRNG key, consumed.

    Returns:
        action: integer index of the chosen action, shape (...,).
        new_key: split key for downstream use.
    """
    key, subkey = random.split(key)
    max_val = jnp.max(motor, axis=-1, keepdims=True)
    is_max = motor == max_val
    noise = random.uniform(subkey, motor.shape)
    # Multiplying noise by is_max means only tied entries get jittered.
    # Untied entries stay at their original values; the global argmax
    # picks the unique max if there is one, else a random tied entry.
    return jnp.argmax(motor + noise * is_max, axis=-1), key
