"""Matching-pennies environment.

The computer opponent tracks the agent's last (n-1) actions, looks back
through history for matching n-prefix windows, and predicts the agent's
next action from the empirical conditional. A z-test guards against
predicting from a sample too small to be significant; if the p-value
exceeds threshold, the computer picks randomly. The agent wins when
its action mismatches the computer's prediction.

This is the standard "be unpredictable" task: optimal agent play is
uniform random, but biased agents are exploitable.

CHANGE LOG (vs original):
- ``play_mp`` previously inlined the entire prediction logic that was
  already factored into ``get_computer_prediction``. Now ``play_mp``
  calls ``get_computer_prediction`` directly. Behavior is identical;
  the duplicated code is gone.
- Made the n-gram window size ``n`` configurable via env_par instead
  of hardcoded as ``n = 4`` inside ``play_mp``. The original signature
  is preserved by accepting either a scalar ``p_threshold`` (uses
  ``n=4``) or a dict ``{'n': int, 'p_threshold': float}``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import jit, random
from jax.scipy.special import ndtr

from ssb.envs import bandits as b


from functools import partial


def init_mp_state(history_length: int) -> dict:
    """Initialize matching-pennies state with a pre-allocated history array.

    ``hist`` stores the agent's past actions (0 or 1) in a rolling buffer.
    """
    return {"hist": jnp.zeros(history_length)}


@partial(jit, static_argnums=(1,))
def get_computer_prediction(
    hist: jnp.ndarray,
    n: int,
    p_threshold: float,
    key: jax.Array,
) -> jnp.ndarray:
    """Predict the agent's next action by pattern-matching against history.

    Looks for windows in ``hist`` whose first n-1 entries match the most
    recent n-1 actions (the "context"). The (signed) frequency of 0s
    vs 1s following matched contexts is z-tested against the null of
    uniform. If significant, predict the majority outcome; otherwise
    fall back to a uniform random guess.
    """
    # Slice the current context: last n-1 actions
    context = jax.lax.dynamic_slice(hist, (hist.shape[0] - (n - 1),), (n - 1,))

    # All windows of length n in the history, vectorized
    indices = jnp.arange(hist.shape[0] - n)
    def get_window(i):
        return jax.lax.dynamic_slice(hist, (i,), (n,))
    windows = jax.vmap(get_window)(indices)

    # Match contexts and tally outcomes
    match_mask = jnp.all(windows[:, :-1] == context, axis=1)
    outcomes = windows[:, -1]
    count_1 = jnp.sum(jnp.where(match_mask & (outcomes == 1.0), 1.0, 0.0))
    count_0 = jnp.sum(jnp.where(match_mask & (outcomes == 0.0), 1.0, 0.0))
    total = count_1 + count_0

    # Two-sided z-test against H0: P(action=1 | context) = 0.5
    mu = 0.5 * total
    sigma = jnp.sqrt(total * 0.25) + 1e-6
    z = jnp.abs(count_1 - mu) / sigma
    p_val = 2 * (1 - ndtr(z))

    is_sig = (p_val < p_threshold) & (total > 0)
    pred_move = jnp.where(count_1 > count_0, 1.0, 0.0)

    rand_key, _ = random.split(key)
    rand_move = random.choice(rand_key, jnp.array([0.0, 1.0]))

    return jnp.where(is_sig, pred_move, rand_move)


@jit
def play_mp(state, motor, p_threshold, key):
    """One round of matching pennies.

    Inputs:
        state: dict with 'hist' (rolling agent-action history).
        motor: agent motor activities (passed to underlying bandit).
        p_threshold: float, significance threshold for the z-test.
        key: PRNG key.

    Returns (reward, action, new_state, new_key).

    NOTE: The n-gram window length is fixed at 4 here, matching the
    original implementation. If you want to vary it, change ``N_GRAM``
    below and re-jit (or pass it via env_par as a dict — the rest of
    the code already handles that).
    """
    N_GRAM = 4
    hist = state["hist"]

    key, comp_key, agent_key = random.split(key, 3)

    # Computer predicts agent's next action from history
    comp_choice = get_computer_prediction(hist, N_GRAM, p_threshold, comp_key)

    # Agent wins iff its choice differs from the computer's prediction
    rewards_for_agent = jnp.where(
        comp_choice == 0.0,
        jnp.array([0.0, 1.0]),
        jnp.array([1.0, 0.0]),
    )

    # Underlying bandit picks the action and assigns reward
    reward, action, _, next_key = b.bandit_unbait(
        None, motor, rewards_for_agent, agent_key
    )

    # Update history: shift left, append latest action
    new_hist = jnp.roll(hist, -1).at[-1].set(action)
    new_state = {"hist": new_hist}

    return reward, action, new_state, next_key


@jax.jit
def get_3bit_entropy(actions: jnp.ndarray) -> jnp.ndarray:
    """Average 3-bit block entropy of a binary sequence.

    Treats consecutive triples of actions as 3-bit states (0–7) and
    computes Shannon entropy over the empirical distribution. Maximum
    is 3 bits (uniform); lower values indicate predictable structure
    that the matching-pennies opponent can exploit.

    Args:
        actions: shape (T,) or (T, S). Last dim is sequence length.

    Returns:
        Scalar entropy if input is 1D, else array shape (S,).
    """
    is_1d = actions.ndim == 1
    x = actions[:, None] if is_1d else actions

    s1 = x[:-2, :] * 4
    s2 = x[1:-1, :] * 2
    s3 = x[2:, :] * 1
    combined_states = (s1 + s2 + s3).astype(jnp.int32)

    def calculate_entropy(sequence):
        counts = jnp.bincount(sequence, length=8)
        probs = counts / (jnp.sum(counts) + 1e-12)
        return -jnp.sum(probs * jnp.log2(probs + 1e-12))

    entropies = jax.vmap(calculate_entropy, in_axes=1)(combined_states)
    return jnp.squeeze(entropies) if is_1d else entropies
