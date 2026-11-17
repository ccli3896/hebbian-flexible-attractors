"""Two-alternative forced-choice (2AFC) environments.

Every env in this module follows the standard contract:

    env(state, motor, env_par, key) -> (reward, action, new_state, new_key)

so it slots directly into ``ssb.utils.run.run_env``.

We have - action A
        - environment state B
        - environment response (or "reward") S.

A ∈ {0,1}, representing two options.
B is a 2-element binary vector with the available reward for each action.
The environment function ENV(A, B) = S indexes B by A and returns S.

EXAMPLE:
    If B=[0,0], ENV(A=0, B) = 0 and ENV(A=1, B) = 0.
    If B=[1,0], ENV(A=0, B) = 1 and ENV(A=1, B) = 0.

THE ENVIRONMENTS:

1. Deterministic (``bandit_det``): B = env_par directly.

2. Two-armed bandits. Bernoulli trials parameterized by ``probs`` (2 elements).
   Probabilities typically 0.3; Bari et al. 2019 use 0.45 for mice.

   - ``bandit_unbait``: Bernoulli trials redrawn every trial. Standard RL bandit.
   - ``bandit_bait``: Bernoulli trials only redrawn when the slot is empty. If the
     reward was placed and not collected, it stays. Encourages exploration: if
     P[A] > 0, waiting long enough guarantees a reward.

3. ``bandit_peak``: 2D foraging — reward iff the step heads toward ``peak``.

Linear-motor variants (``bandit_linear*``) take a scalar motor activity rather
than a 2-d argmax. Used in the smallest networks where the motor layer is one
neuron.

CHANGE LOG (vs original):
- Fixed a subtle key-reuse bug in ``bandit_unbait`` and ``bandit_linear_unbait``:
  the same subkey was used for the tie-breaking noise AND the bernoulli reward
  draw, which statistically couples the random part of the action choice to
  the reward outcome. The fix splits an additional subkey for the bernoulli
  draw. Behavior is identical in distribution for the deterministic-action
  case; the change only affects ties, but matters for probability-matching
  curves accumulated over millions of steps.
- Tie-breaking logic centralized in ``ssb.envs._common.argmax_with_random_tiebreak``.
- Bait state-update logic centralized as ``_baited_state_update``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random

from ssb.envs._common import argmax_with_random_tiebreak


# ============================================================================
# Internal helper for baited bandits
# ============================================================================

def _baited_state_update(
    state: jnp.ndarray,
    probs: jnp.ndarray,
    action: jnp.ndarray,
    key: jax.Array,
) -> tuple[jnp.ndarray, jax.Array]:
    """Re-bait empty slots after action consumption.

    For each slot independently:
      - If the slot had reward AND was chosen, set to 0.0 (consumed).
      - Otherwise, leave the existing reward if any.
      - If now empty, sample a fresh bernoulli(prob) reward.

    This is the shared body that ``bandit_bait`` and ``bandit_linear_bait``
    were duplicating.
    """
    keys = random.split(key, len(state) + 1)

    def update_action(one_state, k, prob, action_idx, prev_action):
        # Branchless: consume reward if (slot had reward) AND (slot was chosen).
        one_state = jax.lax.cond(
            (one_state != 0.0) & (action_idx == prev_action),
            lambda _: 0.0,
            lambda _: one_state,
            operand=None,
        )
        # Re-bait empty slots probabilistically.
        bait = random.bernoulli(k, prob).astype(jnp.float32)
        return jnp.where(one_state == 0.0, bait, one_state)

    v_update = jax.vmap(update_action, in_axes=(0, 0, 0, 0, None))
    new_state = v_update(state, keys[:-1], probs, jnp.arange(len(state)), action)
    return new_state, keys[-1]


# ============================================================================
# Foraging
# ============================================================================

@jit
def bandit_peak(state, motor, peak, key):
    """2D foraging: reward if the step heads toward ``peak``.

    motor: shape (4,) -> [N, E, S, W]
    state: shape (2,) -> (x, y) position
    peak:  shape (2,) -> (x_peak, y_peak)
    reward = 1 if the chosen step has positive dot-product with (peak - state).
    """
    action, key = argmax_with_random_tiebreak(motor, key)

    step = jax.lax.switch(
        action,
        [
            lambda: jnp.array([0.0, 1.0]),   # North
            lambda: jnp.array([1.0, 0.0]),   # East
            lambda: jnp.array([0.0, -1.0]),  # South
            lambda: jnp.array([-1.0, 0.0]),  # West
        ],
    )

    new_state = state + step
    to_peak = peak - state
    dot = jnp.dot(step, to_peak)
    reward = jnp.where(dot > 0, 1.0, 0.0)

    return jnp.array([reward]), action, new_state, key


# ============================================================================
# Deterministic and probabilistic 2AFC
# ============================================================================

@jit
def bandit_det(_, motor, rewards, key):
    """Deterministic 2AFC: ``rewards`` is the reward vector indexed by action."""
    action, key = argmax_with_random_tiebreak(motor, key)
    return jnp.array([rewards[action]]), action, _, key


@jit
def bandit_unbait(_, motor, probs, key):
    """Unbaited 2-armed bandit. Rewards redrawn every trial.

    Standard RL bandit: action is argmax(motor) with tie-breaking,
    reward ~ Bernoulli(probs[action]).
    """
    action, key = argmax_with_random_tiebreak(motor, key)
    # Fresh subkey for the bernoulli draw — previously this reused the
    # tie-break subkey, statistically coupling reward to the random part
    # of action choice. Split independently.
    key, reward_key = random.split(key)
    reward = random.bernoulli(reward_key, probs[action]).astype(jnp.int32)
    return jnp.array([reward]), action, _, key


@jit
def bandit_bait(state, motor, probs, key):
    """Baited 2-armed bandit. Rewards persist until collected.

    Inputs:
        state: shape (2,) — current reward presence for each arm.
        motor: motor activities.
        probs: baiting probabilities (re-bait when slot is empty).
        key: PRNG key.

    Returns reward (collected from chosen arm), action, new state
    (post-consumption + re-bait), key.
    """
    action, key = argmax_with_random_tiebreak(motor, key)
    reward = jnp.array(state[action])
    new_state, new_key = _baited_state_update(state, probs, action, key)
    return jnp.array([reward]), action, new_state, new_key


# ============================================================================
# Linear-motor variants (scalar motor activity, sign decides action)
# ============================================================================

@jit
def bandit_linear(_, motor, rewards, key):
    """Deterministic 2AFC with scalar motor.

    motor > 0 → action 1
    motor < 0 → action 0
    motor == 0 → random
    """
    key, subkey = random.split(key)

    def zero_case(_):
        return random.randint(subkey, (), 0, 2)

    def non_zero_case(_):
        return (motor > 0).astype(jnp.int32)

    action = jax.lax.cond(motor == 0, zero_case, non_zero_case, operand=None)
    return rewards[action], action, _, key


@jit
def bandit_linear_unbait(_, motor, probs, key):
    """Unbaited 2AFC with scalar motor. See ``bandit_linear`` for action rule."""
    key, action_key = random.split(key)

    def zero_case(_):
        return random.randint(action_key, (), 0, 2)

    def non_zero_case(_):
        return (motor > 0).astype(jnp.int32)

    action = jax.lax.cond(motor == 0, zero_case, non_zero_case, operand=None)
    # Independent subkey for the reward draw — see bandit_unbait note.
    key, reward_key = random.split(key)
    reward = random.bernoulli(reward_key, probs[action]).astype(jnp.int32)
    return reward, action, _, key


@jit
def bandit_linear_bait(state, motor, probs, key):
    """Baited 2AFC with scalar motor.

    Action is deterministic (argmax of [0, motor]), no tie-break needed.
    State updates use the shared bait-replenishment helper.
    """
    action = jnp.argmax(jnp.array([0.0, motor]))
    reward = jnp.array(state[action])
    new_state, new_key = _baited_state_update(state, probs, action, key)
    return reward, action, new_state, new_key


# ============================================================================
# ITI variants — interleave null (zero-input) steps between bandit trials
# ============================================================================
#
# State layout: jnp.array([arm0_bait, arm1_bait, phase_counter])
#   phase_counter counts UP each step.
#   When phase_counter % (iti + 1) == 0  → bandit step fires, counter resets.
#   Otherwise                            → ITI step: zero reward, motor ignored.
#
# env_par layout: jnp.array([prob0, prob1, iti])
#   iti is cast to int internally; keep it float in env_par for JAX tracing.
#
# Both unbait and bait variants share the same phase logic via _iti_step.

def _iti_phase(state, motor, env_par, key, _bandit_fn):
    """Shared ITI wrapper. Calls ``_bandit_fn`` on bandit steps, zeros on ITI.

    Args:
        state    : jnp.array([arm0, arm1, counter])
        motor    : motor activities (passed through to bandit on bandit steps)
        env_par  : jnp.array([prob0, prob1, iti])
        key      : PRNG key
        _bandit_fn: callable(bait_state, motor, probs, key) -> (reward, action, new_bait_state, key)

    Returns:
        reward, action, new_state, new_key
    """
    bait_state = state[:2]
    counter    = state[2]
    probs      = env_par[:2]
    iti        = env_par[2].astype(jnp.int32)

    # Bandit step when counter == 0, ITI step otherwise.
    is_bandit = (counter == 0)

    def bandit_step(_):
        reward, action, new_bait, new_key = _bandit_fn(bait_state, motor, probs, key)
        return (jnp.array([reward[0]], dtype=jnp.float32),
                jnp.array(action, dtype=jnp.int32),
                new_bait.astype(jnp.float32),
                new_key)

    def iti_step(_):
        # Zero reward, action 0 (ignored by caller during ITI), state unchanged.
        return (jnp.array([0.0], dtype=jnp.float32),
                jnp.array(0, dtype=jnp.int32),
                bait_state.astype(jnp.float32),
                key)

    reward, action, new_bait, new_key = jax.lax.cond(is_bandit, bandit_step, iti_step, operand=None)

    # Advance counter; reset to iti after a bandit step fires.
    new_counter = jax.lax.cond(
        is_bandit,
        lambda _: iti.astype(jnp.float32),          # just fired → count down from iti
        lambda _: (counter - 1).astype(jnp.float32),  # still in ITI → keep counting down
        operand=None,
    )

    new_state = jnp.array([new_bait[0], new_bait[1], new_counter.astype(jnp.float32)])
    return reward, action, new_state, new_key


@jit
def bandit_unbait_iti(state, motor, env_par, key):
    """Unbaited 2-armed bandit with ITI null steps.

    env_par: jnp.array([prob0, prob1, iti])
    state  : jnp.array([unused0, unused1, phase_counter])
    """
    return _iti_phase(state, motor, env_par, key, bandit_unbait)


@jit
def bandit_bait_iti(state, motor, env_par, key):
    """Baited 2-armed bandit with ITI null steps.

    env_par: jnp.array([prob0, prob1, iti])
    state  : jnp.array([arm0_bait, arm1_bait, phase_counter])
    """
    return _iti_phase(state, motor, env_par, key, bandit_bait)


def make_iti_state(iti: int) -> jnp.ndarray:
    """Initial state for ITI bandits: bait empty, counter starts at iti."""
    return jnp.array([0.0, 0.0, float(iti)])


def make_iti_env_par(prob0: float, prob1: float, iti: int) -> jnp.ndarray:
    """Pack env_par for ITI bandits."""
    return jnp.array([prob0, prob1, float(iti)])


# ============================================================================
# Blocked-exposure variants — bandit trials come in BLOCKS, with a rest gap
# between blocks (rather than a null step between every single trial).
# ============================================================================
#
# The ITI envs above space out *every* trial. The "exposure block" view groups
# trials into blocks: ``block`` real bandit trials fire back-to-back, then
# ``gap`` null (zero-input, motor-ignored) rest steps run before the next block.
# This is the structure you want when asking whether learning efficiency rises
# *per exposure block* (massed block + rest) rather than per single trial — and
# it's during the rest gap that PC weights keep settling / "jump" toward the
# learned fixed point.
#
# State layout: jnp.array([arm0_bait, arm1_bait, position])
#   position counts UP each step, modulo (block + gap).
#   position < block  → bandit step fires.
#   position >= block → rest step: zero reward, motor ignored, state frozen.
#
# env_par layout: jnp.array([prob0, prob1, block, gap])
#   block and gap are cast to int internally; keep them float for JAX tracing.

def _blocked_phase(state, motor, env_par, key, _bandit_fn):
    """Shared blocked-exposure wrapper. Bandit step while position < block,
    rest (null) step otherwise. Mirrors ``_iti_phase`` but with block structure.
    """
    bait_state = state[:2]
    position   = state[2]
    probs      = env_par[:2]
    block      = env_par[2].astype(jnp.int32)
    gap        = env_par[3].astype(jnp.int32)
    cycle      = block + gap

    is_bandit = (position.astype(jnp.int32) < block)

    def bandit_step(_):
        reward, action, new_bait, new_key = _bandit_fn(bait_state, motor, probs, key)
        return (jnp.array([reward[0]], dtype=jnp.float32),
                jnp.array(action, dtype=jnp.int32),
                new_bait.astype(jnp.float32),
                new_key)

    def rest_step(_):
        return (jnp.array([0.0], dtype=jnp.float32),
                jnp.array(0, dtype=jnp.int32),
                bait_state.astype(jnp.float32),
                key)

    reward, action, new_bait, new_key = jax.lax.cond(
        is_bandit, bandit_step, rest_step, operand=None)

    new_position = (position + 1.0) % cycle.astype(jnp.float32)
    new_state = jnp.array([new_bait[0], new_bait[1], new_position])
    return reward, action, new_state, new_key


@jit
def bandit_unbait_blocked(state, motor, env_par, key):
    """Unbaited 2-armed bandit with blocked exposure + rest gaps.

    env_par: jnp.array([prob0, prob1, block, gap])
    state  : jnp.array([unused0, unused1, position])
    """
    return _blocked_phase(state, motor, env_par, key, bandit_unbait)


@jit
def bandit_bait_blocked(state, motor, env_par, key):
    """Baited 2-armed bandit with blocked exposure + rest gaps.

    env_par: jnp.array([prob0, prob1, block, gap])
    state  : jnp.array([arm0_bait, arm1_bait, position])
    """
    return _blocked_phase(state, motor, env_par, key, bandit_bait)


def make_blocked_state(block: int, gap: int) -> jnp.ndarray:
    """Initial state for blocked bandits: bait empty, position starts at 0
    (so the very first step begins a block)."""
    return jnp.array([0.0, 0.0, 0.0])


def make_blocked_env_par(prob0: float, prob1: float,
                         block: int, gap: int) -> jnp.ndarray:
    """Pack env_par for blocked-exposure bandits."""
    return jnp.array([prob0, prob1, float(block), float(gap)])


def blocked_bandit_mask(block: int, gap: int, timesteps: int) -> np.ndarray:
    """Boolean mask over raw steps selecting the real bandit trials, so callers
    can drop rest steps the same way the ITI envs do."""
    pos = np.arange(timesteps) % (block + gap)
    return pos < block