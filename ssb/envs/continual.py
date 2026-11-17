"""Continual-learning task: two contingencies on separate input channels.

Packaged port of ``notebooks/cl_nonlinear.py`` so the continual-learning
measurement can run inside a SLURM sweep rather than only inside a notebook.
The task itself is unchanged; the differences are:

* no module-level ``jax.config.update('jax_platform_name', 'cpu')`` -- the
  sbatch templates set ``JAX_PLATFORMS``, and a hard-coded platform here would
  silently override that;
* the reward magnitude is a parameter rather than a hard-coded ``[0, 1]``;
* the per-block logs are reduced to summary statistics inside this module, so a
  sweep over many cells never materialises the full weight trace.

A 2-input -> 2-output network runs a deterministic 2AFC bandit whose contingency
reverses every ``timesteps`` steps. With ``separate_inputs=True`` the two
contingencies arrive on different input channels, so each is stored on its own
synapse and a rule that gates plasticity by input can hold both at once (the
continual-learning test, F3e-g). With ``separate_inputs=False`` both are routed
through one shared channel and must collide on a single synapse (the
overwriting control, S7).
"""

from __future__ import annotations

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import vmap

from ssb.models import ssb, rules
from ssb.envs import bandits
from ssb.utils import batch_inits
from ssb.utils.architectures import get_sizes

# The task always has 2 input and 2 output units -- both contingencies need
# their own input channel for the separate-inputs condition to be meaningful --
# but any number of hidden layers may sit between them. `cl_sizes` builds the
# layer spec for a named architecture from the shared registry, so 'small' is
# the original [[2,0],[2,0]] and 'big' is [[2,0],[50,0],[2,0]].
CL_SIZES = jnp.array([[2, 0], [2, 0]])          # == cl_sizes('small')


def cl_sizes(arch='small'):
    """Layer spec for the continual-learning task at a named architecture."""
    return jnp.array(get_sizes(arch, 2, 2))


def _layer_bounds(sizes):
    """Neuron index boundaries per layer, from a concrete sizes array."""
    per_layer = [int(e) + int(i) for e, i in np.asarray(sizes)]
    bounds, c = [], 0
    for n in per_layer:
        bounds.append((c, c + n)); c += n
    return bounds, c


def effective_io_block(weights, tern, bounds):
    """Effective 2x2 input->output gain, through however many hidden layers.

    The readout the task cares about is how strongly each input channel drives
    each output. With no hidden layer that is just the input->output weight
    block. With hidden layers the inputs do not touch the outputs directly, so
    the effective gain is the product of the successive layer blocks -- the
    linear-network equivalent of the same quantity. Each block is masked by
    ``tern`` so entries that are not synapses (and therefore never update)
    cannot contribute. For a 2-layer net this returns exactly
    ``weights[:, :2, 2:]``, so the original behaviour is preserved bit-for-bit.
    """
    eff = None
    for (a0, a1), (b0, b1) in zip(bounds[:-1], bounds[1:]):
        blk = weights[:, a0:a1, b0:b1] * jnp.abs(tern[:, a0:a1, b0:b1])
        eff = blk if eff is None else eff @ blk
    return eff

# Clip applied to the instantaneous memory index before averaging, matching the
# notebook analysis (keeps a single diverging seed from dominating the mean).
MEM_CLIP = 0.3


def run_cl(hps_batch, weight_rule, timesteps, separate_inputs=True, num_revs=10,
           settle_time=1, update_acts=ssb.update_acts, reward=1.0):
    """Run the reversal sequence; yields per-block (actions, input->output weights).

    The architecture is taken from ``hps_batch['sizes']``, so any number of
    hidden layers is allowed as long as there are 2 inputs and 2 outputs.

    Returns a list of ``num_revs`` tuples ``(action_batch, weights)`` where
    ``action_batch`` is (timesteps, batch) and ``weights`` is
    (timesteps, batch, 2, 2), the effective input->output gain indexed
    ``[input_i, output_j]`` (see ``effective_io_block``).
    """
    sizes = np.asarray(hps_batch['sizes'][0])
    if sizes[0][0] != 2 or sizes[-1][0] != 2:
        raise ValueError(
            "continual-learning task needs 2 input and 2 output units; got "
            f"sizes {sizes.tolist()}. Hidden layers between them are fine.")
    bounds, n_total = _layer_bounds(sizes)
    batch_size = len(hps_batch['seed'])

    update_weights, update_states = rules.rule2fns(weight_rule)

    update_acts_batch = vmap(update_acts)
    update_weights_batch = vmap(update_weights)
    act_noise_batch = vmap(ssb.act_noise)
    update_states_batch = vmap(update_states)
    bandit_batch = vmap(bandits.bandit_unbait)

    base = jnp.array([1., 0.]) if separate_inputs else jnp.array([0., 1.])

    @partial(jax.jit, static_argnums=(1,))
    def step_fn(state, t):
        env_state_batch, acts, states, weights, tern, key, rewards = state

        set_r_batch = jnp.tile(rewards, (batch_size, 1))
        reward_batch, action_batch, env_state_batch, key = bandit_batch(
            env_state_batch, acts[:, -2:], set_r_batch, key)

        set_r_batch = jnp.tile(rewards, (batch_size, 1))
        rb = jnp.broadcast_to(reward_batch.reshape(batch_size, 1), (batch_size, 2))
        input_neurons = rb * set_r_batch * base[0] + rb * jnp.array([1., 0.]) * base[1]

        # Pad the stimulus out to the full neuron count: only the two input
        # units are driven, every hidden and output unit gets zero.
        stimuli = jnp.hstack(
            [input_neurons, jnp.zeros((batch_size, n_total - 2))])

        def full_phase(carry, stimulus_input):
            acts, states, weights, key = carry

            def settle_step(carry, _):
                acts, states, weights, key = carry
                acts = update_acts_batch(stimulus_input, acts, states, weights,
                                         tern, hps_batch)
                return (acts, states, weights, key), None

            (acts, states, weights, key), _ = jax.lax.scan(
                settle_step, (acts, states, weights, key), jnp.arange(settle_time))

            acts, key = act_noise_batch(acts, key, hps_batch)
            states = update_states_batch(acts, states, hps_batch)
            weights = update_weights_batch(stimulus_input, acts, states, weights,
                                           tern, hps_batch)
            return (acts, states, weights, key), None

        carry = acts, states, weights, key
        carry, _ = jax.lax.scan(lambda c, _: full_phase(c, stimuli), carry, None, length=1)
        acts, states, weights, key = carry

        return (env_state_batch, acts, states, weights, tern, key, rewards), \
            (action_batch, effective_io_block(weights, tern, bounds))

    env_state_batch = jnp.array([jnp.zeros(2) for _ in range(batch_size)])
    acts, states, weights, tern, key = batch_inits.init_params_batch(hps_batch)

    revlogs = []
    rewards = jnp.array([0., float(reward)])
    state = (env_state_batch, acts, states, weights, tern, key, rewards)
    for _ in range(num_revs):
        rewards = jnp.flip(rewards)
        state = (*state[:-1], rewards)
        state, logs = jax.lax.scan(step_fn, state, None, length=timesteps)
        revlogs.append(logs)
    return revlogs


def summarize_cl(revlogs, batch_size, timesteps, num_revs, mem_clip=MEM_CLIP):
    """Reduce per-block logs to the statistics the sweep stores.

    Returns a dict of arrays:
      acc       (batch, blocks)  accuracy over the 2nd half of each block --
                                 whether the current contingency was acquired
      end_w     (batch, blocks, 2, 2)  end-of-block input->output weights
      memory    (batch,)   the retention index: both channels held at once,
                           averaged over the 2nd half of the run
      d0_end    (batch, blocks)  channel-0 weight difference at block end
      d1_end    (batch, blocks)  channel-1 weight difference at block end

    ``memory`` is the notebook's index: for each step take
    ``min(-(w01-w00), (w11-w10))``, i.e. the weaker of "channel 0 encodes A" and
    "channel 1 encodes B". It is high only when BOTH associations are held
    simultaneously, which is what distinguishes continual learning from simply
    tracking the current block.
    """
    S, NR, TS = batch_size, num_revs, timesteps
    d0 = np.zeros((S, NR, TS), dtype=np.float32)
    d1 = np.zeros((S, NR, TS), dtype=np.float32)
    end_w = np.zeros((S, NR, 2, 2), dtype=np.float32)
    acc = np.zeros((S, NR), dtype=np.float32)

    for r in range(NR):
        a = np.asarray(revlogs[r][0]).T                    # (batch, T)
        w = np.swapaxes(np.asarray(revlogs[r][1]), 0, 1)   # (batch, T, 2, 2)
        d0[:, r] = w[:, :, 0, 1] - w[:, :, 0, 0]
        d1[:, r] = w[:, :, 1, 1] - w[:, :, 1, 0]
        end_w[:, r] = w[:, -1]
        # Blocks alternate which arm is correct; block r rewards arm (r % 2).
        acc[:, r] = (a[:, TS // 2:] == (0 if r % 2 == 0 else 1)).mean(1)

    inst = np.clip(np.minimum(-d0.reshape(S, NR * TS), d1.reshape(S, NR * TS)),
                   -mem_clip, mem_clip)
    memory = inst[:, inst.shape[1] // 2:].mean(1).astype(np.float32)

    return {
        'acc': acc,
        'end_w': end_w,
        'memory': memory,
        'd0_end': d0[:, :, -1].astype(np.float32),
        'd1_end': d1[:, :, -1].astype(np.float32),
    }
