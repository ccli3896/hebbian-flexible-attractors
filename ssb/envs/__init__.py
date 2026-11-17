"""Environments: bandit, slow-features, matching-pennies, foraging.

Every env follows the common signature:

    env(state, motor, env_par, key) -> (reward, action, new_state, new_key)

so they can be passed interchangeably to ``ssb.utils.run.run_env``.

Submodules:
    bandits         — 2AFC environments (deterministic, unbaited, baited, foraging)
    matchingpennies — adversarial pattern-matching opponent
    streams         — non-interactive input streams for representation learning
"""

from ssb.envs import bandits
from ssb.envs import matchingpennies
from ssb.envs import streams
