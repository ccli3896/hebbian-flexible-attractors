"""Registry of network architectures used in the paper.

Each entry maps an architecture name to a function that takes
``(input_dim, n_motor)`` and returns the ``hps['sizes']`` list of
``[N_excitatory, N_inhibitory]`` pairs per layer.

Motor-neuron count depends on the environment, not the architecture:
- Decision tasks (bandit_bait, bandit_unbait, det, matchingpennies) use n_motor=2:
  the action is argmax over two motor neurons.
- Perceptual tasks (selectivity, slowfeature) use n_motor=1: the
  motor neuron's response is measured directly.
"""

from __future__ import annotations

# Environments that emit a discrete 2-arm action via argmax of motor.
DECISION_ENVS = {'bandit_bait', 'bandit_unbait', 'det', 'matchingpennies'}


def get_n_motor(env_name: str) -> int:
    """Return the motor-layer width for the given environment.

    Decision envs get 2 (action = argmax over two motor neurons).
    Perceptual envs get 1 (motor activity is measured directly).
    """
    return 2 if env_name in DECISION_ENVS else 1


ARCHITECTURES = {
    # Legacy names preserved for the F2/F4 master sweep.
    'small': lambda i, m: [[i, 0], [m, 0]],
    'big':   lambda i, m: [[i, 0], [50, 0], [m, 0]],

    # F5 width scan at L=2 (one hidden layer of width N).
    'N2_L2':   lambda i, m: [[i, 0], [2, 0],   [m, 0]],
    'N10_L2':  lambda i, m: [[i, 0], [10, 0],  [m, 0]],
    'N50_L2':  lambda i, m: [[i, 0], [50, 0],  [m, 0]],
    'N200_L2': lambda i, m: [[i, 0], [200, 0], [m, 0]],

    # F5 depth scan at N=50.
    'N50_L3':  lambda i, m: [[i, 0], [50, 0], [50, 0], [m, 0]],
}


def get_sizes(arch_name: str, input_dim: int, n_motor: int) -> list:
    """Build hps['sizes'] for the given architecture, input dim, motor count.

    Raises KeyError if arch_name is not in the registry.
    """
    if arch_name not in ARCHITECTURES:
        raise KeyError(
            f"Unknown architecture {arch_name!r}. "
            f"Known: {sorted(ARCHITECTURES.keys())}"
        )
    return ARCHITECTURES[arch_name](input_dim, n_motor)


def list_architectures() -> list[str]:
    """Return all registered architecture names, sorted."""
    return sorted(ARCHITECTURES.keys())
