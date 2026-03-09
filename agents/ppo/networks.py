"""Flax neural-network definitions for the PPO agent.

Architecture
------------
Both policy and value networks are **MLPs** with ``tanh`` activations
and layer sizes ``(512, 256, 128)`` (configurable).

The **policy** outputs ``(mean, log_std)`` for a diagonal-Gaussian
action distribution.

The **value** network outputs a single scalar state-value.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


# ── Policy Network ───────────────────────────────────────────────────


class PolicyNetwork(nn.Module):
    """Diagonal-Gaussian policy MLP.

    Attributes:
        action_size: dimensionality of the action space.
        hidden_sizes: tuple of hidden-layer widths.
        init_log_std: initial value for the learned log-std parameter.
    """

    action_size: int
    hidden_sizes: Sequence[int] = (512, 256, 128)
    init_log_std: float = -0.5

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            obs: (…, obs_dim) observation tensor.

        Returns:
            ``(mean, log_std)`` each of shape ``(…, action_size)``.
        """
        x = obs
        for size in self.hidden_sizes:
            x = nn.Dense(size, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(x)
            x = nn.tanh(x)

        mean = nn.Dense(
            self.action_size,
            kernel_init=nn.initializers.orthogonal(0.01),
        )(x)

        # Learnable per-action log-std (state-independent)
        log_std = self.param(
            "log_std",
            lambda _rng, shape: jnp.full(shape, self.init_log_std),
            (self.action_size,),
        )
        # Broadcast to batch dims
        log_std = jnp.broadcast_to(log_std, mean.shape)
        return mean, log_std


# ── Value Network ────────────────────────────────────────────────────


class ValueNetwork(nn.Module):
    """Scalar state-value MLP.

    Attributes:
        hidden_sizes: tuple of hidden-layer widths.
    """

    hidden_sizes: Sequence[int] = (512, 256, 128)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            obs: (…, obs_dim) observation tensor.

        Returns:
            Scalar value ``(…,)``.
        """
        x = obs
        for size in self.hidden_sizes:
            x = nn.Dense(size, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(x)
            x = nn.tanh(x)

        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return jnp.squeeze(value, axis=-1)


# ── Combined container ───────────────────────────────────────────────


class PPONetworks:
    """Lightweight container holding both policy and value networks.

    Not a Flax module itself – just a convenience wrapper for parameter
    initialisation and forward passes.

    Attributes:
        policy: :class:`PolicyNetwork` instance.
        value: :class:`ValueNetwork` instance.
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_sizes: Sequence[int] = (512, 256, 128),
        init_log_std: float = -0.5,
    ):
        self.obs_size = obs_size
        self.action_size = action_size
        self.policy = PolicyNetwork(
            action_size=action_size,
            hidden_sizes=hidden_sizes,
            init_log_std=init_log_std,
        )
        self.value = ValueNetwork(hidden_sizes=hidden_sizes)

    def init(self, rng: jax.Array):
        """Initialise both networks and return a combined param dict.

        Returns:
            ``{"policy": …, "value": …}`` Flax parameter trees.
        """
        rng_p, rng_v = jax.random.split(rng)
        dummy_obs = jnp.zeros((self.obs_size,))
        policy_params = self.policy.init(rng_p, dummy_obs)
        value_params = self.value.init(rng_v, dummy_obs)
        return {"policy": policy_params, "value": value_params}
