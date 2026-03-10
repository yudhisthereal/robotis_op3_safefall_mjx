"""PPO rollout buffer – stores one rollout's worth of transitions.

This is a **stateless** buffer implemented as a frozen dataclass so that
it is fully compatible with ``jax.jit`` (no Python-side mutation).

Usage pattern inside the training loop::

    buffer = RolloutBuffer.create(num_envs, rollout_length, obs_dim, act_dim)
    buffer = buffer.append(step_idx, obs, actions, rewards, values, log_probs, dones, goals)
    ...
    batch = buffer.as_batch(last_values, gamma, gae_lambda)
"""

from __future__ import annotations

from typing import Optional

from flax import struct
import jax
import jax.numpy as jnp


@struct.dataclass
class RolloutBuffer:
    """Immutable PPO rollout storage."""

    obs: jnp.ndarray         # (T, N, obs_dim)
    actions: jnp.ndarray     # (T, N, act_dim)
    rewards: jnp.ndarray     # (T, N)
    values: jnp.ndarray      # (T, N)
    log_probs: jnp.ndarray   # (T, N)
    dones: jnp.ndarray       # (T, N)
    goals: jnp.ndarray       # (T, N, goal_dim)  – may be zeros if unused
    advantages: jnp.ndarray  # (T, N)
    returns: jnp.ndarray     # (T, N)

    @classmethod
    def create(
        cls,
        num_envs: int,
        rollout_length: int,
        obs_dim: int,
        act_dim: int,
        goal_dim: int = 0,
    ) -> "RolloutBuffer":
        """Allocate an empty buffer."""
        T, N = rollout_length, num_envs
        return cls(
            obs=jnp.zeros((T, N, obs_dim)),
            actions=jnp.zeros((T, N, act_dim)),
            rewards=jnp.zeros((T, N)),
            values=jnp.zeros((T, N)),
            log_probs=jnp.zeros((T, N)),
            dones=jnp.zeros((T, N)),
            goals=jnp.zeros((T, N, max(goal_dim, 1))),
            advantages=jnp.zeros((T, N)),
            returns=jnp.zeros((T, N)),
        )

    def store(
        self,
        t: int,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        log_probs: jnp.ndarray,
        dones: jnp.ndarray,
        goals: Optional[jnp.ndarray] = None,
    ) -> "RolloutBuffer":
        """Store data for time-step *t*. Returns a new buffer."""
        new_goals = self.goals if goals is None else self.goals.at[t].set(goals)
        return self.replace(
            obs=self.obs.at[t].set(obs),
            actions=self.actions.at[t].set(actions),
            rewards=self.rewards.at[t].set(rewards),
            values=self.values.at[t].set(values),
            log_probs=self.log_probs.at[t].set(log_probs),
            dones=self.dones.at[t].set(dones),
            goals=new_goals,
        )

    def compute_gae(
        self,
        last_values: jnp.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> "RolloutBuffer":
        """Compute GAE advantages and returns. Returns a new buffer.

        Args:
            last_values: (N,) value estimates for the state *after* the
                last stored transition.
            gamma: discount factor.
            gae_lambda: GAE lambda.

        Returns:
            Updated buffer with ``advantages`` and ``returns`` filled in.
        """

        def _scan_fn(carry, t_data):
            last_gae, next_values = carry
            rewards, values, dones = t_data
            delta = rewards + gamma * next_values * (1.0 - dones) - values
            gae = delta + gamma * gae_lambda * (1.0 - dones) * last_gae
            return (gae, values), gae

        # Reverse scan (from T-1 to 0)
        init_carry = (jnp.zeros_like(last_values), last_values)
        scan_data = (self.rewards, self.values, self.dones)

        # Reverse the time axis for the scan
        reversed_data = jax.tree.map(lambda x: x[::-1], scan_data)
        _, advantages_rev = jax.lax.scan(_scan_fn, init_carry, reversed_data)
        advantages = advantages_rev[::-1]  # un-reverse

        returns = advantages + self.values
        return self.replace(advantages=advantages, returns=returns)

    def flatten(self):
        """Flatten (T, N, ...) → (T*N, ...) for minibatch sampling."""
        T, N = self.obs.shape[0], self.obs.shape[1]
        flat = lambda x: x.reshape((T * N,) + x.shape[2:])
        return self.replace(
            obs=flat(self.obs),
            actions=flat(self.actions),
            rewards=flat(self.rewards),
            values=flat(self.values),
            log_probs=flat(self.log_probs),
            dones=flat(self.dones),
            goals=flat(self.goals),
            advantages=flat(self.advantages),
            returns=flat(self.returns),
        )
