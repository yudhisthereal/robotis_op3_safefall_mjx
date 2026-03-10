"""PPO Agent – orchestrates network init, rollout collection, and updates.

Designed for a **fully JIT-compiled training loop** with ``jax.lax.scan``
over rollout steps and minibatch updates.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Tuple

from flax import struct
import jax
import jax.numpy as jnp
import optax

from agents.ppo.losses import compute_gae, gaussian_log_prob, ppo_loss
from agents.ppo.networks import PPONetworks
from utils.config import Config
from utils.replay_buffer import RolloutBuffer


# ── Training state ───────────────────────────────────────────────────


@struct.dataclass
class TrainState:
    """Immutable PPO training state carried through the training loop."""
    params: Any           # {"policy": …, "value": …}
    opt_state: Any        # Optax optimiser state
    step: jnp.ndarray     # scalar – global gradient step
    rng: jax.Array        # PRNG key

# ── PPO Agent ────────────────────────────────────────────────────────


class PPOAgent:
    """Stateless PPO agent operating on :class:`TrainState`.

    All heavy methods return *new* ``TrainState`` objects; nothing is
    mutated in-place.

    Args:
        config: project-wide configuration.
        obs_size: observation dimensionality.
        action_size: action dimensionality.
    """

    def __init__(self, config: Config, obs_size: int, action_size: int):
        self.config = config
        self.obs_size = obs_size
        self.action_size = action_size

        # Networks
        self.networks = PPONetworks(
            obs_size=obs_size,
            action_size=action_size,
            hidden_sizes=config.hidden_sizes,
            init_log_std=config.init_log_std,
        )

        # Optimiser
        if config.lr_schedule == "linear":
            lr_schedule = optax.linear_schedule(
                init_value=config.learning_rate,
                end_value=0.0,
                transition_steps=config.total_timesteps,
            )
        else:
            lr_schedule = config.learning_rate

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(lr_schedule, eps=1e-5),
        )
        self._lr_schedule = lr_schedule

    # ── Initialisation ───────────────────────────────────────────────

    def init(self, rng: jax.Array) -> TrainState:
        """Create the initial training state."""
        rng, rng_net = jax.random.split(rng)
        params = self.networks.init(rng_net)
        opt_state = self.optimizer.init(params)
        return TrainState(
            params=params,
            opt_state=opt_state,
            step=jnp.array(0, dtype=jnp.int32),
            rng=rng,
        )

    # ── Policy forward passes ────────────────────────────────────────

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_action(
        self, params: Any, obs: jnp.ndarray, rng: jax.Array
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample actions from the current policy.

        Args:
            params: network parameters.
            obs: (N, obs_dim) observations.
            rng: PRNG key.

        Returns:
            ``(actions, log_probs)`` each of shape ``(N, act_dim)`` / ``(N,)``.
        """
        mean, log_std = self.networks.policy.apply(params["policy"], obs)
        std = jnp.exp(log_std)
        noise = jax.random.normal(rng, mean.shape)
        actions = mean + std * noise
        log_probs = gaussian_log_prob(mean, log_std, actions)
        return actions, log_probs

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_value(self, params: Any, obs: jnp.ndarray) -> jnp.ndarray:
        """Compute state values.

        Args:
            params: network parameters.
            obs: (N, obs_dim) observations.

        Returns:
            (N,) value estimates.
        """
        return self.networks.value.apply(params["value"], obs)

    # ── PPO update ───────────────────────────────────────────────────

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        train_state: TrainState,
        buffer: RolloutBuffer,
        last_values: jnp.ndarray,
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        """Run a full PPO update epoch on the collected rollout.

        1. Compute GAE advantages.
        2. Flatten buffer to a single batch.
        3. Shuffle and split into minibatches.
        4. Run *update_epochs* passes of gradient descent.

        Returns:
            ``(new_train_state, metrics_dict)``.
        """
        cfg = self.config

        # GAE
        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            last_values, cfg.gamma, cfg.gae_lambda,
        )
        buffer = buffer.replace(advantages=advantages, returns=returns)

        # Flatten (T, N, …) → (T*N, …)
        flat = buffer.flatten()
        batch_size = flat.obs.shape[0]

        def _epoch_step(carry, _epoch_rng):
            params, opt_state = carry
            perm = jax.random.permutation(_epoch_rng, batch_size)

            def _minibatch_step(carry2, start_idx):
                params2, opt_state2 = carry2
                idx = jax.lax.dynamic_slice(perm, (start_idx,), (cfg.minibatch_size,))
                mb_obs = flat.obs[idx]
                mb_actions = flat.actions[idx]
                mb_old_lp = flat.log_probs[idx]
                mb_adv = flat.advantages[idx]
                mb_ret = flat.returns[idx]

                # Normalise advantages
                mb_adv = jnp.where(
                    cfg.normalize_advantages,
                    (mb_adv - jnp.mean(mb_adv)) / (jnp.std(mb_adv) + 1e-8),
                    mb_adv,
                )

                loss_fn = functools.partial(
                    ppo_loss,
                    self.networks.policy.apply,
                    self.networks.value.apply,
                    clip_eps=cfg.clip_eps,
                    entropy_coeff=cfg.entropy_coeff,
                    value_loss_coeff=cfg.value_loss_coeff,
                )
                grad_fn = jax.grad(loss_fn, argnums=0, has_aux=True)
                grads, info = grad_fn(
                    params2, mb_obs, mb_actions, mb_old_lp, mb_adv, mb_ret,
                )
                updates, new_opt = self.optimizer.update(grads, opt_state2, params2)
                new_params = optax.apply_updates(params2, updates)
                return (new_params, new_opt), info

            starts = jnp.arange(0, batch_size, cfg.minibatch_size)
            (params, opt_state), infos = jax.lax.scan(
                _minibatch_step, (params, opt_state), starts
            )
            return (params, opt_state), infos

        rng, epoch_rng = jax.random.split(train_state.rng)
        epoch_keys = jax.random.split(epoch_rng, cfg.update_epochs)

        (new_params, new_opt), all_infos = jax.lax.scan(
            _epoch_step, (train_state.params, train_state.opt_state), epoch_keys
        )

        # Average metrics across epochs and minibatches
        metrics = jax.tree.map(lambda x: jnp.mean(x), all_infos)

        # Current learning rate
        if callable(self._lr_schedule):
            lr = self._lr_schedule(train_state.step)
        else:
            lr = self._lr_schedule
        metrics["learning_rate"] = lr

        new_state = TrainState(
            params=new_params,
            opt_state=new_opt,
            step=train_state.step + 1,
            rng=rng,
        )
        return new_state, metrics
