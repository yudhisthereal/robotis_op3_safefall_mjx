"""Deterministic evaluation rollout for OP3 SafeFall environments.

Runs a fixed number of episodes using the **mean** of the policy
(no sampling) and reports aggregate statistics.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from utils.config import Config


def evaluate(
    policy_apply_fn: Callable,
    params: Any,
    reset_fn: Callable,
    step_fn: Callable,
    rng: jax.Array,
    config: Config,
    num_eval_episodes: int = 16,
) -> Dict[str, float]:
    """Run deterministic evaluation episodes.

    Args:
        policy_apply_fn: ``apply(params, obs) -> (mean, log_std)``.
        params: network parameter tree.
        reset_fn: **single-env** ``reset(rng) -> state``.
        step_fn: **single-env** ``step(state, action, rng) -> state``.
        rng: JAX PRNG key.
        config: project configuration.
        num_eval_episodes: how many episodes to run.

    Returns:
        ``{"mean_reward": …, "episode_length": …, "success_rate": …}``
    """

    def _run_one_episode(rng: jax.Array) -> Tuple[float, int, bool]:
        rng_reset, rng_step = jax.random.split(rng)
        state = reset_fn(rng_reset)

        def _step_fn(carry, _):
            state, rng, total_reward, length, done_flag = carry
            rng, rng_act, rng_env = jax.random.split(rng, 3)

            # Deterministic: use mean of policy
            mean, _log_std = policy_apply_fn(params, state.obs)
            action = mean  # deterministic

            next_state = step_fn(state, action, rng_env)

            # Only accumulate reward while not done
            still_alive = 1.0 - done_flag
            total_reward = total_reward + next_state.reward * still_alive
            length = length + jnp.int32(1) * jnp.int32(still_alive > 0.5)
            done_flag = jnp.maximum(done_flag, next_state.done)

            return (next_state, rng, total_reward, length, done_flag), None

        init = (state, rng_step, 0.0, jnp.int32(0), 0.0)
        (final_state, _, total_reward, length, done), _ = jax.lax.scan(
            _step_fn, init, None, length=config.episode_max_steps
        )
        success = done > 0.5  # episode ended naturally (not timeout)
        return total_reward, length, success

    keys = jax.random.split(rng, num_eval_episodes)
    rewards, lengths, successes = jax.vmap(_run_one_episode)(keys)

    return {
        "mean_reward": float(jnp.mean(rewards)),
        "episode_length": float(jnp.mean(lengths)),
        "success_rate": float(jnp.mean(successes.astype(jnp.float32))),
    }
