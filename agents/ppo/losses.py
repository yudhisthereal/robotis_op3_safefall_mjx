"""PPO loss functions – clipped surrogate objective, value loss, entropy.

All functions are **pure JAX** and designed to be called inside a
``jax.jit``-compiled training step.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


# ── GAE computation (standalone, matches replay_buffer but usable separately)


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_values: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalised Advantage Estimation.

    Args:
        rewards: (T, N) rewards.
        values: (T, N) value predictions.
        dones: (T, N) episode-done flags.
        last_values: (N,) bootstrap values for t = T.
        gamma: discount factor.
        gae_lambda: GAE lambda.

    Returns:
        ``(advantages, returns)`` each of shape ``(T, N)``.
    """

    def _scan_fn(carry, t_data):
        last_gae, next_val = carry
        rew, val, done = t_data
        delta = rew + gamma * next_val * (1.0 - done) - val
        gae = delta + gamma * gae_lambda * (1.0 - done) * last_gae
        return (gae, val), gae

    init = (jnp.zeros_like(last_values), last_values)
    reversed_data = (rewards[::-1], values[::-1], dones[::-1])
    _, advantages_rev = jax.lax.scan(_scan_fn, init, reversed_data)
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


# ── Log-probability of a diagonal-Gaussian ───────────────────────────


def gaussian_log_prob(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    actions: jnp.ndarray,
) -> jnp.ndarray:
    """Log-probability under a diagonal Gaussian.

    Args:
        mean: (…, A) means.
        log_std: (…, A) log standard deviations.
        actions: (…, A) sampled actions.

    Returns:
        (…,) log-probabilities (summed over action dims).
    """
    std = jnp.exp(log_std)
    log_p = -0.5 * (
        ((actions - mean) / (std + 1e-8)) ** 2
        + 2.0 * log_std
        + jnp.log(2.0 * jnp.pi)
    )
    return jnp.sum(log_p, axis=-1)


def gaussian_entropy(log_std: jnp.ndarray) -> jnp.ndarray:
    """Entropy of a diagonal Gaussian.

    Args:
        log_std: (…, A) log standard deviations.

    Returns:
        (…,) entropy (summed over action dims).
    """
    return jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)


# ── PPO loss ─────────────────────────────────────────────────────────


def ppo_loss(
    policy_apply,
    value_apply,
    params,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    clip_eps: float = 0.2,
    entropy_coeff: float = 0.01,
    value_loss_coeff: float = 0.5,
) -> Tuple[jnp.ndarray, dict]:
    """Compute the PPO clipped surrogate loss.

    Args:
        policy_apply: ``fn(params["policy"], obs) -> (mean, log_std)``.
        value_apply: ``fn(params["value"], obs) -> value``.
        params: ``{"policy": …, "value": …}`` parameter dict.
        obs: (B, obs_dim) observations.
        actions: (B, act_dim) actions taken.
        old_log_probs: (B,) log-probs under the old policy.
        advantages: (B,) GAE advantages.
        returns: (B,) discounted returns (targets for value network).
        clip_eps: PPO clip parameter.
        entropy_coeff: entropy bonus weight.
        value_loss_coeff: value-loss weight.

    Returns:
        ``(total_loss, info_dict)`` where *info_dict* contains
        ``policy_loss``, ``value_loss``, ``entropy``.
    """
    # ── Policy loss ──────────────────────────────────────────────────
    mean, log_std = policy_apply(params["policy"], obs)
    new_log_probs = gaussian_log_prob(mean, log_std, actions)

    ratio = jnp.exp(new_log_probs - old_log_probs)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))

    # ── Entropy ──────────────────────────────────────────────────────
    entropy = jnp.mean(gaussian_entropy(log_std))

    # ── Value loss ───────────────────────────────────────────────────
    values = value_apply(params["value"], obs)
    value_loss = 0.5 * jnp.mean((values - returns) ** 2)

    # ── Total ────────────────────────────────────────────────────────
    total_loss = policy_loss - entropy_coeff * entropy + value_loss_coeff * value_loss

    info = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": jnp.mean((ratio - 1.0) - jnp.log(ratio)),
        "clip_fraction": jnp.mean(jnp.abs(ratio - 1.0) > clip_eps),
    }
    return total_loss, info
