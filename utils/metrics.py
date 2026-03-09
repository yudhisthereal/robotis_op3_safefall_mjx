"""Shared metrics computation for all OP3 SafeFall environments.

Every environment calls these pure functions to compute standardised
metrics that are then logged to WandB.
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp


# ── Per-step metric helpers ──────────────────────────────────────────


def compute_peak_torque(ctrl: jnp.ndarray) -> jnp.ndarray:
    """Maximum absolute joint torque across all actuators for a single step.

    Args:
        ctrl: (num_actuators,) array of applied control signals / torques.

    Returns:
        Scalar – peak |torque| this step.
    """
    return jnp.max(jnp.abs(ctrl))


def compute_peak_contact_force(contact_force: jnp.ndarray) -> jnp.ndarray:
    """Maximum contact force magnitude across all contacts for a single step.

    Args:
        contact_force: (..., 3) array of contact forces (from mjx_data).

    Returns:
        Scalar – peak contact force magnitude this step.
    """
    if contact_force.ndim == 1:
        return jnp.linalg.norm(contact_force)
    norms = jnp.linalg.norm(contact_force, axis=-1)
    return jnp.max(norms)


# ── Episode-level aggregation ────────────────────────────────────────


def aggregate_episode_metrics(
    episode_rewards: jnp.ndarray,
    episode_lengths: jnp.ndarray,
    peak_torques: jnp.ndarray,
    peak_contact_forces: jnp.ndarray,
    policy_loss: float,
    value_loss: float,
    entropy: float,
    learning_rate: float,
) -> Dict[str, float]:
    """Aggregate per-episode metrics into a dict suitable for WandB logging.

    All array inputs are expected to already be host-side NumPy / scalars.

    Returns:
        Dictionary with keys matching the required WandB metric names.
    """
    return {
        "episode_reward": float(jnp.mean(episode_rewards)),
        "episode_reward_std": float(jnp.std(episode_rewards)),
        "episode_length": float(jnp.mean(episode_lengths)),
        "peak_torque": float(jnp.max(peak_torques)),
        "peak_contact_force": float(jnp.max(peak_contact_forces)),
        "policy_loss": float(policy_loss),
        "value_loss": float(value_loss),
        "entropy": float(entropy),
        "learning_rate": float(learning_rate),
    }


# ── Vectorised helpers (for use inside jax.vmap / jit) ───────────────


@jax.jit
def batch_peak_torque(ctrls: jnp.ndarray) -> jnp.ndarray:
    """Vectorised peak torque across a batch of environments.

    Args:
        ctrls: (num_envs, num_actuators) array.

    Returns:
        (num_envs,) peak torques.
    """
    return jnp.max(jnp.abs(ctrls), axis=-1)


@jax.jit
def batch_peak_contact_force(contact_forces: jnp.ndarray) -> jnp.ndarray:
    """Vectorised peak contact force across a batch of environments.

    Args:
        contact_forces: (num_envs, num_contacts, 3) array.

    Returns:
        (num_envs,) peak contact forces.
    """
    norms = jnp.linalg.norm(contact_forces, axis=-1)  # (num_envs, num_contacts)
    return jnp.max(norms, axis=-1)  # (num_envs,)
