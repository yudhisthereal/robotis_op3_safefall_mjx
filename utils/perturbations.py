"""Shared perturbation system for all OP3 SafeFall environments.

Implements SafeFall-style failure factors that are applied **randomly
during rollouts** (i.e. at each simulation step or at episode reset).

Every perturbation function is a **pure JAX function** compatible with
``jax.jit`` and ``jax.vmap``.

Perturbation types
------------------
* External push – random force applied to torso body.
* Foot slip – horizontal velocity perturbation on stance foot.
* Foot trip – vertical impulse on swing foot.
* Joint noise – additive Gaussian noise on joint positions / velocities.
* Motor delay – circular buffer that delays actions by *N* steps.
* Sensor noise – additive Gaussian noise on observations.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct


# ── Perturbation state ──────────────────────────────────────────────


@struct.dataclass
class PerturbationState:
    """Immutable state carried between steps for stateful perturbations."""

    # Motor delay circular buffer: (delay_steps, num_actions)
    action_buffer: jnp.ndarray
    buffer_idx: jnp.ndarray  # scalar int


def init_perturbation_state(
    num_actions: int,
    delay_steps: int,
    rng: jax.Array,
) -> PerturbationState:
    """Create a fresh perturbation state."""
    del rng  # may be used in future for random init
    return PerturbationState(
        action_buffer=jnp.zeros((delay_steps, num_actions)),
        buffer_idx=jnp.array(0, dtype=jnp.int32),
    )


# ── External push ───────────────────────────────────────────────────


def apply_external_push(
    xfrc_applied: jnp.ndarray,
    body_index: int,
    rng: jax.Array,
    prob: float = 0.1,
    force_range: Tuple[float, float] = (5.0, 20.0),
) -> jnp.ndarray:
    """Apply a random external force to the torso body.

    Args:
        xfrc_applied: (nbody, 6) external wrench array.
        body_index: index of the torso body in the MuJoCo model.
        rng: JAX PRNG key.
        prob: probability of applying the push this step.
        force_range: (min, max) magnitude of the applied force.

    Returns:
        Updated xfrc_applied array (same shape).
    """
    rng_apply, rng_dir, rng_mag = jax.random.split(rng, 3)
    do_push = jax.random.uniform(rng_apply) < prob
    direction = jax.random.normal(rng_dir, shape=(3,))
    direction = direction / (jnp.linalg.norm(direction) + 1e-8)
    magnitude = jax.random.uniform(
        rng_mag, minval=force_range[0], maxval=force_range[1]
    )
    force = direction * magnitude  # (3,)
    push_wrench = jnp.concatenate([force, jnp.zeros(3)])  # (6,)
    new_row = jnp.where(do_push, push_wrench, jnp.zeros(6))
    return xfrc_applied.at[body_index].set(new_row)


# ── Foot slip ────────────────────────────────────────────────────────


def apply_foot_slip(
    qvel: jnp.ndarray,
    foot_body_indices: Tuple[int, int],
    rng: jax.Array,
    prob: float = 0.05,
    vel_range: Tuple[float, float] = (0.1, 0.5),
) -> jnp.ndarray:
    """Add a horizontal velocity perturbation to a stance foot.

    Simulates a slippery surface by injecting random horizontal velocity
    into the stance foot body.

    Args:
        qvel: (nv,) generalised velocity vector.
        foot_body_indices: tuple of two ints – left and right foot body ids.
        rng: JAX PRNG key.
        prob: probability of slip this step.
        vel_range: (min, max) horizontal slip velocity.

    Returns:
        Updated qvel array.
    """
    rng_apply, rng_foot, rng_dir, rng_mag = jax.random.split(rng, 4)
    do_slip = jax.random.uniform(rng_apply) < prob
    # Pick left (0) or right (1) foot
    foot_choice = jax.random.randint(rng_foot, (), 0, 2)
    # Random horizontal direction
    angle = jax.random.uniform(rng_dir, minval=0.0, maxval=2.0 * jnp.pi)
    mag = jax.random.uniform(rng_mag, minval=vel_range[0], maxval=vel_range[1])
    slip_vel = jnp.array([jnp.cos(angle) * mag, jnp.sin(angle) * mag])

    # Apply to first 2 components of the chosen foot's velocity DOFs
    # For floating base robot: qvel[0:3] = base linear vel, qvel[3:6] = base angular vel
    # We perturb the base linear velocity x/y as a proxy (foot slip propagates through contact)
    perturbed_qvel = qvel.at[0:2].add(jnp.where(do_slip, slip_vel, jnp.zeros(2)))
    return perturbed_qvel


# ── Foot trip ────────────────────────────────────────────────────────


def apply_foot_trip(
    xfrc_applied: jnp.ndarray,
    foot_body_indices: Tuple[int, int],
    rng: jax.Array,
    prob: float = 0.05,
    impulse_range: Tuple[float, float] = (2.0, 8.0),
) -> jnp.ndarray:
    """Simulate a foot catching on terrain by applying a vertical impulse.

    Args:
        xfrc_applied: (nbody, 6) external wrench array.
        foot_body_indices: tuple of two ints – left and right foot body ids.
        rng: JAX PRNG key.
        prob: probability of trip this step.
        impulse_range: (min, max) upward impulse magnitude.

    Returns:
        Updated xfrc_applied array.
    """
    rng_apply, rng_foot, rng_mag = jax.random.split(rng, 3)
    do_trip = jax.random.uniform(rng_apply) < prob
    foot_choice = jax.random.randint(rng_foot, (), 0, 2)
    mag = jax.random.uniform(rng_mag, minval=impulse_range[0], maxval=impulse_range[1])
    trip_wrench = jnp.array([0.0, 0.0, mag, 0.0, 0.0, 0.0])
    foot_idx = jnp.where(foot_choice == 0, foot_body_indices[0], foot_body_indices[1])
    new_wrench = jnp.where(do_trip, trip_wrench, jnp.zeros(6))
    return xfrc_applied.at[foot_idx].set(
        xfrc_applied[foot_idx] + new_wrench
    )


# ── Joint noise ──────────────────────────────────────────────────────


def apply_joint_noise(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    rng: jax.Array,
    qpos_noise_std: float = 0.02,
    qvel_noise_std: float = 0.02,
    joint_start: int = 7,  # skip free-joint DOFs (pos: 7, vel: 6)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Add Gaussian noise to joint positions and velocities.

    Skips the free-joint (first 7 qpos, first 6 qvel) to avoid
    corrupting the base pose.

    Returns:
        (noisy_qpos, noisy_qvel) tuple.
    """
    rng_q, rng_v = jax.random.split(rng)
    nq_joints = qpos.shape[0] - joint_start
    nv_joints = qvel.shape[0] - (joint_start - 1)  # free-joint: 6 vel DOFs

    qpos_noise = jax.random.normal(rng_q, (nq_joints,)) * qpos_noise_std
    qvel_noise = jax.random.normal(rng_v, (nv_joints,)) * qvel_noise_std

    noisy_qpos = qpos.at[joint_start:].add(qpos_noise)
    noisy_qvel = qvel.at[joint_start - 1 :].add(qvel_noise)
    return noisy_qpos, noisy_qvel


# ── Motor delay ──────────────────────────────────────────────────────


def apply_motor_delay(
    action: jnp.ndarray,
    perturbation_state: PerturbationState,
    delay_steps: int,
) -> Tuple[jnp.ndarray, PerturbationState]:
    """Simulate actuator latency using a circular buffer.

    The *current* action is stored in the buffer and the action from
    ``delay_steps`` ago is returned.

    Returns:
        (delayed_action, updated_perturbation_state).
    """
    buf = perturbation_state.action_buffer
    idx = perturbation_state.buffer_idx

    # Read delayed action
    read_idx = (idx + 1) % delay_steps
    delayed_action = buf[read_idx]

    # Write current action
    new_buf = buf.at[idx].set(action)
    new_idx = (idx + 1) % delay_steps

    new_state = PerturbationState(action_buffer=new_buf, buffer_idx=new_idx)
    return delayed_action, new_state


# ── Sensor noise ─────────────────────────────────────────────────────


def apply_sensor_noise(
    obs: jnp.ndarray,
    rng: jax.Array,
    noise_std: float = 0.01,
) -> jnp.ndarray:
    """Add Gaussian noise to the observation vector."""
    noise = jax.random.normal(rng, obs.shape) * noise_std
    return obs + noise


# ── Convenience: apply all perturbations in one call ─────────────────


def apply_all_perturbations(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    xfrc_applied: jnp.ndarray,
    action: jnp.ndarray,
    perturbation_state: PerturbationState,
    rng: jax.Array,
    torso_body_index: int,
    foot_body_indices: Tuple[int, int],
    *,
    push_prob: float = 0.1,
    push_force_range: Tuple[float, float] = (5.0, 20.0),
    slip_prob: float = 0.05,
    slip_vel_range: Tuple[float, float] = (0.1, 0.5),
    trip_prob: float = 0.05,
    joint_noise_std: float = 0.02,
    motor_delay_steps: int = 2,
    sensor_noise_std: float = 0.01,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, PerturbationState]:
    """Apply the full SafeFall-style perturbation pipeline.

    Returns:
        (qpos, qvel, xfrc_applied, delayed_action, perturbation_state)
    """
    del sensor_noise_std
    keys = jax.random.split(rng, 5)

    # Joint noise
    qpos, qvel = apply_joint_noise(qpos, qvel, keys[0], joint_noise_std, joint_noise_std)

    # External push
    xfrc_applied = apply_external_push(
        xfrc_applied, torso_body_index, keys[1], push_prob, push_force_range
    )

    # Foot slip
    qvel = apply_foot_slip(qvel, foot_body_indices, keys[2], slip_prob, slip_vel_range)

    # Foot trip
    xfrc_applied = apply_foot_trip(
        xfrc_applied, foot_body_indices, keys[3], trip_prob
    )

    # Motor delay
    delayed_action, perturbation_state = apply_motor_delay(
        action, perturbation_state, motor_delay_steps
    )

    return qpos, qvel, xfrc_applied, delayed_action, perturbation_state
