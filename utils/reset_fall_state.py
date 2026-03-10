"""Shared stochastic falling-state initialization for OP3 environments.

This module provides pure JAX helpers to initialise episodes *after* the
fall has already started (no fall predictor required).
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


def _euler_xyz_to_quat(roll: jnp.ndarray, pitch: jnp.ndarray, yaw: jnp.ndarray) -> jnp.ndarray:
    """Convert XYZ euler angles to MuJoCo quaternion format ``[w, x, y, z]``."""
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return jnp.array([w, x, y, z], dtype=jnp.float32)


def sample_falling_state(
    qpos_nominal: jnp.ndarray,
    qvel_nominal: jnp.ndarray,
    rng: jax.Array,
    *,
    joint_pos_noise_std: float = 0.10,
    joint_vel_noise_std: float = 1.00,
    base_linvel_std: float = 1.25,
    base_angvel_std: float = 2.50,
    min_tilt_rad: float = jnp.deg2rad(15.0),
    min_base_height: float = 0.22,
    max_tries: int = 8,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a stochastic unstable falling state with simple rejection.

    The rejection criteria are approximations suitable for JAX-only setup:
    - reject if base height is too low (ground intersection proxy)
    - reject if orientation is too close to upright (fully stable proxy)
    """

    def _one_sample(key: jax.Array):
        k_roll, k_pitch, k_yaw, k_h, k_qp, k_qv, k_bv_lin, k_bv_ang = jax.random.split(key, 8)

        roll = jax.random.uniform(k_roll, (), minval=-jnp.deg2rad(60.0), maxval=jnp.deg2rad(60.0))
        pitch = jax.random.uniform(k_pitch, (), minval=-jnp.deg2rad(60.0), maxval=jnp.deg2rad(60.0))
        yaw = jax.random.uniform(k_yaw, (), minval=-jnp.pi, maxval=jnp.pi)
        base_h = jax.random.uniform(k_h, (), minval=0.3, maxval=0.6)

        quat = _euler_xyz_to_quat(roll, pitch, yaw)

        qpos = qpos_nominal
        qpos = qpos.at[0:3].set(jnp.array([0.0, 0.0, base_h], dtype=qpos.dtype))
        qpos = qpos.at[3:7].set(quat.astype(qpos.dtype))
        qpos = qpos.at[7:].add(jax.random.normal(k_qp, qpos[7:].shape) * joint_pos_noise_std)

        qvel = qvel_nominal
        base_linvel = jax.random.normal(k_bv_lin, (3,)) * base_linvel_std
        base_angvel = jax.random.normal(k_bv_ang, (3,)) * base_angvel_std
        qvel = qvel.at[0:3].set(base_linvel.astype(qvel.dtype))
        qvel = qvel.at[3:6].set(base_angvel.astype(qvel.dtype))
        qvel = qvel.at[6:].add(jax.random.normal(k_qv, qvel[6:].shape) * joint_vel_noise_std)

        tilt = jnp.sqrt(roll * roll + pitch * pitch)
        valid_height = base_h > min_base_height
        valid_unstable = tilt > min_tilt_rad
        valid = valid_height & valid_unstable
        return qpos, qvel, valid

    def _cond(carry):
        _qpos, _qvel, valid, i, _key = carry
        return jnp.logical_and(~valid, i < max_tries)

    def _body(carry):
        _qpos, _qvel, _valid, i, key = carry
        key, sample_key = jax.random.split(key)
        qpos, qvel, valid = _one_sample(sample_key)
        return qpos, qvel, valid, i + 1, key

    init_key, first_key = jax.random.split(rng)
    qpos0, qvel0, valid0 = _one_sample(first_key)
    qpos, qvel, _valid, _i, _key = jax.lax.while_loop(
        _cond,
        _body,
        (qpos0, qvel0, valid0, jnp.array(1, dtype=jnp.int32), init_key),
    )
    return qpos, qvel
