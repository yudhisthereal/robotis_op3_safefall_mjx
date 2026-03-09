"""Environment utilities – model loading, vectorisation, RNG management.

Provides helpers to:
* load a MuJoCo model from XML and convert it to an MJX model;
* create vectorised (vmapped) environment step / reset functions;
* manage per-environment PRNG keys.
"""

from __future__ import annotations

import os
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from utils.config import Config


# ── Model loading ────────────────────────────────────────────────────


def load_mujoco_model(config: Config) -> mujoco.MjModel:
    """Load the MuJoCo XML model from disk.

    Resolves the asset directory so that ``scene.xml`` (which includes
    ``op3.xml``) can find the mesh files.

    Returns:
        ``mujoco.MjModel`` instance (CPU-side).
    """
    xml_path = config.scene_xml_path
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"Scene XML not found at {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    return model


def load_mjx_model(config: Config) -> Tuple[mujoco.MjModel, mjx.Model]:
    """Load both CPU MuJoCo and GPU MJX model.

    Returns:
        (mj_model, mjx_model) tuple.
    """
    mj_model = load_mujoco_model(config)
    mjx_model = mjx.put_model(mj_model)
    return mj_model, mjx_model


# ── Body / joint index helpers ───────────────────────────────────────


def get_body_index(mj_model: mujoco.MjModel, name: str) -> int:
    """Get the body index by name."""
    return mj_model.body(name).id


def get_joint_index(mj_model: mujoco.MjModel, name: str) -> int:
    """Get the joint index by name."""
    return mj_model.joint(name).id


def get_sensor_adr(mj_model: mujoco.MjModel, name: str) -> Tuple[int, int]:
    """Return (start, end) address slice into sensordata for a named sensor."""
    sensor_id = mj_model.sensor(name).id
    adr = mj_model.sensor_adr[sensor_id]
    dim = mj_model.sensor_dim[sensor_id]
    return int(adr), int(adr + dim)


# ── RNG management ───────────────────────────────────────────────────


def make_rng_keys(rng: jax.Array, num_envs: int) -> jax.Array:
    """Split a single PRNG key into *num_envs* independent keys.

    Returns:
        Array of shape ``(num_envs, 2)`` containing PRNG keys.
    """
    return jax.random.split(rng, num_envs)


def advance_rng(keys: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Advance every per-environment key by one split.

    Args:
        keys: (num_envs, 2) array of PRNG keys.

    Returns:
        (new_keys, subkeys) each of shape (num_envs, 2).
    """
    new_keys, subkeys = jax.vmap(lambda k: tuple(jax.random.split(k)))(keys)
    return new_keys, subkeys


# ── Vectorised env construction ──────────────────────────────────────


def vectorize_env(
    reset_fn: Callable,
    step_fn: Callable,
) -> Tuple[Callable, Callable]:
    """Wrap single-env ``reset`` and ``step`` with ``jax.vmap``.

    Assumes signatures::

        reset(rng) -> state
        step(state, action, rng) -> state

    Returns:
        (v_reset, v_step) – vmapped versions.
    """
    v_reset = jax.vmap(reset_fn, in_axes=(0,))
    v_step = jax.vmap(step_fn, in_axes=(0, 0, 0))
    return v_reset, v_step
