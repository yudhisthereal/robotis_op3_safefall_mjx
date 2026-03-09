"""Shared domain randomization for all OP3 SafeFall environments.

At every episode reset the physical parameters of the MJX model are
randomised to improve sim-to-real robustness.

All functions are **pure JAX** and compatible with ``jax.jit`` / ``jax.vmap``.
They operate on the raw model arrays rather than mutating MjModel objects,
because MJX models are immutable pytrees.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import mujoco.mjx as mjx


def randomize_model(
    model: mjx.Model,
    rng: jax.Array,
    *,
    mass_range: Tuple[float, float] = (0.8, 1.2),
    damping_range: Tuple[float, float] = (0.7, 1.3),
    friction_range: Tuple[float, float] = (0.5, 1.5),
    ground_friction_range: Tuple[float, float] = (0.4, 1.6),
) -> mjx.Model:
    """Return a new MJX model with randomised physical parameters.

    The original ``model`` is treated as the *nominal* model – randomisation
    multiplies nominal values by uniform scale factors so that the model
    stays physically plausible.

    Args:
        model: Nominal MJX model (immutable pytree).
        rng: JAX PRNG key.
        mass_range: (lo, hi) multiplicative scale for body masses.
        damping_range: (lo, hi) multiplicative scale for joint damping.
        friction_range: (lo, hi) multiplicative scale for geom friction.
        ground_friction_range: (lo, hi) multiplicative scale for ground geom friction.

    Returns:
        A new ``mjx.Model`` with randomised parameters.
    """
    rng_mass, rng_damp, rng_fric, rng_gfric = jax.random.split(rng, 4)

    # ── Body masses ──────────────────────────────────────────────────
    mass_scale = jax.random.uniform(
        rng_mass,
        shape=model.body_mass.shape,
        minval=mass_range[0],
        maxval=mass_range[1],
    )
    new_mass = model.body_mass * mass_scale

    # ── Joint damping ────────────────────────────────────────────────
    damp_scale = jax.random.uniform(
        rng_damp,
        shape=model.dof_damping.shape,
        minval=damping_range[0],
        maxval=damping_range[1],
    )
    new_damping = model.dof_damping * damp_scale

    # ── Geom friction ────────────────────────────────────────────────
    fric_scale = jax.random.uniform(
        rng_fric,
        shape=model.geom_friction.shape,
        minval=friction_range[0],
        maxval=friction_range[1],
    )
    new_friction = model.geom_friction * fric_scale

    # ── Ground friction (geom index 0 by convention in scene.xml) ───
    gfric_scale = jax.random.uniform(
        rng_gfric,
        shape=(model.geom_friction.shape[-1],),
        minval=ground_friction_range[0],
        maxval=ground_friction_range[1],
    )
    # Override the first geom's friction (the floor plane)
    new_friction = new_friction.at[0].set(model.geom_friction[0] * gfric_scale)

    # ── Assemble new model ───────────────────────────────────────────
    new_model = model.replace(
        body_mass=new_mass,
        dof_damping=new_damping,
        geom_friction=new_friction,
    )
    return new_model


# ── Vectorised version for batch of environments ─────────────────────


def batch_randomize_model(
    model: mjx.Model,
    rng: jax.Array,
    num_envs: int,
    **kwargs,
) -> mjx.Model:
    """Randomise model parameters independently for *num_envs* environments.

    Uses ``jax.vmap`` over the PRNG keys; returns a batched model pytree
    where every leaf has an extra leading dimension of size ``num_envs``.
    """
    keys = jax.random.split(rng, num_envs)
    return jax.vmap(lambda k: randomize_model(model, k, **kwargs))(keys)
