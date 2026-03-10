"""SafeFall OP3 Training Environment (MJX).

Inspired by *SafeFall: Learning Protective Control for Humanoid Robots*.

The robot is subject to random perturbations (pushes, slips, trips, …)
and must learn **protective fall control** that minimises injury risk
while maximising stability recovery.

Design
------
* Fully functional JAX API: ``reset(rng) -> State`` / ``step(state, action, rng) -> State``.
* Immutable ``State`` dataclass.
* Compatible with ``jax.jit`` and ``jax.vmap`` for parallel simulation.

Reward
------
``r = r_safety + r_stability`` with uprightness, impact, effort, and smoothness terms.
"""

from __future__ import annotations

import os
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
import mujoco
import mujoco.mjx as mjx

from utils.config import Config
from utils.domain_randomization import randomize_model
from utils.perturbations import (
    PerturbationState,
    apply_all_perturbations,
    apply_sensor_noise,
    init_perturbation_state,
)
from utils.reset_fall_state import sample_falling_state
from utils.metrics import compute_peak_contact_force, compute_peak_torque


# ── Environment state ────────────────────────────────────────────────


@struct.dataclass
class SafeFallState:
    """Immutable environment state."""

    obs: jnp.ndarray           # (obs_dim,)
    reward: jnp.ndarray        # scalar
    done: jnp.ndarray          # scalar bool-like float
    qpos: jnp.ndarray          # explicit generalized positions
    qvel: jnp.ndarray          # explicit generalized velocities
    act: jnp.ndarray           # explicit applied action/control
    mjx_model: mjx.Model       # per-env randomized MJX model
    mjx_data: mjx.Data         # MJX physics state
    info: dict                 # auxiliary info dict
    perturbation_state: PerturbationState
    step_count: jnp.ndarray    # scalar int


# ── Environment class ────────────────────────────────────────────────


class SafeFallOP3Env:
    """SafeFall-style MJX environment for the OP3 humanoid.

    Instantiate once, then call the (jittable) ``reset`` / ``step``
    methods.
    """

    def __init__(self, config: Config, mj_model: mujoco.MjModel, mjx_model: mjx.Model):
        self.config = config
        self.mj_model = mj_model
        self.mjx_model = mjx_model

        # Cache useful body / sensor indices
        self.torso_body_idx = mj_model.body("body_link").id
        self.l_foot_body_idx = mj_model.body("l_ank_roll_link").id
        self.r_foot_body_idx = mj_model.body("r_ank_roll_link").id
        self.foot_body_indices = (self.l_foot_body_idx, self.r_foot_body_idx)

        # Sensor addresses
        self._accel_adr = self._sensor_slice("torso_accel")
        self._gyro_adr = self._sensor_slice("torso_gyro")
        self._quat_adr = self._sensor_slice("torso_quat")
        self._linvel_adr = self._sensor_slice("torso_linvel")
        self._angvel_adr = self._sensor_slice("torso_angvel")

        # Observation / action dims
        # obs = joint_pos(20) + joint_vel(20) + accel(3) + gyro(3) + quat(4) + linvel(3) + angvel(3) + contact(~11)
        # We fix obs_dim after a trial build:
        self.num_joints = config.num_joints  # 20
        self._obs_dim: int | None = None  # lazily determined

    # ── helpers ───────────────────────────────────────────────────────

    def _sensor_slice(self, name: str) -> Tuple[int, int]:
        sid = self.mj_model.sensor(name).id
        adr = int(self.mj_model.sensor_adr[sid])
        dim = int(self.mj_model.sensor_dim[sid])
        return (adr, adr + dim)

    @property
    def obs_dim(self) -> int:
        if self._obs_dim is None:
            # joint_pos(20) + joint_vel(20) + accel(3) + gyro(3) + quat(4) + linvel(3) + angvel(3)
            self._obs_dim = 20 + 20 + 3 + 3 + 4 + 3 + 3  # = 56
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self.config.num_actions

    # ── observation builder ──────────────────────────────────────────

    def _build_obs(self, data: mjx.Data) -> jnp.ndarray:
        """Construct observation vector from MJX data."""
        joint_pos = data.qpos[7:]  # skip free-joint (7 DOFs)
        joint_vel = data.qvel[6:]  # skip free-joint (6 DOFs)
        sd = data.sensordata
        accel = sd[self._accel_adr[0]:self._accel_adr[1]]
        gyro = sd[self._gyro_adr[0]:self._gyro_adr[1]]
        quat = sd[self._quat_adr[0]:self._quat_adr[1]]
        linvel = sd[self._linvel_adr[0]:self._linvel_adr[1]]
        angvel = sd[self._angvel_adr[0]:self._angvel_adr[1]]
        return jnp.concatenate([joint_pos, joint_vel, accel, gyro, quat, linvel, angvel])

    # ── reward ───────────────────────────────────────────────────────

    def _compute_reward(
        self, data: mjx.Data, prev_data: mjx.Data, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute SafeFall reward with stability and safety shaping."""
        torso_z = data.qpos[2]

        # Orientation from torso quaternion sensor (w, x, y, z)
        sd = data.sensordata
        torso_quat = sd[self._quat_adr[0]:self._quat_adr[1]]
        upright = jnp.clip(jnp.abs(torso_quat[0]), 0.0, 1.0)

        # Stability
        height_term = jnp.clip(torso_z / 0.30, 0.0, 1.2)
        linear_vel_penalty = -0.02 * jnp.linalg.norm(data.qvel[0:3])
        angular_vel_penalty = -0.01 * jnp.linalg.norm(data.qvel[3:6])

        # Safety / efficiency
        torque_penalty = -0.001 * jnp.mean(jnp.square(action))
        contact_penalty = -0.0002 * jnp.mean(jnp.square(data.qfrc_constraint))
        smoothness_penalty = -0.0005 * jnp.mean(jnp.square(action - prev_data.ctrl))

        # Terminal shaping
        fallen_penalty = jnp.where(torso_z < 0.05, -2.0, 0.0)
        alive_bonus = jnp.where(torso_z > 0.08, 0.2, -0.2)

        return (
            1.0 * upright
            + 0.6 * height_term
            + linear_vel_penalty
            + angular_vel_penalty
            + torque_penalty
            + contact_penalty
            + smoothness_penalty
            + fallen_penalty
            + alive_bonus
        )

    # ── termination ──────────────────────────────────────────────────

    def _check_termination(
        self, data: mjx.Data, step_count: jnp.ndarray
    ) -> jnp.ndarray:
        """Shared termination conditions."""
        torso_z = data.qpos[2]

        # Robot fully on the ground (fallen and stabilised)
        fallen = torso_z < 0.05

        # Time limit
        timeout = step_count >= self.config.episode_max_steps

        # Out of bounds
        xy = data.qpos[0:2]
        out_of_bounds = jnp.any(jnp.abs(xy) > 5.0)

        # Numerical instability
        nan_detected = jnp.any(jnp.isnan(data.qpos)) | jnp.any(jnp.isnan(data.qvel))

        done = jnp.any(jnp.array([fallen, timeout, out_of_bounds, nan_detected]))
        return done.astype(jnp.float32)

    # ── reset ────────────────────────────────────────────────────────

    def reset(self, rng: jax.Array) -> SafeFallState:
        """Reset the environment. Pure-functional, jittable.

        Args:
            mjx_model: (possibly domain-randomised) MJX model.
            rng: JAX PRNG key.

        Returns:
            Initial ``SafeFallState``.
        """
        rng, rng_dr, rng_perturb = jax.random.split(rng, 3)
        model = randomize_model(
            self.mjx_model,
            rng_dr,
            mass_range=self.config.dr_mass_range,
            damping_range=self.config.dr_damping_range,
            friction_range=self.config.dr_friction_range,
            ground_friction_range=self.config.dr_ground_friction_range,
            sensor_noise_scale_range=self.config.dr_sensor_noise_scale_range,
            contact_solref_scale_range=self.config.dr_contact_solref_scale_range,
            contact_solimp_scale_range=self.config.dr_contact_solimp_scale_range,
        )
        data = mjx.make_data(model)

        # Stochastic post-fall initial state (no fall predictor)
        rng, rng_fall = jax.random.split(rng)
        qpos_init, qvel_init = sample_falling_state(data.qpos, data.qvel, rng_fall)
        data = data.replace(qpos=qpos_init, qvel=qvel_init)

        # Forward-simulate one step to settle initial state
        data = mjx.step(model, data)

        obs = self._build_obs(data)
        perturb_state = init_perturbation_state(
            self.config.num_actions, self.config.perturb_motor_delay_steps, rng_perturb
        )
        return SafeFallState(
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            qpos=data.qpos,
            qvel=data.qvel,
            act=data.ctrl,
            mjx_model=model,
            mjx_data=data,
            info={
                "episode_reward": jnp.float32(0.0),
                "peak_torque": jnp.float32(0.0),
                "peak_contact_force": jnp.float32(0.0),
            },
            perturbation_state=perturb_state,
            step_count=jnp.int32(0),
        )

    # ── step ─────────────────────────────────────────────────────────

    def step(
        self,
        state: SafeFallState,
        action: jnp.ndarray,
        rng: jax.Array,
    ) -> SafeFallState:
        """Take one environment step. Pure-functional, jittable.

        Args:
            mjx_model: MJX model.
            state: current state.
            action: (num_actions,) action vector.
            rng: PRNG key.

        Returns:
            Next ``SafeFallState``.
        """
        rng_perturb, rng_obs = jax.random.split(rng)
        data = state.mjx_data
        mjx_model = state.mjx_model

        # Apply perturbations
        qpos, qvel, xfrc, delayed_action, new_perturb = apply_all_perturbations(
            data.qpos, data.qvel, data.xfrc_applied, action,
            state.perturbation_state, rng_perturb,
            torso_body_index=self.torso_body_idx,
            foot_body_indices=self.foot_body_indices,
            push_prob=self.config.perturb_external_push_prob,
            push_force_range=self.config.perturb_external_push_force_range,
            slip_prob=self.config.perturb_foot_slip_prob,
            slip_vel_range=self.config.perturb_foot_slip_vel_range,
            trip_prob=self.config.perturb_foot_trip_prob,
            joint_noise_std=self.config.perturb_joint_noise_std,
            motor_delay_steps=self.config.perturb_motor_delay_steps,
        )

        # Update data with perturbed values
        data = data.replace(
            qpos=qpos,
            qvel=qvel,
            xfrc_applied=xfrc,
            ctrl=jnp.clip(delayed_action, -1.0, 1.0),
        )

        # Physics step (multiple sub-steps for control_dt)
        def _physics_substep(data_i, _):
            return mjx.step(mjx_model, data_i), None

        next_data, _ = jax.lax.scan(
            _physics_substep, data, None,
            length=self.config.physics_steps_per_control,
        )

        # Observation & reward
        new_obs = apply_sensor_noise(
            self._build_obs(next_data),
            rng_obs,
            self.config.perturb_sensor_noise_std,
        )
        reward = self._compute_reward(next_data, state.mjx_data, delayed_action)
        new_step = state.step_count + 1
        done = self._check_termination(next_data, new_step)

        # Metrics
        pt = jnp.maximum(state.info["peak_torque"], compute_peak_torque(delayed_action))
        pcf_raw = jnp.sum(jnp.abs(next_data.qfrc_constraint))
        pcf = jnp.maximum(state.info["peak_contact_force"], pcf_raw)

        info = {
            "episode_reward": state.info["episode_reward"] + reward,
            "peak_torque": pt,
            "peak_contact_force": pcf,
        }

        return SafeFallState(
            obs=new_obs,
            reward=reward,
            done=done,
            qpos=next_data.qpos,
            qvel=next_data.qvel,
            act=next_data.ctrl,
            mjx_model=mjx_model,
            mjx_data=next_data,
            info=info,
            perturbation_state=new_perturb,
            step_count=new_step,
        )
