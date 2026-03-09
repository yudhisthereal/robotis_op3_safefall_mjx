"""Low-Level Goal-Conditioned Fall Controller (MJX).

The agent receives a **goal** (one-hot encoded fall strategy) and must
execute the corresponding protective manoeuvre:

    0 – arm_bracing
    1 – roll
    2 – squat_drop_forward
    3 – squat_drop_backward
    4 – side_collapse

Observation = ``robot_state ‖ goal_one_hot``

Reward = ``r_safety + r_strategy`` with strategy-target tracking and safety shaping.

Shares the same perturbation, domain-randomisation, termination, and
metrics systems as the other environments.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from utils.config import Config
from utils.domain_randomization import randomize_model
from utils.perturbations import (
    PerturbationState,
    apply_all_perturbations,
    init_perturbation_state,
)
from utils.metrics import compute_peak_torque

# ── Strategy identifiers ─────────────────────────────────────────────

STRATEGIES = ("arm_bracing", "roll", "squat_drop_forward", "squat_drop_backward", "side_collapse")
NUM_STRATEGIES = len(STRATEGIES)


# ── Environment state ────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class LowLevelFallState:
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    mjx_model: mjx.Model
    mjx_data: mjx.Data
    info: dict
    perturbation_state: PerturbationState
    step_count: jnp.ndarray
    goal: jnp.ndarray  # (NUM_STRATEGIES,) one-hot


# ── Environment ──────────────────────────────────────────────────────


class OP3LowLevelFallEnv:
    """Goal-conditioned low-level fall controller.

    The *goal* is sampled uniformly at reset and kept fixed for the
    episode.  The policy must learn to execute the specified strategy.
    """

    def __init__(self, config: Config, mj_model: mujoco.MjModel, mjx_model: mjx.Model):
        self.config = config
        self.mj_model = mj_model
        self.mjx_model = mjx_model

        self.torso_body_idx = mj_model.body("body_link").id
        self.l_foot_body_idx = mj_model.body("l_ank_roll_link").id
        self.r_foot_body_idx = mj_model.body("r_ank_roll_link").id
        self.foot_body_indices = (self.l_foot_body_idx, self.r_foot_body_idx)

        # Sensor slices
        self._accel_adr = self._ss("torso_accel")
        self._gyro_adr = self._ss("torso_gyro")
        self._quat_adr = self._ss("torso_quat")
        self._linvel_adr = self._ss("torso_linvel")
        self._angvel_adr = self._ss("torso_angvel")

    def _ss(self, name: str) -> Tuple[int, int]:
        sid = self.mj_model.sensor(name).id
        adr = int(self.mj_model.sensor_adr[sid])
        dim = int(self.mj_model.sensor_dim[sid])
        return (adr, adr + dim)

    @property
    def robot_obs_dim(self) -> int:
        return 20 + 20 + 3 + 3 + 4 + 3 + 3  # 56

    @property
    def obs_dim(self) -> int:
        return self.robot_obs_dim + NUM_STRATEGIES

    @property
    def action_dim(self) -> int:
        return self.config.num_actions

    # ── observation builder ──────────────────────────────────────────

    def _build_robot_obs(self, data: mjx.Data) -> jnp.ndarray:
        joint_pos = data.qpos[7:]
        joint_vel = data.qvel[6:]
        sd = data.sensordata
        parts = [
            joint_pos, joint_vel,
            sd[self._accel_adr[0]:self._accel_adr[1]],
            sd[self._gyro_adr[0]:self._gyro_adr[1]],
            sd[self._quat_adr[0]:self._quat_adr[1]],
            sd[self._linvel_adr[0]:self._linvel_adr[1]],
            sd[self._angvel_adr[0]:self._angvel_adr[1]],
        ]
        return jnp.concatenate(parts)

    def _build_obs(self, data: mjx.Data, goal: jnp.ndarray) -> jnp.ndarray:
        robot_obs = self._build_robot_obs(data)
        return jnp.concatenate([robot_obs, goal])

    def _strategy_target_action(self, goal: jnp.ndarray) -> jnp.ndarray:
        """Return a simple strategy-specific target action template."""
        # Joint groups in actuator order:
        # 0:head_pan, 1:head_tilt,
        # 2:l_sho_pitch, 3:l_sho_roll, 4:l_el,
        # 5:r_sho_pitch, 6:r_sho_roll, 7:r_el,
        # 8:l_hip_yaw, 9:l_hip_roll, 10:l_hip_pitch,
        # 11:l_knee, 12:l_ank_pitch, 13:l_ank_roll,
        # 14:r_hip_yaw, 15:r_hip_roll, 16:r_hip_pitch,
        # 17:r_knee, 18:r_ank_pitch, 19:r_ank_roll
        arm_bracing = jnp.array([
            0.0, 0.1,
            0.8, 0.2, -0.8,
            0.8, -0.2, -0.8,
            0.0, 0.0, 0.1,
            -0.1, 0.0, 0.0,
            0.0, 0.0, 0.1,
            -0.1, 0.0, 0.0,
        ], dtype=jnp.float32)
        roll = jnp.array([
            0.0, 0.0,
            0.3, 0.6, -0.3,
            0.3, -0.6, -0.3,
            0.0, 0.4, 0.7,
            -1.0, 0.4, 0.3,
            0.0, -0.4, 0.7,
            -1.0, 0.4, -0.3,
        ], dtype=jnp.float32)
        squat_fwd = jnp.array([
            0.0, 0.2,
            0.2, 0.0, -0.2,
            0.2, 0.0, -0.2,
            0.0, 0.0, 1.0,
            -1.2, 0.7, 0.0,
            0.0, 0.0, 1.0,
            -1.2, 0.7, 0.0,
        ], dtype=jnp.float32)
        squat_bwd = jnp.array([
            0.0, -0.1,
            -0.2, 0.0, 0.2,
            -0.2, 0.0, 0.2,
            0.0, 0.0, -0.8,
            1.0, -0.6, 0.0,
            0.0, 0.0, -0.8,
            1.0, -0.6, 0.0,
        ], dtype=jnp.float32)
        side = jnp.array([
            0.0, 0.0,
            0.4, 0.9, -0.2,
            0.1, -0.2, -0.2,
            0.0, 0.8, 0.5,
            -0.8, 0.2, 0.6,
            0.0, -0.2, 0.1,
            -0.2, 0.1, -0.5,
        ], dtype=jnp.float32)

        templates = jnp.stack([arm_bracing, roll, squat_fwd, squat_bwd, side], axis=0)
        return jnp.sum(templates * goal[:, None], axis=0)

    # ── reward (strategy-conditioned) ────────────────────────────────

    def _compute_reward(
        self, data: mjx.Data, prev_data: mjx.Data, action: jnp.ndarray, goal: jnp.ndarray
    ) -> jnp.ndarray:
        """``r = r_safety + r_strategy`` with strategy-target tracking."""
        torso_z = data.qpos[2]
        torso_quat = data.sensordata[self._quat_adr[0]:self._quat_adr[1]]
        upright = jnp.clip(jnp.abs(torso_quat[0]), 0.0, 1.0)

        torque_pen = -0.001 * jnp.mean(jnp.square(action))
        contact_pen = -0.0002 * jnp.mean(jnp.square(data.qfrc_constraint))
        smoothness_pen = -0.0005 * jnp.mean(jnp.square(action - prev_data.ctrl))

        target_action = self._strategy_target_action(goal)
        strategy_err = jnp.mean(jnp.square(jnp.tanh(action) - jnp.tanh(target_action)))
        strategy_bonus = 0.6 * jnp.exp(-4.0 * strategy_err)

        alive = jnp.where(torso_z > 0.05, 0.2, -0.5)
        fallen_penalty = jnp.where(torso_z < 0.05, -1.5, 0.0)

        return 0.8 * upright + 0.4 * jnp.clip(torso_z / 0.3, 0.0, 1.2) + strategy_bonus + torque_pen + contact_pen + smoothness_pen + alive + fallen_penalty

    # ── termination (shared logic) ───────────────────────────────────

    def _check_termination(self, data: mjx.Data, step_count: jnp.ndarray) -> jnp.ndarray:
        torso_z = data.qpos[2]
        fallen = torso_z < 0.05
        timeout = step_count >= self.config.episode_max_steps
        oob = jnp.any(jnp.abs(data.qpos[0:2]) > 5.0)
        nan_det = jnp.any(jnp.isnan(data.qpos)) | jnp.any(jnp.isnan(data.qvel))
        return jnp.any(jnp.array([fallen, timeout, oob, nan_det])).astype(jnp.float32)

    # ── reset ────────────────────────────────────────────────────────

    def reset(self, rng: jax.Array) -> LowLevelFallState:
        rng, rng_dr, rng_goal, rng_perturb = jax.random.split(rng, 4)
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
        data = mjx.step(model, data)

        # Sample a random strategy (one-hot goal)
        strategy_idx = jax.random.randint(rng_goal, (), 0, NUM_STRATEGIES)
        goal = jax.nn.one_hot(strategy_idx, NUM_STRATEGIES)

        obs = self._build_obs(data, goal)
        ps = init_perturbation_state(self.config.num_actions, self.config.perturb_motor_delay_steps, rng_perturb)

        return LowLevelFallState(
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            mjx_model=model,
            mjx_data=data,
            info={"episode_reward": jnp.float32(0.0), "peak_torque": jnp.float32(0.0), "peak_contact_force": jnp.float32(0.0)},
            perturbation_state=ps,
            step_count=jnp.int32(0),
            goal=goal,
        )

    # ── step ─────────────────────────────────────────────────────────

    def step(
        self, state: LowLevelFallState, action: jnp.ndarray, rng: jax.Array
    ) -> LowLevelFallState:
        data = state.mjx_data
        mjx_model = state.mjx_model

        qpos, qvel, xfrc, obs_noisy, delayed_action, new_ps = apply_all_perturbations(
            data.qpos, data.qvel, data.xfrc_applied, state.obs, action,
            state.perturbation_state, rng,
            torso_body_index=self.torso_body_idx,
            foot_body_indices=self.foot_body_indices,
            push_prob=self.config.perturb_external_push_prob,
            push_force_range=self.config.perturb_external_push_force_range,
            slip_prob=self.config.perturb_foot_slip_prob,
            slip_vel_range=self.config.perturb_foot_slip_vel_range,
            trip_prob=self.config.perturb_foot_trip_prob,
            joint_noise_std=self.config.perturb_joint_noise_std,
            motor_delay_steps=self.config.perturb_motor_delay_steps,
            sensor_noise_std=self.config.perturb_sensor_noise_std,
        )

        data = data.replace(qpos=qpos, qvel=qvel, xfrc_applied=xfrc, ctrl=jnp.clip(delayed_action, -1.0, 1.0))

        def _substep(d, _):
            return mjx.step(mjx_model, d), None
        next_data, _ = jax.lax.scan(_substep, data, None, length=self.config.physics_steps_per_control)

        new_obs = self._build_obs(next_data, state.goal)
        reward = self._compute_reward(next_data, state.mjx_data, delayed_action, state.goal)
        new_step = state.step_count + 1
        done = self._check_termination(next_data, new_step)

        pt = jnp.maximum(state.info["peak_torque"], compute_peak_torque(delayed_action))
        pcf = jnp.maximum(state.info["peak_contact_force"], jnp.sum(jnp.abs(next_data.qfrc_constraint)))

        info = {"episode_reward": state.info["episode_reward"] + reward, "peak_torque": pt, "peak_contact_force": pcf}

        return LowLevelFallState(
            obs=new_obs, reward=reward, done=done, mjx_data=next_data,
            mjx_model=mjx_model, info=info, perturbation_state=new_ps, step_count=new_step, goal=state.goal,
        )
