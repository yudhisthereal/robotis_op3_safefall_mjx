"""High-Level Fall Strategy Selector (MJX).

A PPO policy that **selects a fall strategy** from a discrete set.  The
chosen strategy is then executed by the low-level goal-conditioned
controller (``envs/op3_low_level_fall.py``).

Action space: ``Discrete(NUM_STRATEGIES)`` – output is a strategy index
    0 – arm_bracing
    1 – roll
    2 – squat_drop_forward
    3 – squat_drop_backward
    4 – side_collapse

The high-level policy observes the robot state (same as SafeFall obs)
and produces a categorical distribution over strategies.  At each
*decision point* it selects a strategy that is passed to the low-level
controller which then runs for the remainder of the episode.

Reward: strategy-conditioned low-level proxy control with uprightness,
impact, and effort shaping.

Shares perturbation / DR / termination / metrics systems.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from envs.op3_low_level_fall import NUM_STRATEGIES
from utils.config import Config
from utils.perturbations import (
    PerturbationState,
    init_perturbation_state,
)


# ── State ────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class HighLevelFallState:
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    mjx_data: mjx.Data
    info: dict
    perturbation_state: PerturbationState
    step_count: jnp.ndarray
    selected_strategy: jnp.ndarray  # scalar int


# ── Environment ──────────────────────────────────────────────────────


class OP3HighLevelFallEnv:
    """High-level strategy selector trained with a continuous PPO head.

    The policy outputs strategy preferences; the selected strategy is
    converted into a low-level action template and executed for one
    control interval.
    """

    def __init__(
        self,
        config: Config,
        mj_model: mujoco.MjModel,
    ):
        self.config = config
        self.mj_model = mj_model

        self.torso_body_idx = mj_model.body("body_link").id
        self.l_foot_body_idx = mj_model.body("l_ank_roll_link").id
        self.r_foot_body_idx = mj_model.body("r_ank_roll_link").id
        self.foot_body_indices = (self.l_foot_body_idx, self.r_foot_body_idx)

        # Sensor slices (same as SafeFall)
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
    def obs_dim(self) -> int:
        """Same as SafeFall robot-only observation."""
        return 20 + 20 + 3 + 3 + 4 + 3 + 3  # 56

    @property
    def num_strategies(self) -> int:
        return NUM_STRATEGIES

    @property
    def action_dim(self) -> int:
        """High-level policy outputs logits/preferences over strategies."""
        return NUM_STRATEGIES

    # ── observation ──────────────────────────────────────────────────

    def _build_obs(self, data: mjx.Data) -> jnp.ndarray:
        joint_pos = data.qpos[7:]
        joint_vel = data.qvel[6:]
        sd = data.sensordata
        return jnp.concatenate([
            joint_pos, joint_vel,
            sd[self._accel_adr[0]:self._accel_adr[1]],
            sd[self._gyro_adr[0]:self._gyro_adr[1]],
            sd[self._quat_adr[0]:self._quat_adr[1]],
            sd[self._linvel_adr[0]:self._linvel_adr[1]],
            sd[self._angvel_adr[0]:self._angvel_adr[1]],
        ])

    def _strategy_to_action(self, strategy_idx: jnp.ndarray) -> jnp.ndarray:
        """Map a discrete strategy id to a low-level action template."""
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
        return templates[strategy_idx]

    # ── termination (shared) ─────────────────────────────────────────

    def _check_termination(self, data: mjx.Data, step_count: jnp.ndarray) -> jnp.ndarray:
        torso_z = data.qpos[2]
        fallen = torso_z < 0.05
        timeout = step_count >= self.config.episode_max_steps
        oob = jnp.any(jnp.abs(data.qpos[0:2]) > 5.0)
        nan_det = jnp.any(jnp.isnan(data.qpos)) | jnp.any(jnp.isnan(data.qvel))
        return jnp.any(jnp.array([fallen, timeout, oob, nan_det])).astype(jnp.float32)

    # ── reset ────────────────────────────────────────────────────────

    def reset(self, mjx_model: mjx.Model, rng: jax.Array) -> HighLevelFallState:
        rng, rng_perturb = jax.random.split(rng)
        data = mjx.make_data(mjx_model)
        data = mjx.step(mjx_model, data)
        obs = self._build_obs(data)
        ps = init_perturbation_state(
            self.config.num_actions, self.config.perturb_motor_delay_steps, rng_perturb,
        )
        return HighLevelFallState(
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            mjx_data=data,
            info={
                "episode_reward": jnp.float32(0.0),
                "peak_torque": jnp.float32(0.0),
                "peak_contact_force": jnp.float32(0.0),
            },
            perturbation_state=ps,
            step_count=jnp.int32(0),
            selected_strategy=jnp.int32(-1),
        )

    # ── step (one strategy selection per call) ───────────────────────

    def step(
        self,
        mjx_model: mjx.Model,
        state: HighLevelFallState,
        action: jnp.ndarray,
        rng: jax.Array,
    ) -> HighLevelFallState:
        """Execute one high-level strategy selection and low-level proxy step."""
        # Discretise continuous action → strategy index
        strategy_idx = jnp.argmax(action[:NUM_STRATEGIES])

        data = state.mjx_data
        low_level_action = self._strategy_to_action(strategy_idx)
        data = data.replace(ctrl=jnp.clip(low_level_action, -1.0, 1.0))

        def _substep(d, _):
            return mjx.step(mjx_model, d), None

        next_data, _ = jax.lax.scan(_substep, data, None, length=self.config.physics_steps_per_control)

        obs = self._build_obs(next_data)
        new_step = state.step_count + 1
        done = self._check_termination(next_data, new_step)

        torso_z = next_data.qpos[2]
        torso_quat = next_data.sensordata[self._quat_adr[0]:self._quat_adr[1]]
        upright = jnp.clip(jnp.abs(torso_quat[0]), 0.0, 1.0)
        impact_penalty = -0.0002 * jnp.mean(jnp.square(next_data.qfrc_constraint))
        effort_penalty = -0.001 * jnp.mean(jnp.square(low_level_action))
        alive = jnp.where(torso_z > 0.05, 0.2, -0.8)
        reward = 0.8 * upright + 0.4 * jnp.clip(torso_z / 0.3, 0.0, 1.2) + impact_penalty + effort_penalty + alive

        info = {
            "episode_reward": state.info["episode_reward"] + reward,
            "peak_torque": jnp.maximum(state.info["peak_torque"], jnp.max(jnp.abs(low_level_action))),
            "peak_contact_force": jnp.maximum(
                state.info["peak_contact_force"],
                jnp.sum(jnp.abs(next_data.qfrc_constraint)),
            ),
        }

        return HighLevelFallState(
            obs=obs,
            reward=reward,
            done=done,
            mjx_data=next_data,
            info=info,
            perturbation_state=state.perturbation_state,
            step_count=new_step,
            selected_strategy=strategy_idx,
        )
