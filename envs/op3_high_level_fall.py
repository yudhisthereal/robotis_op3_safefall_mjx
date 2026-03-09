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

Reward (placeholder): The high-level agent's reward is the **total
episode reward accumulated by the low-level controller**.

Shares perturbation / DR / termination / metrics systems.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from envs.op3_low_level_fall import NUM_STRATEGIES, OP3LowLevelFallEnv, LowLevelFallState
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
    """High-level strategy selector using discrete PPO.

    This environment is a **wrapper** around :class:`OP3LowLevelFallEnv`.
    The high-level policy selects a strategy once per episode; the
    low-level policy then runs for ``episode_max_steps`` time-steps.

    For training the high-level policy, we:
    1. Reset the low-level env with the chosen strategy.
    2. Execute the low-level policy for the entire episode.
    3. Return the accumulated reward to the high-level policy.

    This file exposes a simplified ``reset`` / ``step`` API for the
    outer training loop, where ``step`` is called **once per episode**
    with the strategy index as the action.
    """

    def __init__(
        self,
        config: Config,
        mj_model: mujoco.MjModel,
        low_level_env: OP3LowLevelFallEnv | None = None,
    ):
        self.config = config
        self.mj_model = mj_model
        self.low_level_env = low_level_env or OP3LowLevelFallEnv(config, mj_model)

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
        """Execute one *high-level* step.

        ``action`` is a **strategy index** (integer or one-hot logits).
        We treat it as the argmax of a continuous action vector to keep
        the API compatible with the PPO agent (continuous output that is
        discretised here).

        The reward returned is a placeholder; in the full hierarchical
        pipeline the low-level episode reward would be fed back here.
        """
        # Discretise continuous action → strategy index
        strategy_idx = jnp.argmax(action[:NUM_STRATEGIES])

        # For now, we just run one physics step as a proxy – the full
        # hierarchical pipeline would unroll the low-level controller.
        data = state.mjx_data
        # Zero-action step (high-level doesn't directly control joints)
        data = data.replace(ctrl=jnp.zeros(self.config.num_actions))
        next_data = mjx.step(mjx_model, data)

        obs = self._build_obs(next_data)
        new_step = state.step_count + 1
        done = self._check_termination(next_data, new_step)

        # Placeholder reward – in full pipeline this comes from low-level
        torso_z = next_data.qpos[2]
        reward = torso_z / 0.3 + jnp.where(torso_z > 0.05, 1.0, 0.0)

        info = {
            "episode_reward": state.info["episode_reward"] + reward,
            "peak_torque": state.info["peak_torque"],
            "peak_contact_force": state.info["peak_contact_force"],
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
