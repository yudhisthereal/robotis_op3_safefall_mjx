"""Configuration module for OP3 SafeFall MJX training.

Central configuration dataclass holding all hyperparameters, environment
settings, and training options.  Uses JAX-friendly defaults throughout.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional, Tuple


@dataclasses.dataclass(frozen=True)
class Config:
    """Immutable project-wide configuration."""

    # ── Environment ──────────────────────────────────────────────────
    env_name: str = "safefall_op3"
    num_envs: int = 1024
    episode_max_steps: int = 1000
    sim_dt: float = 0.002
    control_dt: float = 0.01  # 100 Hz control
    physics_steps_per_control: int = 5  # sim_dt * 5 = control_dt

    # ── Assets ───────────────────────────────────────────────────────
    asset_dir: str = os.path.join(os.path.dirname(__file__), "..", "envs", "assets")
    scene_xml: str = "scene.xml"

    # ── OP3 robot constants ──────────────────────────────────────────
    num_joints: int = 20  # 20 actuated DOF
    num_actions: int = 20
    # Joint ordering matching the MJCF actuator list:
    joint_names: Tuple[str, ...] = (
        "head_pan", "head_tilt",
        "l_sho_pitch", "l_sho_roll", "l_el",
        "r_sho_pitch", "r_sho_roll", "r_el",
        "l_hip_yaw", "l_hip_roll", "l_hip_pitch",
        "l_knee", "l_ank_pitch", "l_ank_roll",
        "r_hip_yaw", "r_hip_roll", "r_hip_pitch",
        "r_knee", "r_ank_pitch", "r_ank_roll",
    )

    # ── Observation dimensions (computed later, but defaults) ────────
    obs_size_safefall: int = 56  # robot_state only
    obs_size_low_level: int = 61  # robot_state (56) + one-hot goal (5)
    obs_size_high_level: int = 56  # same as safefall obs
    num_strategies: int = 5  # arm_bracing, roll, squat_fwd, squat_bwd, side

    # ── PPO ──────────────────────────────────────────────────────────
    agent: str = "ppo"
    learning_rate: float = 3e-4
    lr_schedule: str = "linear"  # linear | constant
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    num_minibatches: int = 4
    update_epochs: int = 4
    rollout_length: int = 256  # steps per rollout before PPO update
    normalize_advantages: bool = True

    # ── Network ──────────────────────────────────────────────────────
    hidden_sizes: Tuple[int, ...] = (512, 256, 128)
    activation: str = "tanh"
    init_log_std: float = -0.5

    # ── Training ─────────────────────────────────────────────────────
    total_timesteps: int = int(1e10)  # effectively infinite
    seed: int = 0
    device: str = "gpu"

    # Plateau stopping
    plateau_window_episodes: int = 50_000
    plateau_improvement_threshold: float = 0.05  # 5 %

    # ── Checkpointing ────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval_episodes: int = 1000

    # ── Logging ──────────────────────────────────────────────────────
    wandb_project: str = "op3-safefall-mjx"
    wandb_entity: Optional[str] = None
    log_interval: int = 10  # updates between WandB logs
    eval_interval: int = 50  # updates between deterministic evaluations
    num_eval_episodes: int = 16

    # ── Rendering ────────────────────────────────────────────────────
    render_interval_episodes: int = 1000
    render_width: int = 640
    render_height: int = 480

    # ── Perturbations ────────────────────────────────────────────────
    perturb_external_push_prob: float = 0.1
    perturb_external_push_force_range: Tuple[float, float] = (5.0, 20.0)
    perturb_foot_slip_prob: float = 0.05
    perturb_foot_slip_vel_range: Tuple[float, float] = (0.1, 0.5)
    perturb_foot_trip_prob: float = 0.05
    perturb_joint_noise_std: float = 0.02
    perturb_motor_delay_steps: int = 2
    perturb_sensor_noise_std: float = 0.01

    # ── Domain Randomization ─────────────────────────────────────────
    dr_mass_range: Tuple[float, float] = (0.8, 1.2)
    dr_damping_range: Tuple[float, float] = (0.7, 1.3)
    dr_friction_range: Tuple[float, float] = (0.5, 1.5)
    dr_sensor_noise_scale_range: Tuple[float, float] = (0.5, 2.0)
    dr_ground_friction_range: Tuple[float, float] = (0.4, 1.6)
    dr_contact_solref_scale_range: Tuple[float, float] = (0.8, 1.2)
    dr_contact_solimp_scale_range: Tuple[float, float] = (0.9, 1.1)

    # ── Derived helpers ──────────────────────────────────────────────
    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_length

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def scene_xml_path(self) -> str:
        return os.path.join(self.asset_dir, self.scene_xml)


def make_config(**overrides) -> Config:
    """Create a Config with optional keyword overrides."""
    return Config(**overrides)
