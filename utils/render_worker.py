"""Asynchronous rendering worker (CPU process).

Runs in a separate ``multiprocessing.Process`` so that rendering
**never blocks GPU training**.

Workflow
--------
1. Training process saves a checkpoint and puts its path into a
   ``multiprocessing.Queue``.
2. This worker picks up the path, loads the policy, runs a short
   evaluation rollout on **CPU MuJoCo** (not MJX), and renders
   the result with ``mujoco.viewer.launch_passive``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time
from typing import Optional

import numpy as np


def _render_loop(
    queue: mp.Queue,
    scene_xml_path: str,
    num_actions: int,
    episode_max_steps: int,
    hidden_sizes: tuple = (512, 256, 128),
    obs_dim: int = 56,
    headless: bool = False,
):
    """Main loop executed in the render worker process.

    This function **imports MuJoCo inside the subprocess** to avoid
    GPU-context issues.
    """
    import mujoco
    import jax
    import jax.numpy as jnp

    # Force JAX to CPU in this process
    jax.config.update("jax_platform_name", "cpu")

    from agents.ppo.networks import PPONetworks

    mj_model = mujoco.MjModel.from_xml_path(scene_xml_path)
    mj_data = mujoco.MjData(mj_model)

    print("[RenderWorker] Waiting for checkpoint paths …")

    while True:
        # Block until a checkpoint path arrives (or sentinel None)
        msg = queue.get()
        if msg is None:
            print("[RenderWorker] Received shutdown signal.")
            break

        ckpt_path: str = msg
        if not os.path.isfile(ckpt_path):
            print(f"[RenderWorker] Checkpoint not found: {ckpt_path}")
            continue

        print(f"[RenderWorker] Loading checkpoint: {ckpt_path}")
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        params = ckpt["params"]

        # Build network on CPU
        networks = PPONetworks(
            obs_size=obs_dim,
            action_size=num_actions,
            hidden_sizes=hidden_sizes,
        )
        rng = jax.random.PRNGKey(42)

        # Run rollout on CPU MuJoCo
        mujoco.mj_resetData(mj_model, mj_data)
        frames = []

        if not headless:
            try:
                viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
            except Exception:
                viewer = None
        else:
            viewer = None

        for step_i in range(episode_max_steps):
            # Build observation (simplified – matches safefall obs builder)
            obs = _build_obs_from_mjdata(mj_model, mj_data)
            obs_jnp = jnp.array(obs)

            # Policy forward pass (deterministic)
            mean, _ = networks.policy.apply(params["policy"], obs_jnp)
            action = np.array(mean)

            # Set ctrl and step
            mj_data.ctrl[:] = np.clip(action, -1.0, 1.0)
            mujoco.mj_step(mj_model, mj_data)

            if viewer is not None:
                viewer.sync()
                time.sleep(mj_model.opt.timestep)

        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass

        print(f"[RenderWorker] Finished rendering episode (ckpt step {ckpt.get('step', '?')}).")


def _build_obs_from_mjdata(mj_model, mj_data) -> np.ndarray:
    """Build a flat observation vector from CPU MuJoCo data.

    Mirrors the observation construction in the MJX environments but
    using NumPy / MuJoCo-C data structures.
    """
    qpos = mj_data.qpos.copy()
    qvel = mj_data.qvel.copy()
    sensordata = mj_data.sensordata.copy()

    # Joint positions (skip free-joint 7 DOFs)
    joint_pos = qpos[7:]
    joint_vel = qvel[6:]

    # Sensor slices matching envs/safefall_op3.py
    def _ss(name: str):
        sid = mj_model.sensor(name).id
        adr = int(mj_model.sensor_adr[sid])
        dim = int(mj_model.sensor_dim[sid])
        return adr, adr + dim

    accel_adr = _ss("torso_accel")
    gyro_adr = _ss("torso_gyro")
    quat_adr = _ss("torso_quat")
    linvel_adr = _ss("torso_linvel")
    angvel_adr = _ss("torso_angvel")

    accel = sensordata[accel_adr[0]:accel_adr[1]]
    gyro = sensordata[gyro_adr[0]:gyro_adr[1]]
    quat = sensordata[quat_adr[0]:quat_adr[1]]
    linvel = sensordata[linvel_adr[0]:linvel_adr[1]]
    angvel = sensordata[angvel_adr[0]:angvel_adr[1]]

    # Match training observation (56 dims)
    obs = np.concatenate([
        joint_pos,
        joint_vel,
        accel,
        gyro,
        quat,
        linvel,
        angvel,
    ])
    return obs.astype(np.float32)


# ── Public API ───────────────────────────────────────────────────────


def start_render_worker(
    scene_xml_path: str,
    num_actions: int = 20,
    episode_max_steps: int = 1000,
    obs_dim: int = 56,
    hidden_sizes: tuple = (512, 256, 128),
    headless: bool = False,
) -> tuple:
    """Spawn the rendering subprocess.

    Returns:
        ``(process, queue)`` – put checkpoint paths into *queue*;
        put ``None`` to shut down.
    """
    queue: mp.Queue = mp.Queue(maxsize=4)
    proc = mp.Process(
        target=_render_loop,
        args=(queue, scene_xml_path, num_actions, episode_max_steps,
              hidden_sizes, obs_dim, headless),
        daemon=True,
    )
    proc.start()
    return proc, queue
