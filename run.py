#!/usr/bin/env python3
"""Main entry-point for OP3 SafeFall MJX training.

Usage::

    python run.py --agent ppo --env safefall_op3
    python run.py --env op3_low_level_fall --num_envs 512
    python run.py --env op3_high_level_fall --num_envs 256

The script:
1. Parses CLI arguments and builds a :class:`Config`.
2. Loads the MuJoCo / MJX model.
3. Initialises the chosen environment and PPO agent.
4. Optionally spawns an async rendering worker.
5. Runs the JIT-compiled training loop with WandB logging.
6. Saves checkpoints and tracks the best model.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

# Force JAX to use GPU if available
jax.config.update("jax_default_matmul_precision", "float32")

import wandb

from utils.config import Config, make_config
from utils.env import load_mjx_model, make_rng_keys
from utils.checkpoint import save_checkpoint, BestModelTracker
from utils.metrics import aggregate_episode_metrics, batch_peak_torque
from utils.render_worker import start_render_worker
from utils.replay_buffer import RolloutBuffer

from agents.ppo.ppo import PPOAgent

from envs.safefall_op3 import SafeFallOP3Env
from envs.op3_low_level_fall import OP3LowLevelFallEnv
from envs.op3_high_level_fall import OP3HighLevelFallEnv


# ── Environment registry ─────────────────────────────────────────────

ENV_REGISTRY = {
    "safefall_op3": SafeFallOP3Env,
    "op3_low_level_fall": OP3LowLevelFallEnv,
    "op3_high_level_fall": OP3HighLevelFallEnv,
}


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OP3 SafeFall MJX Training")
    p.add_argument("--agent", type=str, default="ppo", choices=["ppo"])
    p.add_argument("--env", type=str, default="safefall_op3", choices=list(ENV_REGISTRY))
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rollout_length", type=int, default=256)
    p.add_argument("--update_epochs", type=int, default=4)
    p.add_argument("--num_minibatches", type=int, default=4)
    p.add_argument("--wandb_project", type=str, default="op3-safefall-mjx")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--no_render", action="store_true", help="Disable async render worker")
    p.add_argument("--headless", action="store_true", help="Render worker runs headless")
    p.add_argument("--episode_max_steps", type=int, default=1000)
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # ── Config ───────────────────────────────────────────────────────
    config = make_config(
        env_name=args.env,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
        learning_rate=args.lr,
        rollout_length=args.rollout_length,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        checkpoint_dir=args.checkpoint_dir,
        episode_max_steps=args.episode_max_steps,
    )

    print(f"[run.py] Environment : {config.env_name}")
    print(f"[run.py] Num envs    : {config.num_envs}")
    print(f"[run.py] Device      : {config.device}")
    print(f"[run.py] JAX devices : {jax.devices()}")

    # ── Model ────────────────────────────────────────────────────────
    mj_model, mjx_model = load_mjx_model(config)
    print(f"[run.py] Loaded MJX model — nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}")

    # ── Environment ──────────────────────────────────────────────────
    EnvClass = ENV_REGISTRY[config.env_name]
    env = EnvClass(config, mj_model)
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    print(f"[run.py] obs_dim={obs_dim}, action_dim={action_dim}")

    # ── Agent ────────────────────────────────────────────────────────
    agent = PPOAgent(config, obs_dim, action_dim)
    rng = jax.random.PRNGKey(config.seed)
    train_state = agent.init(rng)
    print(f"[run.py] PPO agent initialised.")

    # ── Vectorised env functions ─────────────────────────────────────
    v_reset = jax.jit(jax.vmap(env.reset, in_axes=(None, 0)))
    v_step = jax.jit(jax.vmap(env.step, in_axes=(None, 0, 0, 0)))

    # ── WandB ────────────────────────────────────────────────────────
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        config=vars(args),
        name=f"{config.env_name}_{config.num_envs}envs_{int(time.time())}",
    )
    run_id = wandb.run.id if wandb.run else "local"

    # ── Checkpointing ────────────────────────────────────────────────
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_tracker = BestModelTracker(config.checkpoint_dir, run_id=run_id)

    # ── Async render worker ──────────────────────────────────────────
    render_proc, render_queue = None, None
    if not args.no_render:
        render_proc, render_queue = start_render_worker(
            scene_xml_path=config.scene_xml_path,
            num_actions=action_dim,
            episode_max_steps=config.episode_max_steps,
            obs_dim=obs_dim,
            hidden_sizes=config.hidden_sizes,
            headless=args.headless,
        )
        print("[run.py] Render worker started.")

    # ── Training loop ────────────────────────────────────────────────
    rng, rng_reset = jax.random.split(train_state.rng)
    env_keys = jax.random.split(rng_reset, config.num_envs)
    states = v_reset(mjx_model, env_keys)

    global_step = 0
    global_episode = 0
    reward_history: deque = deque(maxlen=config.plateau_window_episodes)
    update_count = 0

    print("[run.py] Starting training loop …")
    t_start = time.time()

    try:
        while True:
            # ── Collect rollout ──────────────────────────────────────
            buffer = RolloutBuffer.create(
                config.num_envs, config.rollout_length, obs_dim, action_dim,
            )

            rng, rng_rollout = jax.random.split(rng)
            rollout_keys = jax.random.split(rng_rollout, config.rollout_length)

            episode_rewards_accum = jnp.zeros(config.num_envs)
            episode_lengths_accum = jnp.zeros(config.num_envs, dtype=jnp.int32)
            peak_torques_accum = jnp.zeros(config.num_envs)
            peak_cforces_accum = jnp.zeros(config.num_envs)

            for t in range(config.rollout_length):
                obs = states.obs
                rng_act = rollout_keys[t]

                # Get actions and values
                actions, log_probs = agent.get_action(train_state.params, obs, rng_act)
                values = agent.get_value(train_state.params, obs)

                # Step environments
                step_keys = jax.random.split(rollout_keys[t], config.num_envs)
                next_states = v_step(mjx_model, states, actions, step_keys)

                # Store transition
                buffer = buffer.store(
                    t, obs, actions, next_states.reward, values, log_probs, next_states.done,
                )

                # Track per-env metrics
                episode_rewards_accum = episode_rewards_accum + next_states.reward
                episode_lengths_accum = episode_lengths_accum + 1
                peak_torques_accum = jnp.maximum(peak_torques_accum, batch_peak_torque(actions))

                # Auto-reset done environments
                rng, rng_autoreset = jax.random.split(rng)
                done_mask = next_states.done > 0.5
                any_done = jnp.any(done_mask)

                # Count completed episodes
                n_done = jnp.sum(done_mask).astype(jnp.int32)
                global_episode += int(n_done)

                # Record finished episode rewards
                if int(any_done):
                    finished_rewards = jnp.where(done_mask, episode_rewards_accum, 0.0)
                    for r in finished_rewards[done_mask]:
                        reward_history.append(float(r))
                    # Reset accumulators for done envs
                    episode_rewards_accum = jnp.where(done_mask, 0.0, episode_rewards_accum)
                    episode_lengths_accum = jnp.where(done_mask, 0, episode_lengths_accum)

                # Reset done environments
                reset_keys = jax.random.split(rng_autoreset, config.num_envs)
                fresh_states = v_reset(mjx_model, reset_keys)
                # Merge: use fresh state where done, keep next_state otherwise
                states = jax.tree.map(
                    lambda fresh, cont: jnp.where(
                        done_mask.reshape((-1,) + (1,) * (fresh.ndim - 1)), fresh, cont
                    ),
                    fresh_states,
                    next_states,
                )

                global_step += config.num_envs

            # ── PPO update ───────────────────────────────────────────
            last_values = agent.get_value(train_state.params, states.obs)
            train_state, update_metrics = agent.update(train_state, buffer, last_values)
            update_count += 1

            # ── Logging ──────────────────────────────────────────────
            if update_count % config.log_interval == 0:
                elapsed = time.time() - t_start
                sps = global_step / max(elapsed, 1e-6)
                mean_reward = float(np.mean(list(reward_history))) if reward_history else 0.0

                log_data = {
                    "episode_reward": mean_reward,
                    "policy_loss": float(update_metrics["policy_loss"]),
                    "value_loss": float(update_metrics["value_loss"]),
                    "entropy": float(update_metrics["entropy"]),
                    "learning_rate": float(update_metrics["learning_rate"]),
                    "global_step": global_step,
                    "global_episode": global_episode,
                    "sps": sps,
                    "update": update_count,
                }
                wandb.log(log_data, step=global_step)
                print(
                    f"[{update_count:>6}] step={global_step:>10,}  ep={global_episode:>8,}  "
                    f"rew={mean_reward:>8.2f}  sps={sps:>10,.0f}  "
                    f"ploss={float(update_metrics['policy_loss']):>8.4f}  "
                    f"vloss={float(update_metrics['value_loss']):>8.4f}  "
                    f"ent={float(update_metrics['entropy']):>7.4f}"
                )

            # ── Checkpointing ────────────────────────────────────────
            if global_episode > 0 and global_episode % config.checkpoint_interval_episodes < config.num_envs:
                ckpt_path = save_checkpoint(
                    train_state.params,
                    train_state.opt_state,
                    step=global_step,
                    episode=global_episode,
                    reward=float(np.mean(list(reward_history))) if reward_history else 0.0,
                    checkpoint_dir=config.checkpoint_dir,
                )
                print(f"[run.py] Checkpoint saved: {ckpt_path}")

                # Best model tracking
                mean_rew = float(np.mean(list(reward_history))) if reward_history else 0.0
                if best_tracker.update(
                    train_state.params, train_state.opt_state,
                    global_step, global_episode, mean_rew,
                ):
                    print(f"[run.py] New best model! reward={mean_rew:.4f}")

                # Send checkpoint to render worker
                if render_queue is not None:
                    render_queue.put(ckpt_path)

            # ── Plateau stopping ─────────────────────────────────────
            if len(reward_history) >= config.plateau_window_episodes:
                recent = list(reward_history)
                half = len(recent) // 2
                old_mean = float(np.mean(recent[:half]))
                new_mean = float(np.mean(recent[half:]))
                improvement = abs(new_mean - old_mean) / (abs(old_mean) + 1e-8)
                if improvement < config.plateau_improvement_threshold:
                    print(
                        f"[run.py] Plateau detected: improvement={improvement:.4f} "
                        f"< threshold={config.plateau_improvement_threshold}. Stopping."
                    )
                    break

    except KeyboardInterrupt:
        print("\n[run.py] Training interrupted by user.")

    # ── Cleanup ──────────────────────────────────────────────────────
    # Final checkpoint
    final_path = save_checkpoint(
        train_state.params, train_state.opt_state,
        step=global_step, episode=global_episode,
        reward=float(np.mean(list(reward_history))) if reward_history else 0.0,
        checkpoint_dir=config.checkpoint_dir,
        filename="final.pkl",
    )
    print(f"[run.py] Final checkpoint: {final_path}")

    if render_queue is not None:
        render_queue.put(None)  # shutdown signal
    if render_proc is not None:
        render_proc.join(timeout=5)

    wandb.finish()
    print("[run.py] Done.")


if __name__ == "__main__":
    main()
