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
import time
from collections import deque

# Reduce GPU memory fragmentation / preallocation pressure for large MJX graphs.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax
import jax.numpy as jnp
import numpy as np

# Keep matmul precision stable across backends
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "float32")

import wandb

from utils.config import Config, make_config
from utils.env import load_mjx_model
from utils.checkpoint import save_checkpoint, BestModelTracker
from utils.evaluator import evaluate
from utils.metrics import batch_peak_torque
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

    # Respect CLI device request before backend is first used.
    try:
        jax.config.update("jax_platform_name", args.device)
    except Exception as exc:  # pragma: no cover (depends on local backend)
        print(f"[run.py] Warning: failed to enforce device '{args.device}': {exc}")

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
    env = EnvClass(config, mj_model, mjx_model)
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    print(f"[run.py] obs_dim={obs_dim}, action_dim={action_dim}")

    # ── Agent ────────────────────────────────────────────────────────
    agent = PPOAgent(config, obs_dim, action_dim)
    rng = jax.random.PRNGKey(config.seed)
    train_state = agent.init(rng)
    print(f"[run.py] PPO agent initialised.")

    # ── Vectorised env functions ─────────────────────────────────────
    v_reset = jax.jit(jax.vmap(env.reset, in_axes=(0,)))
    v_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

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
    states = v_reset(env_keys)

    global_step = 0
    global_episode = 0
    reward_history: deque = deque(maxlen=config.plateau_window_episodes)
    length_history: deque = deque(maxlen=config.plateau_window_episodes)
    peak_torque_history: deque = deque(maxlen=config.plateau_window_episodes)
    peak_contact_history: deque = deque(maxlen=config.plateau_window_episodes)
    finished_count_total = 0
    finished_reward_total = 0.0
    finished_length_total = 0.0
    finished_peak_torque_total = 0.0
    finished_peak_contact_total = 0.0
    update_count = 0

    print("[run.py] Starting training loop …")
    t_start = time.time()

    has_goal = hasattr(states, "goal")
    goal_dim = int(states.goal.shape[-1]) if has_goal else 0

    @jax.jit
    def _train_iteration(train_state, states, rng):
        buffer = RolloutBuffer.create(
            config.num_envs,
            config.rollout_length,
            obs_dim,
            action_dim,
            goal_dim=goal_dim,
        )
        rollout_keys = jax.random.split(rng, config.rollout_length + 1)
        next_rng = rollout_keys[0]
        step_keys = rollout_keys[1:]

        episode_rewards_accum = jnp.zeros(config.num_envs)
        episode_lengths_accum = jnp.zeros(config.num_envs, dtype=jnp.int32)
        peak_torques_accum = jnp.zeros(config.num_envs)
        peak_cforces_accum = jnp.zeros(config.num_envs)

        def _scan_step(carry, xs):
            t, key = xs
            cur_states, cur_buffer, ep_rew, ep_len, ep_pt, ep_pcf = carry

            key_act, key_env, key_reset = jax.random.split(key, 3)
            obs = cur_states.obs
            actions, log_probs = agent.get_action(train_state.params, obs, key_act)
            values = agent.get_value(train_state.params, obs)

            env_step_keys = jax.random.split(key_env, config.num_envs)
            next_states = v_step(cur_states, actions, env_step_keys)

            goals = cur_states.goal if has_goal else None
            cur_buffer = cur_buffer.store(
                t,
                obs,
                actions,
                next_states.reward,
                values,
                log_probs,
                next_states.done,
                goals=goals,
            )

            ep_rew = ep_rew + next_states.reward
            ep_len = ep_len + 1
            ep_pt = jnp.maximum(ep_pt, batch_peak_torque(actions))
            ep_pcf = jnp.maximum(
                ep_pcf,
                jnp.sum(jnp.abs(next_states.mjx_data.qfrc_constraint), axis=-1),
            )

            done_mask = next_states.done > 0.5
            finished_rewards = jnp.where(done_mask, ep_rew, 0.0)
            finished_lengths = jnp.where(done_mask, ep_len, 0)
            finished_peak_torque = jnp.where(done_mask, ep_pt, 0.0)
            finished_peak_cforce = jnp.where(done_mask, ep_pcf, 0.0)

            reset_keys = jax.random.split(key_reset, config.num_envs)

            def _reset_if_done(xs):
                state_i, key_i, done_i = xs

                def _do_reset(_):
                    return env.reset(key_i)

                def _keep_state(_):
                    return state_i

                return jax.lax.cond(done_i, _do_reset, _keep_state, operand=None)

            merged_states = jax.lax.map(_reset_if_done, (next_states, reset_keys, done_mask))

            ep_rew = jnp.where(done_mask, 0.0, ep_rew)
            ep_len = jnp.where(done_mask, 0, ep_len)
            ep_pt = jnp.where(done_mask, 0.0, ep_pt)
            ep_pcf = jnp.where(done_mask, 0.0, ep_pcf)

            new_carry = (merged_states, cur_buffer, ep_rew, ep_len, ep_pt, ep_pcf)
            out = (
                jnp.sum(done_mask.astype(jnp.int32)),
                jnp.sum(finished_rewards),
                jnp.sum(finished_lengths.astype(jnp.float32)),
                jnp.sum(finished_peak_torque),
                jnp.sum(finished_peak_cforce),
            )
            return new_carry, out

        t_idx = jnp.arange(config.rollout_length, dtype=jnp.int32)
        (states, buffer, _, _, _, _), rollout_out = jax.lax.scan(
            _scan_step,
            (states, buffer, episode_rewards_accum, episode_lengths_accum, peak_torques_accum, peak_cforces_accum),
            (t_idx, step_keys),
        )

        done_count_t, finished_rewards_sum_t, finished_lengths_sum_t, finished_pt_sum_t, finished_pcf_sum_t = rollout_out
        total_done = jnp.sum(done_count_t)
        total_rewards = jnp.sum(finished_rewards_sum_t)
        total_lengths = jnp.sum(finished_lengths_sum_t)
        total_peak_torque = jnp.sum(finished_pt_sum_t)
        total_peak_cforce = jnp.sum(finished_pcf_sum_t)

        last_values = agent.get_value(train_state.params, states.obs)
        train_state, update_metrics = agent.update(train_state, buffer, last_values)

        stats = {
            "num_finished": total_done,
            "sum_reward_finished": total_rewards,
            "sum_length_finished": total_lengths,
            "sum_pt_finished": total_peak_torque,
            "sum_pcf_finished": total_peak_cforce,
        }
        return train_state, states, next_rng, update_metrics, stats

    try:
        while True:
            train_state, states, rng, update_metrics, stats = _train_iteration(train_state, states, rng)

            global_step += config.num_envs * config.rollout_length
            num_finished = int(stats["num_finished"])
            global_episode += num_finished

            if num_finished > 0:
                finished_count_total += num_finished
                finished_reward_total += float(stats["sum_reward_finished"])
                finished_length_total += float(stats["sum_length_finished"])
                finished_peak_torque_total += float(stats["sum_pt_finished"])
                finished_peak_contact_total += float(stats["sum_pcf_finished"])

                # Keep plateau logic functional with one scalar per update.
                reward_history.append(float(stats["sum_reward_finished"]) / num_finished)
                length_history.append(float(stats["sum_length_finished"]) / num_finished)
                peak_torque_history.append(float(stats["sum_pt_finished"]) / num_finished)
                peak_contact_history.append(float(stats["sum_pcf_finished"]) / num_finished)

            update_count += 1

            # ── Logging ──────────────────────────────────────────────
            if update_count % config.log_interval == 0:
                elapsed = time.time() - t_start
                sps = global_step / max(elapsed, 1e-6)
                if finished_count_total > 0:
                    mean_reward = finished_reward_total / finished_count_total
                    mean_length = finished_length_total / finished_count_total
                    mean_peak_torque = finished_peak_torque_total / finished_count_total
                    mean_peak_contact = finished_peak_contact_total / finished_count_total
                else:
                    mean_reward = 0.0
                    mean_length = 0.0
                    mean_peak_torque = 0.0
                    mean_peak_contact = 0.0

                log_data = {
                    "episode_reward": mean_reward,
                    "episode_length": mean_length,
                    "policy_loss": float(update_metrics["policy_loss"]),
                    "value_loss": float(update_metrics["value_loss"]),
                    "entropy": float(update_metrics["entropy"]),
                    "learning_rate": float(update_metrics["learning_rate"]),
                    "peak_torque": mean_peak_torque,
                    "peak_contact_force": mean_peak_contact,
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

            # ── Deterministic evaluation ────────────────────────────
            if update_count % config.eval_interval == 0:
                rng, rng_eval = jax.random.split(rng)
                eval_metrics = evaluate(
                    policy_apply_fn=agent.networks.policy.apply,
                    params=train_state.params["policy"],
                    reset_fn=env.reset,
                    step_fn=env.step,
                    rng=rng_eval,
                    config=config,
                    num_eval_episodes=config.num_eval_episodes,
                )
                wandb.log(
                    {
                        "eval/mean_reward": eval_metrics["mean_reward"],
                        "eval/episode_length": eval_metrics["episode_length"],
                        "eval/success_rate": eval_metrics["success_rate"],
                        "global_step": global_step,
                        "update": update_count,
                    },
                    step=global_step,
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
