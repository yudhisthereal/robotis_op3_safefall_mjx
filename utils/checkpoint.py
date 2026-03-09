"""Checkpoint save / load / best-model tracking.

Checkpoints are stored as msgpack-serialised Flax parameter dicts using
``flax.serialization``.  A ``best_<run_id>.ckpt`` is maintained alongside
periodic snapshots.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import jax.numpy as jnp

try:
    from flax import serialization
except ImportError:  # graceful fallback
    serialization = None


# ── Types ────────────────────────────────────────────────────────────

Params = Any  # Flax FrozenDict / dict tree


# ── Checkpoint I/O ───────────────────────────────────────────────────


def save_checkpoint(
    params: Params,
    opt_state: Any,
    step: int,
    episode: int,
    reward: float,
    checkpoint_dir: str,
    filename: Optional[str] = None,
) -> str:
    """Persist a training checkpoint to disk.

    Args:
        params: Flax parameter tree.
        opt_state: Optax optimiser state.
        step: global training step.
        episode: global episode count.
        reward: mean episode reward at this point.
        checkpoint_dir: directory to save into.
        filename: override filename (default: ``ckpt_{step}.pkl``).

    Returns:
        Absolute path to the saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    fname = filename or f"ckpt_{step}.pkl"
    path = os.path.join(checkpoint_dir, fname)

    ckpt = {
        "params": params,
        "opt_state": opt_state,
        "step": step,
        "episode": episode,
        "reward": reward,
    }
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    return path


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load a checkpoint dict from disk.

    Returns:
        Dictionary with keys ``params``, ``opt_state``, ``step``,
        ``episode``, ``reward``.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Best-model tracking ─────────────────────────────────────────────


class BestModelTracker:
    """Track and persist the best model seen so far.

    Attributes:
        best_reward: highest mean reward seen so far.
        best_path: path to the best checkpoint file.
    """

    def __init__(self, checkpoint_dir: str, run_id: str = "run"):
        self.checkpoint_dir = checkpoint_dir
        self.run_id = run_id
        self.best_reward: float = -float("inf")
        self.best_path: Optional[str] = None

    def update(
        self,
        params: Params,
        opt_state: Any,
        step: int,
        episode: int,
        reward: float,
    ) -> bool:
        """Save if *reward* beats the current best. Returns True on save."""
        if reward > self.best_reward:
            self.best_reward = reward
            fname = f"best_{self.run_id}.ckpt"
            self.best_path = save_checkpoint(
                params, opt_state, step, episode, reward,
                self.checkpoint_dir, filename=fname,
            )
            return True
        return False
