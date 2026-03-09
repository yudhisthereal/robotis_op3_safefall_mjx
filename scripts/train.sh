#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# OP3 SafeFall MJX Training Script
# ──────────────────────────────────────────────────────────────────────
# Usage:
#   bash scripts/train.sh                       # defaults
#   ENV=op3_low_level_fall bash scripts/train.sh  # override env
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults (override via environment variables) ────────────────────
ENV="${ENV:-safefall_op3}"
NUM_ENVS="${NUM_ENVS:-1024}"
DEVICE="${DEVICE:-gpu}"
SEED="${SEED:-0}"
LR="${LR:-3e-4}"
ROLLOUT="${ROLLOUT:-256}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MB="${NUM_MB:-4}"
WANDB_PROJECT="${WANDB_PROJECT:-op3-safefall-mjx}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ── Run ──────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  OP3 SafeFall MJX Training                              ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Env          : ${ENV}"
echo "║  Num envs     : ${NUM_ENVS}"
echo "║  Device       : ${DEVICE}"
echo "║  Seed         : ${SEED}"
echo "║  LR           : ${LR}"
echo "║  Rollout len  : ${ROLLOUT}"
echo "║  Update epochs: ${UPDATE_EPOCHS}"
echo "║  Minibatches  : ${NUM_MB}"
echo "╚══════════════════════════════════════════════════════════╝"

python run.py \
    --env "${ENV}" \
    --num_envs "${NUM_ENVS}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --lr "${LR}" \
    --rollout_length "${ROLLOUT}" \
    --update_epochs "${UPDATE_EPOCHS}" \
    --num_minibatches "${NUM_MB}" \
    --wandb_project "${WANDB_PROJECT}" \
    --checkpoint_dir "${CKPT_DIR}" \
    ${EXTRA_ARGS}
