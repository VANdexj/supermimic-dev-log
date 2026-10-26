#!/bin/bash
# Foreground small-VRAM training. Arguments after the run name override the defaults.
set -euo pipefail
cd "$(dirname "$0")"
RUN=${1:-local_3060}
if [ "$#" -gt 0 ]; then shift; fi
if [[ ! "$RUN" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Run name must contain only letters, digits, underscores or hyphens." >&2
  exit 1
fi
RESUMING=0
for ARG in "$@"; do
  case "$ARG" in --resume|--resume=*) RESUMING=1 ;; esac
done
if [ -f "logs/$RUN/last.pt" ] && [ "$RESUMING" = 0 ]; then
  echo "Run already has a checkpoint. Use another name or --resume logs/$RUN/last.pt." >&2
  exit 1
fi
PROFILE=(--num_envs 64 --buffer_steps 256 --wm_batch 64 --wm_batches 2 --pi_batch 32)
if [ "${LOW_VRAM:-0}" = 1 ]; then
  PROFILE=(--num_envs 16 --buffer_steps 256 --wm_batch 16 --wm_batches 2 --pi_batch 8)
fi
docker exec supermimic mkdir -p "logs/$RUN"
docker exec supermimic python supermimic/train.py \
  "${PROFILE[@]}" --horizon 32 --wm_window 8 --wm_only_iters 500 \
  --ckpt_every 200 --log_every 50 --amp bf16 --no_compile \
  --max_train_seconds 7200 --log_dir "logs/$RUN" "$@" \
  2>&1 | tee -a "logs/$RUN/stdout.log"
