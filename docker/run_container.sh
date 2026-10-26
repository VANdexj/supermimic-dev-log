#!/bin/bash
# Start the long-lived 'supermimic' container with one GPU and the repository mounted.
# Usage: docker/run_container.sh [GPU_INDEX] [MEMORY_LIMIT] [SHM_SIZE]
set -euo pipefail
GPU=${1:-0}
MEM=${2:-10g}
SHM=${3:-2g}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO/.cache"
docker run -d --name supermimic --gpus "\"device=${GPU}\"" --memory="$MEM" --memory-swap="$MEM" \
  --shm-size="$SHM" --user "$(id -u):$(id -g)" \
  -e WARP_CACHE_PATH=/workspace/supermimic/.cache/warp -e MPLCONFIGDIR=/tmp/mpl \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e TORCHINDUCTOR_COMPILE_THREADS=1 \
  -e XDG_CACHE_HOME=/workspace/supermimic/.cache \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/supermimic/.cache/inductor \
  -v "$REPO":/workspace/supermimic -w /workspace/supermimic supermimic:latest sleep infinity
