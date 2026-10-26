#!/bin/bash
# Serial setup -> smoke/resume/budget checks -> two-hour training -> evaluation/rendering.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-local_3060}
if [[ ! "$RUN" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Run name must contain only letters, digits, underscores or hyphens." >&2
  exit 1
fi
if [ -e "logs/$RUN" ] || [ -e "logs/${RUN}_smoke" ]; then
  echo "Run already exists. Choose another name, or resume with run_local.sh and --resume logs/$RUN/last.pt." >&2
  exit 1
fi
tools/setup.sh
if ! docker inspect supermimic >/dev/null 2>&1; then
  docker/run_container.sh 0 10g 2g
else
  docker start supermimic >/dev/null
fi
docker exec supermimic nvidia-smi
docker exec supermimic python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print("bf16:", torch.cuda.is_bf16_supported())'
tools/download_data.sh
docker exec supermimic python tools/test_env.py --num_envs 16
docker exec supermimic python -m unittest discover -s tests -v
mkdir -p "logs/${RUN}_smoke" "logs/$RUN"
docker exec supermimic python -m pip freeze > "logs/$RUN/packages.txt"
docker inspect supermimic --format '{{.Image}}' > "logs/$RUN/image.txt"
docker exec supermimic nvidia-smi > "logs/$RUN/gpu.txt"
docker exec supermimic python -VV > "logs/$RUN/python_version.txt"
docker version > "logs/$RUN/docker_version.txt"
nvidia-ctk --version > "logs/$RUN/nvidia_toolkit_version.txt"
git -C IsaacLab rev-parse HEAD > "logs/$RUN/isaaclab_commit.txt"
cp data/source.json "logs/$RUN/data_source.json"
SMOKE=(--num_envs 16 --buffer_steps 64 --warmup_steps 64 --wm_batch 16 --pi_batch 8
       --wm_batches 1 --wm_only_iters 1 --no_compile --log_every 1 --ckpt_every 2)
docker exec supermimic python supermimic/train.py "${SMOKE[@]}" \
  --iterations 5 --log_dir "logs/${RUN}_smoke" 2>&1 | tee "logs/${RUN}_smoke/stdout.log"
docker exec supermimic python supermimic/train.py "${SMOKE[@]}" \
  --iterations 7 --log_dir "logs/${RUN}_smoke" --resume "logs/${RUN}_smoke/last.pt" \
  2>&1 | tee -a "logs/${RUN}_smoke/stdout.log"
# A tiny budget must save an off-interval checkpoint/log after the first resumed update.
docker exec supermimic python supermimic/train.py "${SMOKE[@]}" --ckpt_every 200 --log_every 50 \
  --iterations 100 --max_train_seconds 0.001 --log_dir "logs/${RUN}_smoke" \
  --resume "logs/${RUN}_smoke/last.pt" 2>&1 | tee -a "logs/${RUN}_smoke/stdout.log"
docker exec supermimic python tools/verify_run.py "logs/${RUN}_smoke" --expected_iteration 8
docker exec supermimic python tools/finish_run.py "logs/${RUN}_smoke" --out "results/${RUN}_smoke" --smoke
./run_local.sh "$RUN"
docker exec supermimic python tools/finish_run.py "logs/$RUN" --out "results/$RUN"
