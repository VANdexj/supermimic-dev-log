#!/bin/bash
# Launch the bounded local reproduction controller in the existing Docker container.
set -euo pipefail
cd "$(dirname "$0")"
RUN=${1:-local_3060_v2}
if [ "$#" -gt 0 ]; then shift; fi
if [[ ! "$RUN" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo 'Run name must contain only letters, digits, underscores or hyphens.' >&2
  exit 1
fi
if docker exec supermimic tmux has-session -t "$RUN" 2>/dev/null; then
  echo "Controller session already exists: $RUN" >&2
  exit 1
fi
mkdir -p logs
printf -v COMMAND '%q ' python -u tools/retrain_local.py "$RUN" "$@"
docker exec supermimic tmux new-session -d -s "$RUN" \
  "exec $COMMAND >> logs/$RUN.controller.log 2>&1"
printf 'Started %s; log: logs/%s.controller.log; state: logs/%s/controller_state.json\n' "$RUN" "$RUN" "$RUN"
