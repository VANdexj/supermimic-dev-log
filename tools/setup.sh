#!/bin/bash
# Clone Isaac Lab at the pinned commit and build the kit-less image + the SuperMimic image.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v docker >/dev/null; then
  echo "Docker is missing. Run: bash docker/install_runtime.sh" >&2
  exit 1
fi
docker info >/dev/null
ISAACLAB_COMMIT=46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
if [ ! -d IsaacLab ]; then
  git init IsaacLab
  git -C IsaacLab remote add origin https://github.com/isaac-sim/IsaacLab.git
fi
if ! git -C IsaacLab cat-file -e "$ISAACLAB_COMMIT^{commit}" 2>/dev/null; then
  git -C IsaacLab fetch --depth 1 origin "$ISAACLAB_COMMIT"
fi
git -C IsaacLab checkout "$ISAACLAB_COMMIT"
mkdir -p .cache
# Keep the pinned checkout untouched; constrain native builds in a generated Dockerfile.
sed '/^ENV UV_NO_CACHE=1$/a ENV UV_CONCURRENT_BUILDS=2 UV_CONCURRENT_INSTALLS=2 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2' \
  IsaacLab/docker/Dockerfile.kitless > .cache/Dockerfile.kitless
# Stable parent manifests keep repeated local builds from invalidating the runtime cache.
docker build --provenance=false -f .cache/Dockerfile.kitless -t isaac-lab-kitless:main-46d6cc3 IsaacLab/
docker build --provenance=false -t supermimic:latest docker/
