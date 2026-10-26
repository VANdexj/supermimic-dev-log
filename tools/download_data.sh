#!/bin/bash
# Download, validate and convert assets using only the container's Python.
set -euo pipefail
cd "$(dirname "$0")/.."
docker exec supermimic python tools/download_data.py
docker exec supermimic python tools/urdf_to_mjcf.py
docker exec supermimic python tools/mjcf_to_usd.py
