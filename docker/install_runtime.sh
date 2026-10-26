#!/bin/bash
# Host infrastructure only. All project dependencies are installed by docker build.
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then
  exec sudo bash "$0" "$@"
fi
source /etc/os-release
if [ "$ID" != ubuntu ] || [ "$(dpkg --print-architecture)" != amd64 ]; then
  echo "This installer targets Ubuntu amd64." >&2
  exit 1
fi
nvidia-smi
apt-get update
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $VERSION_CODENAME
Components: stable
Architectures: amd64
Signed-By: /etc/apt/keyrings/docker.asc
EOF
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --yes -o /etc/apt/keyrings/nvidia-container-toolkit.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin nvidia-container-toolkit
if [ -f /etc/docker/daemon.json ] && [ ! -f /etc/docker/daemon.json.before-supermimic ]; then
  cp -p /etc/docker/daemon.json /etc/docker/daemon.json.before-supermimic
fi
nvidia-ctk runtime configure --runtime=docker
systemctl enable --now docker
systemctl restart docker
if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ]; then
  usermod -aG docker "$SUDO_USER"
fi
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
echo "Runtime ready. Use a new login, or: sg docker -c './tools/reproduce.sh local_3060'"
